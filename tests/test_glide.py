"""CPU contract checks for the independent GLIDE integration."""

import torch

from seahorse.models.event_models.glide import GLIDEEventModel
from seahorse.models.state_models.glide import GLIDEStateModel
from seahorse.models.configs.glide import GLIDEConfig
from seahorse.models.unified_model import UnifiedSTPP


def _model(*, timesteps=4):
    state = GLIDEStateModel(
        d_model=16, temporal_windows=(2,), spatial_windows=(2,), spatial_grid_sizes=(.1,), graph_layers=1
    )
    event = GLIDEEventModel(cond_dim=16, hidden_units=16, timesteps=timesteps)
    return UnifiedSTPP(state, event)


def _batch():
    times = torch.tensor([[0., 1., 3., 6.], [0., 2., 0., 0.]])
    locs = torch.tensor([[[0., 0.], [1., 2.], [2., 2.], [4., 3.]],
                         [[3., 1.], [4., 4.], [0., 0.], [0., 0.]]])
    return times, locs, torch.tensor([4, 2])


def test_glide_weighted_objective_backward_ema_and_leap_sample():
    torch.manual_seed(0)
    model = _model(); times, locs, lengths = _batch()
    out = model(times, locs, lengths)
    assert torch.isfinite(out["loss"])
    assert out["total_events"].item() == 4  # (4 - 1) + (2 - 1)
    out["loss"].backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step(); model.event_model.update_ema()
    ctx = model.state_model.encode_sampling_history(times=times, locations=locs, lengths=lengths)
    sample = model.event_model.sample_native(state=ctx)
    assert sample.shape == (2, 1, 3)
    assert torch.isfinite(sample).all()


def test_glide_vlb_is_explicit_diagnostic_and_not_nll_capability():
    model = _model(timesteps=3); times, locs, lengths = _batch()
    ctx = model.state_model.encode_history(times=times, locations=locs, lengths=lengths)
    out = model.event_model.eval_nll(times=times, locations=locs, lengths=lengths, state=ctx)
    assert "glide_transformed_vlb_diagnostic" in out["extra_metrics"]
    assert model.event_model.capabilities.nll_kind == "none"
    assert not model.event_model.capabilities.has_eval_nll


def test_glide_padding_and_future_suffix_do_not_change_prefix_conditions():
    state = _model().state_model
    prefix_t = torch.tensor([[0., 2., 5.]])
    prefix_x = torch.tensor([[[0., 0.], [1., 1.], [3., 1.]]])
    full_t = torch.tensor([[0., 2., 5., 9., 14.]], requires_grad=True)
    full_x = torch.tensor([[[0., 0.], [1., 1.], [3., 1.], [9., 4.], [2., 8.]]], requires_grad=True)
    prefix = state.encode_history(times=prefix_t, locations=prefix_x, lengths=torch.tensor([3])).payload
    padded = state.encode_history(times=torch.cat((prefix_t, torch.zeros(1, 2)), dim=1),
                                  locations=torch.cat((prefix_x, torch.zeros(1, 2, 2)), dim=1),
                                  lengths=torch.tensor([3])).payload
    extended = state.encode_history(times=full_t, locations=full_x, lengths=torch.tensor([5])).payload
    assert torch.allclose(prefix["glide_cond"][:, :2], padded["glide_cond"][:, :2])
    # The causal Transformer changes floating-point reduction order when a
    # longer masked sequence is supplied.  It produces sub-micro-unit drift in
    # forward values, but no computational path from a future suffix to a
    # prefix condition.  Test the semantic property directly with gradients.
    torch.testing.assert_close(prefix["glide_cond"][:, :2], extended["glide_cond"][:, :2], rtol=1e-5, atol=2e-6)
    grad_t, grad_x = torch.autograd.grad(extended["glide_cond"][:, :2].square().sum(), (full_t, full_x))
    assert torch.count_nonzero(grad_t[:, 3:]) == 0
    assert torch.count_nonzero(grad_x[:, 3:]) == 0


def test_glide_ema_is_checkpointed_and_token_units_are_explicit():
    model = _model(); times, locs, lengths = _batch()
    loss = model(times, locs, lengths)["loss"]
    loss.backward(); torch.optim.AdamW(model.parameters(), lr=1e-3).step(); model.event_model.update_ema()
    checkpoint = model.event_model.state_dict()
    restored = GLIDEEventModel(cond_dim=16, hidden_units=16, timesteps=4)
    restored.load_state_dict(checkpoint)
    assert any(key.startswith("ema.shadow_") for key in checkpoint)
    assert torch.equal(checkpoint["ema.shadow_0"], restored.state_dict()["ema.shadow_0"])
    state = GLIDEStateModel(d_model=16, temporal_windows=(2,), spatial_windows=(2,), spatial_grid_sizes=(.1,),
                            graph_layers=1, delta_min=10., delta_range=2., loc_min=(100., 200.), loc_range=(5., 10.))
    ctx = state.encode_history(times=torch.tensor([[0., 12.]]), locations=torch.tensor([[[100., 200.], [105., 210.]]]),
                               lengths=torch.tensor([2])).payload
    token = ctx["glide_tokens"][0, 0]
    assert torch.allclose(token, torch.ones(3))
    assert torch.equal(token[:1] * ctx["glide_token_delta_range"] + ctx["glide_token_delta_min"], torch.tensor([12.]))
    assert torch.equal(token[1:] * ctx["glide_token_loc_range"] + ctx["glide_token_loc_min"], torch.tensor([105., 210.]))


def test_glide_data_init_uses_train_split_only():
    class Dataset:
        normalize_time = normalize_space = False
        time_mean, time_std = 0., 1.
        loc_mean, loc_std = torch.zeros(2).numpy(), torch.ones(2).numpy()
        def __init__(self, seq): self.sequences = [seq]
    train = Dataset({"times": [1., 3., 6.], "locations": [[10., 20.], [20., 30.], [40., 50.]]})
    held_out = Dataset({"times": [1000., 2000.], "locations": [[-100., -100.], [999., 999.]]})
    class Bundle: train_dataset, val_dataset, test_dataset = train, held_out, held_out
    class DM: _bundle = Bundle()
    stats = GLIDEConfig.data_init_overrides(DM())
    assert stats["abs_time_min"] == 1.
    assert stats["abs_time_range"] == 5.
    assert stats["loc_min"] == (10., 20.)
    assert stats["loc_range"] == (30., 30.)


def test_glide_causal_feature_and_edge_reference_values():
    state = GLIDEStateModel(d_model=16, temporal_windows=(2,), spatial_windows=(2,), spatial_grid_sizes=(0.,),
                            graph_layers=1, num_time_harmonics=0, abs_time_range=3., delta_std=1.,
                            loc_range=(1., 1.), loc_delta_std=(1., 1.))
    features, _dt, loc = state._features(torch.tensor([0., 1., 3.]),
                                         torch.tensor([[0., 0.], [.2, .4], [.5, .4]]))
    # Source formulas (preprocess.py): causal mean/std/trend or displacement,
    # with trend/displacement divided by valid count.  No shifted target row.
    expected_second = torch.tensor([1/3, 1., .2, .4, .2, .4, .5, .5, .5, .1, .2, .1, .2, .1, .2])
    assert torch.allclose(features[1], expected_second, atol=1e-6)
    edge_index, edge_attr = state._graph(torch.tensor([0., .25]), torch.tensor([[0., 0.], [.5, 0.]]))
    assert torch.equal(edge_index, torch.tensor([[0], [1]]))
    assert torch.allclose(edge_attr[0], torch.tensor([
        .25 * torch.sigmoid(torch.tensor(-1.25)), .5 * torch.exp(torch.tensor(-.5)) * 1.2, 0.
    ]))
