"""Analytic limits, causality, quadrature, gradients and reporting for DNSK."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from seahorse.evaluation.likelihood import (
    compute_next_event_test_nll,
    _prefix_difference_next_event_nlls,
)
from seahorse.models.event_models.dnsk import DNSKEventModel
from seahorse.registry import build_model


def _model(**kwargs):
    torch.manual_seed(3)
    return build_model(
        preset="dnsk",
        config={
            "decoder": {
                "width": 8,
                "temporal_nodes": 17,
                "spatial_nodes": 12,
                "barrier_time_nodes": 2,
                "barrier_space_nodes": 3,
                **kwargs,
            }
        },
    )


def _batch():
    return {
        "times": torch.tensor([[0.1, 0.2, 0.35, 0.5], [0.2, 0.4, 0.0, 0.0]]),
        "locations": torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                [[0.8, 0.2], [0.4, 0.6], [0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        "lengths": torch.tensor([4, 2]),
    }


def test_zero_kernel_is_exact_poisson_on_finite_rectangle():
    model = _model(mu=2.0).eval()
    with torch.no_grad():
        model.event_model.alpha.zero_()
        out = model.eval_forward(**_batch())
    expected = -np.log(2.0) + 2 * np.array([0.1, 0.1, 0.15, 0.15, 0.2, 0.2])
    np.testing.assert_allclose(out["nll_matrix"][out["mask"]].numpy(), expected, atol=1e-6)
    assert float(out["nll"]) == pytest.approx(expected.mean(), abs=1e-6)
    assert out["eventwise_target_indices"][1].tolist() == [0, 1, -1, -1]


def test_temporal_antiderivative_matches_interpolated_basis():
    event = _model().event_model.double()
    times = torch.tensor([0.013, 0.043, 0.181], dtype=torch.float64, requires_grad=True)
    table = event._tables()
    integral = event.temporal_basis(times, table, integral=True).sum(-1)
    derivative = torch.autograd.grad(integral.sum(), times)[0]
    torch.testing.assert_close(
        derivative, event.temporal_basis(times, table).sum(-1), atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(
        event.temporal_basis(torch.tensor([-1.0], dtype=torch.float64), table, integral=True),
        torch.zeros(1, 2, dtype=torch.float64),
    )


def test_finite_region_spatial_integral_respects_boundaries():
    event = DNSKEventModel(width=4, spatial_nodes=80, radius=0.5)
    for net in event.v.nets:
        for parameter in net.parameters():
            parameter.data.zero_()
        net[-1].bias.data.fill_(1.0)
    mass = event.spatial_integrals(torch.tensor([[[0.5, 0.5], [0.0, 0.5], [0.0, 0.0]]]))[0, :, 0]
    torch.testing.assert_close(mass / mass[0], torch.tensor([1.0, 0.5, 0.25]), atol=0.02, rtol=0.02)
    assert float(mass[0].detach()) == pytest.approx(np.pi * 0.5**2, rel=0.02)


def test_raw_query_intensity_and_density_jacobian():
    from seahorse.data.transforms import ZScoreTransformArtifact

    transform = ZScoreTransformArtifact(time_std=2.0, loc_mean=(10.0, 20.0), loc_std=(3.0, 4.0))
    model = build_model(
        preset="dnsk",
        config={"input_transform": transform.serialize(), "decoder": {"width": 4, "mu": 2.0}},
    )
    model.event_model.alpha.data.zero_()
    state = model.state_model.encode_history(
        times=torch.tensor([[0.2, 0.4]]),
        locations=torch.tensor([[[10.3, 20.4], [10.6, 20.8]]]),
        lengths=torch.tensor([2]),
    )
    rate = model.event_model.intensity(
        state=state,
        query_times=torch.tensor([[0.6]]),
        query_locations=torch.tensor([[[11.5, 22.0]]]),
    )
    torch.testing.assert_close(rate, torch.tensor([[2.0 / (2.0 * 3.0 * 4.0)]]))


def test_signed_inhibition_and_invalid_intensity_are_not_clipped():
    model = _model(temporal_rank=1, spatial_rank=1, mu=2.0)
    event = model.event_model
    for bank in (event.psi, event.phi, event.u, event.v):
        for net in bank.nets:
            for p in net.parameters():
                p.data.zero_()
            net[-1].bias.data.fill_(1.0)
    event.alpha.data.fill_(-0.5)
    state = model.state_model.encode_history(
        times=torch.tensor([[0.1]]),
        locations=torch.tensor([[[0.5, 0.5]]]),
        lengths=torch.tensor([1]),
    )
    value = event.intensity(
        state=state, query_times=torch.tensor([[0.2]]), query_locations=torch.tensor([[[0.5, 0.5]]])
    )
    torch.testing.assert_close(value, torch.tensor([[1.5]]))
    event.alpha.data.fill_(-3.0)
    with pytest.raises(ValueError, match="non-positive"):
        event.intensity(
            state=state,
            query_times=torch.tensor([[0.2]]),
            query_locations=torch.tensor([[[0.5, 0.5]]]),
        )


def test_gradients_and_checkpointed_barrier_schedule():
    model = _model()
    out = model(**_batch())
    out["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    before = float(model.event_model.barrier_weight)
    model.event_model.advance_barrier_schedule()
    assert float(model.event_model.barrier_weight) < before
    assert model.event_model.state_dict()["optimizer_updates"].item() == 1


def test_eventwise_scores_match_prefix_recomputation_and_are_causal():
    model = _model().eval()
    batch = _batch()
    seq = {"times": batch["times"][0].numpy(), "locations": batch["locations"][0].numpy()}
    runner = SimpleNamespace(
        model=model,
        norm_stats={"normalize": False},
        config=SimpleNamespace(training=SimpleNamespace(test_nll_space="native")),
    )
    with torch.no_grad():
        actual = compute_next_event_test_nll(runner, [seq], device=torch.device("cpu"))
        expected = _prefix_difference_next_event_nlls(runner, seq, device=torch.device("cpu"))
    np.testing.assert_allclose(actual["per_context_nll"], expected, rtol=1e-5, atol=1e-5)
    changed = {"times": seq["times"].copy(), "locations": seq["locations"].copy()}
    changed["times"][-1] = 0.9
    changed["locations"][-1] = [0.9, 0.1]
    other = compute_next_event_test_nll(runner, [changed], device=torch.device("cpu"))
    np.testing.assert_allclose(actual["per_context_nll"][:-1], other["per_context_nll"][:-1])


def test_invalid_domain_is_rejected():
    batch = _batch()
    batch["locations"][0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="observation rectangle"):
        _model()(**batch)
