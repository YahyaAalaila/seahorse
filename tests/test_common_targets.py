"""A conditional experiment must count and optimize each target exactly once."""
import pytest
import torch

from seahorse.models.targets import last_target_output, last_target_state
from seahorse.models.abstractions import StateContext
from seahorse.registry import build_model


def test_last_target_selection_changes_loss_and_gradient_not_history():
    terms = torch.tensor([[100., 2., 3.], [4., 8., 999.]], requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    old = terms[mask].mean()
    out = last_target_output({"nll": old, "loss": old + .3, "nll_matrix": terms, "mask": mask,
        "extra_metrics": {"raw_space_nll": float(old.detach()) + 2., "raw_reporting_correction": 2.}}, torch.tensor([3, 2]))
    torch.testing.assert_close(out["nll"], torch.tensor(5.5))
    torch.testing.assert_close(out["loss"], torch.tensor(5.8))
    assert float(out["extra_metrics"]["raw_space_nll"]) == 7.5
    out["loss"].backward()
    torch.testing.assert_close(terms.grad, torch.tensor([[0., 0., .5], [0., .5, 0.]]))


def test_flat_generative_targets_keep_matching_condition_and_target():
    p = {"diff_img": torch.arange(5).reshape(5, 1, 1), "diff_cond": torch.arange(5).reshape(5, 1, 1) + 10,
         "diff_total_events": torch.tensor(5.), "diff_cond_last": torch.ones(2, 1, 3)}
    state = last_target_state(StateContext(payload=p), torch.tensor([4, 3])).payload
    assert state["diff_img"].flatten().tolist() == [2, 4]
    assert state["diff_cond"].flatten().tolist() == [12, 14]
    assert float(state["diff_total_events"]) == 2


@pytest.mark.parametrize("preset", ["auto_stpp", "deep_stpp", "thp_gmm", "rmtpp_gmm", "dnsk", "gaussian_poisson", "static_hawkes"])
def test_model_common_final_target_equals_its_eventwise_score(preset):
    torch.manual_seed(7)
    cfg = {"decoder": {"lookback": 3, "max_history": 3, "n_prodnet": 2, "hidden_size": 8, "num_layers": 1}} if preset == "auto_stpp" else {}
    if preset == "deep_stpp": cfg = {"decoder": {"seq_len": 3, "num_points": 2, "n_layers": 1}, "encoder": {"num_layers": 1}}
    if preset == "dnsk": cfg = {"decoder": {"width": 4, "barrier_space_nodes": 2, "barrier_time_nodes": 2}}
    model = build_model(preset=preset, config=cfg, hidden_dim=8).eval()
    t = torch.tensor([[.1, .2, .3, .4]])
    x = torch.tensor([[[.1, .2], [.3, .2], [.4, .3], [.5, .6]]])
    torch.manual_seed(5); full = model.eval_forward(t, x, torch.tensor([4]))
    torch.manual_seed(5); last = model.eval_forward(t, x, torch.tensor([4]), target_scope="last")
    assert int(last["total_events"]) == 1
    torch.testing.assert_close(last["nll"], full["nll_matrix"][0, -1])


def test_glide_test_does_not_log_diagnostic_as_nll():
    from seahorse.training.lightning_module import STPPLightningModule
    from seahorse.config.schema import TrainingConfig
    model = build_model(preset="glide", config={"encoder": {"d_model": 16}, "decoder": {"hidden_units": 16, "timesteps": 3}})
    lm = STPPLightningModule(model, TrainingConfig(loss_target_scope="last"))
    logged = {}; lm.log = lambda name, value, **kwargs: logged.update({name: value})
    lm.test_step({"times": torch.tensor([[0., .1, .2]]), "locations": torch.rand(1, 3, 2), "lengths": torch.tensor([3])}, 0)
    assert "test/nll" not in logged
    assert any(k.startswith("test/") for k in logged)


def test_glide_validation_logs_objective_once():
    from seahorse.training.lightning_module import STPPLightningModule
    from seahorse.config.schema import TrainingConfig
    model=build_model(preset="glide",config={"encoder":{"d_model":16},"decoder":{"hidden_units":16,"timesteps":3}})
    lm=STPPLightningModule(model,TrainingConfig(loss_target_scope="last"))
    names=[];lm.log=lambda name,*args,**kwargs:names.append(name)
    lm.validation_step({"times":torch.tensor([[0.,.1,.2]]),"locations":torch.rand(1,3,2),"lengths":torch.tensor([3])},0)
    assert names.count("val/glide_loss") == 1
