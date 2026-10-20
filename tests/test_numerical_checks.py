from types import SimpleNamespace

import numpy as np
import torch

from seahorse.experiments.numerical_checks import refinement_decision, run_numerical_nll_checks
from seahorse.registry import build_model


def _batch():
    return {"times": torch.tensor([[.1, .2, .4], [.2, .5, .0]]),
            "locations": torch.tensor([[[.1, .2], [.3, .4], [.5, .6]], [[.6, .2], [.4, .7], [0., 0.]]]),
            "lengths": torch.tensor([3, 2])}


def test_dnsk_zero_kernel_is_quadrature_invariant_without_checkpoint_mutation():
    model = build_model(preset="dnsk", config={"decoder": {"width": 4, "temporal_nodes": 9, "spatial_nodes": 6}}).eval()
    model.event_model.alpha.data.zero_()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    runner = SimpleNamespace(model=model, config=SimpleNamespace(model=SimpleNamespace(preset="dnsk")))
    result = run_numerical_nll_checks(runner, _batch())
    assert result["status"] == "pass"
    assert result["default_vs_finest"]["max_absolute_change"] < 1e-6
    assert result["positivity"]["query_grid"].startswith("passed")
    for key, value in model.state_dict().items(): torch.testing.assert_close(value, before[key])


def test_refinement_decision_rejects_default_to_finest_perturbation_even_if_adjacent_not_examined():
    result = refinement_decision(np.asarray([1., 1.]), np.asarray([1.002, 1.2]),
                                 {"absolute_tolerance": 1e-3, "relative_tolerance": 1e-2})
    assert not result["passed"]
    assert result["violating_targets"] == 1


def test_thp_uses_deterministic_midpoint_refinement_not_resampled_mc():
    model = build_model(
        preset="thp_gmm",
        config={"hidden_size": 4, "n_heads": 2, "n_layers": 1, "dropout": 0.0},
    ).eval()
    runner = SimpleNamespace(model=model, config=SimpleNamespace(model=SimpleNamespace(preset="thp_gmm")))
    result = run_numerical_nll_checks(runner, _batch())
    assert result["status"] == "pass"
    assert [record["setting"]["midpoint_nodes"] for record in result["checks"]] == [30, 60, 120]
    assert model.event_model.temporal_model.MC_SAMPLES == 30
