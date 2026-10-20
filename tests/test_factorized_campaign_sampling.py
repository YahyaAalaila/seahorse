"""Contracts for full-plane sampling from factorized GMM campaign baselines."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from seahorse.experiments.factorized_sampling import (
    _require_model_device,
    _resolve_sampler_device,
    sample_factorized,
)
from seahorse.models.configs.factorized import (
    HawkesGMMConfig,
    PoissonGMMConfig,
    RMTPPGMMConfig,
    SelfCorrectingGMMConfig,
    THPGMMConfig,
)


def _runner(model, *, normalize: bool = False, time_mean: float = 0.0, time_std: float = 1.0, loc_mean=(0.0, 0.0), loc_std=(1.0, 1.0)):
    return SimpleNamespace(
        model=model,
        norm_stats={
            "normalize": normalize,
            "time_mean": time_mean,
            "time_std": time_std,
            "loc_mean": list(loc_mean),
            "loc_std": list(loc_std),
        },
    )


def _model(config_cls):
    model = config_cls(hidden_dim=8, spatial_dim=2).build_model()
    model.eval()
    return model


def _history():
    return {
        "times": np.asarray([4.0, 4.5, 5.0], dtype=np.float64),
        "locations": np.asarray([[3.0, -2.0], [3.0, -2.0], [3.0, -2.0]], dtype=np.float64),
    }


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def test_unindexed_cuda_alias_resolves_to_current_device_and_keeps_real_mismatch_rejection():
    assert _resolve_sampler_device("cuda", current_cuda_device=3) == torch.device("cuda:3")
    assert _resolve_sampler_device("cuda:1", current_cuda_device=3) == torch.device("cuda:1")
    _require_model_device(torch.device("cuda:3"), torch.device("cuda:3"))
    with pytest.raises(ValueError, match="cuda:2.*cuda:3"):
        _require_model_device(torch.device("cuda:2"), torch.device("cuda:3"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA alias needs a realized CUDA device")
@pytest.mark.parametrize("config_cls", [
    PoissonGMMConfig, HawkesGMMConfig, SelfCorrectingGMMConfig, RMTPPGMMConfig, THPGMMConfig,
])
def test_actual_cuda_alias_samples_from_model_on_current_device(config_cls):
    resolved = _resolve_sampler_device("cuda")
    assert torch.Generator(device=resolved).device == resolved
    model = _model(config_cls).to(resolved)
    output = sample_factorized(_runner(model), _history(), k=2, device="cuda", seed=9)
    finite = ~output["no_event"]
    assert output["success"].all()
    assert np.isfinite(output["next_times"][finite]).all()
    assert np.isfinite(output["next_locations"][finite]).all()
    assert np.isnan(output["next_times"][~finite]).all()
    assert np.isnan(output["next_locations"][~finite]).all()


def test_poisson_gmm_wait_and_spatial_law_are_reproducible_and_untruncated():
    model = _model(PoissonGMMConfig)
    temporal = model.event_model.temporal_model
    spatial = model.event_model.spatial_model
    with torch.no_grad():
        temporal.log_lambda.fill_(_inverse_softplus(2.0))
        spatial.log_sigma_kernel.fill_(_inverse_softplus(1.25))
        spatial.log_tau.fill_(_inverse_softplus(1.0))
    runner = _runner(model)

    out = sample_factorized(runner, _history(), k=3000, device="cpu", seed=716)
    repeat = sample_factorized(runner, _history(), k=3000, device="cpu", seed=716)
    np.testing.assert_array_equal(out["next_times"], repeat["next_times"])
    np.testing.assert_array_equal(out["next_locations"], repeat["next_locations"])
    assert not out["no_event"].any()
    waits = out["next_times"] - _history()["times"][-1]
    assert waits.mean() == pytest.approx(0.5, abs=0.035)
    residuals = out["next_locations"] - _history()["locations"][-1]
    assert residuals.mean(axis=0) == pytest.approx([0.0, 0.0], abs=0.075)
    assert residuals.var(axis=0) == pytest.approx([1.25**2, 1.25**2], abs=0.15)
    # The source support is full R^2: this deliberately produces draws outside
    # a conventional [0, 1]^2 plotting rectangle instead of clipping them.
    assert np.any((out["next_locations"] < 0.0) | (out["next_locations"] > 1.0))
    assert out["diagnostics"]["spatial_support"] == "R^D (untruncated Gaussian mixture)"


@pytest.mark.parametrize(
    ("config_cls", "expected_route"),
    [
        (HawkesGMMConfig, "hawkes_inverse_integrated_hazard"),
        (SelfCorrectingGMMConfig, "selfcorrecting_inverse_integrated_hazard"),
    ],
)
def test_parametric_factorized_temporal_heads_use_verified_routes(config_cls, expected_route):
    model = _model(config_cls)
    out = sample_factorized(_runner(model), _history(), k=8, device="cpu", seed=31)
    assert out["success"].all()
    assert not out["no_event"].any()
    assert np.isfinite(out["next_times"]).all()
    assert all(d["route"] == expected_route for d in out["diagnostics"]["draws"])


def test_rmtpp_explicitly_reports_its_defective_no_event_mass():
    model = _model(RMTPPGMMConfig)
    temporal = model.event_model.temporal_model
    # a=b=softplus(0), hence total future hazard is one and P(no event)=e^-1.
    with torch.no_grad():
        for parameter in temporal.mlp.parameters():
            parameter.zero_()
    out = sample_factorized(_runner(model), _history(), k=512, device="cpu", seed=9)
    assert out["no_event"].any()
    assert np.all(np.isnan(out["next_times"][out["no_event"]]))
    assert np.all(np.isnan(out["next_locations"][out["no_event"]]))
    assert all(d["route"] == "rmtpp_defective_inverse_hazard" for d in out["diagnostics"]["draws"])


def test_thp_constant_head_uses_actual_checkpoint_parameters():
    model = _model(THPGMMConfig)
    temporal = model.event_model.temporal_model
    with torch.no_grad():
        temporal.f.weight.zero_()
        temporal.f.bias.zero_()  # c=0, softplus(c)=log(2)
        temporal.beta.zero_()
    out = sample_factorized(_runner(model), _history(), k=1200, device="cpu", seed=73)
    waits = out["next_times"] - _history()["times"][-1]
    assert waits.mean() == pytest.approx(1.0 / np.log(2.0), abs=0.10)
    assert all(d["route"] == "thp_constant_inverse_hazard" for d in out["diagnostics"]["draws"])


def test_runner_and_family_transforms_are_inverted_back_to_raw_units():
    model = _model(PoissonGMMConfig)
    # The state model maps runner-normalized coordinates once more through this
    # fitted artifact.  Sampling must invert both layers before returning values.
    model.state_model._input_transform_spec = {
        "type": "zscore",
        "normalize_time": True,
        "normalize_space": True,
        "time_mean": 1.0,
        "time_std": 2.0,
        "loc_mean": [1.0, -1.0],
        "loc_std": [3.0, 4.0],
    }
    from seahorse.data.transforms import transform_from_spec
    model.state_model._input_transform = transform_from_spec(model.state_model._input_transform_spec)
    runner = _runner(model, normalize=True, time_mean=10.0, time_std=5.0, loc_mean=(100.0, -100.0), loc_std=(10.0, 20.0))
    history = {"times": np.asarray([12.0], dtype=np.float64), "locations": np.asarray([[110.0, -80.0]], dtype=np.float64)}
    out = sample_factorized(runner, history, k=4, device="cpu", seed=101)
    assert np.all(out["next_times"] >= 12.0)
    assert np.isfinite(out["next_locations"]).all()
    # A raw history location is decoded by the same transform composition; no
    # accidental internal z-score coordinate should appear in the output.
    assert np.max(np.abs(out["next_locations"] - np.asarray([110.0, -80.0]))) > 1.0


def test_unknown_temporal_head_is_rejected():
    model = _model(PoissonGMMConfig)
    model.event_model.temporal_model = torch.nn.Identity()
    with pytest.raises(TypeError, match="Unsupported temporal model"):
        sample_factorized(_runner(model), _history(), k=1, device="cpu", seed=0)

@pytest.mark.parametrize("beta", [0.35, -0.35])
def test_thp_adaptive_hazard_route_handles_learned_positive_and_negative_slopes(beta):
    model = _model(THPGMMConfig)
    temporal = model.event_model.temporal_model
    with torch.no_grad():
        temporal.f.weight.zero_()
        temporal.f.bias.zero_()
        temporal.beta.fill_(beta)
    out = sample_factorized(_runner(model), _history(), k=12, device="cpu", seed=154)
    assert all("thp_adaptive_integral" in d["route"] for d in out["diagnostics"]["draws"])
    assert np.isfinite(out["next_times"][~out["no_event"]]).all()
