"""Parametric Hawkes/Poisson baselines: registry, likelihood decomposition, nesting."""

from __future__ import annotations

import math

import numpy as np
import torch

from seahorse.models.configs import ConfigRegistry
from seahorse.registry import build_model

TIMES = torch.tensor([[0.05, 0.13, 0.21, 0.34, 0.46, 0.58, 0.67, 0.79, 0.88]])
LOCS = torch.tensor([[
    [0.12, 0.20], [0.24, 0.17], [0.31, 0.42], [0.48, 0.51], [0.55, 0.39],
    [0.62, 0.66], [0.71, 0.58], [0.83, 0.74], [0.91, 0.62],
]])
LENGTHS = torch.tensor([9])


def _forward(preset, config=None):
    torch.manual_seed(0)
    model = build_model(preset=preset, config=config or {}, hidden_dim=8).eval()
    with torch.no_grad():
        return model, model.eval_forward(times=TIMES, locations=LOCS, lengths=LENGTHS)


def test_presets_are_registered_as_canonical():
    for preset in ("static_hawkes", "gaussian_poisson"):
        assert ConfigRegistry.is_registered(preset)
        assert ConfigRegistry.canonical_status(preset) == "canonical"


def test_eventwise_matrix_averages_to_scalar_and_is_finite():
    for preset in ("static_hawkes", "gaussian_poisson"):
        _, out = _forward(preset)
        matrix, mask = out["nll_matrix"], out["mask"]
        assert torch.isfinite(matrix[mask > 0]).all()
        mean = float((matrix * mask).sum() / mask.sum())
        assert abs(mean - float(out["nll"])) < 1e-6
        assert float(mask[0, 0]) == 0.0  # the first event is not a next-event target


def test_gaussian_poisson_has_no_triggering_and_matches_closed_form():
    model, out = _forward("gaussian_poisson")
    event = model.event_model
    assert float(event.eta) == 0.0
    mu = float(event.mu)
    var = event.background_variance.detach().numpy()
    times = TIMES[0].numpy()
    locs = LOCS[0].numpy()
    expected = []
    for j in range(1, len(times)):
        temporal = -math.log(mu) + mu * (times[j] - times[j - 1])
        spatial = 0.5 * (2 * math.log(2 * math.pi) + np.log(var).sum() + (locs[j] ** 2 / var).sum())
        expected.append(temporal + spatial)
    np.testing.assert_allclose(out["nll_matrix"][0, 1:].numpy(), expected, rtol=1e-5, atol=1e-5)


def test_static_hawkes_nests_the_poisson_null_when_eta_vanishes():
    hawkes, _ = _forward("static_hawkes")
    with torch.no_grad():
        hawkes.event_model.raw_eta.fill_(-60.0)  # eta -> 0
        hawkes_out = hawkes.eval_forward(times=TIMES, locations=LOCS, lengths=LENGTHS)
    _, poisson_out = _forward("gaussian_poisson")
    torch.testing.assert_close(hawkes_out["nll_matrix"], poisson_out["nll_matrix"], rtol=1e-5, atol=1e-5)


def test_config_block_controls_initialisation():
    model = build_model(
        preset="static_hawkes",
        config={"static_hawkes": {"mu_init": 3.0, "static_std_init": 0.2, "eta_init": 0.3}},
        hidden_dim=8,
    )
    event = model.event_model
    assert abs(float(event.mu) - 3.0) < 1e-5
    assert abs(float(event.eta) - 0.3) < 1e-5
    assert abs(float(event.static_variance[0].sqrt()) - 0.2) < 1e-3
