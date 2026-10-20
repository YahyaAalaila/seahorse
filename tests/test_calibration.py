"""Closed-form homogeneous-Poisson checks for the compensator-based calibration metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from seahorse.evaluation.v1 import calibration as C
from seahorse.evaluation.v1.calibration_families import ParametricKernelCalib
from seahorse.evaluation.v1.scoring import (
    crps_exponential_truncated,
    crps_from_cdf_grid,
    pit_from_compensator,
    tau_grid_quadratic,
)
from seahorse.registry import build_model

RATE = 2.5
C_THRESHOLD = 1.0


def _exponential_cumhazard_grid(y_rows: int, tau_grid: np.ndarray) -> np.ndarray:
    return np.repeat((RATE * tau_grid)[None, :], y_rows, axis=0)


def test_pit_is_one_minus_exp_of_rate_times_wait():
    y = np.array([0.0, 0.1, 0.5, 2.0])
    np.testing.assert_allclose(pit_from_compensator(RATE * y), 1.0 - np.exp(-RATE * y), rtol=0, atol=1e-15)


def test_grid_crps_matches_the_analytic_truncated_exponential_crps():
    tau = tau_grid_quadratic(4.0 * C_THRESHOLD, 1024)
    y = np.array([0.02, 0.3, 0.7, 0.95, 1.4])
    cdf = -np.expm1(-_exponential_cumhazard_grid(len(y), tau))
    grid = crps_from_cdf_grid(cdf, tau, y, C_THRESHOLD)
    analytic = np.array([crps_exponential_truncated(RATE, v, C_THRESHOLD) for v in y])
    np.testing.assert_allclose(grid, analytic, rtol=0, atol=2e-6)


def test_quantiles_from_the_cumulative_hazard_are_the_exponential_quantiles():
    tau = tau_grid_quadratic(4.0 * C_THRESHOLD, 4096)
    lg = _exponential_cumhazard_grid(3, tau)
    q = C.quantiles_from_cumhazard(lg, tau)
    for p in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(q[p], -math.log1p(-p) / RATE, rtol=1e-4)
    censored = C.quantiles_from_cumhazard(np.repeat((0.01 * tau)[None, :], 2, axis=0), tau)
    assert np.isinf(censored[0.9]).all()


def test_bisection_quantiles_agree_with_the_closed_form_in_numpy_and_torch():
    expected = {p: -math.log1p(-p) / RATE for p in (0.1, 0.5, 0.9)}
    q_np = C.bisect_quantiles(lambda t: RATE * t, 4, 4.0, iters=40)
    ref = torch.zeros(4)
    q_torch = C.bisect_quantiles(lambda t: RATE * t, 4, 4.0, iters=40, xp=torch, ref=ref)
    for p, value in expected.items():
        np.testing.assert_allclose(q_np[p], value, rtol=1e-8)
        np.testing.assert_allclose(q_torch[p], value, rtol=1e-6)


def test_hpd_pit_from_cells_is_the_randomised_mass_above_the_observation():
    rng = np.random.default_rng(0)
    probs = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.25, 0.25, 0.5]])
    cell_obs = np.array([0, 2, -1])
    u = C.hpd_pit_from_cells(probs, cell_obs, rng)
    assert 0.0 <= u[0] <= 0.5  # mode: only the tie term, V * 0.5
    assert 0.8 <= u[1] <= 1.0  # least likely cell: mass above = 0.8 plus V * 0.2
    assert np.isnan(u[2])


def test_cell_index_round_trips_grid_centres_and_flags_outside_points():
    box = (0.0, 2.0, -1.0, 1.0)
    centres = C.grid_centres(box, 4)
    np.testing.assert_array_equal(C.cell_index(centres, box, 4), np.arange(16))
    assert C.cell_index(np.array([[2.5, 0.0], [1.0, 0.0]]), box, 4).tolist()[0] == -1


def test_metrics_block_reports_uniform_pit_and_finite_crps():
    rng = np.random.default_rng(1)
    n = 2000
    arrays = {
        "pit": rng.uniform(0, 1, n), "crps_raw": rng.uniform(0, 0.1, n), "surv_at_c": rng.uniform(0, 1, n),
        "y_raw": rng.exponential(1.0 / RATE, n), "q10_raw": np.full(n, 0.05), "q50_raw": np.full(n, 0.3),
        "q90_raw": np.full(n, 1.0), "hpd_pit": rng.uniform(0, 1, n), "spatial_mass_in_box": np.ones(n),
    }
    block = C.metrics_block(arrays, np.ones(n, bool), C_THRESHOLD, 24.0)
    assert block["n_selected"] == n
    assert block["pit"]["ks"]["ks"] < block["pit"]["ks"]["ks_crit_95"]
    assert block["crps"]["n"] == n
    assert block["point_forecast"]["n"] == n
    assert block["spatial"]["n_finite"] == n


def _poisson_windows(n_rows: int, history: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0.0, 1.0, (n_rows, history + 1)), axis=1)
    locs = rng.uniform(0.1, 0.9, (n_rows, history + 1, 2))
    return (
        torch.tensor(times, dtype=torch.float32),
        torch.tensor(locs, dtype=torch.float32),
        torch.full((n_rows,), history + 1, dtype=torch.long),
    )


def test_gaussian_poisson_adapter_reproduces_the_homogeneous_poisson_closed_forms():
    """The gaussian_poisson preset is a homogeneous temporal Poisson process."""
    torch.manual_seed(0)
    model = build_model(preset="gaussian_poisson", config={"gaussian_poisson": {"mu_init": RATE}}, hidden_dim=8).eval()
    runner = type("Runner", (), {"model": model})()
    adapter = ParametricKernelCalib(runner, torch.device("cpu"), (0.0, 1.0, 0.0, 1.0))
    times, locs, lengths = _poisson_windows(n_rows=6, history=5)
    tau = torch.tensor(tau_grid_quadratic(4.0 * C_THRESHOLD, 257), dtype=torch.float32)
    with torch.no_grad():
        output = model.eval_forward(times=times, locations=locs, lengths=lengths)
    terms = adapter.terms(times, locs, lengths, tau, output=output)
    mu = float(model.event_model.mu.detach())
    waits = (times[:, -1] - times[:, -2]).numpy()
    np.testing.assert_allclose(terms["d_lambda"].numpy(), mu * waits, rtol=1e-5)
    assert terms["check_temporal_nll"] < 1e-5
    np.testing.assert_allclose(terms["lam_grid"].numpy(), np.broadcast_to(mu * tau.numpy()[None, :], (6, 257)), rtol=1e-5, atol=1e-6)
    # PIT and CRPS from the adapter's compensator equal the closed forms of Exp(mu).
    pit = pit_from_compensator(terms["d_lambda"].numpy())
    np.testing.assert_allclose(pit, 1.0 - np.exp(-mu * waits), rtol=1e-5)
    cdf = -np.expm1(-np.maximum.accumulate(terms["lam_grid"].numpy().astype(np.float64), axis=1))
    crps = crps_from_cdf_grid(cdf, tau.numpy().astype(np.float64), waits, C_THRESHOLD)
    analytic = np.array([crps_exponential_truncated(mu, float(w), C_THRESHOLD) for w in waits])
    np.testing.assert_allclose(crps, analytic, rtol=0, atol=5e-5)
    probes = torch.tensor([[0.1, 0.5, 1.0]] * 6)
    np.testing.assert_allclose(adapter.cumhaz_rows(probes).numpy(), mu * probes.numpy(), rtol=1e-5)
    probs, cells, mass = adapter.spatial_cells(times, locs, lengths, 16)
    assert probs.shape == (6, 256) and (cells >= 0).all()
    np.testing.assert_allclose(probs.sum(1), 1.0, rtol=1e-9)


def test_campaign_calibrate_subcommand_is_registered(tmp_path):
    import argparse

    from seahorse.cli import campaign

    parser = argparse.ArgumentParser()
    campaign.add_subparser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "campaign", "calibrate", "--manifest", str(tmp_path / "missing_manifest.json"),
        "--job", "j", "--out", str(tmp_path / "out"),
    ])
    assert args.campaign_command == "calibrate"
    with pytest.raises(FileNotFoundError):
        campaign.execute(args)
