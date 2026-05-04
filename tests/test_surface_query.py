"""Tests for SurfaceEvaluator and the spatial-query model methods.

Phased structure mirrors the rollout plan:

  Phase 1 unit — temporal intensity_at, spatial log_spatial_density_at
  Phase 2 unit — FactorizedEventModel.intensity() / density()
  Phase 1 integration — SurfaceEvaluator for deep_stpp, auto_stpp
  Phase 2 integration — SurfaceEvaluator for all 9 factorized presets
  Phase 3 integration — SurfaceEvaluator proxy_kde for smash, diffusion_stpp
"""

import unittest

import numpy as np
import torch

from unified_stpp.models.configs import ConfigRegistry
from unified_stpp.registry import build_model

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TIMES_T = torch.tensor([[0.0, 0.2, 0.5, 0.9]], dtype=torch.float32)
_LOCS_T  = torch.tensor(
    [[[0.0, 0.0], [0.3, -0.1], [0.1, 0.2], [-0.2, 0.1]]], dtype=torch.float32
)
_LENGTHS_T = torch.tensor([4], dtype=torch.long)

# Original-space arrays for SurfaceEvaluator (before normalization)
_TIMES_NP = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32)
_LOCS_NP  = np.array([[0.0, 0.0], [0.3, -0.1], [0.1, 0.2], [-0.2, 0.1]], dtype=np.float32)

_M = 5  # query batch size for unit tests
_T_QUERY = torch.full((_M,), 1.2)
_HIST_T  = _TIMES_T.expand(_M, -1).float()  # (M, 4)
_HIST_S  = _LOCS_T.expand(_M, -1, -1).float()  # (M, 4, 2)
_HIST_MK = torch.ones(_M, 4)


# ---------------------------------------------------------------------------
# Mock runner for SurfaceEvaluator integration tests
# ---------------------------------------------------------------------------

class _MockDataset:
    """Minimal dataset stub with identity normalization stats."""
    time_mean: float = 0.5
    time_std:  float = 1.0
    loc_mean = np.array([0.0, 0.0], dtype=np.float32)
    loc_std  = np.array([1.0, 1.0], dtype=np.float32)


class _MockDataModule:
    _train_dataset = _MockDataset()


class _MockRunner:
    """Minimal runner stub: real model, identity norm stats."""

    def __init__(self, preset, config=None):
        torch.manual_seed(0)
        self.model = build_model(
            config=config or {},
            preset=preset,
            spatial_dim=2,
            hidden_dim=16,
            event_cov_dim=0,
            field_cov_dim=0,
        )
        self.model.eval()
        self.norm_stats = {
            "normalize": False,
            "time_mean": _MockDataset.time_mean,
            "time_std":  _MockDataset.time_std,
            "loc_mean":  list(_MockDataset.loc_mean),
            "loc_std":   list(_MockDataset.loc_std),
        }


# Config used for CNF presets to keep ODE solves fast during tests
_FAST_CNF_CONFIG = {"hidden_dims": [8, 8], "tol": 1e-2}


# ---------------------------------------------------------------------------
# Phase 1 — temporal model intensity_at()
# ---------------------------------------------------------------------------

class TestTemporalIntensityAt(unittest.TestCase):
    """intensity_at() returns finite non-negative (M,) for all temporal processes."""

    def _check(self, cls):
        model = cls()
        model.eval()
        with torch.no_grad():
            out = model.intensity_at(_T_QUERY, _HIST_T, _HIST_MK)
        self.assertEqual(out.shape, (_M,), f"{cls.__name__}: wrong shape {out.shape}")
        self.assertTrue(torch.isfinite(out).all(), f"{cls.__name__}: not finite: {out}")
        self.assertTrue((out >= 0).all(), f"{cls.__name__}: negative values: {out}")

    def test_poisson(self):
        from unified_stpp.models.temporal_models.parametric_processes import (
            HomogeneousPoissonProcess,
        )
        self._check(HomogeneousPoissonProcess)

    def test_hawkes(self):
        from unified_stpp.models.temporal_models.parametric_processes import HawkesProcess
        self._check(HawkesProcess)

    def test_selfcorrecting(self):
        from unified_stpp.models.temporal_models.parametric_processes import SelfCorrectingProcess
        self._check(SelfCorrectingProcess)


# ---------------------------------------------------------------------------
# Phase 1 — GMM log_spatial_density_at()
# ---------------------------------------------------------------------------

class TestGMMLogSpatialDensityAt(unittest.TestCase):
    """GaussianMixtureSpatialModel.log_spatial_density_at returns finite (M,)."""

    def test_shape_and_finite(self):
        from unified_stpp.models.spatial_models.gaussian_mixture import (
            GaussianMixtureSpatialModel,
        )
        torch.manual_seed(0)
        model = GaussianMixtureSpatialModel()
        s_q = torch.randn(_M, 2)
        with torch.no_grad():
            out = model.log_spatial_density_at(
                _T_QUERY, s_q, _HIST_T, _HIST_S, _HIST_MK
            )
        self.assertEqual(out.shape, (_M,))
        self.assertTrue(torch.isfinite(out).all(), f"not finite: {out}")

    def test_no_history_uses_prior(self):
        """All-zero mask triggers Gaussian prior fallback — result still finite."""
        from unified_stpp.models.spatial_models.gaussian_mixture import (
            GaussianMixtureSpatialModel,
        )
        model = GaussianMixtureSpatialModel()
        empty_mask = torch.zeros(_M, 4)
        s_q = torch.zeros(_M, 2)
        with torch.no_grad():
            out = model.log_spatial_density_at(
                _T_QUERY, s_q, _HIST_T, _HIST_S, empty_mask
            )
        self.assertEqual(out.shape, (_M,))
        self.assertTrue(torch.isfinite(out).all())

    def test_consistent_with_logprob(self):
        """Point-query density at event i should agree with logprob for a single
        query point using the matching history."""
        from unified_stpp.models.spatial_models.gaussian_mixture import (
            GaussianMixtureSpatialModel,
        )
        torch.manual_seed(1)
        model = GaussianMixtureSpatialModel()
        # Query at event index i=2, history = events 0 and 1
        t_i  = _TIMES_T[0, 2:3]           # (1,)
        s_i  = _LOCS_T[0, 2:3, :]        # (1, 2)  — already (M=1, D=2)
        h_t  = _TIMES_T[:, :2].float()   # (1, 2) — past events only
        h_s  = _LOCS_T[:, :2, :].float() # (1, 2, 2)
        h_m  = torch.ones(1, 2)
        with torch.no_grad():
            point_ll = model.log_spatial_density_at(
                t_i, s_i, h_t, h_s, h_m  # t_i: (1,), s_i: (1, 2)
            )  # (1,)
            # logprob on the 3-event prefix: result[:, 2] is log p(s_2 | s_0, s_1)
            full_t  = _TIMES_T[:, :3].float()
            full_s  = _LOCS_T[:, :3, :].float()
            full_m  = torch.ones(1, 3)
            seq_ll = model.logprob(full_t, full_s, full_m)  # (1, 3)

        self.assertAlmostEqual(
            point_ll[0].item(), seq_ll[0, 2].item(), places=4,
            msg="log_spatial_density_at disagrees with logprob at event i=2",
        )


# ---------------------------------------------------------------------------
# Phase 1 — IndependentCNF log_spatial_density_at()
# ---------------------------------------------------------------------------

class TestIndependentCNFLogSpatialDensityAt(unittest.TestCase):
    """IndependentCNF.log_spatial_density_at accepts history args and returns finite (M,)."""

    def _build_cnf(self, squash_time=True):
        from unified_stpp.models.spatial_models.independent_cnf import IndependentCNF
        return IndependentCNF(
            dim=2, hidden_dims=(8, 8), tol=1e-2, squash_time=squash_time
        )

    def test_squash_time_true(self):
        torch.manual_seed(0)
        model = self._build_cnf(squash_time=True)
        model.eval()
        s_q = torch.randn(_M, 2)
        with torch.no_grad():
            out = model.log_spatial_density_at(
                _T_QUERY, s_q, _HIST_T, _HIST_S, _HIST_MK
            )
        self.assertEqual(out.shape, (_M,))
        self.assertTrue(torch.isfinite(out).all())

    def test_squash_time_false(self):
        torch.manual_seed(0)
        model = self._build_cnf(squash_time=False)
        model.eval()
        s_q = torch.randn(_M, 2)
        with torch.no_grad():
            out = model.log_spatial_density_at(
                _T_QUERY, s_q, _HIST_T, _HIST_S, _HIST_MK
            )
        self.assertEqual(out.shape, (_M,))
        self.assertTrue(torch.isfinite(out).all())

    def test_history_args_ignored(self):
        """History args are accepted but unused; omitting them gives identical results."""
        torch.manual_seed(0)
        model = self._build_cnf(squash_time=True)
        model.eval()
        s_q = torch.randn(_M, 2)
        with torch.no_grad():
            out_with = model.log_spatial_density_at(
                _T_QUERY, s_q, _HIST_T, _HIST_S, _HIST_MK
            )
            out_without = model.log_spatial_density_at(_T_QUERY, s_q)
        self.assertTrue(
            torch.allclose(out_with, out_without),
            "History args should have no effect on IndependentCNF output",
        )


# ---------------------------------------------------------------------------
# Phase 2 — FactorizedEventModel capabilities + intensity() + density()
# ---------------------------------------------------------------------------

class TestFactorizedCapabilities(unittest.TestCase):
    """FactorizedEventModel declares has_intensity=True, has_density=True."""

    def test_gmm_capabilities(self):
        model = build_model(
            config={}, preset="hawkes_gmm", spatial_dim=2, hidden_dim=16,
            event_cov_dim=0, field_cov_dim=0,
        )
        caps = model.event_model.capabilities
        self.assertTrue(caps.has_intensity)
        self.assertTrue(caps.has_density)

    def test_cnf_capabilities(self):
        model = build_model(
            config=_FAST_CNF_CONFIG, preset="hawkes_cnf", spatial_dim=2, hidden_dim=16,
            event_cov_dim=0, field_cov_dim=0,
        )
        caps = model.event_model.capabilities
        self.assertTrue(caps.has_intensity)
        self.assertTrue(caps.has_density)


class TestFactorizedIntensityDensity(unittest.TestCase):
    """FactorizedEventModel.intensity() and density() return correct shapes."""

    def _state(self):
        from unified_stpp.models.abstractions import StateContext
        return StateContext(payload={
            "times":     _TIMES_T,
            "locations": _LOCS_T,
            "lengths":   _LENGTHS_T,
        })

    def _check_intensity(self, preset, config=None):
        torch.manual_seed(0)
        model = build_model(
            config=config or {}, preset=preset, spatial_dim=2, hidden_dim=16,
            event_cov_dim=0, field_cov_dim=0,
        )
        model.eval()
        state = self._state()
        qt = torch.full((_M, 1), 1.2)
        ql = torch.randn(_M, 2)
        with torch.no_grad():
            out = model.event_model.intensity(
                state=state, query_times=qt, query_locations=ql
            )
        self.assertEqual(out.shape, (_M,), f"{preset}: wrong shape {out.shape}")
        self.assertTrue(torch.isfinite(out).all(), f"{preset}: not finite: {out}")
        self.assertTrue((out >= 0).all(), f"{preset}: negative intensity: {out}")

    def _check_density(self, preset, config=None):
        torch.manual_seed(0)
        model = build_model(
            config=config or {}, preset=preset, spatial_dim=2, hidden_dim=16,
            event_cov_dim=0, field_cov_dim=0,
        )
        model.eval()
        state = self._state()
        qt = torch.full((_M, 1), 1.2)
        ql = torch.randn(_M, 2)
        with torch.no_grad():
            out = model.event_model.density(
                state=state, query_times=qt, query_locations=ql
            )
        self.assertEqual(out.shape, (_M,), f"{preset}: wrong shape {out.shape}")
        self.assertTrue(torch.isfinite(out).all(), f"{preset}: not finite: {out}")
        self.assertTrue((out >= 0).all(), f"{preset}: negative density: {out}")

    def test_gmm_intensity(self):
        for preset in ("poisson_gmm", "hawkes_gmm", "selfcorrecting_gmm"):
            with self.subTest(preset=preset):
                self._check_intensity(preset)

    def test_gmm_density(self):
        for preset in ("poisson_gmm", "hawkes_gmm", "selfcorrecting_gmm"):
            with self.subTest(preset=preset):
                self._check_density(preset)

    def test_cnf_intensity(self):
        for preset in ("poisson_cnf", "hawkes_cnf", "selfcorrecting_cnf"):
            with self.subTest(preset=preset):
                self._check_intensity(preset, config=_FAST_CNF_CONFIG)

    def test_cnf_density(self):
        for preset in ("poisson_cnf", "hawkes_cnf", "selfcorrecting_cnf"):
            with self.subTest(preset=preset):
                self._check_density(preset, config=_FAST_CNF_CONFIG)

    def test_tvcnf_intensity(self):
        for preset in ("poisson_tvcnf", "hawkes_tvcnf", "selfcorrecting_tvcnf"):
            with self.subTest(preset=preset):
                self._check_intensity(preset, config=_FAST_CNF_CONFIG)


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

_N_GRID = 4  # small grid to keep ODE-based presets fast


def _check_surface_result(tc, result, surface_type, comparable, n_grid=_N_GRID):
    """Assert the structural contract of a SurfaceResult."""
    from unified_stpp.evaluation.surface import SurfaceResult

    tc.assertIsInstance(result, SurfaceResult)
    tc.assertEqual(result.surface_type, surface_type)
    tc.assertEqual(result.comparable, comparable)
    tc.assertEqual(result.values.shape, (n_grid, n_grid))
    tc.assertTrue(np.isfinite(result.values).all(), f"values not finite: {result.values}")
    tc.assertGreater(len(result.xs), 0)
    tc.assertGreater(len(result.ys), 0)


def _evaluate_surface(
    runner,
    *,
    history_times=_TIMES_NP,
    history_locs=_LOCS_NP,
    t_query=1.0,
    n_grid=_N_GRID,
    n_samples=500,
):
    from unified_stpp.evaluation.surface import SurfaceEvaluator

    evaluator = SurfaceEvaluator(runner.model, runner.norm_stats)
    return evaluator.evaluate_frame(
        history_times=history_times,
        history_locs=history_locs,
        t_query=t_query,
        n_grid=n_grid,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Phase 1 integration — deep_stpp, auto_stpp
# ---------------------------------------------------------------------------

class TestSurfaceQueryPhase1(unittest.TestCase):
    """deep_stpp and auto_stpp variants return surface_type='intensity'."""

    def _check(self, preset):
        runner = _MockRunner(preset)
        result = _evaluate_surface(runner)
        _check_surface_result(self, result, "intensity", comparable=True)

    def test_deep_stpp(self):
        self._check("deep_stpp")

    def test_auto_stpp(self):
        self._check("auto_stpp")


# ---------------------------------------------------------------------------
# Phase 2 integration — factorized GMM and CNF presets
# ---------------------------------------------------------------------------

class TestSurfaceQueryPhase2GMM(unittest.TestCase):
    """Factorized GMM presets return surface_type='intensity'."""

    def _check(self, preset):
        runner = _MockRunner(preset)
        result = _evaluate_surface(runner)
        _check_surface_result(self, result, "intensity", comparable=True)

    def test_poisson_gmm(self):
        self._check("poisson_gmm")

    def test_hawkes_gmm(self):
        self._check("hawkes_gmm")

    def test_selfcorrecting_gmm(self):
        self._check("selfcorrecting_gmm")


class TestSurfaceQueryPhase2CNF(unittest.TestCase):
    """Factorized CNF and TVCNF presets return surface_type='intensity'."""

    def _check(self, preset):
        runner = _MockRunner(preset, config=_FAST_CNF_CONFIG)
        result = _evaluate_surface(runner)
        _check_surface_result(self, result, "intensity", comparable=True)

    def test_poisson_cnf(self):
        self._check("poisson_cnf")

    def test_hawkes_cnf(self):
        self._check("hawkes_cnf")

    def test_selfcorrecting_cnf(self):
        self._check("selfcorrecting_cnf")

    def test_poisson_tvcnf(self):
        self._check("poisson_tvcnf")

    def test_hawkes_tvcnf(self):
        self._check("hawkes_tvcnf")

    def test_selfcorrecting_tvcnf(self):
        self._check("selfcorrecting_tvcnf")


# ---------------------------------------------------------------------------
# Phase 3 integration — SMASH and DiffusionSTPP (proxy_kde)
# ---------------------------------------------------------------------------

class TestSurfaceQueryPhase3(unittest.TestCase):
    """smash and diffusion_stpp return surface_type='proxy_kde', comparable=False."""

    def _check(self, preset, n_samples=20):
        runner = _MockRunner(preset)
        result = _evaluate_surface(runner, n_samples=n_samples)
        _check_surface_result(self, result, "proxy_kde", comparable=False)
        self.assertEqual(result.n_samples, n_samples)

    def test_smash(self):
        self._check("smash", n_samples=20)

    def test_diffusion_stpp(self):
        self._check("diffusion_stpp", n_samples=20)


# ---------------------------------------------------------------------------
# SurfaceResult — field contract
# ---------------------------------------------------------------------------

class TestSurfaceResultFields(unittest.TestCase):
    """SurfaceResult exposes all required fields with correct types."""

    def _make_intensity_result(self):
        from unified_stpp.evaluation.surface import SurfaceResult
        return SurfaceResult(
            surface_type="intensity",
            values=np.ones((4, 4), dtype=np.float32),
            xs=np.linspace(0, 1, 4, dtype=np.float32),
            ys=np.linspace(0, 1, 4, dtype=np.float32),
            t_query=1.0,
            label="Conditional intensity λ*(t,s|H)",
            unit="events / (unit_time × unit_area)",
            comparable=True,
        )

    def test_intensity_comparable(self):
        r = self._make_intensity_result()
        self.assertTrue(r.comparable)
        self.assertIsNone(r.n_samples)

    def test_proxy_kde_not_comparable(self):
        from unified_stpp.evaluation.surface import SurfaceResult
        r = SurfaceResult(
            surface_type="proxy_kde",
            values=np.ones((4, 4), dtype=np.float32),
            xs=np.linspace(0, 1, 4, dtype=np.float32),
            ys=np.linspace(0, 1, 4, dtype=np.float32),
            t_query=1.0,
            label="Spatial proxy from 50 samples (KDE, not comparable)",
            unit="proxy (not comparable)",
            comparable=False,
            n_samples=50,
        )
        self.assertFalse(r.comparable)
        self.assertEqual(r.n_samples, 50)

    def test_exported_from_evaluation(self):
        from unified_stpp.evaluation.surface import SurfaceEvaluator, SurfaceResult
        self.assertIsNotNone(SurfaceResult)
        self.assertIsNotNone(SurfaceEvaluator)

    def test_history_fields_present(self):
        """SurfaceResult carries history_times and history_locs."""
        from unified_stpp.evaluation.surface import SurfaceResult
        r = SurfaceResult(
            surface_type="intensity",
            values=np.ones((4, 4), dtype=np.float32),
            xs=np.linspace(0, 1, 4, dtype=np.float32),
            ys=np.linspace(0, 1, 4, dtype=np.float32),
            t_query=1.0,
            label="L",
            unit="U",
            comparable=True,
            history_times=np.array([0.1, 0.5], dtype=np.float32),
            history_locs=np.zeros((2, 2), dtype=np.float32),
        )
        self.assertIsNotNone(r.history_times)
        self.assertIsNotNone(r.history_locs)

    def test_model_name_field(self):
        """SurfaceResult.model_name defaults to None and can be set."""
        from unified_stpp.evaluation.surface import SurfaceResult
        r = SurfaceResult(
            surface_type="intensity",
            values=np.ones((4, 4), dtype=np.float32),
            xs=np.linspace(0, 1, 4, dtype=np.float32),
            ys=np.linspace(0, 1, 4, dtype=np.float32),
            t_query=1.0,
            label="L",
            unit="U",
            comparable=True,
        )
        self.assertIsNone(r.model_name)
        r.model_name = "my_model"
        self.assertEqual(r.model_name, "my_model")


# ---------------------------------------------------------------------------
# query_surface() contract — direct unit tests per model family
# ---------------------------------------------------------------------------
#
# These tests call event_model.query_surface() directly, bypassing
# SurfaceEvaluator.  They verify the minimal contract:
#   - output shape (G,)
#   - output dtype float32
#   - all values non-negative
#
# Uses tiny random weights (no training) and G=4 grid points.
# ---------------------------------------------------------------------------

_G = 4  # grid size for contract tests


def _fake_state(model, device="cpu"):
    """Return a StateContext by calling state_model.encode_history with tiny data."""
    import torch

    times     = torch.tensor([[0.0, 0.2, 0.5, 0.9]], dtype=torch.float32, device=device)
    locations = torch.tensor(
        [[[0.0, 0.0], [0.3, -0.1], [0.1, 0.2], [-0.2, 0.1]]], dtype=torch.float32, device=device
    )
    lengths   = torch.tensor([4], dtype=torch.long, device=device)
    with torch.no_grad():
        state = model.state_model.encode_history(
            times=times, locations=locations, lengths=lengths
        )
    return state


class TestQuerySurfaceContract(unittest.TestCase):
    """event_model.query_surface() satisfies the (G,) / float32 / non-negative contract."""

    def _check(self, preset, config=None, n_samples=20):
        import torch

        torch.manual_seed(42)
        model = build_model(
            config=config or {},
            preset=preset,
            spatial_dim=2,
            hidden_dim=16,
            event_cov_dim=0,
            field_cov_dim=0,
        )
        model.eval()

        state = _fake_state(model)
        grid_times = torch.zeros(_G, dtype=torch.float32)
        grid_locs  = torch.randn(_G, 2, dtype=torch.float32)

        out = model.event_model.query_surface(
            state=state,
            grid_times=grid_times,
            grid_locs=grid_locs,
            n_samples=n_samples,
        )

        self.assertEqual(out.shape, (_G,), f"{preset}: shape {out.shape} ≠ ({_G},)")
        self.assertEqual(out.dtype, torch.float32, f"{preset}: dtype {out.dtype} ≠ float32")
        self.assertTrue(
            (out >= 0).all(),
            f"{preset}: surface values must be non-negative, got {out}",
        )

    # ---- intensity-family presets (one-liner → self.intensity()) -----------

    def test_deep_stpp(self):
        self._check("deep_stpp")

    def test_auto_stpp(self):
        self._check("auto_stpp")

    def test_auto_stpp_legacy(self):
        self._check("auto_stpp_legacy")

    def test_factorized_gmm(self):
        self._check("hawkes_gmm")

    def test_factorized_cnf(self):
        self._check("hawkes_cnf", config=_FAST_CNF_CONFIG)

    # ---- proxy_kde presets (explicit KDE-based overrides) ------------------

    def test_smash(self):
        self._check("smash", n_samples=20)

    def test_diffusion_stpp(self):
        self._check("diffusion_stpp", n_samples=20)

    # ---- surface_query_type declaration ------------------------------------

    def test_surface_query_type_intensity_families(self):
        for preset in ("deep_stpp", "auto_stpp", "auto_stpp_legacy", "hawkes_gmm"):
            with self.subTest(preset=preset):
                model = build_model(
                    config={}, preset=preset, spatial_dim=2, hidden_dim=16,
                    event_cov_dim=0, field_cov_dim=0,
                )
                self.assertEqual(
                    model.event_model.surface_query_type, "intensity",
                    f"{preset}: expected surface_query_type='intensity'",
                )

    def test_surface_query_type_proxy_kde_families(self):
        for preset in ("smash", "diffusion_stpp"):
            with self.subTest(preset=preset):
                model = build_model(
                    config={}, preset=preset, spatial_dim=2, hidden_dim=16,
                    event_cov_dim=0, field_cov_dim=0,
                )
                self.assertEqual(
                    model.event_model.surface_query_type, "proxy_kde",
                    f"{preset}: expected surface_query_type='proxy_kde'",
                )

    def test_paper_neural_presets_are_benchmark_supported(self):
        for preset in ("njsde", "neural_jumpcnf", "neural_attncnf"):
            with self.subTest(preset=preset):
                self.assertEqual(ConfigRegistry.canonical_status(preset), "canonical")

    def test_base_event_model_raises(self):
        """Base EventModel.query_surface() must raise NotImplementedError."""
        import torch
        from unified_stpp.models.abstractions import EventModel, StateContext

        class _Bare(EventModel):
            @property
            def capabilities(self):
                from unified_stpp.models.abstractions import EventCapabilities
                return EventCapabilities()

            def training_loss(self, *, times, locations, lengths, state, **_):
                del times, locations, lengths, state
                return {}

        bare = _Bare()
        state = StateContext()
        with self.assertRaises(NotImplementedError):
            bare.query_surface(
                state=state,
                grid_times=torch.zeros(4),
                grid_locs=torch.zeros(4, 2),
            )


if __name__ == "__main__":
    unittest.main()
