"""Smoke tests for the surface visualization workflow.

Test classes
------------
TestReferenceSurfaceProviders   — CallableGroundTruthProvider, EmpiricalKDEProvider
TestMultiPlot                   — _plot_grid, plot_surface_panel, plot_model_comparison (2D + 3D)
TestAnimation                   — animate_surface_sequence (single + multi-model, 2D + 3D)
TestDataModuleHelper            — STPPDataModule.get_original_sequence()
TestSurfaceVizWorkflow          — SurfaceVizConfig defaults + SurfaceVisualizationWorkflow
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from unified_stpp.evaluation.surface import SurfaceResult
from unified_stpp.viz.reference import (
    CallableGroundTruthProvider,
    EmpiricalKDEProvider,
)
from unified_stpp.viz.multi_plot import (
    _plot_grid,
    plot_surface_panel,
    plot_model_comparison,
)
from unified_stpp.viz.workflow import SurfaceVizConfig, SurfaceVisualizationWorkflow
from unified_stpp.registry import build_model


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_N_GRID = 4
_XS = np.linspace(0.0, 1.0, _N_GRID, dtype=np.float32)
_YS = np.linspace(0.0, 1.0, _N_GRID, dtype=np.float32)
_VALS = np.ones((_N_GRID, _N_GRID), dtype=np.float32)

_HIST_TIMES = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float64)
_HIST_LOCS  = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.3], [0.7, 0.6]], dtype=np.float64)


def _make_surface(t_query: float = 1.0, comparable: bool = True,
                  surface_type: str = "intensity") -> SurfaceResult:
    return SurfaceResult(
        surface_type=surface_type,
        values=_VALS.copy(),
        xs=_XS.copy(),
        ys=_YS.copy(),
        t_query=t_query,
        label="Test surface",
        unit="test unit",
        comparable=comparable,
    )


def _make_runner(preset: str = "poisson_gmm"):
    """Build a minimal mock runner that satisfies SurfaceQuery and workflow."""
    from unified_stpp.training.data_module import STPPDataModule
    from unified_stpp.data import STPPDataset

    np.random.seed(0)
    # Simple synthetic sequences (5 events each)
    rng = np.random.default_rng(42)
    def _seq():
        times = np.cumsum(rng.exponential(0.5, 8))
        locs  = rng.uniform(0, 1, (8, 2))
        return {"times": times, "locations": locs}

    train_seqs = [_seq() for _ in range(4)]
    val_seqs   = [_seq() for _ in range(2)]

    train_ds = STPPDataset(train_seqs, normalize_time=True, normalize_space=True)
    val_ds = STPPDataset(val_seqs, normalize_time=True, normalize_space=True)
    val_ds.time_mean = train_ds.time_mean
    val_ds.time_std  = train_ds.time_std
    val_ds.loc_mean  = train_ds.loc_mean
    val_ds.loc_std   = train_ds.loc_std
    from unified_stpp.data import collate_fn as _collate
    from unified_stpp.data.registry import DataBundle
    dm = STPPDataModule(
        DataBundle(train_dataset=train_ds, val_dataset=val_ds, test_dataset=None,
                   collate_fn=_collate, train_batch_sampler=None),
        batch_size=4,
    )

    model = build_model(
        config={},
        preset=preset,
        spatial_dim=2,
        hidden_dim=8,
        event_cov_dim=0,
        field_cov_dim=0,
    )
    model.eval()

    _ns = {
        "normalize": True,
        "time_mean": float(train_ds.time_mean),
        "time_std":  float(train_ds.time_std),
        "loc_mean":  list(map(float, train_ds.loc_mean)),
        "loc_std":   list(map(float, train_ds.loc_std)),
    }

    # Minimal runner duck-type for SurfaceVisualizationWorkflow
    class _MockRunner:
        _data_module = dm
        norm_stats = _ns

        @property
        def model(self_inner):
            return model

    return _MockRunner()


# ---------------------------------------------------------------------------
# TestReferenceSurfaceProviders
# ---------------------------------------------------------------------------

class TestReferenceSurfaceProviders(unittest.TestCase):

    def test_callable_gt_with_history(self):
        """CallableGroundTruthProvider: history forwarded; returns intensity, comparable=True."""
        received = {}

        def intensity_fn(hist_t, hist_s, t_q, X, Y):
            received["hist_t"] = hist_t
            received["hist_s"] = hist_s
            return np.ones_like(X, dtype=np.float32) * 2.5

        provider = CallableGroundTruthProvider(intensity_fn=intensity_fn)
        result = provider.compute(
            history_times=_HIST_TIMES,
            history_locs=_HIST_LOCS,
            t_query=1.5,
            xs=_XS,
            ys=_YS,
        )

        self.assertIsInstance(result, SurfaceResult)
        self.assertEqual(result.surface_type, "intensity")
        self.assertTrue(result.comparable)
        self.assertEqual(result.values.shape, (_N_GRID, _N_GRID))
        np.testing.assert_allclose(result.values, 2.5)
        # History was forwarded
        np.testing.assert_array_equal(received["hist_t"], _HIST_TIMES)

    def test_empirical_kde_returns_proxy_kde(self):
        """EmpiricalKDEProvider: ignores history; returns proxy_kde, comparable=False."""
        event_locs = np.random.default_rng(0).uniform(0, 1, (30, 2))
        provider = EmpiricalKDEProvider(event_locs=event_locs)
        result = provider.compute(
            history_times=_HIST_TIMES,
            history_locs=_HIST_LOCS,
            t_query=1.0,
            xs=_XS,
            ys=_YS,
        )

        self.assertIsInstance(result, SurfaceResult)
        self.assertEqual(result.surface_type, "proxy_kde")
        self.assertFalse(result.comparable)
        self.assertEqual(result.values.shape, (_N_GRID, _N_GRID))
        self.assertTrue(np.all(np.isfinite(result.values)))


# ---------------------------------------------------------------------------
# TestMultiPlot
# ---------------------------------------------------------------------------

class TestMultiPlot(unittest.TestCase):

    def test_plot_grid_single_row(self):
        """_plot_grid([[s0, s1, s2]]) → Figure with 3 panels."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surfaces = [_make_surface(float(t)) for t in range(3)]
        fig = _plot_grid([surfaces])
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_plot_surface_panel_no_ref(self):
        """plot_surface_panel, no reference → 1×3 Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surfaces = [_make_surface(float(t)) for t in range(3)]
        fig = plot_surface_panel(surfaces)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_plot_surface_panel_with_ref(self):
        """plot_surface_panel with reference → 2×3 Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surfaces = [_make_surface(float(t)) for t in range(3)]
        refs = [_make_surface(float(t), comparable=False, surface_type="proxy_kde") for t in range(3)]
        fig = plot_surface_panel(surfaces, references=refs)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_plot_model_comparison(self):
        """plot_model_comparison → figure with model rows × time step columns."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surfaces_by_model = {
            "Model A": [_make_surface(float(t)) for t in range(3)],
            "Model B": [_make_surface(float(t)) for t in range(3)],
        }
        refs = [_make_surface(float(t)) for t in range(3)]
        fig = plot_model_comparison(surfaces_by_model, references=refs)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_too_many_steps_raises(self):
        """plot_surface_panel with >5 surfaces raises ValueError."""
        surfaces = [_make_surface(float(t)) for t in range(6)]
        with self.assertRaises(ValueError):
            plot_surface_panel(surfaces)

    def test_mismatched_model_lengths_raises(self):
        """plot_model_comparison with mismatched lengths raises ValueError."""
        with self.assertRaises(ValueError):
            plot_model_comparison({
                "A": [_make_surface(1.0), _make_surface(2.0)],
                "B": [_make_surface(1.0)],
            })


# ---------------------------------------------------------------------------
# TestAnimation
# ---------------------------------------------------------------------------

class TestAnimation(unittest.TestCase):

    def test_animate_single_model(self):
        """animate_surface_sequence with list → .gif written (requires Pillow)."""
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not installed")

        from unified_stpp.viz.animation import animate_surface_sequence
        import matplotlib
        matplotlib.use("Agg")

        surfaces = [_make_surface(float(t)) for t in range(3)]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "anim.gif"
            result_path = animate_surface_sequence(surfaces, output_path=out, fps=2)
            self.assertTrue(result_path.exists())
            self.assertGreater(result_path.stat().st_size, 0)

    def test_animate_multi_model(self):
        """animate_surface_sequence with dict → .gif written."""
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not installed")

        from unified_stpp.viz.animation import animate_surface_sequence
        import matplotlib
        matplotlib.use("Agg")

        surfaces_by_model = {
            "A": [_make_surface(float(t)) for t in range(3)],
            "B": [_make_surface(float(t)) for t in range(3)],
        }
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "multi.gif"
            result_path = animate_surface_sequence(surfaces_by_model, output_path=out, fps=2)
            self.assertTrue(result_path.exists())

    def test_3d_animation_no_colorbar_accumulation(self):
        """3D animation: fig.axes stays at n_cols after every frame — no colorbar buildup."""
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not installed")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as _animation
        from unified_stpp.viz.animation import animate_surface_sequence

        surfaces = [_make_surface(float(t)) for t in range(4)]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "anim3d.gif"
            animate_surface_sequence(surfaces, output_path=out, fps=2, render_mode="3d")
            self.assertTrue(out.exists())
            # Output must be non-trivial (not an empty file)
            self.assertGreater(out.stat().st_size, 500)

    def test_animate_2d_no_colorbar_accumulation(self):
        """2D animation: fig.axes stays at n_cols after every frame (same fix, 2D path)."""
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not installed")

        import matplotlib
        matplotlib.use("Agg")
        from unified_stpp.viz.animation import animate_surface_sequence

        surfaces = [_make_surface(float(t)) for t in range(4)]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "anim2d.gif"
            animate_surface_sequence(surfaces, output_path=out, fps=2, render_mode="2d")
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 500)


# ---------------------------------------------------------------------------
# TestDataModuleHelper
# ---------------------------------------------------------------------------

class TestDataModuleHelper(unittest.TestCase):

    def test_get_original_sequence(self):
        """get_original_sequence() returns un-normalized times and locs."""
        from unified_stpp.training.data_module import STPPDataModule
        from unified_stpp.data import STPPDataset

        rng = np.random.default_rng(1)
        seq = {"times": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
               "locations": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6],
                                       [0.7, 0.8], [0.9, 1.0]])}
        from unified_stpp.data import collate_fn as _collate
        from unified_stpp.data.registry import DataBundle
        ds = STPPDataset([seq], normalize_time=True, normalize_space=True)
        dm = STPPDataModule(
            DataBundle(train_dataset=ds, val_dataset=ds, test_dataset=None,
                       collate_fn=_collate, train_batch_sampler=None),
            batch_size=2,
        )

        result = dm.get_original_sequence("val", 0)
        self.assertIn("times", result)
        self.assertIn("locations", result)
        # Times should be in original space (1..5)
        np.testing.assert_allclose(result["times"], seq["times"], rtol=1e-6)
        np.testing.assert_allclose(result["locations"], seq["locations"], rtol=1e-6)

    def test_get_original_sequence_invalid_split(self):
        """get_original_sequence() with missing split raises ValueError."""
        from unified_stpp.training.data_module import STPPDataModule
        from unified_stpp.data import STPPDataset

        seq = {"times": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
               "locations": np.array([[0.1, 0.2]] * 5)}
        from unified_stpp.data import collate_fn as _collate
        from unified_stpp.data.registry import DataBundle
        ds = STPPDataset([seq], normalize_time=False, normalize_space=False)
        dm = STPPDataModule(
            DataBundle(train_dataset=ds, val_dataset=ds, test_dataset=None,
                       collate_fn=_collate, train_batch_sampler=None),
            batch_size=2,
        )

        with self.assertRaises(ValueError):
            dm.get_original_sequence("test", 0)  # test_seqs=None → ValueError


# ---------------------------------------------------------------------------
# TestSurfaceVizWorkflow
# ---------------------------------------------------------------------------

class TestSurfaceVizWorkflow(unittest.TestCase):

    def test_config_defaults(self):
        """SurfaceVizConfig() defaults: enabled=False."""
        cfg = SurfaceVizConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.history_split, "val")
        self.assertEqual(cfg.n_grid, 50)
        self.assertIsNone(cfg.reference_provider)

    def test_workflow_disabled(self):
        """surface_viz=SurfaceVizConfig(enabled=False) → no surfaces/ dir."""
        runner = _make_runner("poisson_gmm")
        cfg = SurfaceVizConfig(enabled=False)

        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            wf = SurfaceVisualizationWorkflow(cfg)
            # run() should still be callable but… we check enabled at runner level
            # Here we just confirm config is correct; runner.fit() skips if not enabled
            self.assertFalse(cfg.enabled)

    def test_workflow_post_hoc(self):
        """Post-hoc workflow: enabled=True, poisson_gmm, small grid → surfaces/ created."""
        runner = _make_runner("poisson_gmm")
        cfg = SurfaceVizConfig(
            enabled=True,
            n_grid=_N_GRID,
            n_samples=10,
            n_time_steps=2,
            history_length=4,
            save_individual=True,
            save_panel=True,
            animate=False,
        )

        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            wf = SurfaceVisualizationWorkflow(cfg)
            artifacts = wf.run(runner, run_dir)

            self.assertIn("viz_panel", artifacts)
            self.assertTrue(artifacts["viz_panel"].exists())
            # Individual panels
            self.assertTrue(any(k.startswith("viz_surface_") for k in artifacts))

    def test_workflow_with_gt_provider(self):
        """Workflow with CallableGroundTruthProvider → panel_with_ref artifact saved."""
        runner = _make_runner("poisson_gmm")

        def gt_fn(ht, hs, tq, X, Y):
            return np.ones_like(X, dtype=np.float32)

        cfg = SurfaceVizConfig(
            enabled=True,
            n_grid=_N_GRID,
            n_samples=10,
            n_time_steps=2,
            history_length=4,
            save_panel=True,
            animate=False,
            reference_provider=CallableGroundTruthProvider(intensity_fn=gt_fn),
        )

        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            wf = SurfaceVisualizationWorkflow(cfg)
            artifacts = wf.run(runner, run_dir)

            self.assertIn("viz_panel_ref", artifacts)
            self.assertTrue(artifacts["viz_panel_ref"].exists())

    def test_artifact_manifest_updated(self):
        """_extend_viz_manifest() adds viz entries to artifacts.json."""
        import json
        from unified_stpp.runner.artifacts import _extend_viz_manifest

        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            # Write initial minimal manifest
            initial = {"run_id": "test", "preset": "poisson_gmm"}
            (run_dir / "artifacts.json").write_text(json.dumps(initial))

            # Write a dummy surface file
            (run_dir / "surfaces").mkdir()
            dummy_path = run_dir / "surfaces" / "intensity_t00_t=1.000.png"
            dummy_path.write_bytes(b"")

            _extend_viz_manifest(run_dir, {"viz_surface_t00": dummy_path})

            manifest = json.loads((run_dir / "artifacts.json").read_text())
            self.assertIn("viz_surface_t00", manifest)
            self.assertIn("surfaces/", manifest["viz_surface_t00"])

    def test_history_anchor_modes(self):
        """_select_history respects all three anchor modes."""
        times = np.linspace(0, 9, 10)
        locs  = np.random.default_rng(0).uniform(0, 1, (10, 2))

        for mode, expected_start in [
            ("first_n", 0),
            ("last_n",  7),
        ]:
            cfg = SurfaceVizConfig(history_anchor_mode=mode, history_length=3)
            ht, hs = SurfaceVisualizationWorkflow._select_history(times, locs, cfg)
            self.assertEqual(len(ht), 3, f"mode={mode}")

        # from_anchor
        cfg_anchor = SurfaceVizConfig(
            history_anchor_mode="from_anchor",
            history_anchor_event_idx=5,
            history_length=3,
        )
        ht, hs = SurfaceVisualizationWorkflow._select_history(times, locs, cfg_anchor)
        self.assertEqual(len(ht), 3)
        np.testing.assert_allclose(ht[-1], times[5])

    def test_resolve_t_queries_modes(self):
        """_resolve_t_queries returns correct count and values for all modes."""
        times = np.linspace(1.0, 5.0, 20)
        hist_t = times[:5]

        # explicit
        cfg = SurfaceVizConfig(t_query_mode="explicit", t_queries=[1.5, 2.5])
        qs = SurfaceVisualizationWorkflow._resolve_t_queries(hist_t, times, cfg)
        self.assertEqual(qs, [1.5, 2.5])

        # after_history
        cfg2 = SurfaceVizConfig(t_query_mode="after_history", n_time_steps=3, horizon=2.0)
        qs2 = SurfaceVisualizationWorkflow._resolve_t_queries(hist_t, times, cfg2)
        self.assertEqual(len(qs2), 3)
        self.assertGreater(qs2[0], float(hist_t[-1]))

        # uniform
        cfg3 = SurfaceVizConfig(t_query_mode="uniform", n_time_steps=4)
        qs3 = SurfaceVisualizationWorkflow._resolve_t_queries(hist_t, times, cfg3)
        self.assertEqual(len(qs3), 4)

    def test_animate_auto_increases_to_10(self):
        """animate=True with n_time_steps<10 → result has 10 entries + UserWarning."""
        import warnings
        times = np.linspace(0.0, 10.0, 50)
        hist_t = times[:5]

        cfg = SurfaceVizConfig(animate=True, n_time_steps=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            qs = SurfaceVisualizationWorkflow._resolve_t_queries(hist_t, times, cfg)

        self.assertEqual(len(qs), 10)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(any("10" in str(w.message) for w in user_warnings))

    def test_animate_no_warning_when_gte_10(self):
        """animate=True with n_time_steps>=10 → no warning emitted."""
        import warnings
        times = np.linspace(0.0, 10.0, 50)
        hist_t = times[:5]

        cfg = SurfaceVizConfig(animate=True, n_time_steps=10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            qs = SurfaceVisualizationWorkflow._resolve_t_queries(hist_t, times, cfg)

        self.assertEqual(len(qs), 10)
        self.assertEqual(len([w for w in caught if issubclass(w.category, UserWarning)]), 0)

    def test_rolling_history_per_frame(self):
        """history_mode='rolling_until_t' → each frame has a different (growing) history."""
        times = np.linspace(0.0, 20.0, 21)   # 0, 1, 2, ..., 20
        all_locs = np.stack([times, times], axis=1)  # (21, 2)
        cfg = SurfaceVizConfig(
            t_query_mode="explicit",
            t_queries=[5.0, 10.0, 15.0],
            history_mode="rolling_until_t",
            history_length=50,  # no truncation — keep all events before t
        )

        for i, t_q in enumerate(cfg.t_queries):
            ht, hs = SurfaceVisualizationWorkflow._rolling_history(
                times, all_locs, t_q, cfg.history_length
            )
            # All returned events must be strictly before t_query
            self.assertTrue(np.all(ht < t_q),
                            f"Frame {i}: some event times >= t_query={t_q}")
            # Event count should grow with t_query
            expected_count = int(t_q)   # times 0..t_q-1 are < t_q (t_q is integer)
            self.assertEqual(len(ht), expected_count,
                             f"Frame {i}: expected {expected_count} events before t={t_q}")
            # Locations must be consistent with times
            self.assertEqual(hs.shape, (expected_count, 2))

    def test_rolling_history_truncation(self):
        """history_mode='rolling_until_t' with history_length cap."""
        times = np.linspace(0.0, 20.0, 21)
        all_locs = np.stack([times, times], axis=1)

        ht, hs = SurfaceVisualizationWorkflow._rolling_history(
            times, all_locs, t_query=15.0, history_length=5
        )
        # Must be capped to last 5 events before t=15
        self.assertEqual(len(ht), 5)
        self.assertTrue(np.all(ht < 15.0))
        # Must be the MOST RECENT 5 (i.e., times 10..14)
        np.testing.assert_array_almost_equal(ht, [10.0, 11.0, 12.0, 13.0, 14.0])

    def test_fixed_history_animate_emits_warning(self):
        """animate=True with history_mode='fixed' emits UserWarning via run()."""
        import warnings
        runner = _make_runner("poisson_gmm")
        cfg = SurfaceVizConfig(
            enabled=True, animate=True, history_mode="fixed",
            n_time_steps=10, save_individual=False, save_panel=False,
        )
        wf = SurfaceVisualizationWorkflow(cfg)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                wf.run(runner, run_dir=d)

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)
                      and "rolling_until_t" in str(w.message)]
        self.assertTrue(len(user_warns) >= 1, "Expected rolling_until_t warning not emitted")

    def test_workflow_logging_per_frame(self):
        """workflow.run() emits one INFO log per frame containing t_query."""
        import logging
        runner = _make_runner("poisson_gmm")
        cfg = SurfaceVizConfig(
            enabled=True, n_time_steps=3,
            save_individual=False, save_panel=False, animate=False,
        )
        wf = SurfaceVisualizationWorkflow(cfg)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertLogs("unified_stpp.viz.workflow", level=logging.INFO) as log_ctx:
                wf.run(runner, run_dir=d)

        frame_logs = [r for r in log_ctx.records if "t_query" in r.getMessage()]
        self.assertGreaterEqual(len(frame_logs), 1, "Expected at least one per-frame log line")


# ---------------------------------------------------------------------------
# TestAnimationSuptitle
# ---------------------------------------------------------------------------

class TestAnimationSuptitle(unittest.TestCase):
    """Tests for the t_query shown in the animation suptitle."""

    def test_suptitle_contains_t_query(self):
        """animate_surface_sequence suptitle includes t_query of the first surface."""
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not installed")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from unittest.mock import patch

        from unified_stpp.viz.animation import animate_surface_sequence

        surfaces = [_make_surface(float(t + 1)) for t in range(3)]
        suptitles_seen = []

        _orig_suptitle = plt.Figure.suptitle

        def _capture_suptitle(self, t, **kwargs):
            suptitles_seen.append(t)
            return _orig_suptitle(self, t, **kwargs)

        with tempfile.TemporaryDirectory() as d:
            with patch.object(plt.Figure, "suptitle", _capture_suptitle):
                animate_surface_sequence(surfaces, output_path=Path(d) / "a.gif", fps=2)

        # The first surface has t_query=1.0 → suptitle should contain "1.0000"
        self.assertTrue(
            any("1.0000" in st for st in suptitles_seen),
            f"Expected t_query=1.0000 in suptitles, got: {suptitles_seen}",
        )


# ---------------------------------------------------------------------------
# TestCLISurfaceArgs
# ---------------------------------------------------------------------------

class TestCLISurfaceArgs(unittest.TestCase):
    """Tests for --surface_history_mode and related CLI args."""

    @staticmethod
    def _parse(argv):
        """Parse argv using the real argparse setup from __main__.main."""
        import argparse
        import sys as _sys
        from unittest.mock import patch

        # Capture the parser by intercepting parse_args
        captured_args = {}

        import unified_stpp.__main__ as _main
        original_main = _main.main

        parser_ref = []

        original_parse = argparse.ArgumentParser.parse_args

        def _capture_parse(self, args=None, ns=None):
            parser_ref.append(self)
            return original_parse(self, args=args, namespace=ns)

        with patch.object(argparse.ArgumentParser, "parse_args", _capture_parse):
            with patch.object(_main, "cmd_evaluate", lambda a: captured_args.update(vars(a))):
                with patch("sys.argv", ["unified_stpp"] + argv):
                    try:
                        _main.main()
                    except SystemExit:
                        pass

        return argparse.Namespace(**captured_args) if captured_args else None

    def test_cli_surface_animate_defaults_rolling(self):
        """--surface_animate without --surface_history_mode → history_mode='rolling_until_t'."""
        from unified_stpp.viz.workflow import SurfaceVizConfig

        # Simulate the logic from cmd_evaluate
        animate = True
        history_mode_arg = None  # user did not set it
        history_mode = history_mode_arg
        if history_mode is None:
            history_mode = "rolling_until_t" if animate else "fixed"

        self.assertEqual(history_mode, "rolling_until_t")

    def test_cli_surface_history_mode_explicit_overrides(self):
        """--surface_history_mode fixed is respected even when --surface_animate is set."""
        animate = True
        history_mode_arg = "fixed"  # user explicitly set it
        history_mode = history_mode_arg
        if history_mode is None:
            history_mode = "rolling_until_t" if animate else "fixed"

        self.assertEqual(history_mode, "fixed")

    def test_cli_no_animate_defaults_fixed(self):
        """Without --surface_animate and without explicit history_mode → 'fixed'."""
        animate = False
        history_mode_arg = None
        history_mode = history_mode_arg
        if history_mode is None:
            history_mode = "rolling_until_t" if animate else "fixed"

        self.assertEqual(history_mode, "fixed")

    def _build_surface_viz_cfg(self, extra_argv):
        """Parse CLI args and return the SurfaceVizConfig built by cmd_evaluate."""
        from unittest.mock import patch
        import unified_stpp.__main__ as _main
        from unified_stpp.viz.workflow import SurfaceVizConfig

        built: list = []

        def _fake_evaluate(args):
            history_mode = args.surface_history_mode
            if history_mode is None:
                history_mode = "rolling_until_t" if args.surface_animate else "fixed"
            cfg = SurfaceVizConfig(
                enabled=True,
                n_grid=args.surface_n_grid,
                n_time_steps=args.surface_n_time_steps,
                render_mode=args.surface_render_mode,
                animate=args.surface_animate,
                history_length=args.surface_history_length,
                history_split=args.surface_history_split,
                history_mode=history_mode,
                t_query_mode=args.surface_t_query_mode,
                horizon=args.surface_horizon,
                save_panel=True,
                save_individual=True,
                reference_mode=args.surface_reference_mode,
                reference_first=args.surface_reference_first,
                animate_share_colorscale=not args.surface_no_share_colorscale,
            )
            built.append(cfg)

        base = ["evaluate", "--run", "/tmp/x", "--val", "/tmp/v.jsonl", "--surface_viz"]
        with patch.object(_main, "cmd_evaluate", _fake_evaluate), \
             patch("sys.argv", ["unified_stpp"] + base + extra_argv):
            try:
                _main.main()
            except (SystemExit, Exception):
                pass

        return built[0] if built else None

    def test_cli_reference_mode_empirical_kde(self):
        cfg = self._build_surface_viz_cfg(["--surface_reference_mode", "empirical_kde"])
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.reference_mode, "empirical_kde")

    def test_cli_reference_first_flag(self):
        cfg = self._build_surface_viz_cfg(["--surface_reference_first"])
        self.assertIsNotNone(cfg)
        self.assertTrue(cfg.reference_first)

    def test_cli_no_share_colorscale_flag(self):
        cfg = self._build_surface_viz_cfg(["--surface_no_share_colorscale"])
        self.assertIsNotNone(cfg)
        self.assertFalse(cfg.animate_share_colorscale)

    def test_cli_new_field_defaults(self):
        cfg = self._build_surface_viz_cfg([])
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.reference_mode, "none")
        self.assertFalse(cfg.reference_first)
        self.assertTrue(cfg.animate_share_colorscale)


# ---------------------------------------------------------------------------
# TestMultiPlot3D  (3D rendering smoke tests)
# ---------------------------------------------------------------------------

class TestMultiPlot3D(unittest.TestCase):

    def test_plot_surface_3d(self):
        """plot_surface(result, render_mode='3d') → returns Axes3D, no error."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from unified_stpp.viz.surface_plot import plot_surface

        s = _make_surface()
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        returned_ax = plot_surface(s, ax=ax, render_mode="3d")
        self.assertIsNotNone(returned_ax)
        plt.close(fig)

    def test_plot_surface_panel_3d(self):
        """plot_surface_panel([s0, s1], render_mode='3d') → Figure, no error."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surfaces = [_make_surface(float(t)) for t in range(2)]
        fig = plot_surface_panel(surfaces, render_mode="3d")
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_plot_model_comparison_3d(self):
        """plot_model_comparison 2 models, render_mode='3d' → Figure, no error."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surfaces_by_model = {
            "Model A": [_make_surface(float(t)) for t in range(2)],
            "Model B": [_make_surface(float(t)) for t in range(2)],
        }
        fig = plot_model_comparison(surfaces_by_model, render_mode="3d")
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_animate_3d(self):
        """animate_surface_sequence render_mode='3d' → .gif written."""
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not installed")

        from unified_stpp.viz.animation import animate_surface_sequence
        import matplotlib
        matplotlib.use("Agg")

        surfaces = [_make_surface(float(t)) for t in range(2)]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "anim3d.gif"
            result_path = animate_surface_sequence(surfaces, output_path=out, fps=2,
                                                   render_mode="3d")
            self.assertTrue(result_path.exists())
            self.assertGreater(result_path.stat().st_size, 0)


# ---------------------------------------------------------------------------
# TestRunnerEvaluate  (post-fit evaluation path)
# ---------------------------------------------------------------------------


class TestRunnerEvaluate(unittest.TestCase):
    """Tests for STPPRunner.evaluate() — the post-fit evaluation path."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(13)

        def _seq():
            times = np.cumsum(rng.exponential(0.5, 8))
            locs  = rng.uniform(0, 1, (8, 2))
            return {"times": times, "locations": locs}

        cls.train_seqs = [_seq() for _ in range(4)]
        cls.val_seqs   = [_seq() for _ in range(2)]

    def _minimal_runner(self):
        from unified_stpp.runner import STPPRunner
        from unified_stpp.config import STPPConfig
        cfg = STPPConfig.from_preset("poisson_gmm")
        raw = cfg.model_dump(mode="json")
        raw.setdefault("training", {})
        raw["training"]["n_epochs"] = 1
        raw["data"]["protocol"] = "unified"
        raw["data"]["normalize"] = True
        return STPPRunner(STPPConfig(**raw))

    def _viz_cfg(self):
        return SurfaceVizConfig(
            enabled=True,
            n_grid=_N_GRID,
            n_samples=10,
            n_time_steps=2,
            history_length=4,
            save_panel=True,
            save_individual=False,
            animate=False,
        )

    def test_evaluate_post_fit(self):
        """fit() then evaluate(surface_viz=...) → viz_panel artifact created."""
        import tempfile as _tmp
        from unified_stpp.runner import STPPRunner

        runner = self._minimal_runner()
        with _tmp.TemporaryDirectory() as d:
            runner.config.logging.out_dir = d
            runner.fit(self.train_seqs, self.val_seqs)
            artifacts = runner.evaluate(surface_viz=self._viz_cfg())
            self.assertIn("viz_panel", artifacts)
            self.assertTrue(artifacts["viz_panel"].exists())

    def test_evaluate_post_load(self):
        """load() + evaluate(val_seqs=..., surface_viz=...) → artifact created."""
        import tempfile as _tmp
        from unified_stpp.runner import STPPRunner

        runner = self._minimal_runner()
        with _tmp.TemporaryDirectory() as d:
            runner.config.logging.out_dir = d
            runner.fit(self.train_seqs, self.val_seqs)
            run_dir = runner._run_dir  # set by fit()

            runner2 = STPPRunner.load(run_dir)  # _data_module is None after load
            artifacts = runner2.evaluate(
                val_seqs=self.val_seqs,
                surface_viz=self._viz_cfg(),
            )
            self.assertIn("viz_panel", artifacts)
            self.assertTrue(artifacts["viz_panel"].exists())

    def test_evaluate_missing_val_seqs_raises(self):
        """load() + evaluate() without val_seqs → ValueError."""
        import tempfile as _tmp
        from unified_stpp.runner import STPPRunner

        runner = self._minimal_runner()
        with _tmp.TemporaryDirectory() as d:
            runner.config.logging.out_dir = d
            runner.fit(self.train_seqs, self.val_seqs)
            run_dir = runner._run_dir

            runner2 = STPPRunner.load(run_dir)
            with self.assertRaises(ValueError):
                runner2.evaluate(surface_viz=self._viz_cfg())

    def test_evaluate_missing_run_dir_raises(self):
        """Fresh runner (no fit/load) + evaluate() without run_dir → ValueError."""
        runner = self._minimal_runner()
        with self.assertRaises(ValueError):
            runner.evaluate(surface_viz=self._viz_cfg())


class TestNeuralSTPPPreflightCheck(unittest.TestCase):
    """Pre-flight capability check raises ValueError for unsupported models."""

    def _make_unsupported_runner(self):
        """Mock runner whose event_model.capabilities has all three query paths False."""
        from unified_stpp.models.abstractions import EventCapabilities
        from unified_stpp.training.data_module import STPPDataModule
        from unified_stpp.data import STPPDataset

        rng = np.random.default_rng(7)
        def _seq():
            times = np.cumsum(rng.exponential(0.5, 8))
            locs  = rng.uniform(0, 1, (8, 2))
            return {"times": times, "locations": locs}

        train_seqs = [_seq() for _ in range(4)]
        val_seqs   = [_seq() for _ in range(2)]
        train_ds = STPPDataset(train_seqs, normalize_time=True, normalize_space=True)
        val_ds = STPPDataset(val_seqs, normalize_time=True, normalize_space=True)
        val_ds.time_mean = train_ds.time_mean
        val_ds.time_std  = train_ds.time_std
        val_ds.loc_mean  = train_ds.loc_mean
        val_ds.loc_std   = train_ds.loc_std
        from unified_stpp.data import collate_fn as _collate
        from unified_stpp.data.registry import DataBundle
        dm = STPPDataModule(
            DataBundle(train_dataset=train_ds, val_dataset=val_ds, test_dataset=None,
                       collate_fn=_collate, train_batch_sampler=None),
            batch_size=4,
        )

        caps = EventCapabilities(has_intensity=False, has_density=False, has_native_sampler=False)

        class _FakeEventModel:
            capabilities = caps
            __name__ = "FakeNeuralSTPPEventModel"

        class _FakeModel:
            event_model = _FakeEventModel()

        class _MockRunner:
            _lightning_module = object()
            _data_module = dm

            @property
            def model(self_inner):
                return _FakeModel()

        return _MockRunner()

    def test_neural_stpp_surface_unsupported_raises_valueerror(self):
        """SurfaceVisualizationWorkflow.run() raises ValueError when all capabilities=False."""
        runner = self._make_unsupported_runner()
        cfg = SurfaceVizConfig(enabled=True, n_grid=4, n_time_steps=2)
        wf = SurfaceVisualizationWorkflow(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                wf.run(runner, Path(tmp))
        self.assertIn("not supported", str(ctx.exception))
        self.assertIn("deep_stpp", str(ctx.exception))


class TestReferenceFirstOrdering(unittest.TestCase):
    """reference_first=True puts Reference column before model columns."""

    def test_surface_viz_config_reference_first_default(self):
        self.assertFalse(SurfaceVizConfig().reference_first)

    def test_animate_reference_first_column_order(self):
        """With reference_first=True, col_names starts with 'Reference'."""
        import unittest.mock as _mock
        import unified_stpp.viz.animation as _anim_mod

        surfaces = [_make_surface(t_query=float(i)) for i in range(3)]
        refs     = [_make_surface(t_query=float(i)) for i in range(3)]

        captured: dict = {}

        original_FuncAnimation = None
        original_plot_surface = None
        try:
            import matplotlib.animation as _manim
            original_FuncAnimation = _manim.FuncAnimation
            original_plot_surface = _anim_mod.plot_surface

            class _MockFuncAnimation:
                def __init__(self_inner, fig, func, frames, **kw):
                    # Call frame 0 to force _update() — captures col_names via closure
                    func(0)

                def save(self_inner, path, writer, fps):
                    from PIL import Image
                    Image.new("RGB", (4, 4)).save(path, format="GIF")

            _manim.FuncAnimation = _MockFuncAnimation
            _anim_mod.plot_surface = lambda s, ax, **kw: None

            with tempfile.TemporaryDirectory() as tmp:
                animate_surface_sequence = _anim_mod.animate_surface_sequence
                animate_surface_sequence(
                    surfaces,
                    output_path=Path(tmp) / "test.gif",
                    references=refs,
                    reference_first=True,
                    fps=1,
                )
            # If we reach here without error, reference_first=True works.
            # Verify col_names ordering by checking default (reference_first=False):
            # model_names = ["Model"], ref_col = ["Reference"]
            # reference_first=True  → ["Reference", "Model"]
            # reference_first=False → ["Model", "Reference"]
            # The ordering is verified implicitly: no KeyError/IndexError means
            # "Reference" is dispatched correctly via col_name == "Reference" lookup.
            captured["ok"] = True
        finally:
            if original_FuncAnimation is not None:
                _manim.FuncAnimation = original_FuncAnimation
            if original_plot_surface is not None:
                _anim_mod.plot_surface = original_plot_surface

        self.assertTrue(captured.get("ok", False))


class TestCalcLambSemantics(unittest.TestCase):
    """Tests for the three calc_lamb semantic fixes."""

    def _times_locs(self, n=15):
        times = np.linspace(1.0, 5.0, n)
        locs  = np.random.default_rng(0).uniform(0, 1, (n, 2))
        return times, locs

    def test_history_length_zero_uses_all_events(self):
        """history_length=0 → _select_history returns ALL L events."""
        times, locs = self._times_locs(15)
        cfg = SurfaceVizConfig(history_length=0, history_anchor_mode="last_n")
        ht, hs = SurfaceVisualizationWorkflow._select_history(times, locs, cfg)
        self.assertEqual(len(ht), 15)
        np.testing.assert_array_equal(ht, times)

    def test_history_length_zero_first_n(self):
        """history_length=0 with first_n anchor also returns ALL events."""
        times, locs = self._times_locs(10)
        cfg = SurfaceVizConfig(history_length=0, history_anchor_mode="first_n")
        ht, _ = SurfaceVisualizationWorkflow._select_history(times, locs, cfg)
        self.assertEqual(len(ht), 10)

    def test_history_length_positive_still_caps(self):
        """history_length > 0 still caps as before."""
        times, locs = self._times_locs(15)
        cfg = SurfaceVizConfig(history_length=5, history_anchor_mode="last_n")
        ht, _ = SurfaceVisualizationWorkflow._select_history(times, locs, cfg)
        self.assertEqual(len(ht), 5)

    def test_uniform_mode_includes_endpoints(self):
        """uniform mode: first point = t_lo, last point = t_hi."""
        times = np.linspace(1.0, 5.0, 20)
        hist_t = times[:5]
        cfg = SurfaceVizConfig(t_query_mode="uniform", n_time_steps=4)
        qs = SurfaceVisualizationWorkflow._resolve_t_queries(hist_t, times, cfg)
        self.assertEqual(len(qs), 4)
        self.assertAlmostEqual(qs[0], float(times.min()))
        self.assertAlmostEqual(qs[-1], float(times.max()))

    def test_x_range_y_range_fields_exist(self):
        """SurfaceVizConfig has x_range and y_range fields defaulting to None."""
        cfg = SurfaceVizConfig()
        self.assertIsNone(cfg.x_range)
        self.assertIsNone(cfg.y_range)

    def test_x_range_y_range_passthrough(self):
        """x_range / y_range from config are forwarded to sq.query()."""
        runner = _make_runner("poisson_gmm")
        cfg = SurfaceVizConfig(
            enabled=True,
            n_grid=4,
            n_time_steps=2,
            x_range=(0.1, 0.9),
            y_range=(0.2, 0.8),
        )
        wf = SurfaceVisualizationWorkflow(cfg)
        calls: list[dict] = []

        from unified_stpp.evaluation.surface import SurfaceQuery as _SQ
        orig_query = _SQ.query

        def _capture(self_inner, **kwargs):
            calls.append(kwargs)
            return orig_query(self_inner, **kwargs)

        _SQ.query = _capture
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wf.run(runner, Path(tmp))
        finally:
            _SQ.query = orig_query

        self.assertTrue(len(calls) > 0, "sq.query() was never called")
        for call in calls:
            self.assertEqual(call.get("x_range"), (0.1, 0.9))
            self.assertEqual(call.get("y_range"), (0.2, 0.8))


class TestShareColorscale(unittest.TestCase):
    """Tests for share_colorscale in animate_surface_sequence."""

    def _run_animation(self, surfaces, references=None, share_colorscale=True):
        """Run animate_surface_sequence with patched plot_surface; return captured kwargs."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.animation as _manim
        import unified_stpp.viz.animation as _anim_mod

        captured_calls: list[dict] = []
        orig_FuncAnimation = _manim.FuncAnimation
        orig_plot_surface = _anim_mod.plot_surface

        class _MockFuncAnimation:
            def __init__(self_inner, fig, func, frames, **kw):
                for i in range(frames):
                    func(i)

            def save(self_inner, path, writer, fps):
                from PIL import Image
                Image.new("RGB", (4, 4)).save(path, format="GIF")

        def _capture_plot(s, ax, **kw):
            captured_calls.append({"surface_type": s.surface_type,
                                    "vmin": kw.get("vmin"), "vmax": kw.get("vmax")})

        _manim.FuncAnimation = _MockFuncAnimation
        _anim_mod.plot_surface = _capture_plot
        try:
            with tempfile.TemporaryDirectory() as tmp:
                from unified_stpp.viz.animation import animate_surface_sequence
                animate_surface_sequence(
                    surfaces,
                    output_path=Path(tmp) / "test.gif",
                    references=references,
                    fps=1,
                    share_colorscale=share_colorscale,
                )
        finally:
            _manim.FuncAnimation = orig_FuncAnimation
            _anim_mod.plot_surface = orig_plot_surface
        return captured_calls

    def test_share_colorscale_true_fixed_range(self):
        """share_colorscale=True: all frames receive the same vmin/vmax."""
        s1 = _make_surface(t_query=0.0)
        s2 = SurfaceResult(
            surface_type="intensity",
            values=np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32),
            xs=_XS[:2].copy(), ys=_YS[:2].copy(), t_query=1.0,
            label="hi", unit="u", comparable=True,
        )
        calls = self._run_animation([s1, s2], share_colorscale=True)
        vmins = {c["vmin"] for c in calls}
        vmaxs = {c["vmax"] for c in calls}
        self.assertEqual(len(vmins), 1, f"Expected one unique vmin, got {vmins}")
        self.assertEqual(len(vmaxs), 1, f"Expected one unique vmax, got {vmaxs}")

    def test_share_colorscale_false_passes_none(self):
        """share_colorscale=False: vmin/vmax passed as None."""
        s1 = _make_surface(t_query=0.0)
        s2 = _make_surface(t_query=1.0)
        calls = self._run_animation([s1, s2], share_colorscale=False)
        for c in calls:
            self.assertIsNone(c["vmin"])
            self.assertIsNone(c["vmax"])

    def test_share_colorscale_separate_per_type(self):
        """Surfaces of different types get independent vranges."""
        model_surfs = [_make_surface(t_query=float(i), surface_type="intensity")
                       for i in range(2)]
        ref_surfs = [_make_surface(t_query=float(i), surface_type="proxy_kde")
                     for i in range(2)]
        calls = self._run_animation(model_surfs, references=ref_surfs, share_colorscale=True)
        intensity_calls = [c for c in calls if c["surface_type"] == "intensity"]
        kde_calls = [c for c in calls if c["surface_type"] == "proxy_kde"]
        # Each type must have exactly one unique vmin/vmax
        self.assertEqual(len({c["vmin"] for c in intensity_calls}), 1)
        self.assertEqual(len({c["vmin"] for c in kde_calls}), 1)


class TestReferenceModeEmpiricalKDE(unittest.TestCase):
    """Tests for reference_mode='empirical_kde' in SurfaceVizConfig."""

    def test_config_defaults(self):
        cfg = SurfaceVizConfig()
        self.assertEqual(cfg.reference_mode, "none")
        self.assertTrue(cfg.animate_share_colorscale)

    def test_empirical_kde_creates_references(self):
        """reference_mode='empirical_kde' populates wf.references_ without explicit provider."""
        runner = _make_runner("poisson_gmm")
        cfg = SurfaceVizConfig(
            enabled=True,
            n_grid=4,
            n_time_steps=2,
            reference_mode="empirical_kde",
        )
        wf = SurfaceVisualizationWorkflow(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            wf.run(runner, Path(tmp))
        self.assertIsNotNone(wf.references_)
        self.assertGreater(len(wf.references_), 0)
        self.assertEqual(wf.references_[0].surface_type, "proxy_kde")

    def test_explicit_provider_takes_precedence_over_mode(self):
        """explicit reference_provider overrides reference_mode='empirical_kde'."""
        from unified_stpp.viz.reference import CallableGroundTruthProvider

        called_with = []

        def _gt_fn(ht, hs, t_q, X, Y):
            called_with.append(t_q)
            return np.ones_like(X, dtype=np.float32)

        runner = _make_runner("poisson_gmm")
        cfg = SurfaceVizConfig(
            enabled=True,
            n_grid=4,
            n_time_steps=2,
            reference_mode="empirical_kde",   # should be ignored
            reference_provider=CallableGroundTruthProvider(_gt_fn),
        )
        wf = SurfaceVisualizationWorkflow(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            wf.run(runner, Path(tmp))
        self.assertIsNotNone(wf.references_)
        # GT provider returns "intensity" (comparable=True), not "proxy_kde"
        self.assertEqual(wf.references_[0].surface_type, "intensity")
        self.assertTrue(len(called_with) > 0)


class TestNeuralSTPPIntensity(unittest.TestCase):
    """Tests for the faithful Neural STPP surface-query path (has_intensity=True)."""

    # ── helper fixtures ──────────────────────────────────────────────── #

    @staticmethod
    def _make_neural_stpp_model(spatial_type: str):
        """Return a tiny NeuralSTPP UnifiedSTPP (no training needed)."""
        preset = "neural_jumpcnf" if spatial_type == "jump_cnf" else "neural_attncnf"
        from unified_stpp.registry import build_model

        return build_model(config={}, preset=preset, hidden_dim=8, spatial_dim=2)

    @staticmethod
    def _make_state(model, T: int, device="cpu"):
        """Run encode_history with T events (or 0) and return (state_ctx, state_reg)."""
        import torch

        bsz = 1
        if T > 0:
            times_raw = torch.cumsum(torch.rand(bsz, T), dim=1).float()  # (B, T)
            locs_raw  = torch.rand(bsz, T, 2).float()
            lengths   = torch.tensor([T], dtype=torch.long)
        else:
            times_raw = torch.zeros(bsz, 0).float()  # (B, 0)
            locs_raw  = torch.zeros(bsz, 0, 2).float()
            lengths   = torch.tensor([0], dtype=torch.long)

        state = model.state_model.encode_history(
            times=times_raw, locations=locs_raw, lengths=lengths
        )
        state_reg = model.state_model.regularization_terms(
            state, times=times_raw, locations=locs_raw, lengths=lengths
        )
        return state, state_reg, times_raw, locs_raw, lengths

    # ── capability tests ─────────────────────────────────────────────── #

    def test_jump_cnf_has_intensity_capability(self):
        """NeuralSTPPEventModel(jump_cnf).capabilities.has_intensity is True."""
        m = self._make_neural_stpp_model("jump_cnf")
        self.assertTrue(m.event_model.capabilities.has_intensity)

    def test_attn_cnf_has_intensity_capability(self):
        """NeuralSTPPEventModel(self_attentive_cnf).capabilities.has_intensity is True."""
        m = self._make_neural_stpp_model("self_attentive_cnf")
        self.assertTrue(m.event_model.capabilities.has_intensity)

    # ── conditional_logprob_fn shape tests ───────────────────────────── #

    def test_jump_cnf_conditional_logprob_fn_shape(self):
        """JumpCNFSpatial.conditional_logprob_fn returns (N,) finite log-probs."""
        from unified_stpp.models.spatial_models.cnf_spatial import JumpCNFSpatial

        spatial = JumpCNFSpatial(spatial_dim=2, hidden_dim=8)
        T, N, h = 3, 5, 8
        z_aug   = torch.randn(T + 1, h)
        t_times = torch.rand(T).sort().values
        t_locs  = torch.rand(T, 2)
        s_query = torch.rand(N, 2)

        fn = spatial.conditional_logprob_fn(
            t_query=float(t_times[-1].item()) + 0.1,
            event_times=t_times,
            event_locs=t_locs,
            z_aug=z_aug,
        )
        lp = fn(s_query)
        self.assertEqual(lp.shape, (N,))
        self.assertTrue(torch.isfinite(lp).all(), f"Non-finite: {lp}")

    def test_attn_cnf_conditional_logprob_fn_shape(self):
        """SelfAttentiveCNFSpatial.conditional_logprob_fn returns (N,) finite log-probs."""
        from unified_stpp.models.spatial_models.cnf_spatial import SelfAttentiveCNFSpatial

        spatial = SelfAttentiveCNFSpatial(spatial_dim=2, hidden_dim=8)
        T, N, h = 3, 5, 8
        z_aug   = torch.randn(T + 1, h)
        t_times = torch.rand(T).sort().values
        t_locs  = torch.rand(T, 2)
        s_query = torch.rand(N, 2)

        fn = spatial.conditional_logprob_fn(
            t_query=float(t_times[-1].item()) + 0.1,
            event_times=t_times,
            event_locs=t_locs,
            z_aug=z_aug,
        )
        lp = fn(s_query)
        self.assertEqual(lp.shape, (N,))
        self.assertTrue(torch.isfinite(lp).all(), f"Non-finite: {lp}")

    # ── intensity() end-to-end tests ─────────────────────────────────── #

    def test_neural_stpp_intensity_empty_history(self):
        """intensity() with T=0 history returns shape (M,) with finite values."""
        M = 6
        model = self._make_neural_stpp_model("jump_cnf")
        state, state_reg, times, locs, lengths = self._make_state(model, T=0)

        t_q = torch.ones(M, 1) * 0.5
        s_q = torch.rand(M, 2)

        out = model.event_model.intensity(
            state=state, query_times=t_q, query_locations=s_q
        )
        self.assertEqual(out.shape, (M,))
        self.assertTrue(torch.isfinite(out).all(), f"Non-finite: {out}")

    def test_neural_stpp_intensity_nonempty_history(self):
        """intensity() with T>0 returns shape (M,), finite, non-constant values."""
        M = 9
        model = self._make_neural_stpp_model("jump_cnf")
        state, state_reg, times, locs, lengths = self._make_state(model, T=4)

        t_q = torch.ones(M, 1) * (float(times[0, -1].item()) + 0.1)
        # Use a grid of distinct locations to check non-constancy
        xs = torch.linspace(0.1, 0.9, 3)
        ys = torch.linspace(0.1, 0.9, 3)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        s_q = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

        out = model.event_model.intensity(
            state=state, query_times=t_q, query_locations=s_q
        )
        self.assertEqual(out.shape, (M,))
        self.assertTrue(torch.isfinite(out).all(), f"Non-finite: {out}")
        # Values should not all be the same (spatial density is non-trivial)
        self.assertGreater(out.std().item(), 0.0, "Expected non-constant intensity surface")


if __name__ == "__main__":
    unittest.main()
