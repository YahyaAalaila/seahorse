from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from unified_stpp.evaluation.bundle_io import load_surface_bundle, write_surface_bundle
from unified_stpp.evaluation.runtime import HistoryQuery, RunTarget
from unified_stpp.evaluation.surface import (
    SurfaceDiagnosticEvaluator,
    SurfaceDiagnosticSpec,
)
from unified_stpp.viz import SurfaceRenderConfig, render_surface_bundle

from tests.eval_test_helpers import assert_finite_array, make_saved_run, write_history_jsonl


class TestSurfaceDiagnostics(unittest.TestCase):
    def test_history_frame_surface_bundle_and_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_path = write_history_jsonl(root / "history.jsonl")
            run_dir = make_saved_run(root, preset="auto_stpp", label="auto")

            result = SurfaceDiagnosticEvaluator().evaluate(
                RunTarget(run=run_dir),
                HistoryQuery(
                    history_path=history_path,
                    split="test",
                    seq_idx=0,
                    history_length=0,
                ),
                SurfaceDiagnosticSpec(
                    profile="history_frame",
                    x_nstep=5,
                    y_nstep=5,
                    t_nstep=4,
                    frame_index=1,
                    device="cpu",
                ),
            )

            self.assertEqual(result.profile, "history_frame")
            self.assertFalse(result.provisional)
            self.assertEqual(result.primary_cube.shape, (4, 5, 5))
            assert_finite_array(self, result.primary_cube)

            out_dir = root / "surface_notebook"
            write_surface_bundle(out_dir, result)
            loaded = load_surface_bundle(out_dir)
            artifacts = render_surface_bundle(
                loaded,
                out_dir,
                SurfaceRenderConfig(interactive=False),
            )

            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "data.npz").exists())
            self.assertIn("intensity_heatmap", artifacts)
            self.assertTrue(artifacts["intensity_heatmap"].exists())

    def test_future_exact_surface_is_marked_provisional(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_path = write_history_jsonl(root / "history.jsonl")
            run_dir = make_saved_run(root, preset="neural_cond_gmm", label="neural")

            fake_payload = {
                "profile": "future_exact",
                "split": "test",
                "preset": "neural_cond_gmm",
                "device": "cpu",
                "history_times": np.asarray([0.32, 0.58, 0.91, 1.23], dtype=np.float32),
                "history_locs": np.asarray(
                    [[0.22, 0.18], [0.35, 0.44], [0.52, 0.49], [0.68, 0.61]],
                    dtype=np.float32,
                ),
                "t_grid": np.asarray([1.35, 1.55, 1.73], dtype=np.float32),
                "x_grid": np.linspace(0.0, 1.0, 5, dtype=np.float32),
                "y_grid": np.linspace(0.0, 1.0, 5, dtype=np.float32),
                "lambda_t": np.asarray([0.8, 0.7, 0.6], dtype=np.float32),
                "spatial_density": np.ones((3, 5, 5), dtype=np.float32) / 25.0,
                "primary_cube": np.ones((3, 5, 5), dtype=np.float32) * 0.2,
                "primary_value_name": "joint_intensity",
                "primary_value_label": "joint intensity",
                "future_horizon": 0.5,
                "spatial_chunk_size": 8,
                "auto_coarsened_grid": False,
                "notes": ["Neural exact-family packaged support is provisional until parity is proven."],
                "query_complexity": {"total_chunk_calls": 15},
                "provisional": True,
            }
            with mock.patch(
                "unified_stpp.evaluation.surface.diagnostics.evaluate_neural_future_exact",
                return_value=fake_payload,
            ):
                result = SurfaceDiagnosticEvaluator().evaluate(
                    RunTarget(run=run_dir),
                    HistoryQuery(
                        history_path=history_path,
                        split="test",
                        seq_idx=0,
                        history_length=4,
                    ),
                    SurfaceDiagnosticSpec(
                        profile="future_exact",
                        x_nstep=5,
                        y_nstep=5,
                        t_nstep=3,
                        future_horizon=0.5,
                        spatial_chunk_size=8,
                        device="cpu",
                    ),
                )

            self.assertTrue(result.provisional)
            self.assertTrue(any("provisional" in note.lower() for note in result.notes))
            self.assertIn("lambda_t", result.extra_arrays)
            self.assertIn("spatial_density", result.extra_arrays)
            assert_finite_array(self, result.primary_cube)
            assert_finite_array(self, result.extra_arrays["lambda_t"])
            assert_finite_array(self, result.extra_arrays["spatial_density"])

            out_dir = root / "surface_future_exact"
            write_surface_bundle(out_dir, result)
            loaded = load_surface_bundle(out_dir)
            artifacts = render_surface_bundle(
                loaded,
                out_dir,
                SurfaceRenderConfig(interactive=False),
            )

            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "data.npz").exists())
            self.assertIn("temporal_curve", artifacts)
            self.assertTrue(artifacts["temporal_curve"].exists())


if __name__ == "__main__":
    unittest.main()
