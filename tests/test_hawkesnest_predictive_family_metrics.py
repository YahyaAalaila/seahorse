from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tests.eval_test_helpers import make_saved_run, write_history_jsonl


def _load_module():
    path = Path("scripts/hawkesnest_predictive_family_metrics.py").resolve()
    spec = importlib.util.spec_from_file_location("hawkesnest_predictive_family_metrics_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HawkesnestPredictiveFamilyMetricsTest(unittest.TestCase):
    def test_compute_metric_row_on_cached_predictive_bundle(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bench_root = root / "bench"
            splits_root = root / "splits"
            out_dir = root / "analysis"

            run_dir = make_saved_run(bench_root / "diffusion_run", preset="diffusion_stpp", label="diffusion")
            run_result_path = run_dir / "run_result.json"
            run_result = json.loads(run_result_path.read_text())
            run_result["dataset_id"] = "toy_T0"
            run_result_path.write_text(json.dumps(run_result, indent=2))

            write_history_jsonl(splits_root / "toy_T0" / "test.jsonl")

            args = Namespace(
                force=False,
                split="test",
                seq_idx=0,
                start_event_idx=1,
                history_length=3,
                rollout_mode="teacher_forced",
                n_frames=1,
                horizon=0.4,
                step_size=0.4,
                n_rollouts=1,
                grid_size=8,
                bandwidth=0.25,
                lambda_bar=5.0,
                max_events_per_window=3,
                bridge_retries=4,
                adaptive_thinning=True,
                exact_proposal="coarse",
                exact_time_bins=4,
                exact_spatial_bins=4,
                exact_safety=2.0,
                color_percentile=99.0,
                eval_seed=17,
                device="cpu",
                with_renders=True,
                plot_style="both",
                gif=False,
                fps=2.0,
            )

            records = module._discover_runs(bench_root, "diffusion_stpp")
            self.assertEqual(len(records), 1)
            row = module._compute_metric_for_record(
                records[0],
                args,
                splits_dir=splits_root,
                out_dir=out_dir,
            )

            self.assertEqual(row["preset"], "diffusion_stpp")
            self.assertEqual(row["dataset_id"], "toy_T0")
            self.assertIsNotNone(row["rmse_mean"])
            self.assertIsNotNone(row["mae_mean"])
            self.assertIsNotNone(row["count_mae_mean"])
            self.assertTrue(Path(row["bundle_dir"]).exists())
            self.assertTrue(Path(row["render_dir"]).exists())
            self.assertTrue(Path(row["panel_frame0_2d"]).exists())
            self.assertTrue(Path(row["panel_frame0_3d"]).exists())


if __name__ == "__main__":
    unittest.main()
