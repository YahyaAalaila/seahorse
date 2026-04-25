from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    path = Path("scripts/aggregate_hawkesnest_predictive_kde.py").resolve()
    spec = importlib.util.spec_from_file_location("aggregate_hawkesnest_predictive_kde_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AggregateHawkesnestPredictiveKDETest(unittest.TestCase):
    def test_build_aggregate_writes_tables_and_suite_plots(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite_root = root / "suite"
            suite_root.mkdir(parents=True, exist_ok=True)
            with open(suite_root / "metadata.json", "w") as f:
                json.dump(
                    {
                        "suite": "suite3_entanglement",
                        "levels": [
                            {"label": "L0", "description": "baseline"},
                            {"label": "L1", "description": "harder"},
                        ],
                    },
                    f,
                )

            campaign_root = root / "runs" / "hawkesnest_campaigns" / "suite3_entanglement" / "s3ent_v2__gen__04241035"
            manifests = campaign_root / "manifests"
            manifests.mkdir(parents=True, exist_ok=True)
            with open(manifests / "campaign_manifest.json", "w") as f:
                json.dump(
                    {
                        "suite": "suite3_entanglement",
                        "suite_path": str(suite_root),
                    },
                    f,
                )

            for preset, offset in (("smash", 0.0), ("diffusion_stpp", 0.2)):
                eval_root = campaign_root / "evaluate" / "predictive_kde" / preset
                eval_root.mkdir(parents=True, exist_ok=True)
                rows = [
                    {
                        "family": "L",
                        "level": 0,
                        "dataset_id": "L0",
                        "preset": preset,
                        "seed": 42,
                        "run_dir": str(campaign_root / "fit" / "L0" / preset / "seed_42"),
                        "bundle_dir": str(eval_root / "bundles" / "L0"),
                        "bundle_reused": False,
                        "render_dir": str(eval_root / "bundles" / "L0" / "renders"),
                        "panel_frame0_2d": str(eval_root / "bundles" / "L0" / "renders" / "panels" / "frame_000.png"),
                        "panel_frame0_3d": str(eval_root / "bundles" / "L0" / "renders" / "panels" / "frame_000_3d.png"),
                        "model_label": preset,
                        "split": "test",
                        "seq_idx": 0,
                        "start_event_idx": 20,
                        "n_frames": 2,
                        "n_rollouts": 16,
                        "grid_size": 32,
                        "horizon": 1.0,
                        "step_size": 1.0,
                        "bandwidth": "",
                        "test_nll": -0.5,
                        "val_objective": -0.4,
                        "rmse_mean": 0.6 + offset,
                        "mae_mean": 0.4 + offset,
                        "count_mae_mean": 0.2 + offset,
                        "count_bias_mean": 0.1 + offset,
                        "rmse_per_frame": json.dumps([0.6 + offset]),
                        "mae_per_frame": json.dumps([0.4 + offset]),
                        "count_mae_per_frame": json.dumps([0.2 + offset]),
                        "count_bias_per_frame": json.dumps([0.1 + offset]),
                        "true_event_counts": json.dumps([4]),
                        "pred_mean_event_counts": json.dumps([4.1]),
                        "eval_elapsed_sec": 5.0,
                    },
                    {
                        "family": "L",
                        "level": 1,
                        "dataset_id": "L1",
                        "preset": preset,
                        "seed": 42,
                        "run_dir": str(campaign_root / "fit" / "L1" / preset / "seed_42"),
                        "bundle_dir": str(eval_root / "bundles" / "L1"),
                        "bundle_reused": False,
                        "render_dir": "",
                        "panel_frame0_2d": "",
                        "panel_frame0_3d": "",
                        "model_label": preset,
                        "split": "test",
                        "seq_idx": 0,
                        "start_event_idx": 20,
                        "n_frames": 2,
                        "n_rollouts": 16,
                        "grid_size": 32,
                        "horizon": 1.0,
                        "step_size": 1.0,
                        "bandwidth": "",
                        "test_nll": -0.3,
                        "val_objective": -0.2,
                        "rmse_mean": 0.9 + offset,
                        "mae_mean": 0.7 + offset,
                        "count_mae_mean": 0.3 + offset,
                        "count_bias_mean": 0.15 + offset,
                        "rmse_per_frame": json.dumps([0.9 + offset]),
                        "mae_per_frame": json.dumps([0.7 + offset]),
                        "count_mae_per_frame": json.dumps([0.3 + offset]),
                        "count_bias_per_frame": json.dumps([0.15 + offset]),
                        "true_event_counts": json.dumps([5]),
                        "pred_mean_event_counts": json.dumps([5.1]),
                        "eval_elapsed_sec": 6.0,
                    },
                ]
                with open(eval_root / "metrics_by_run.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)

            out_dir = root / "analysis"
            manifest = module.build_aggregate(roots=[campaign_root], out_dir=out_dir)

            self.assertEqual(manifest["campaign_count"], 1)
            self.assertEqual(manifest["metric_row_count"], 4)
            self.assertEqual(manifest["summary_row_count"], 4)
            self.assertTrue((out_dir / "report.html").exists())
            self.assertTrue((out_dir / "table_metrics_by_run.csv").exists())
            self.assertTrue((out_dir / "table_metrics_by_level.csv").exists())
            self.assertTrue((out_dir / "table_render_artifacts.csv").exists())

            with open(out_dir / "table_metrics_by_level.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            presets = sorted({row["preset"] for row in rows})
            self.assertEqual(presets, ["diffusion_stpp", "smash"])

            plots = sorted((out_dir / "plots").rglob("*.png"))
            self.assertGreaterEqual(len(plots), 4)


if __name__ == "__main__":
    unittest.main()
