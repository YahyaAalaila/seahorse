from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_module():
    path = Path("scripts/generate_hawkesnest_suite_ground_truth.py").resolve()
    spec = importlib.util.spec_from_file_location("generate_hawkesnest_suite_ground_truth_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GenerateHawkesnestSuiteGroundTruthTest(unittest.TestCase):
    def test_suite4_cluster_mix_bundle_is_written(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            suite = Path(td) / "suite4_heterogeneity"
            (suite / "jsonl" / "H1").mkdir(parents=True, exist_ok=True)
            (suite / "sequences").mkdir(parents=True, exist_ok=True)

            with (suite / "metadata.json").open("w") as f:
                json.dump(
                    {
                        "suite": "suite4_heterogeneity",
                        "kernel": {
                            "type": "separable",
                            "temporal_decay": 0.3,
                            "spatial_sigma": 0.15,
                        },
                        "levels": [
                            {
                                "label": "H1",
                                "bg": {
                                    "type": "function",
                                    "name": "cluster_mix",
                                    "centers": [[0.5, 0.5]],
                                    "sigma": 0.2,
                                    "a0": 0.5,
                                    "amp": 3.0,
                                },
                                "adj": 1.2,
                            }
                        ],
                    },
                    f,
                )

            row = {"times": [0.1, 0.4, 0.9], "locations": [[0.2, 0.2], [0.4, 0.5], [0.7, 0.8]]}
            with (suite / "jsonl" / "H1" / "test.jsonl").open("w") as f:
                f.write(json.dumps(row) + "\n")
            with (suite / "jsonl" / "H1" / "manifest.jsonl").open("w") as f:
                f.write(
                    json.dumps(
                        {
                            "suite": "suite4_heterogeneity",
                            "config": "H1",
                            "seed": 2,
                            "source_npz": "H1_r2.npz",
                            "chunk_idx": 18,
                            "split": "test",
                        }
                    )
                    + "\n"
                )

            np.savez(
                suite / "sequences" / "H1_r0.npz",
                domain_bounds=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            )

            written = module.generate_suite_ground_truth(
                suite_path=suite,
                x_resolution=6,
                y_resolution=5,
                t_resolution=4,
                overwrite=False,
            )
            self.assertEqual(len(written), 2)

            gt_dir = suite / "ground_truth"
            npz_path = gt_dir / "H1_intensity_grid_r0.npz"
            params_path = gt_dir / "H1_params.json"
            self.assertTrue(npz_path.exists())
            self.assertTrue(params_path.exists())

            payload = np.load(npz_path)
            self.assertEqual(payload["lambda_"].shape, (4, 6, 5))
            self.assertEqual(payload["x_grid"].shape, (6,))
            self.assertEqual(payload["y_grid"].shape, (5,))
            self.assertEqual(payload["t_grid"].shape, (4,))
            self.assertGreaterEqual(float(payload["lambda_"].min()), 0.0)

            params = json.loads(params_path.read_text())
            self.assertEqual(params["suite"], "suite4_heterogeneity")
            self.assertEqual(params["config"], "H1")
            self.assertEqual(np.asarray(params["background_grid"]).shape, (4, 6, 5))


if __name__ == "__main__":
    unittest.main()
