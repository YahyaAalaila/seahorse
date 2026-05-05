from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_reprocess_module():
    path = Path("scripts/reprocess_hawkesnest_easy_hard_v2.py").resolve()
    spec = importlib.util.spec_from_file_location("reprocess_hawkesnest_easy_hard_v2_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReprocessHawkesNestEasyHardV2Test(unittest.TestCase):
    def _make_suite(self, root: Path) -> Path:
        suite = root / "combined"
        (suite / "sequences").mkdir(parents=True, exist_ok=True)
        metadata = {
            "suite": "combined",
            "n_seeds": 1,
            "levels": [{"label": "C0"}],
        }
        (suite / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        times = np.array(
            [
                0.1,
                2.0,
                5.0,
                7.0,
                10.0,
                np.nextafter(np.float64(10.0), np.float64(np.inf)),
                15.0,
                19.5,
            ],
            dtype=np.float64,
        )
        locations = np.stack([np.linspace(0.0, 1.0, len(times)), np.linspace(1.0, 0.0, len(times))], axis=1).astype(
            np.float32
        )
        np.savez(
            suite / "sequences" / "C0_r0.npz",
            times=times,
            locations=locations,
            train_idx=np.arange(0, 4, dtype=np.int64),
            val_idx=np.arange(4, 6, dtype=np.int64),
            test_idx=np.arange(6, 8, dtype=np.int64),
            T_window=np.array(20.0, dtype=np.float64),
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        )
        return suite

    def test_reprocess_rebuilds_short_chunked_jsonl_and_logs_repair_stats(self):
        module = _load_reprocess_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite = self._make_suite(root)
            rc = module.main(
                [
                    "--root",
                    str(root),
                    "--suite",
                    "combined",
                    "--n-windows",
                    "4",
                    "--train-windows",
                    "2",
                    "--val-windows",
                    "1",
                    "--test-windows",
                    "1",
                ]
            )
            self.assertEqual(rc, 0)

            config_dir = suite / "jsonl" / "C0"
            train_rows = [json.loads(line) for line in (config_dir / "train.jsonl").read_text().splitlines()]
            val_rows = [json.loads(line) for line in (config_dir / "val.jsonl").read_text().splitlines()]
            test_rows = [json.loads(line) for line in (config_dir / "test.jsonl").read_text().splitlines()]
            manifest_rows = [json.loads(line) for line in (config_dir / "manifest.jsonl").read_text().splitlines()]

            self.assertEqual(len(train_rows), 2)
            self.assertEqual(len(val_rows), 1)
            self.assertEqual(len(test_rows), 1)
            self.assertEqual(len(manifest_rows), 4)
            self.assertEqual([row["chunk_idx"] for row in manifest_rows], [0, 1, 2, 3])
            self.assertEqual([row["split"] for row in manifest_rows], ["train", "train", "val", "test"])
            self.assertTrue(all(len(row["times"]) <= 2 for row in train_rows + val_rows + test_rows))

            repair_log = json.loads((suite / "repair_log.json").read_text())
            self.assertEqual(repair_log["C0"][0]["source_npz"], "C0_r0.npz")
            self.assertEqual(repair_log["C0"][0]["n_time_ties_repaired"], 1)
            self.assertEqual(repair_log["C0"][0]["time_repair_policy"], "nextafter_forward_pass")

            metadata = json.loads((suite / "metadata.json").read_text())
            self.assertEqual(metadata["serialization"]["n_windows"], 4)
            self.assertEqual(metadata["serialization"]["split_windows"]["train"], 2)
