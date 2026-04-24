from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    path = Path("scripts/resolve_bench_eval_targets.py").resolve()
    spec = importlib.util.spec_from_file_location("resolve_bench_eval_targets_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "times": [0.0, 1.0, 2.0],
        "locations": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    }
    with open(path, "w") as f:
        f.write(json.dumps(payload) + "\n")


class ResolveBenchEvalTargetsTest(unittest.TestCase):
    def test_single_dataset_bench_root_resolves_paths(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            splits_dir = root / "covid-stpp"
            for split in ("train", "val", "test"):
                _write_jsonl(splits_dir / f"{split}.jsonl")

            bench_root = root / "bench" / "covid-stpp__gen__04231151"
            bench_root.mkdir(parents=True, exist_ok=True)
            run_dir = bench_root / "fit" / "smash" / "covid-stpp" / "seed_42" / "run_0"
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(bench_root / "bench_meta.json", "w") as f:
                json.dump({"bench_id": "covid-stpp__gen__04231151", "splits_dir": str(splits_dir)}, f)
            with open(bench_root / "cell_index.json", "w") as f:
                json.dump(
                    [
                        {
                            "preset": "smash",
                            "dataset_id": "covid-stpp",
                            "seed": 42,
                            "run_dir": str(run_dir),
                        }
                    ],
                    f,
                )

            targets = module.build_targets(bench_root=bench_root, split="test")
            self.assertEqual(len(targets), 1)
            target = targets[0]
            self.assertEqual(target["bench_id"], "covid-stpp__gen__04231151")
            self.assertEqual(target["dataset_id"], "covid-stpp")
            self.assertEqual(target["preset"], "smash")
            self.assertEqual(target["seed"], 42)
            self.assertTrue(target["data_path"].endswith("/covid-stpp/test.jsonl"))
            self.assertTrue(target["train_data"].endswith("/covid-stpp/train.jsonl"))

    def test_multi_dataset_bench_root_resolves_nested_dataset_paths(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            splits_dir = root / "suite_root"
            for split in ("train", "val", "test"):
                _write_jsonl(splits_dir / "H0" / f"{split}.jsonl")

            bench_root = root / "bench" / "suite4__rest__04231151"
            bench_root.mkdir(parents=True, exist_ok=True)
            run_dir = bench_root / "fit" / "auto_stpp" / "H0" / "seed_3" / "run_0"
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(bench_root / "bench_meta.json", "w") as f:
                json.dump({"bench_id": "suite4__rest__04231151", "splits_dir": str(splits_dir)}, f)
            with open(bench_root / "cell_index.json", "w") as f:
                json.dump(
                    [
                        {
                            "preset": "auto_stpp",
                            "dataset_id": "H0",
                            "seed": 3,
                            "run_dir": str(run_dir),
                        }
                    ],
                    f,
                )

            targets = module.build_targets(bench_root=bench_root, split="test")
            self.assertEqual(len(targets), 1)
            target = targets[0]
            self.assertEqual(target["dataset_id"], "H0")
            self.assertTrue(target["data_path"].endswith("/suite_root/H0/test.jsonl"))
            self.assertTrue(target["train_data"].endswith("/suite_root/H0/train.jsonl"))

    def test_dataset_backed_bench_root_emits_dataset_reference(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            splits_dir = root / "snapshot_root"
            for split in ("train", "val", "test"):
                _write_jsonl(splits_dir / f"{split}.jsonl")

            bench_root = root / "bench" / "covid-stpp__rest__04231151"
            bench_root.mkdir(parents=True, exist_ok=True)
            run_dir = bench_root / "fit" / "auto_stpp" / "covid-stpp" / "seed_42" / "run_0"
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(bench_root / "bench_meta.json", "w") as f:
                json.dump({"bench_id": "covid-stpp__rest__04231151", "splits_dir": str(splits_dir)}, f)
            with open(bench_root / "data_manifest.json", "w") as f:
                json.dump(
                    {
                        "requested": {
                            "dataset": "yahya021/covid-stpp",
                            "dataset_revision": "v1",
                        }
                    },
                    f,
                )
            with open(bench_root / "cell_index.json", "w") as f:
                json.dump(
                    [
                        {
                            "preset": "auto_stpp",
                            "dataset_id": "covid-stpp",
                            "seed": 42,
                            "run_dir": str(run_dir),
                        }
                    ],
                    f,
                )

            targets = module.build_targets(bench_root=bench_root, split="test")
            self.assertEqual(len(targets), 1)
            target = targets[0]
            self.assertEqual(target["dataset_ref"], "yahya021/covid-stpp")
            self.assertEqual(target["dataset_revision"], "v1")
            self.assertIsNone(target["data_path"])
            self.assertIsNone(target["train_data"])


if __name__ == "__main__":
    unittest.main()
