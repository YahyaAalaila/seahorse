from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from unified_stpp.cli import bench, fit, tune
from unified_stpp.config.schema import DataConfig, STPPConfig
from unified_stpp.data.resolution import ResolvedBenchmarkData, ResolvedDataPaths


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def _records() -> list[dict]:
    return [{"times": [0.1, 0.2], "locations": [[0.0, 0.0], [1.0, 1.0]]}]


class DataConfigResolutionTest(unittest.TestCase):
    def test_resolve_data_named_dataset_delegates_to_hub_with_revision(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "topology_T5"
            _write_jsonl(root / "train.jsonl", _records())
            _write_jsonl(root / "val.jsonl", _records())
            _write_jsonl(root / "test.jsonl", _records())

            cfg = DataConfig(
                dataset="hawkesnest_hard_v2/topology_T5",
                dataset_revision="rev-1",
            )

            with patch("unified_stpp.data.resolution.download_dataset", return_value=root) as download:
                resolved = cfg.resolve_data(mode="single", include_test=True)

        self.assertEqual(resolved.train_path, root / "train.jsonl")
        self.assertEqual(resolved.val_path, root / "val.jsonl")
        self.assertEqual(resolved.test_path, root / "test.jsonl")
        self.assertEqual(resolved.dataset_id, "topology_T5")
        download.assert_called_once_with(
            "hawkesnest_hard_v2/topology_T5",
            revision="rev-1",
        )

    def test_resolve_data_explicit_paths_preserves_backward_compatibility(self):
        cfg = DataConfig(
            train_path="data/train.jsonl",
            val_path="data/val.jsonl",
            test_path="data/test.jsonl",
        )

        resolved = cfg.resolve_data(mode="single", include_test=True)

        self.assertEqual(resolved.train_path, Path("data/train.jsonl"))
        self.assertEqual(resolved.val_path, Path("data/val.jsonl"))
        self.assertEqual(resolved.test_path, Path("data/test.jsonl"))
        self.assertEqual(resolved.dataset_id, "train")

    def test_resolve_data_benchmark_uses_splits_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "exports"
            ds_dir = root / "topology_T5"
            _write_jsonl(ds_dir / "train.jsonl", _records())
            _write_jsonl(ds_dir / "val.jsonl", _records())
            cfg = DataConfig(
                splits_dir=str(root),
                datasets=["topology_T5"],
            )

            resolved = cfg.resolve_data(mode="benchmark")

        self.assertEqual(resolved.splits_dir, root)
        self.assertEqual(sorted(resolved.splits.keys()), ["topology_T5"])
        self.assertEqual(resolved.files["topology_T5"]["train"], ds_dir / "train.jsonl")
        self.assertEqual(resolved.files["topology_T5"]["val"], ds_dir / "val.jsonl")
        self.assertIsNone(resolved.files["topology_T5"]["test"])


class FitDatasetCliTest(unittest.TestCase):
    def test_fit_execute_delegates_to_data_config_resolution(self):
        args = Namespace(
            preset="deep_stpp",
            config=None,
            dataset="hawkesnest_hard_v2/pulse_P0",
            dataset_revision="rev-a",
            train=None,
            val=None,
            test=None,
            out=None,
            save=None,
            override=[],
        )
        resolved = ResolvedDataPaths(
            train_path=Path("/tmp/train.jsonl"),
            val_path=Path("/tmp/val.jsonl"),
            test_path=Path("/tmp/test.jsonl"),
            dataset_id="pulse_P0",
            source_root=Path("/tmp"),
        )
        runner = Mock()
        runner.config = SimpleNamespace(data=Mock(resolve_data=Mock(return_value=resolved)))
        runner.fit.return_value = Mock(
            val_metric_key="nll",
            val_objective=1.0,
            test_nll=float("nan"),
            run_dir=None,
        )

        with patch("unified_stpp.runner.STPPRunner.from_config_source", return_value=runner) as factory:
            with patch("unified_stpp.utils.load_jsonl", return_value=_records()):
                fit.execute(args)

        runner.config.data.resolve_data.assert_called_once_with(mode="single", include_test=True)
        self.assertEqual(
            factory.call_args.kwargs["cli_values"]["data"],
            {
                "dataset": "hawkesnest_hard_v2/pulse_P0",
                "dataset_revision": "rev-a",
            },
        )
        self.assertEqual(runner.fit.call_args.kwargs["dataset_id"], "pulse_P0")


class TuneDatasetCliTest(unittest.TestCase):
    def test_tune_execute_delegates_to_data_config_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "bg_BG1"
            _write_jsonl(root / "train.jsonl", _records())
            _write_jsonl(root / "val.jsonl", _records())

            args = Namespace(
                preset="deep_stpp",
                config=None,
                dataset="hawkesnest_v20260409/bg_BG1",
                dataset_revision=None,
                train=None,
                val=None,
                out=str(Path(td) / "best_config.yaml"),
                n_trials=None,
                search_alg=None,
                scheduler=None,
                seed=None,
                fail_fast=None,
                max_concurrent_trials=None,
            )
            raw_cfg = {"data": {}, "model": {"preset": "deep_stpp"}, "training": {}}
            best_config = STPPConfig(**raw_cfg)
            resolved = ResolvedDataPaths(
                train_path=root / "train.jsonl",
                val_path=root / "val.jsonl",
                test_path=None,
                dataset_id="bg_BG1",
                source_root=root,
            )

            with patch("unified_stpp.config.schema.STPPConfig.raw_source_dict", return_value=raw_cfg):
                with patch("unified_stpp.config.schema.STPPConfig.split_tuning_dict", return_value=(raw_cfg, {})):
                    with patch("unified_stpp.config.tuning.TuningConfig.from_sources", return_value="tuning"):
                        with patch("unified_stpp.config.schema.DataConfig.resolve_data", return_value=resolved) as resolve_data:
                            with patch("unified_stpp.benchmark.hpo.run_hpo", return_value=best_config) as run_hpo:
                                tune.execute(args)

        resolve_data.assert_called_once_with(mode="single", include_test=False)
        self.assertEqual(run_hpo.call_args.kwargs["train_path"], str(root / "train.jsonl"))
        self.assertEqual(run_hpo.call_args.kwargs["val_path"], str(root / "val.jsonl"))

    def test_tune_execute_persists_dataset_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "resolved_dataset"
            _write_jsonl(root / "train.jsonl", _records())
            _write_jsonl(root / "val.jsonl", _records())
            out_path = Path(td) / "best_config.yaml"

            args = Namespace(
                preset="deep_stpp",
                config=None,
                dataset="hawkesnest_hard_v2/topology_T5",
                dataset_revision="rev-c",
                train=None,
                val=None,
                out=str(out_path),
                n_trials=None,
                search_alg=None,
                scheduler=None,
                seed=None,
                fail_fast=None,
                max_concurrent_trials=None,
            )
            raw_cfg = {"data": {}, "model": {"preset": "deep_stpp"}, "training": {}}
            resolved = ResolvedDataPaths(
                train_path=root / "train.jsonl",
                val_path=root / "val.jsonl",
                test_path=None,
                dataset_id="topology_T5",
                source_root=root,
            )
            best_config = STPPConfig(**raw_cfg)

            with patch("unified_stpp.config.schema.STPPConfig.raw_source_dict", return_value=raw_cfg):
                with patch("unified_stpp.config.schema.STPPConfig.split_tuning_dict", return_value=(raw_cfg, {})):
                    with patch("unified_stpp.config.tuning.TuningConfig.from_sources", return_value="tuning"):
                        with patch("unified_stpp.config.schema.DataConfig.resolve_data", return_value=resolved):
                            with patch("unified_stpp.benchmark.hpo.run_hpo", return_value=best_config):
                                tune.execute(args)

            with open(out_path) as f:
                saved = yaml.safe_load(f)
            self.assertEqual(saved["data"]["dataset"], "hawkesnest_hard_v2/topology_T5")
            self.assertEqual(saved["data"]["dataset_revision"], "rev-c")

            manifest_path = out_path.with_suffix(".data_manifest.json")
            self.assertTrue(manifest_path.exists())
            with open(manifest_path) as f:
                manifest = json.load(f)

            self.assertEqual(manifest["requested"]["dataset"], "hawkesnest_hard_v2/topology_T5")
            self.assertEqual(manifest["requested"]["dataset_revision"], "rev-c")
            self.assertEqual(manifest["resolved"]["dataset_id"], "topology_T5")
            self.assertEqual(manifest["resolved"]["source_root"], str(root.resolve()))
            self.assertEqual(
                manifest["resolved"]["train_path"],
                str((root / "train.jsonl").resolve()),
            )
            self.assertIn("source_fingerprint", manifest)
            self.assertIn("sha256", manifest["files"]["train"])


class BenchDatasetCliTest(unittest.TestCase):
    def test_bench_execute_delegates_to_data_config_resolution_for_named_dataset(self):
        args = Namespace(
            presets=None,
            preset="deep_stpp",
            splits_dir=None,
            dataset="hawkesnest_hard_v2/regime_R2",
            dataset_revision="rev-b",
            datasets=None,
            seeds=["42"],
            out="bench_out",
            n_workers=1,
            tune=False,
            n_trials=50,
            search_alg="random",
            scheduler="asha",
            hpo_configs_dir=None,
            override=[],
            normalize=False,
        )
        resolved = ResolvedBenchmarkData(
            splits_dir=Path("/tmp/regime_R2"),
            splits={"regime_R2": ([], [], None)},
        )
        table = Mock()
        bench_instance = Mock()
        bench_instance.run.return_value = table

        with patch("unified_stpp.config.schema.DataConfig.resolve_data", return_value=resolved) as resolve_data:
            with patch("unified_stpp.benchmark.Benchmark", return_value=bench_instance) as benchmark_cls:
                bench.execute(args)

        resolve_data.assert_called_once_with(mode="benchmark")
        self.assertEqual(benchmark_cls.call_args.kwargs["presets"], ["deep_stpp"])
        table.report.assert_called_once()
