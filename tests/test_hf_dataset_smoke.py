from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from seahorse.cli import bench, fit, tune
from seahorse.config.schema import STPPConfig


HF_SMOKE_ENABLED = os.environ.get("SEAHORSE_RUN_HF_SMOKE") == "1"
HF_SMOKE_DATASET = os.environ.get("SEAHORSE_HF_SMOKE_DATASET", "austin_311_stpp")
HF_SMOKE_DATASET_ID = HF_SMOKE_DATASET.rstrip("/").split("/")[-1]


@unittest.skipUnless(HF_SMOKE_ENABLED, "Set SEAHORSE_RUN_HF_SMOKE=1 to run HF smoke tests.")
class HuggingFaceDatasetSmokeTest(unittest.TestCase):
    def test_fit_dataset_resolves_and_loads_real_hf_repo(self):
        args = Namespace(
            preset="deep_stpp",
            config=None,
            dataset=HF_SMOKE_DATASET,
            dataset_revision=None,
            train=None,
            val=None,
            test=None,
            out=None,
            save=None,
            override=[],
        )
        result = Mock(val_metric_key="nll", val_objective=0.0, test_nll=float("nan"), run_dir=None)

        with patch("seahorse.runner.STPPRunner.fit", return_value=result) as fit_mock:
            fit.execute(args)

        train_seqs, val_seqs, test_seqs = fit_mock.call_args.args[:3]
        self.assertGreater(len(train_seqs), 0)
        self.assertGreater(len(val_seqs), 0)
        self.assertGreater(len(test_seqs), 0)
        self.assertEqual(fit_mock.call_args.kwargs["dataset_id"], HF_SMOKE_DATASET_ID)

    def test_tune_dataset_resolves_and_loads_real_hf_repo(self):
        with tempfile.TemporaryDirectory() as td:
            args = Namespace(
                preset="deep_stpp",
                config=None,
                dataset=HF_SMOKE_DATASET,
                dataset_revision=None,
                train=None,
                val=None,
                out=str(Path(td) / "best_config.yaml"),
                n_trials=1,
                search_alg=None,
                scheduler=None,
                seed=None,
                fail_fast=None,
                max_concurrent_trials=None,
            )

            def _fake_run_hpo(*, config_dict, tuning, train_path, val_path, dataset_id, return_analysis):
                from seahorse.utils import load_jsonl

                self.assertGreater(len(load_jsonl(train_path)), 0)
                self.assertGreater(len(load_jsonl(val_path)), 0)
                self.assertEqual(dataset_id, HF_SMOKE_DATASET_ID)
                return STPPConfig(**config_dict), None

            with patch("seahorse.benchmark.hpo.run_hpo", side_effect=_fake_run_hpo):
                tune.execute(args)

            self.assertTrue((Path(td) / "best_config.yaml").exists())
            self.assertTrue((Path(td) / "best_config.data_manifest.json").exists())

    def test_bench_dataset_resolves_and_loads_real_hf_repo(self):
        args = Namespace(
            preset="deep_stpp",
            presets=None,
            splits_dir=None,
            dataset=HF_SMOKE_DATASET,
            dataset_revision=None,
            datasets=None,
            seeds=["42"],
            out=None,
            n_workers=1,
            tune=False,
            tune_dataset=None,
            hpo_seed=0,
            n_trials=1,
            search_alg="random",
            scheduler="asha",
            hpo_configs_dir=None,
            override=[],
            normalize=False,
        )
        table = Mock()
        bench_instance = Mock()
        bench_instance.run.return_value = table

        def _fake_benchmark(*args, **kwargs):
            splits = kwargs["splits"]
            self.assertEqual(sorted(splits.keys()), [HF_SMOKE_DATASET_ID])
            self.assertGreater(len(splits[HF_SMOKE_DATASET_ID][0]), 0)
            self.assertGreater(len(splits[HF_SMOKE_DATASET_ID][1]), 0)
            return bench_instance

        with patch("seahorse.benchmark.Benchmark", side_effect=_fake_benchmark):
            bench.execute(args)

        table.report.assert_called_once()
