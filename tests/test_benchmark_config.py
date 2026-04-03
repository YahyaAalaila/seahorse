"""Tests for benchmark configuration and scalar reporting behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from unified_stpp.benchmark import Benchmark, BenchmarkTable
from unified_stpp.config import BenchmarkConfig, STPPConfig, TuningConfig
from unified_stpp.runner.results import RunResult


class BenchmarkConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)

        def _seq():
            times = np.cumsum(rng.exponential(0.5, 8))
            locs = rng.uniform(0, 1, (8, 2))
            return {"times": times, "locations": locs}

        cls.train_seqs = [_seq() for _ in range(4)]
        cls.val_seqs = [_seq() for _ in range(2)]
        cls.test_seqs = [_seq() for _ in range(2)]

    def test_primary_metric_is_distinct_from_hpo_metric(self):
        cfg = BenchmarkConfig(
            run_hpo=True,
            tuning=TuningConfig(metric="val_nll"),
            primary_metric="test_nll",
        )
        self.assertEqual(cfg.primary_metric, "test_nll")
        self.assertEqual(cfg.resolved_tuning().metric, "val_nll")

    def test_data_contract_wins_over_base_overrides(self):
        bench = Benchmark(
            presets=["poisson_gmm"],
            splits={"toy": (self.train_seqs, self.val_seqs, self.test_seqs)},
            config=BenchmarkConfig(normalize=True, seeds=[42]),
            base_overrides={
                "data": {"protocol": "standard", "normalize": False},
                "training": {"n_epochs": 1},
            },
        )
        cfg = bench._base_config("poisson_gmm")
        self.assertEqual(cfg.data.protocol, "standard")
        self.assertTrue(cfg.data.normalize)

    def test_run_returns_table_in_sequential_mode(self):
        bench = Benchmark(
            presets=["poisson_gmm"],
            splits={"toy": (self.train_seqs, self.val_seqs, self.test_seqs)},
            config=BenchmarkConfig(seeds=[42], n_workers=1, backend="joblib"),
            base_overrides={"training": {"n_epochs": 1}},
        )
        table = bench.run()
        self.assertIsInstance(table, BenchmarkTable)
        self.assertEqual(len(table.runs), 1)
        self.assertEqual(table.runs[0].preset, "poisson_gmm")

    def test_report_uses_requested_metric(self):
        run = RunResult(
            preset="model_a",
            dataset_id="toy",
            seed=0,
            val_objective=1.23,
            test_nll=2.34,
            train_time_sec=0.1,
            n_params=10,
            effective_config={},
        )
        table = BenchmarkTable(runs=[run])
        with tempfile.TemporaryDirectory() as d:
            table.report(d, metric="val_objective")
            # Report now writes separate exact/all CSVs
            self.assertTrue(
                (Path(d) / "table_val_objective_exact.csv").exists()
                or (Path(d) / "table_val_objective_all.csv").exists()
            )
            html = (Path(d) / "report.html").read_text()
            # New report sections replace "Pivot Table"
            self.assertIn("val_objective", html)
            self.assertNotIn("Surface Comparison", html)


if __name__ == "__main__":
    unittest.main()
