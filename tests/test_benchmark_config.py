"""Tests for benchmark configuration and scalar reporting behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.eval_test_helpers import SAMPLE_SEQUENCES, make_saved_run
from unified_stpp.benchmark import Benchmark, BenchmarkTable
from unified_stpp.config import BenchmarkConfig, STPPConfig, TuningConfig
from unified_stpp.evaluation.likelihood import (
    _prefix_difference_next_event_nlls_unbatched,
    compute_next_event_test_nll,
)
from unified_stpp.runner import STPPRunner
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
        self.assertEqual(cfg.resolved_tuning().metric, "val_objective")

    def test_data_contract_wins_over_base_overrides(self):
        bench = Benchmark(
            presets=["poisson_gmm"],
            splits={"toy": (self.train_seqs, self.val_seqs, self.test_seqs)},
            config=BenchmarkConfig(normalize=True, seeds=[42]),
            base_overrides={
                "data": {"protocol": "standard", "normalize": False},
                "training": {"n_epochs": 1, "predictive_test_nll_samples": 17},
            },
        )
        cfg = bench._base_config("poisson_gmm")
        self.assertEqual(cfg.data.protocol, "raw")
        self.assertTrue(cfg.data.normalize)
        self.assertEqual(cfg.training.checkpoint_select, "best")
        self.assertEqual(cfg.training.test_nll_space, "raw")
        self.assertEqual(cfg.training.predictive_test_nll_samples, 128)

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
        self.assertEqual(table.runs[0].test_nll_method, "exact_next_event_from_eventwise_terms")
        self.assertEqual(
            table.runs[0].test_nll_contexts,
            sum(len(seq["times"]) - 1 for seq in self.test_seqs),
        )
        self.assertEqual(
            table.runs[0].test_nll_contexts,
            table.runs[0].test_nll_scored_contexts,
        )
        self.assertEqual(table.runs[0].test_nll_missing_contexts, 0)
        self.assertTrue(np.isfinite(table.runs[0].native_test_nll))
        self.assertEqual(table.runs[0].nll_report_space, "raw")

    def test_poisson_gmm_eventwise_terms_match_prefix_baseline(self):
        seq = {
            "times": np.asarray(SAMPLE_SEQUENCES[0]["times"][:6], dtype=np.float32),
            "locations": np.asarray(SAMPLE_SEQUENCES[0]["locations"][:6], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_saved_run(Path(tmpdir), preset="poisson_gmm", label="poisson")
            runner = STPPRunner.load(run_dir)
            summary = compute_next_event_test_nll(
                runner,
                [seq],
                device=torch.device("cpu"),
            )
            with torch.no_grad():
                baseline = _prefix_difference_next_event_nlls_unbatched(
                    runner,
                    seq,
                    device=torch.device("cpu"),
                )

        self.assertEqual(summary["method"], "exact_next_event_from_eventwise_terms")
        np.testing.assert_allclose(
            summary["per_context_nll"],
            baseline.astype(np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertAlmostEqual(summary["mean_nll"], float(baseline.mean()), places=6)

    def test_poisson_gmm_exposes_eventwise_next_event_outputs(self):
        seq = {
            "times": np.asarray(SAMPLE_SEQUENCES[0]["times"][:6], dtype=np.float32),
            "locations": np.asarray(SAMPLE_SEQUENCES[0]["locations"][:6], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_saved_run(Path(tmpdir), preset="poisson_gmm", label="poisson")
            runner = STPPRunner.load(run_dir)
            times = torch.tensor(seq["times"], dtype=torch.float32).unsqueeze(0)
            locations = torch.tensor(seq["locations"], dtype=torch.float32).unsqueeze(0)
            lengths = torch.tensor([times.shape[1]], dtype=torch.long)
            with torch.no_grad():
                output = runner.model.eval_forward(
                    times=times,
                    locations=locations,
                    lengths=lengths,
                )

        self.assertIn("tll_matrix", output)
        self.assertIn("temporal_nll_matrix", output)
        self.assertIn("spatial_nll_matrix", output)
        self.assertIn("nll_matrix", output)
        self.assertIn("next_event_mask", output)
        np.testing.assert_array_equal(
            output["next_event_mask"][0].detach().cpu().numpy(),
            np.asarray([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        next_event_terms = output["nll_matrix"][0][output["next_event_mask"][0] > 0]
        self.assertEqual(int(next_event_terms.shape[0]), 5)
        self.assertTrue(torch.isfinite(next_event_terms).all())

    def test_run_hpo_requires_explicit_tune_dataset(self):
        bench = Benchmark(
            presets=["poisson_gmm"],
            splits={"toy": (self.train_seqs, self.val_seqs, self.test_seqs)},
            config=BenchmarkConfig(
                run_hpo=True,
                tuning=TuningConfig(n_trials=1),
                seeds=[42],
            ),
        )
        with self.assertRaisesRegex(ValueError, "tune_dataset"):
            bench.tune_all()

    def test_effective_config_relocks_pretuned_yaml_policy(self):
        tuned = STPPConfig.from_preset("poisson_gmm")
        raw = tuned.model_dump(mode="json")
        raw["data"]["protocol"] = "standard"
        raw["data"]["normalize"] = True
        raw["training"]["checkpoint_select"] = "last"
        raw["training"]["test_nll_space"] = "native"
        raw["training"]["predictive_test_nll_samples"] = 9
        raw["training"]["n_epochs"] = 99
        raw["training"]["patience"] = 7
        tuned = STPPConfig(**raw)

        bench = Benchmark(
            presets=["poisson_gmm"],
            splits={"toy": (self.train_seqs, self.val_seqs, self.test_seqs)},
            config=BenchmarkConfig(seeds=[42]),
            base_overrides={"training": {"n_epochs": 3, "patience": None}},
            hpo_configs={"poisson_gmm": tuned},
        )
        effective = bench._effective_config_for_preset("poisson_gmm")
        self.assertEqual(effective.data.protocol, "raw")
        self.assertFalse(effective.data.normalize)
        self.assertEqual(effective.training.checkpoint_select, "best")
        self.assertEqual(effective.training.test_nll_space, "raw")
        self.assertEqual(effective.training.predictive_test_nll_samples, 128)
        self.assertEqual(effective.training.n_epochs, 3)
        self.assertIsNone(effective.training.patience)

    def test_exact_table_requires_raw_report_space(self):
        table = BenchmarkTable(
            runs=[
                RunResult(
                    preset="exact_native",
                    dataset_id="toy",
                    seed=0,
                    val_objective=1.0,
                    test_nll=2.0,
                    train_time_sec=0.1,
                    n_params=10,
                    effective_config={},
                    nll_kind="exact",
                    nll_report_space="native",
                ),
                RunResult(
                    preset="exact_raw",
                    dataset_id="toy",
                    seed=0,
                    val_objective=1.0,
                    test_nll=1.5,
                    train_time_sec=0.1,
                    n_params=10,
                    effective_config={},
                    nll_kind="exact",
                    nll_report_space="raw",
                ),
            ]
        )
        exact = table.to_dataframe(group="exact")
        self.assertEqual(list(exact.index), ["exact_raw"])

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
