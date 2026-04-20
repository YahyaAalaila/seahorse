from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.eval_test_helpers import SAMPLE_SEQUENCES, make_saved_run
from unified_stpp.evaluation import evaluate
from unified_stpp.evaluation.artifacts import PredictiveSamples
from unified_stpp.evaluation.predictive.benchmark import write_next_event_benchmark_summary
from unified_stpp.evaluation.predictive.rollout import (
    build_exact_intensity_fn,
)
from unified_stpp.evaluation.predictive.sampling import compute_predictive_samples
from unified_stpp.evaluation.profiles import PREDICTIVE_SAMPLES
from unified_stpp.evaluation.registry import metric_by_name
from unified_stpp.evaluation.result import MetricResult, Report
from unified_stpp.runner.runner import STPPRunner


def _toy_sequences() -> list[dict[str, np.ndarray]]:
    seq = SAMPLE_SEQUENCES[0]
    return [
        {
            "times": np.asarray(seq["times"][:4], dtype=np.float32),
            "locations": np.asarray(seq["locations"][:4], dtype=np.float32),
        }
    ]


class TestPredictiveBenchmarkArtifacts(unittest.TestCase):
    def test_context_indexing_matches_teacher_forced_prefixes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_saved_run(Path(tmpdir), preset="diffusion_stpp", label="diffusion")
            runner = STPPRunner.load(run_dir)
            samples = compute_predictive_samples(
                runner,
                _toy_sequences(),
                k=1,
                device=torch.device("cpu"),
                seed=0,
            )

        self.assertEqual(samples.next_times.shape[0], 3)
        np.testing.assert_array_equal(
            samples.target_event_index,
            np.asarray([1, 2, 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            samples.history_length,
            np.asarray([1, 2, 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            samples.is_last_context,
            np.asarray([False, False, True], dtype=np.bool_),
        )

    def test_summary_bundle_writes_context_and_last_context_outputs(self):
        samples = PredictiveSamples(
            next_times=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
            next_locs=np.asarray(
                [[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]],
                dtype=np.float32,
            ),
            true_next_times=np.asarray([1.1, 2.2, 3.3], dtype=np.float32),
            true_next_locs=np.asarray([[0.1, 0.1], [1.2, 1.1], [2.1, 2.2]], dtype=np.float32),
            history_end_times=np.asarray([0.8, 1.8, 2.8], dtype=np.float32),
            sequence_index=np.asarray([0, 0, 1], dtype=np.int64),
            target_event_index=np.asarray([1, 2, 1], dtype=np.int64),
            history_length=np.asarray([1, 2, 1], dtype=np.int64),
            is_last_context=np.asarray([False, True, True], dtype=np.bool_),
            sampling_succeeded=np.asarray([True, False, True], dtype=np.bool_),
            sampling_backend="native_next_event_sampler",
        )
        report = Report(
            results={
                "temporal_mae": MetricResult(
                    value=0.5,
                    per_event=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
                    method="native_next_event_sampler",
                ),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_next_event_benchmark_summary(tmpdir, report, samples)
            summary_path = Path(outputs["summary_path"])
            context_index_path = Path(outputs["context_index_path"])
            self.assertTrue(summary_path.exists())
            self.assertTrue(context_index_path.exists())
            per_context = np.load(Path(outputs["score_files"]["temporal_mae"]["per_context"]))
            per_sequence_mean = np.load(
                Path(outputs["score_files"]["temporal_mae"]["per_sequence_mean"])
            )
            last_context = np.load(
                Path(outputs["score_files"]["temporal_mae"]["last_context_per_sequence"])
            )

            np.testing.assert_allclose(
                per_context,
                np.asarray([0.1, np.nan, 0.3]),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                per_sequence_mean,
                np.asarray([0.1, 0.3]),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                last_context,
                np.asarray([np.nan, 0.3]),
                equal_nan=True,
            )

            with open(summary_path) as handle:
                summary = json.load(handle)
            self.assertEqual(summary["evaluation_task"]["n_contexts"], 3)
            self.assertEqual(summary["evaluation_task"]["n_sampling_failures"], 1)
            self.assertAlmostEqual(
                summary["metrics"]["temporal_mae"]["last_context_mean"],
                0.3,
                places=7,
            )

    def test_exact_family_predictive_smoke(self):
        metric = metric_by_name("temporal_mae")
        for preset in ("poisson_gmm", "deep_stpp", "auto_stpp", "nsmpp"):
            with self.subTest(preset=preset):
                with tempfile.TemporaryDirectory() as tmpdir:
                    run_dir = make_saved_run(Path(tmpdir), preset=preset, label=preset)
                    runner = STPPRunner.load(run_dir)
                    report = evaluate(
                        runner,
                        _toy_sequences(),
                        metrics=[metric],
                        allowed_artifact_families={PREDICTIVE_SAMPLES},
                        artifact_dir=Path(tmpdir) / "artifacts",
                        k_pred=2,
                        seed=0,
                        device="cpu",
                    )
                    self.assertIn("temporal_mae", report.results)
                    self.assertIn(PREDICTIVE_SAMPLES, report.artifact_events)
                    self.assertTrue(report["temporal_mae"].available)

    def test_native_sampler_predictive_smoke(self):
        metric = metric_by_name("temporal_mae")
        for preset in ("smash", "diffusion_stpp"):
            with self.subTest(preset=preset):
                with tempfile.TemporaryDirectory() as tmpdir:
                    run_dir = make_saved_run(Path(tmpdir), preset=preset, label=preset)
                    runner = STPPRunner.load(run_dir)
                    report = evaluate(
                        runner,
                        _toy_sequences(),
                        metrics=[metric],
                        allowed_artifact_families={PREDICTIVE_SAMPLES},
                        artifact_dir=Path(tmpdir) / "artifacts",
                        k_pred=2,
                        seed=0,
                        device="cpu",
                    )
                    self.assertIn("temporal_mae", report.results)
                    self.assertTrue(report["temporal_mae"].available)

    def test_neural_like_exact_intensity_adapter_applies_transform_correction(self):
        class FakeEventModel:
            def intensity(self, *, state, query_times, query_locations, device=None):
                del state, query_times, query_locations, device
                return torch.tensor([8.0], dtype=torch.float32)

        class FakeModel:
            def __init__(self):
                self.event_model = FakeEventModel()

        class FakeRunner:
            def __init__(self):
                self.model = FakeModel()
                self.norm_stats = {"normalize": False}

        class FakeState:
            payload = {
                "input_transform": {
                    "type": "zscore",
                    "normalize_time": False,
                    "normalize_space": True,
                    "time_mean": 0.0,
                    "time_std": 1.0,
                    "loc_mean": [0.0, 0.0],
                    "loc_std": [2.0, 4.0],
                }
            }

        intensity_fn = build_exact_intensity_fn(
            FakeRunner(),
            FakeState(),
            torch.device("cpu"),
        )
        values = intensity_fn(
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([[0.5, -0.5]], dtype=torch.float32),
        )
        self.assertEqual(tuple(values.shape), (1,))
        self.assertTrue(torch.isfinite(values).all())
        self.assertAlmostEqual(float(values.item()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
