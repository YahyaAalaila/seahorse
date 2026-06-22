from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from seahorse.evaluation import evaluate
from seahorse.evaluation.artifacts import PredictiveSamples, merge_predictive_samples
from seahorse.evaluation.profiles import PREDICTIVE_SAMPLES
from seahorse.evaluation.registry import metric_by_name


class _Caps:
    nll_kind = "none"
    has_intensity = True
    has_density = False
    has_native_sampler = False


class _EventModel:
    capabilities = _Caps()


class _Model:
    event_model = _EventModel()


class _ModelConfig:
    preset = "dummy"


class _Config:
    model = _ModelConfig()


class _Runner:
    config = _Config()
    model = _Model()

    def __init__(self, run_dir: Path):
        self._run_dir = run_dir


def _sample_payload() -> PredictiveSamples:
    return PredictiveSamples(
        next_times=np.asarray([[0.9, 1.1]], dtype=np.float32),
        next_locs=np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32),
        true_next_times=np.asarray([1.0], dtype=np.float32),
        true_next_locs=np.asarray([[0.2, 0.3]], dtype=np.float32),
        history_end_times=np.asarray([0.0], dtype=np.float32),
        sequence_index=np.asarray([0], dtype=np.int64),
        target_event_index=np.asarray([1], dtype=np.int64),
        history_length=np.asarray([1], dtype=np.int64),
        is_last_context=np.asarray([True], dtype=np.bool_),
        sampling_succeeded=np.asarray([True], dtype=np.bool_),
        sampling_backend="native_next_event_sampler",
    )


class TestEvalArtifacts(unittest.TestCase):
    def test_predictive_samples_compute_then_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "config.yaml").write_text("model:\n  preset: dummy\n")
            artifact_dir = root / "artifacts"
            runner = _Runner(run_dir)
            test_seqs = [
                {
                    "times": np.asarray([0.0, 1.0], dtype=np.float32),
                    "locations": np.asarray([[0.0, 0.0], [0.2, 0.3]], dtype=np.float32),
                }
            ]
            metric = metric_by_name("temporal_mae")

            with patch(
                "seahorse.evaluation.predictive.sampling.compute_predictive_samples",
                return_value=_sample_payload(),
            ) as compute:
                report = evaluate(
                    runner,
                    test_seqs,
                    metrics=[metric],
                    allowed_artifact_families={PREDICTIVE_SAMPLES},
                    artifact_dir=artifact_dir,
                    k_pred=2,
                    seed=7,
                    device="cpu",
                )
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(report["temporal_mae"].value, 0.0)
            self.assertEqual(
                report.artifact_events[PREDICTIVE_SAMPLES]["status"],
                "computed_and_saved",
            )
            self.assertTrue(any(artifact_dir.glob("predictive_samples/*/manifest.json")))

            with patch(
                "seahorse.evaluation.predictive.sampling.compute_predictive_samples",
                side_effect=AssertionError("should load saved predictive samples"),
            ) as compute_again:
                report = evaluate(
                    runner,
                    test_seqs,
                    metrics=[metric],
                    allowed_artifact_families={PREDICTIVE_SAMPLES},
                    artifact_dir=artifact_dir,
                    artifact_mode="load_only",
                    k_pred=2,
                    seed=7,
                    device="cpu",
                )
            compute_again.assert_not_called()
            self.assertEqual(report["temporal_mae"].value, 0.0)
            self.assertEqual(
                report.artifact_events[PREDICTIVE_SAMPLES]["status"],
                "loaded_from_cache",
            )

    def test_merge_predictive_samples_remaps_context_indices_and_flags(self):
        merged = merge_predictive_samples(
            [
                PredictiveSamples(
                    next_times=np.asarray([[1.0]], dtype=np.float32),
                    next_locs=np.asarray([[[0.0, 0.0]]], dtype=np.float32),
                    true_next_times=np.asarray([1.0], dtype=np.float32),
                    true_next_locs=np.asarray([[0.0, 0.0]], dtype=np.float32),
                    history_end_times=np.asarray([0.5], dtype=np.float32),
                    sequence_index=np.asarray([0], dtype=np.int64),
                    target_event_index=np.asarray([1], dtype=np.int64),
                    history_length=np.asarray([1], dtype=np.int64),
                    is_last_context=np.asarray([True], dtype=np.bool_),
                    sampling_succeeded=np.asarray([True], dtype=np.bool_),
                    sampling_backend="exact_intensity_thinning",
                ),
                PredictiveSamples(
                    next_times=np.asarray([[2.0], [3.0]], dtype=np.float32),
                    next_locs=np.asarray(
                        [[[1.0, 1.0]], [[2.0, 2.0]]],
                        dtype=np.float32,
                    ),
                    true_next_times=np.asarray([2.0, 3.0], dtype=np.float32),
                    true_next_locs=np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
                    history_end_times=np.asarray([1.5, 2.5], dtype=np.float32),
                    sequence_index=np.asarray([0, 1], dtype=np.int64),
                    target_event_index=np.asarray([1, 1], dtype=np.int64),
                    history_length=np.asarray([1, 1], dtype=np.int64),
                    is_last_context=np.asarray([True, True], dtype=np.bool_),
                    sampling_succeeded=np.asarray([False, True], dtype=np.bool_),
                    sampling_backend="exact_intensity_thinning",
                ),
            ]
        )
        np.testing.assert_array_equal(merged.sequence_index, np.asarray([0, 1, 2], dtype=np.int64))
        np.testing.assert_array_equal(
            merged.is_last_context,
            np.asarray([True, True, True], dtype=np.bool_),
        )
        np.testing.assert_array_equal(
            merged.sampling_succeeded,
            np.asarray([True, False, True], dtype=np.bool_),
        )


if __name__ == "__main__":
    unittest.main()
