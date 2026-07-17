from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from seahorse.evaluation.artifacts import PredictiveSamples, build_predictive_samples_key
from seahorse.evaluation.predictive.sampling import _thinning_next_events_adaptive_batch
from seahorse.evaluation.profiles import PREDICTIVE_SAMPLES, resolve_metric_plan
from seahorse.evaluation.registry import metric_by_name


def _geographic_samples() -> PredictiveSamples:
    return PredictiveSamples(
        next_times=np.asarray([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]], dtype=np.float32),
        next_locs=np.asarray(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            ],
            dtype=np.float32,
        ),
        true_next_times=np.asarray([1.5, 2.5], dtype=np.float32),
        true_next_locs=np.asarray([[1.5, 0.0], [0.0, 0.0]], dtype=np.float32),
        history_end_times=np.asarray([1.0, 2.0], dtype=np.float32),
        sequence_index=np.asarray([0, 1], dtype=np.int64),
        target_event_index=np.asarray([1, 1], dtype=np.int64),
        history_length=np.asarray([1, 1], dtype=np.int64),
        is_last_context=np.asarray([True, True], dtype=np.bool_),
        sampling_succeeded=np.asarray([True, False], dtype=np.bool_),
        sampling_backend="fixture",
    )


class TestOperationalNextEventMetric(unittest.TestCase):
    def test_reports_sample_medoid_miss_in_kilometres(self):
        class Context:
            samples_predictive = _geographic_samples()

        result = metric_by_name("next_event_distance_km").compute(Context())

        self.assertTrue(result.available)
        self.assertEqual(result.method, "predictive_sample_medoid_haversine_km")
        self.assertAlmostEqual(float(result.value), 55.59754, places=4)
        self.assertAlmostEqual(float(result.per_event[0]), 55.59754, places=4)
        self.assertTrue(np.isnan(result.per_event[1]))

    def test_rejects_non_geographic_coordinates(self):
        samples = _geographic_samples()
        samples.next_locs[0, 0, 1] = 100.0

        class Context:
            samples_predictive = samples

        with self.assertRaisesRegex(ValueError, "longitude/latitude"):
            metric_by_name("next_event_distance_km").compute(Context())

    def test_operational_profile_is_explicit_and_sample_backed(self):
        plan = resolve_metric_plan(
            metric_profile_name="operational-geographic",
            metrics=None,
            allowed_artifact_families=None,
            allow_heavy_artifacts=False,
        )

        self.assertEqual(
            plan.metric_names,
            ("temporal_mae", "next_event_distance_km"),
        )
        self.assertEqual(plan.allowed_artifact_families, frozenset({PREDICTIVE_SAMPLES}))


class TestAdaptiveExactSampling(unittest.TestCase):
    def test_configured_window_expansions_reach_low_rate_tail(self):
        calls = 0

        def fake_thinning(*args, **kwargs):
            nonlocal calls
            del args
            calls += 1
            t_start = float(kwargs["t_start"])
            t_max = float(kwargs["t_max"])
            if calls < 3:
                return (
                    np.asarray([t_max], dtype=np.float32),
                    np.asarray([[0.0, 0.0]], dtype=np.float32),
                )
            return (
                np.asarray([t_start + 0.1], dtype=np.float32),
                np.asarray([[0.5, 0.5]], dtype=np.float32),
            )

        history = {
            "times": np.asarray([0.0], dtype=np.float32),
            "locations": np.asarray([[0.0, 0.0]], dtype=np.float32),
        }
        with (
            patch(
                "seahorse.evaluation.predictive.sampling.build_state_from_history",
                return_value=object(),
            ),
            patch(
                "seahorse.evaluation.predictive.sampling.build_exact_intensity_fn",
                return_value=lambda t, s: torch.ones(t.shape[0]),
            ),
            patch(
                "seahorse.evaluation.predictive.sampling._build_exact_proposal_cache",
                return_value=(object(), {}),
            ),
            patch(
                "seahorse.evaluation.predictive.sampling._thinning_k_chains_batched",
                side_effect=fake_thinning,
            ),
        ):
            next_t, next_s, success = _thinning_next_events_adaptive_batch(
                SimpleNamespace(),
                history,
                1,
                initial_horizon=1.0,
                xmin=0.0,
                xmax=1.0,
                ymin=0.0,
                ymax=1.0,
                device=torch.device("cpu"),
                exact_max_window_expansions=2,
            )

        self.assertEqual(calls, 3)
        self.assertTrue(bool(success[0]))
        self.assertAlmostEqual(float(next_t[0]), 3.1, places=5)
        np.testing.assert_allclose(next_s[0], [0.5, 0.5])

    def test_artifact_key_tracks_window_expansion_budget(self):
        caps = SimpleNamespace(nll_kind="exact", has_intensity=True, has_native_sampler=False)
        runner = SimpleNamespace(
            _run_dir=None,
            model=SimpleNamespace(event_model=SimpleNamespace(capabilities=caps)),
            config=SimpleNamespace(model=SimpleNamespace(preset="poisson_gmm")),
        )
        seqs = [
            {
                "times": np.asarray([0.0, 1.0], dtype=np.float32),
                "locations": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            }
        ]

        key_a = build_predictive_samples_key(
            runner,
            seqs,
            k=8,
            seed=0,
            device="cpu",
            exact_max_window_expansions=8,
        )
        key_b = build_predictive_samples_key(
            runner,
            seqs,
            k=8,
            seed=0,
            device="cpu",
            exact_max_window_expansions=18,
        )

        self.assertNotEqual(key_a.digest, key_b.digest)
        self.assertEqual(
            key_b.metadata["exact_intensity_sampler"]["max_window_expansions"],
            18,
        )


if __name__ == "__main__":
    unittest.main()
