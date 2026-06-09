from __future__ import annotations

import builtins
import unittest
from unittest.mock import patch

import numpy as np

from unified_stpp.evaluation.context import GenerativeRollouts
from unified_stpp.evaluation import MetricPlanError, evaluate, profile_names
from unified_stpp.evaluation.profiles import (
    GENERATIVE_ROLLOUTS,
    INTENSITY_GRID,
    PREDICTIVE_SAMPLES,
    resolve_metric_plan,
)
from unified_stpp.evaluation.registry import metric_by_name
from unified_stpp.evaluation.result import Metric, MetricResult


class TestMetricProfiles(unittest.TestCase):
    def _autoregressive_rollout_context(self):
        true_times = [
            np.arange(13, dtype=np.float32),
            np.arange(7, dtype=np.float32),
        ]
        true_locs = [
            np.column_stack(
                [
                    np.linspace(0.0, 1.2, 13, dtype=np.float32),
                    np.linspace(1.0, 2.2, 13, dtype=np.float32),
                ]
            ),
            np.column_stack(
                [
                    np.linspace(0.0, 0.6, 7, dtype=np.float32),
                    np.linspace(1.0, 1.6, 7, dtype=np.float32),
                ]
            ),
        ]
        rollout_times = [
            [
                true_times[0][2:12].astype(np.float32) + 0.01,
                true_times[0][2:12].astype(np.float32) + 0.02,
            ],
            [
                true_times[1][2:7].astype(np.float32).tolist()
                + [7.0, 8.0, 9.0, 10.0, 11.0],
                true_times[1][2:7].astype(np.float32).tolist()
                + [7.1, 8.1, 9.1, 10.1, 11.1],
            ],
        ]
        rollout_times = [
            [np.asarray(arr, dtype=np.float32) for arr in seq_rollouts]
            for seq_rollouts in rollout_times
        ]
        rollout_locs = [
            [
                true_locs[0][2:12].astype(np.float32) + 0.01,
                true_locs[0][2:12].astype(np.float32) + 0.02,
            ],
            [
                np.vstack(
                    [
                        true_locs[1][2:7].astype(np.float32) + 0.01,
                        np.tile(np.asarray([[0.8, 1.8]], dtype=np.float32), (5, 1)),
                    ]
                ),
                np.vstack(
                    [
                        true_locs[1][2:7].astype(np.float32) + 0.02,
                        np.tile(np.asarray([[0.9, 1.9]], dtype=np.float32), (5, 1)),
                    ]
                ),
            ],
        ]
        rollouts = GenerativeRollouts(
            rollout_times=rollout_times,
            rollout_locs=rollout_locs,
            true_times=true_times,
            true_locs=true_locs,
            context_lengths=[2, 2],
            method="native",
        )

        class Ctx:
            samples_generative = rollouts

        return Ctx()

    def test_default_evaluate_rejects_unplanned_predictive_sampling_metric(self):
        metric = metric_by_name("temporal_crps")

        with self.assertRaisesRegex(MetricPlanError, "predictive_samples"):
            evaluate(object(), [], metrics=[metric])

    def test_predictive_profile_plans_predictive_samples(self):
        plan = resolve_metric_plan(
            metric_profile_name="predictive",
            metrics=None,
            allowed_artifact_families=None,
            allow_heavy_artifacts=False,
        )

        self.assertIn(PREDICTIVE_SAMPLES, plan.allowed_artifact_families)
        self.assertIn("temporal_crps", plan.metric_names)

    def test_surface_profile_plans_sampler_fallback_artifacts(self):
        plan = resolve_metric_plan(
            metric_profile_name="surface",
            metrics=None,
            allowed_artifact_families=None,
            allow_heavy_artifacts=False,
        )

        self.assertIn(INTENSITY_GRID, plan.allowed_artifact_families)
        self.assertIn(GENERATIVE_ROLLOUTS, plan.allowed_artifact_families)
        self.assertIn("intensity_rmse", plan.metric_names)

    def test_autoregressive_profile_plans_only_generative_rollouts(self):
        plan = resolve_metric_plan(
            metric_profile_name="autoregressive",
            metrics=None,
            allowed_artifact_families=None,
            allow_heavy_artifacts=False,
        )

        self.assertEqual(
            plan.metric_names,
            (
                "rollout_coherence",
                "ar_temporal_crps_h1",
                "ar_spatial_energy_score_h1",
            ),
        )
        self.assertEqual(plan.allowed_artifact_families, frozenset({GENERATIVE_ROLLOUTS}))
        self.assertNotIn(PREDICTIVE_SAMPLES, plan.allowed_artifact_families)
        self.assertEqual(len(plan.metric_names), 3)

    def test_rollout_coherence_uses_sparse_horizons_and_h10_scalar(self):
        result = metric_by_name("rollout_coherence").compute(
            self._autoregressive_rollout_context()
        )

        self.assertTrue(result.available)
        self.assertIsNotNone(result.curve)
        curve = result.curve or {}
        horizon_keys = {str(k) for k in curve if not str(k).startswith("n_h_")}
        self.assertEqual(horizon_keys, {"1", "5", "10"})
        self.assertEqual(result.value, curve["10"])
        self.assertEqual(curve["n_h_1"], 2.0)
        self.assertEqual(curve["n_h_5"], 2.0)
        self.assertEqual(curve["n_h_10"], 1.0)

    def test_rollout_coherence_energy_fallback_available_without_pot(self):
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ot":
                raise ImportError("POT intentionally hidden")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            result = metric_by_name("rollout_coherence").compute(
                self._autoregressive_rollout_context()
            )

        self.assertTrue(result.available)
        self.assertEqual(result.method, "energy_fallback")
        self.assertIsNotNone(result.curve)
        self.assertEqual(result.value, (result.curve or {})["10"])

    def test_nll_names_do_not_hide_sample_kde_fallback(self):
        temporal = metric_by_name("temporal_nll")
        temporal_kde = metric_by_name("temporal_nll_sample_kde")

        self.assertNotIn(PREDICTIVE_SAMPLES, temporal.artifact_families)
        self.assertIn(PREDICTIVE_SAMPLES, temporal_kde.artifact_families)

    def test_deprecated_profile_aliases_resolve(self):
        alias_plan = resolve_metric_plan(
            metric_profile_name="predictive-light",
            metrics=None,
            allowed_artifact_families=None,
            allow_heavy_artifacts=False,
        )
        canonical_plan = resolve_metric_plan(
            metric_profile_name="predictive",
            metrics=None,
            allowed_artifact_families=None,
            allow_heavy_artifacts=False,
        )

        self.assertEqual(alias_plan.metric_names, canonical_plan.metric_names)

    def test_runtime_artifact_guard_is_not_swallowed(self):
        class Caps:
            nll_kind = "none"
            has_intensity = True
            has_density = False
            has_native_sampler = False

        class EventModel:
            capabilities = Caps()

        class Model:
            event_model = EventModel()

            def parameters(self):
                return iter(())

        class Runner:
            model = Model()

        class HiddenSamplingMetric(Metric):
            name = "hidden_sampling"
            requires = frozenset({"samples_predictive"})

            def compute(self, ctx):
                _ = ctx.samples_predictive
                return MetricResult(value=0.0)

        with self.assertRaisesRegex(MetricPlanError, "predictive_samples"):
            evaluate(Runner(), [], metrics=[HiddenSamplingMetric()])

    def test_public_profiles_are_named(self):
        names = profile_names()
        self.assertIn("core", names)
        self.assertIn("predictive", names)
        self.assertIn("generative", names)
        self.assertIn("autoregressive", names)
        self.assertIn("surface", names)
        self.assertIn("full", names)
        self.assertNotIn("cheap", names)
        self.assertNotIn("predictive-light", names)


if __name__ == "__main__":
    unittest.main()
