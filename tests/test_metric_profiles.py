from __future__ import annotations

import unittest

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
        self.assertIn("surface", names)
        self.assertIn("full", names)
        self.assertNotIn("cheap", names)
        self.assertNotIn("predictive-light", names)


if __name__ == "__main__":
    unittest.main()
