"""Metric profiles and heavy-artifact planning for registry evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .result import Metric

PREDICTIVE_SAMPLES = "predictive_samples"
GENERATIVE_ROLLOUTS = "generative_rollouts"
INTENSITY_GRID = "intensity_grid"
NLL_ARRAYS = "nll_arrays"
TRAIN_DATA = "train_data"
GROUND_TRUTH = "ground_truth"
DOMAIN_MASK = "domain_mask"

HEAVY_ARTIFACT_FAMILIES = frozenset(
    {
        PREDICTIVE_SAMPLES,
        GENERATIVE_ROLLOUTS,
        INTENSITY_GRID,
    }
)


class MetricPlanError(ValueError):
    """Raised when metrics require heavy artifacts that were not planned."""


@dataclass(frozen=True)
class MetricProfile:
    name: str
    metric_names: tuple[str, ...]
    allowed_artifact_families: frozenset[str]
    description: str


@dataclass(frozen=True)
class MetricPlan:
    metrics: tuple[Metric, ...]
    profile: MetricProfile | None
    allowed_artifact_families: frozenset[str]

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(metric.name for metric in self.metrics)


CORE_METRICS = (
    "nll",
    "temporal_nll",
    "spatial_nll",
)

NLL_METRICS = (
    *CORE_METRICS,
    "train_test_nll_gap",
    "gt_nll_gap",
    "context_sensitivity",
)

PREDICTIVE_METRICS = (
    "temporal_crps",
    "spatial_energy_score",
    "temporal_pit",
    "spatial_pit",
    "coverage_at_distance",
    "temporal_mae",
    "spatial_mae",
    "spatial_rmse",
    "joint_distance",
)

PREDICTIVE_EXTENDED_METRICS = (
    *PREDICTIVE_METRICS,
    "hotspot_recall",
    "temporal_nll_sample_kde",
    "spatial_nll_sample_kde",
)

GENERATIVE_METRICS = (
    "wasserstein",
    "mmd",
    "temporal_count_chi2",
    "spatial_count_chi2",
    "spatial_ripley_k",
    "temporal_ripley_k",
    "rollout_coherence",
)

SURFACE_METRICS = (
    "intensity_rmse",
    "intensity_relative_error",
    "intensity_correlation",
    "log_intensity_rmse",
    "mass_placement_error",
    "background_trigger_decomposition",
    "support_leakage",
)

METRIC_PROFILES = {
    "core": MetricProfile(
        name="core",
        metric_names=CORE_METRICS,
        allowed_artifact_families=frozenset(),
        description="No sampling artifacts; basic NLL/report metrics only.",
    ),
    "nll": MetricProfile(
        name="nll",
        metric_names=NLL_METRICS,
        allowed_artifact_families=frozenset(),
        description="NLL-family metrics without predictive/generative sampling.",
    ),
    "predictive": MetricProfile(
        name="predictive",
        metric_names=PREDICTIVE_METRICS,
        allowed_artifact_families=frozenset({PREDICTIVE_SAMPLES}),
        description="Predictive metrics from next-event samples; excludes KDE NLL/hotspot extras.",
    ),
    "generative": MetricProfile(
        name="generative",
        metric_names=GENERATIVE_METRICS,
        allowed_artifact_families=frozenset({GENERATIVE_ROLLOUTS}),
        description="Full-rollout generative metrics.",
    ),
    "surface": MetricProfile(
        name="surface",
        metric_names=SURFACE_METRICS,
        allowed_artifact_families=frozenset({INTENSITY_GRID, GENERATIVE_ROLLOUTS}),
        description=(
            "Intensity-grid diagnostics; sampler-only models may approximate "
            "the grid from generative rollouts."
        ),
    ),
    "full": MetricProfile(
        name="full",
        metric_names=(
            *NLL_METRICS,
            *PREDICTIVE_EXTENDED_METRICS,
            *GENERATIVE_METRICS,
            *SURFACE_METRICS,
        ),
        allowed_artifact_families=HEAVY_ARTIFACT_FAMILIES,
        description="All registered benchmark metrics with all heavy artifact families planned.",
    ),
}

DEPRECATED_PROFILE_ALIASES = {
    "cheap": "core",
    "predictive-light": "predictive",
    "grid-diagnostic": "surface",
    "all": "full",
}


def profile_names() -> tuple[str, ...]:
    return tuple(METRIC_PROFILES)


def metric_profile(name: str) -> MetricProfile:
    name = DEPRECATED_PROFILE_ALIASES.get(name, name)
    try:
        return METRIC_PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown metric profile {name!r}. Available: {profile_names()}") from exc


def artifact_families_for_metrics(metrics: Iterable[Metric]) -> frozenset[str]:
    families: set[str] = set()
    for metric in metrics:
        families.update(getattr(metric, "artifact_families", frozenset()))
    return frozenset(families)


def validate_metric_plan(
    metrics: Iterable[Metric],
    *,
    allowed_artifact_families: Iterable[str],
) -> None:
    allowed = frozenset(allowed_artifact_families)
    violations: list[str] = []
    for metric in metrics:
        families = frozenset(getattr(metric, "artifact_families", frozenset()))
        unplanned = sorted((families & HEAVY_ARTIFACT_FAMILIES) - allowed)
        if unplanned:
            violations.append(f"{metric.name}: {unplanned}")
    if violations:
        raise MetricPlanError(
            "Requested metrics require unplanned heavy artifacts. "
            "Use an explicit metric_profile or allowed_artifact_families before "
            "running sampling-heavy evaluation. Violations: "
            + "; ".join(violations)
        )


def resolve_metric_plan(
    *,
    metric_profile_name: str | None,
    metrics: Iterable[Metric] | None,
    allowed_artifact_families: Iterable[str] | None,
    allow_heavy_artifacts: bool,
) -> MetricPlan:
    from .registry import metric_by_name

    profile = metric_profile(metric_profile_name or "core") if metrics is None else None
    resolved_metrics = (
        tuple(metric_by_name(name) for name in profile.metric_names)
        if profile is not None
        else tuple(metrics or ())
    )
    if allowed_artifact_families is None:
        if profile is not None:
            allowed = profile.allowed_artifact_families
        elif allow_heavy_artifacts:
            allowed = artifact_families_for_metrics(resolved_metrics) & HEAVY_ARTIFACT_FAMILIES
        else:
            allowed = frozenset()
    else:
        allowed = frozenset(allowed_artifact_families)
        if allow_heavy_artifacts:
            allowed = allowed | (artifact_families_for_metrics(resolved_metrics) & HEAVY_ARTIFACT_FAMILIES)

    validate_metric_plan(resolved_metrics, allowed_artifact_families=allowed)
    return MetricPlan(
        metrics=resolved_metrics,
        profile=profile,
        allowed_artifact_families=frozenset(allowed),
    )
