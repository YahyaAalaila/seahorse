"""Evaluation-layer exports.

Public API (top-level):
  evaluate      — Run an explicit metric profile against a fitted runner + test set.
  Report        — Collection of MetricResults from one evaluation run.
  MetricResult  — Structured output from a single metric computation.

Everything else (Metric base class, EvalContext, register_metric, Evaluator, …)
is importable via its submodule path for power users writing custom metrics, but
is intentionally absent from this top-level namespace.
"""

# --- New registry-based evaluation API ---
from .evaluator import evaluate
from .profiles import (
    GENERATIVE_ROLLOUTS,
    HEAVY_ARTIFACT_FAMILIES,
    INTENSITY_GRID,
    PREDICTIVE_SAMPLES,
    MetricPlan,
    MetricPlanError,
    MetricProfile,
    artifact_families_for_metrics,
    metric_profile,
    profile_names,
)
from .result import MetricResult, Report

# Trigger metric registration by importing the metrics package.
# The registry is populated as a side-effect of this import.
from . import metrics as _metrics_pkg  # noqa: F401

# --- Legacy evaluation helpers (unchanged) ---
from .common import FrameWindow, HistoryQuery, RunTarget
from .io import (
    load_predictive_bundle,
    load_surface_bundle,
    write_predictive_bundle,
    write_surface_bundle,
)
from .surface_metrics import predictive_surface_mae, predictive_surface_rmse
from .predictive_compare import (
    PredictiveComparator,
    PredictiveCompareSpec,
    PredictiveComparisonResult,
    PredictiveFrameResult,
    PredictiveModelResult,
)
from .predictive_sampling import ExactProposalConfig
from .surface import (
    SurfaceDiagnosticEvaluator,
    SurfaceDiagnosticResult,
    SurfaceDiagnosticSpec,
)

__all__ = [
    # New API
    "evaluate",
    "MetricResult",
    "MetricPlan",
    "MetricPlanError",
    "MetricProfile",
    "PREDICTIVE_SAMPLES",
    "GENERATIVE_ROLLOUTS",
    "INTENSITY_GRID",
    "HEAVY_ARTIFACT_FAMILIES",
    "Report",
    "artifact_families_for_metrics",
    # Legacy
    "ExactProposalConfig",
    "FrameWindow",
    "HistoryQuery",
    "PredictiveComparator",
    "PredictiveCompareSpec",
    "PredictiveComparisonResult",
    "PredictiveFrameResult",
    "PredictiveModelResult",
    "RunTarget",
    "SurfaceDiagnosticEvaluator",
    "SurfaceDiagnosticResult",
    "SurfaceDiagnosticSpec",
    "load_predictive_bundle",
    "load_surface_bundle",
    "metric_profile",
    "predictive_surface_mae",
    "predictive_surface_rmse",
    "profile_names",
    "write_predictive_bundle",
    "write_surface_bundle",
]
