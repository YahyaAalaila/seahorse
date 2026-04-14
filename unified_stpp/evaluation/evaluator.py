"""Evaluator orchestrator and top-level evaluate() public API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .context import EvalContext, GroundTruth
from .profiles import MetricPlanError, resolve_metric_plan
from .result import Metric, MetricResult, Report

if TYPE_CHECKING:
    from unified_stpp.runner.runner import STPPRunner


class Evaluator:
    """Runs a list of metrics against an EvalContext, skipping those whose
    requirements are not satisfied.

    Parameters
    ----------
    metrics:  Pre-instantiated Metric objects to evaluate.
    """

    def __init__(
        self,
        metrics: list[Metric] | tuple[Metric, ...],
        *,
        allowed_artifact_families: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.metrics = list(metrics)
        self.allowed_artifact_families = frozenset(allowed_artifact_families or ())

    def run(self, ctx: EvalContext) -> Report:
        """Evaluate all metrics and return a Report.

        Metrics whose ``requires`` set is not a subset of
        ``ctx.available_capabilities`` are recorded as unavailable (not raised
        as errors).  Any other exception during ``compute()`` is caught and
        stored in the ``reason`` field so one failing metric cannot abort the
        entire evaluation run.
        """
        results: dict[str, MetricResult] = {}
        available = ctx.available_capabilities  # triggers cached_property once

        for m in self.metrics:
            missing = m.requires - available
            if missing:
                results[m.name] = MetricResult(
                    value=None,
                    available=False,
                    reason=f"missing capabilities: {sorted(missing)}",
                )
                continue
            try:
                results[m.name] = m.compute(ctx)
            except MetricPlanError:
                raise
            except Exception as exc:  # noqa: BLE001
                results[m.name] = MetricResult(
                    value=None,
                    available=False,
                    reason=f"error: {type(exc).__name__}: {exc}",
                )

        return Report(results)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    runner: "STPPRunner",
    test_seqs: list[dict[str, np.ndarray]],
    *,
    ground_truth: GroundTruth | None = None,
    domain_mask: np.ndarray | None = None,
    train_data: list[dict[str, np.ndarray]] | None = None,
    k_pred: int = 32,
    k_gen: int = 20,
    exact_time_bins: int = 8,
    exact_spatial_bins: int = 8,
    grid_spec: dict[str, Any] | None = None,
    seed: int = 0,
    device: torch.device | str | None = None,
    skip_requires: set[str] | None = None,
    only_requires: set[str] | None = None,
    metric_profile: str | None = "core",
    allowed_artifact_families: set[str] | frozenset[str] | None = None,
    allow_heavy_artifacts: bool = False,
    artifact_dir: str | Path | None = None,
    artifact_mode: str = "load_or_compute",
    metrics: list[Metric] | None = None,
) -> Report:
    """Evaluate a fitted STPP model on ``test_seqs`` using a metric profile.

    Parameters
    ----------
    runner:
        Loaded STPPRunner with model weights in eval mode.
    test_seqs:
        List of test sequences.  Each is a dict with at least ``"times"`` and
        ``"locations"`` keys holding numpy arrays.
    ground_truth:
        ``GroundTruth`` object for synthetic datasets; pass ``None`` for real data.
    domain_mask:
        Boolean ``(X, Y)`` array marking forbidden spatial regions.
    train_data:
        Training sequences needed for train/test NLL gap (M4).
    k_pred:
        Number of next-event samples per test event for predictive metrics.
    k_gen:
        Number of full-sequence rollouts per test sequence for generative metrics.
    grid_spec:
        Dict with keys ``x_resolution``, ``y_resolution``, ``t_resolution``,
        ``x_range``, ``y_range``.  Controls the intensity grid for grid metrics.
    seed:
        Base random seed for reproducible sampling.
    device:
        Torch device for model inference.  Defaults to the model's current device.
    skip_requires:
        If provided, any metric that has *any* element of ``skip_requires`` in its
        ``requires`` set is skipped.  Useful for excluding expensive artifact
        families (e.g. ``skip_requires={"samples_generative"}``).
    only_requires:
        If provided, only metrics whose ``requires`` set is a subset of
        ``only_requires`` are run.  Useful for quick smoke-checks
        (e.g. ``only_requires=set()`` runs only metrics with no requirements).
    metric_profile:
        Named metric profile to run when ``metrics`` is not provided.  Defaults
        to ``"core"`` so sampling-heavy artifacts are never implicit.
    allowed_artifact_families:
        Heavy artifact families explicitly planned for this evaluation call.
        Sampling/grid metrics fail before execution unless their heavy artifact
        families are present here or selected through a profile.
    allow_heavy_artifacts:
        Escape hatch for explicit ``metrics=...`` calls.  When true, the heavy
        artifact families declared by those metric instances are treated as
        planned.
    artifact_dir:
        Optional root for persisted metric artifacts.
    artifact_mode:
        Artifact read/write policy when artifact_dir is set: "load_or_compute"
        or "load_only".
    metrics:
        Explicit list of pre-instantiated Metric objects.  If ``None``, all
        metrics in ``metric_profile`` are used.

    Returns
    -------
    Report
        Collection of MetricResults keyed by metric name.

    Examples
    --------
    >>> report = evaluate(runner, test_seqs)
    >>> report = evaluate(runner, test_seqs, metric_profile="predictive")
    >>> report = evaluate(runner, test_seqs, ground_truth=gt, train_data=train, metric_profile="nll")
    >>> report.summary()
    >>> report.save("run_042/eval/")
    """
    plan = resolve_metric_plan(
        metric_profile_name=metric_profile,
        metrics=metrics,
        allowed_artifact_families=allowed_artifact_families,
        allow_heavy_artifacts=allow_heavy_artifacts,
    )
    planned_metrics: list[Metric] = list(plan.metrics)

    # Apply skip_requires filter: drop metrics that touch any skipped capability.
    if skip_requires:
        planned_metrics = [m for m in planned_metrics if not (m.requires & skip_requires)]

    # Apply only_requires filter: keep only metrics whose requirements are a
    # subset of the allowed set.
    if only_requires is not None:
        planned_metrics = [m for m in planned_metrics if m.requires <= only_requires]

    # Re-validate after legacy filtering in case explicit metrics were passed.
    plan = resolve_metric_plan(
        metric_profile_name=None,
        metrics=planned_metrics,
        allowed_artifact_families=plan.allowed_artifact_families,
        allow_heavy_artifacts=False,
    )

    ctx = EvalContext(
        runner=runner,
        test_seqs=test_seqs,
        device=device,
        ground_truth=ground_truth,
        domain_mask=domain_mask,
        train_data=train_data,
        k_pred=k_pred,
        k_gen=k_gen,
        exact_time_bins=exact_time_bins,
        exact_spatial_bins=exact_spatial_bins,
        grid_spec=grid_spec,
        seed=seed,
        planned_artifact_families=plan.allowed_artifact_families,
        artifact_dir=artifact_dir,
        artifact_mode=artifact_mode,
    )

    ctx.ensure_artifacts(plan.allowed_artifact_families)
    report = Evaluator(
        plan.metrics,
        allowed_artifact_families=plan.allowed_artifact_families,
    ).run(ctx)
    report.artifact_events = dict(ctx.artifact_events)
    return report
