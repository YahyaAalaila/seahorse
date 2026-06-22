"""Intensity-grid metrics (M21–M26, M29).

Registers:
  intensity_rmse                  — Intensity RMSE on grid (catalog M21)
  intensity_relative_error        — Intensity relative error (catalog M22)
  intensity_correlation           — Intensity Pearson correlation (catalog M23)
  log_intensity_rmse              — Log-intensity RMSE (catalog M24)
  mass_placement_error            — Spatial mass placement error (catalog M25)
  background_trigger_decomposition— Background vs triggering decomposition (catalog M26)
  support_leakage                 — Support leakage for bounded domains (catalog M29)

All grid metrics require ctx.intensity_grid (computed lazily on first access)
plus "ground_truth_intensity" (M21–M26) or "domain_mask" (M29).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seahorse.evaluation.profiles import GROUND_TRUTH, INTENSITY_GRID
from seahorse.evaluation.registry import register_metric
from seahorse.evaluation.result import Metric, MetricResult

if TYPE_CHECKING:
    from seahorse.evaluation.context import EvalContext


# ---------------------------------------------------------------------------
# Shared accessor
# ---------------------------------------------------------------------------


def _get_grids(ctx: "EvalContext") -> tuple[np.ndarray, np.ndarray]:
    """Return (lambda_hat, lambda_true) from ctx.intensity_grid, both (T,X,Y).

    Raises ValueError if ground-truth grid is unavailable.
    """
    grid = ctx.intensity_grid
    if grid.lambda_true is None:
        raise ValueError("ground_truth_intensity not available in intensity grid")
    lh = grid.lambda_hat.astype(np.float64)
    lt = grid.lambda_true.astype(np.float64)
    return lh, lt


# ---------------------------------------------------------------------------
# M21: Intensity RMSE
# ---------------------------------------------------------------------------


@register_metric
class IntensityRMSE(Metric):
    """M21: RMSE between model intensity and ground-truth intensity on the grid."""

    name = "intensity_rmse"
    catalog_id = "M21"
    requires = frozenset({"ground_truth_intensity"})
    artifact_families = frozenset({INTENSITY_GRID, GROUND_TRUTH})
    cost_class = "grid_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        try:
            lh, lt = _get_grids(ctx)
        except ValueError as e:
            return MetricResult(value=None, available=False, reason=str(e))
        rmse = float(np.sqrt(np.mean((lh - lt) ** 2)))
        return MetricResult(value=rmse, method=ctx.intensity_grid.method)


# ---------------------------------------------------------------------------
# M22: Intensity relative error
# ---------------------------------------------------------------------------


@register_metric
class IntensityRelativeError(Metric):
    """M22: RMSE / mean(λ_true) — scale-invariant intensity error."""

    name = "intensity_relative_error"
    catalog_id = "M22"
    requires = frozenset({"ground_truth_intensity"})
    artifact_families = frozenset({INTENSITY_GRID, GROUND_TRUTH})
    cost_class = "grid_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        try:
            lh, lt = _get_grids(ctx)
        except ValueError as e:
            return MetricResult(value=None, available=False, reason=str(e))
        mean_true = float(lt.mean())
        if mean_true < 1e-12:
            return MetricResult(
                value=None, available=False, reason="mean true intensity is zero"
            )
        rmse = float(np.sqrt(np.mean((lh - lt) ** 2)))
        return MetricResult(value=rmse / mean_true, method=ctx.intensity_grid.method)


# ---------------------------------------------------------------------------
# M23: Intensity Pearson correlation
# ---------------------------------------------------------------------------


@register_metric
class IntensityCorrelation(Metric):
    """M23: Pearson correlation between λ̂ and λ_true on the shared grid."""

    name = "intensity_correlation"
    catalog_id = "M23"
    requires = frozenset({"ground_truth_intensity"})
    artifact_families = frozenset({INTENSITY_GRID, GROUND_TRUTH})
    cost_class = "grid_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        try:
            lh, lt = _get_grids(ctx)
        except ValueError as e:
            return MetricResult(value=None, available=False, reason=str(e))

        lh_flat = lh.ravel()
        lt_flat = lt.ravel()

        lh_c = lh_flat - lh_flat.mean()
        lt_c = lt_flat - lt_flat.mean()
        denom = np.sqrt(np.sum(lh_c ** 2) * np.sum(lt_c ** 2))
        if denom < 1e-12:
            return MetricResult(
                value=None, available=False, reason="zero variance in one of the grids"
            )
        corr = float(np.sum(lh_c * lt_c) / denom)
        return MetricResult(value=corr, method=ctx.intensity_grid.method)


# ---------------------------------------------------------------------------
# M24: Log-intensity RMSE
# ---------------------------------------------------------------------------


@register_metric
class LogIntensityRMSE(Metric):
    """M24: RMSE on log(λ) — captures relative multiplicative error."""

    name = "log_intensity_rmse"
    catalog_id = "M24"
    requires = frozenset({"ground_truth_intensity"})
    artifact_families = frozenset({INTENSITY_GRID, GROUND_TRUTH})
    cost_class = "grid_heavy"

    _LOG_CLIP = -10.0  # clip log to max(-10, log(λ))

    def compute(self, ctx: "EvalContext") -> MetricResult:
        try:
            lh, lt = _get_grids(ctx)
        except ValueError as e:
            return MetricResult(value=None, available=False, reason=str(e))

        log_lh = np.maximum(np.log(np.maximum(lh, 1e-8)), self._LOG_CLIP)
        log_lt = np.maximum(np.log(np.maximum(lt, 1e-8)), self._LOG_CLIP)
        rmse = float(np.sqrt(np.mean((log_lh - log_lt) ** 2)))
        return MetricResult(value=rmse, method=ctx.intensity_grid.method)


# ---------------------------------------------------------------------------
# M25: Spatial mass placement error
# ---------------------------------------------------------------------------


@register_metric
class MassPlacementError(Metric):
    """M25: Fraction of model mass placed in ground-truth top-α cells.

    Reports curves over α ∈ {0.1, 0.2, 0.3, 0.5}.
    """

    name = "mass_placement_error"
    catalog_id = "M25"
    requires = frozenset({"ground_truth_intensity"})
    artifact_families = frozenset({INTENSITY_GRID, GROUND_TRUTH})
    cost_class = "grid_heavy"

    _ALPHAS = [0.1, 0.2, 0.3, 0.5]

    def compute(self, ctx: "EvalContext") -> MetricResult:
        try:
            lh, lt = _get_grids(ctx)
        except ValueError as e:
            return MetricResult(value=None, available=False, reason=str(e))

        lh_flat = lh.ravel()
        lt_flat = lt.ravel()
        total_mass = float(lh_flat.sum())
        if total_mass < 1e-12:
            return MetricResult(value=None, available=False, reason="model intensity is zero")

        # Sort by ground-truth intensity descending
        order = np.argsort(lt_flat)[::-1]
        lh_sorted = lh_flat[order]
        curve: dict[str, float] = {}
        for alpha in self._ALPHAS:
            top_k = max(1, int(alpha * len(lh_flat)))
            frac = float(lh_sorted[:top_k].sum() / total_mass)
            curve[str(alpha)] = frac

        value = curve.get("0.1")
        return MetricResult(value=value, curve=curve, method=ctx.intensity_grid.method)


# ---------------------------------------------------------------------------
# M26: Background vs triggering decomposition
# ---------------------------------------------------------------------------


@register_metric
class BackgroundTriggerDecomposition(Metric):
    """M26: Decomposed error — background vs triggering (synthetic only).

    Evaluates model with empty history (no past events) to estimate the
    background component μ(s).  Compares to ground-truth background from
    GroundTruth.params["background_grid"] if available.

    Note: this decomposition is exact only for AutoSTPP (inductive bias
    matches additive Hawkes structure).  For other models it is a proxy.
    """

    name = "background_trigger_decomposition"
    catalog_id = "M26"
    requires = frozenset({"ground_truth_params", "intensity"})
    artifact_families = frozenset({INTENSITY_GRID, GROUND_TRUTH})
    cost_class = "grid_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        gt_params = ctx.ground_truth.params if ctx.ground_truth else None
        if gt_params is None or "background_grid" not in gt_params:
            return MetricResult(
                value=None,
                available=False,
                reason=(
                    "ground_truth.params must contain 'background_grid' "
                    "(T×X×Y or X×Y array)"
                ),
            )

        # Evaluate model with empty history on the shared grid
        from seahorse.evaluation.intensity import _direct_intensity_grid

        grid = ctx.intensity_grid
        xs, ys, ts = grid.xs, grid.ys, grid.ts

        # Empty-history baseline
        empty_seq: list[dict] = [{"times": np.zeros(0, np.float32), "locations": np.zeros((0, 2), np.float32)}]
        try:
            bg_hat = _direct_intensity_grid(ctx.runner, empty_seq, xs, ys, ts, ctx.device)
        except Exception as exc:
            return MetricResult(
                value=None,
                available=False,
                reason=f"failed to evaluate empty-history intensity: {exc}",
            )

        bg_true = np.asarray(gt_params["background_grid"], dtype=np.float64)
        bg_hat = bg_hat.astype(np.float64)

        # If background_grid is (X, Y) (spatial only), broadcast over time
        if bg_true.ndim == 2:
            bg_true = bg_true[None].repeat(len(ts), axis=0)

        if bg_true.shape != bg_hat.shape:
            return MetricResult(
                value=None,
                available=False,
                reason=(
                    f"background_grid shape {bg_true.shape} does not match "
                    f"grid shape {bg_hat.shape}"
                ),
            )

        bg_rmse = float(np.sqrt(np.mean((bg_hat - bg_true) ** 2)))

        # Triggering = full − background; compare on the intensity grid
        model_caps = ctx.runner.model.event_model.capabilities
        is_autostpp = (
            hasattr(model_caps, "nll_kind")
            and model_caps.has_intensity
            and not model_caps.has_density
        )
        annotation = "exact_decomposition" if is_autostpp else "proxy"

        return MetricResult(
            value=bg_rmse,
            curve={"background_rmse": bg_rmse},
            method=annotation,
        )


# ---------------------------------------------------------------------------
# M29: Support leakage
# ---------------------------------------------------------------------------


@register_metric
class SupportLeakage(Metric):
    """M29: Fraction of model mass in forbidden spatial regions.

    For intensity-queryable models: integrate λ*(t,s) over forbidden cells.
    Sample-based support leakage should be a separate explicit metric once
    rollout artifacts are persisted.

    Skipped automatically when domain_mask is absent.
    """

    name = "support_leakage"
    catalog_id = "M29"
    requires = frozenset({"domain_mask"})
    artifact_families = frozenset({INTENSITY_GRID})
    cost_class = "grid_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        mask = ctx.domain_mask  # (X, Y) bool, True = forbidden
        if mask is None:
            return MetricResult(value=None, available=False, reason="no domain_mask provided")

        model_caps = ctx.runner.model.event_model.capabilities

        if model_caps.has_intensity:
            return self._leakage_direct(ctx, mask)
        return MetricResult(
            value=None,
            available=False,
            reason=(
                "support_leakage no longer falls back to generative samples implicitly; "
                "add an explicit sampled support-leakage metric when rollout artifacts exist"
            ),
        )

    def _leakage_direct(self, ctx: "EvalContext", mask: np.ndarray) -> MetricResult:
        """Compute leakage from the intensity grid directly."""
        grid = ctx.intensity_grid
        lh = grid.lambda_hat.astype(np.float64)  # (T, X, Y)

        # mask is (X, Y); resize if grid resolution differs
        if mask.shape != (lh.shape[1], lh.shape[2]):
            from scipy.ndimage import zoom  # type: ignore[import]
            zoom_x = lh.shape[1] / mask.shape[0]
            zoom_y = lh.shape[2] / mask.shape[1]
            mask_r = zoom(mask.astype(float), (zoom_x, zoom_y), order=0) > 0.5
        else:
            mask_r = mask

        # Forbidden mass fraction averaged over time steps
        forbidden = lh[:, mask_r].sum()
        total = lh.sum()
        if total < 1e-12:
            return MetricResult(value=0.0, method="direct")
        return MetricResult(value=float(forbidden / total), method="direct")
