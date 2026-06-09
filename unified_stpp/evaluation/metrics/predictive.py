"""Predictive-quality metrics (M6–M14).

Registers:
  temporal_crps          — Temporal CRPS (catalog M6)
  spatial_energy_score   — Spatial energy score (catalog M7)
  temporal_pit           — Temporal PIT / KS statistic (catalog M8)
  spatial_pit            — Spatial PIT via random projections (catalog M9)
  hotspot_recall         — Top-k hotspot recall (catalog M10)
  coverage_at_distance   — Coverage at distance r (catalog M11)
  temporal_mae           — Temporal MAE (catalog M12)
  spatial_mae            — Spatial MAE (catalog M13)
  spatial_rmse           — Spatial RMSE (catalog M13)
  joint_distance         — Normalized joint event distance (catalog M14)

All metrics in this file share ctx.samples_predictive (K=200 next-event
samples per test event), which is computed lazily once on first access.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from unified_stpp.evaluation.profiles import PREDICTIVE_SAMPLES
from unified_stpp.evaluation.registry import register_metric
from unified_stpp.evaluation.result import Metric, MetricResult

if TYPE_CHECKING:
    from unified_stpp.evaluation.context import EvalContext


def _sampling_success_mask(samples) -> np.ndarray:
    return np.asarray(
        getattr(samples, "sampling_succeeded", np.ones(samples.next_times.shape[0], dtype=np.bool_)),
        dtype=np.bool_,
    )


def _nan_out_failed_contexts(values: np.ndarray, samples) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    mask = _sampling_success_mask(samples)
    if out.ndim == 1 and out.shape[0] == mask.shape[0]:
        out[~mask] = np.nan
    return out


# ---------------------------------------------------------------------------
# M6: Temporal CRPS
# ---------------------------------------------------------------------------


@register_metric
class TemporalCRPS(Metric):
    """M6: Temporal continuous ranked probability score.

    Uses the energy-form CRPS estimator which is O(K log K) after sorting.
    Mean over all test events (excluding the first event of each sequence).
    """

    name = "temporal_crps"
    catalog_id = "M6"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        samples = ctx.samples_predictive
        # Inter-event times: true_next_times − history_end_times
        true_iets = (samples.true_next_times - samples.history_end_times).astype(np.float64)
        pred_iets = (samples.next_times - samples.history_end_times[:, None]).astype(np.float64)
        # Clip to zero — sampling can produce negatives for thinning edge cases
        pred_iets = np.maximum(pred_iets, 0.0)

        per_event = _nan_out_failed_contexts(_crps_energy(pred_iets, true_iets), samples)
        return MetricResult(
            value=float(np.nanmean(per_event)),
            per_event=per_event,
            method=samples.sampling_backend,
        )


def _crps_energy(samples: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Energy-form CRPS: E|X-y| - 0.5 E|X-X'|.

    samples: (N, K)  targets: (N,)  → returns (N,)
    """
    N, K = samples.shape
    # E|X - y|
    term1 = np.mean(np.abs(samples - targets[:, None]), axis=1)  # (N,)
    # E|X - X'| via sorted estimator: for sorted X_1 ≤ ... ≤ X_K:
    # E|X - X'| = 2/K^2 * Σ_k X_k * (2k - K - 1)
    s = np.sort(samples, axis=1)
    k = np.arange(1, K + 1, dtype=np.float64)  # 1-indexed
    weights = 2.0 * k - K - 1.0  # (K,)
    term2 = (2.0 / (K * K)) * (s * weights).sum(axis=1)  # (N,)
    return (term1 - 0.5 * term2).astype(np.float64)


# ---------------------------------------------------------------------------
# M7: Spatial energy score
# ---------------------------------------------------------------------------


@register_metric
class SpatialEnergyScore(Metric):
    """M7: Spatial energy score (multivariate proper scoring rule).

    ES_i = 2 E||X - s_i|| - E||X - X'||, estimated from K samples.
    """

    name = "spatial_energy_score"
    catalog_id = "M7"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        samples = ctx.samples_predictive
        pred_locs = samples.next_locs.astype(np.float64)   # (N, K, 2)
        true_locs = samples.true_next_locs.astype(np.float64)  # (N, 2)

        N, K, _ = pred_locs.shape
        # 2 E||X - y||
        term1 = 2.0 * np.mean(
            np.linalg.norm(pred_locs - true_locs[:, None, :], axis=-1), axis=1
        )  # (N,)
        # E||X - X'|| — approximate with random pairing for speed
        # Draw K//2 independent pairs: even vs odd indices after shuffle
        idx = np.random.permutation(K)
        half = K // 2
        a = pred_locs[:, idx[:half], :]
        b = pred_locs[:, idx[half : 2 * half], :]
        term2 = np.mean(np.linalg.norm(a - b, axis=-1), axis=1)  # (N,)

        per_event = _nan_out_failed_contexts(term1 - term2, samples)
        return MetricResult(
            value=float(np.nanmean(per_event)),
            per_event=per_event,
            method=samples.sampling_backend,
        )


# ---------------------------------------------------------------------------
# M8: Temporal PIT
# ---------------------------------------------------------------------------


@register_metric
class TemporalPIT(Metric):
    """M8: Temporal probability integral transform.

    Under a perfectly calibrated model, PIT values u_i = F*(τ_i | H) are
    Uniform(0,1).  Reports the Kolmogorov-Smirnov statistic and saves the
    raw PIT array as per_event.
    """

    name = "temporal_pit"
    catalog_id = "M8"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        samples = ctx.samples_predictive
        # Empirical CDF of inter-event time samples evaluated at the true IET
        true_iets = (samples.true_next_times - samples.history_end_times).astype(np.float64)
        pred_iets = (samples.next_times - samples.history_end_times[:, None]).astype(np.float64)
        pred_iets = np.maximum(pred_iets, 0.0)

        N, K = pred_iets.shape
        # u_i = fraction of samples < true_iet_i
        pit_values = np.mean(pred_iets < true_iets[:, None], axis=1).astype(np.float64)
        # Clip away exact 0/1 for numerical stability in downstream tests
        pit_values = np.clip(pit_values, 1e-6, 1.0 - 1e-6)
        pit_values = _nan_out_failed_contexts(pit_values, samples)

        # KS statistic: sup |F_n(u) - u|
        valid = pit_values[np.isfinite(pit_values)]
        ks_stat = _ks_uniform(valid)

        return MetricResult(
            value=float(ks_stat),
            per_event=pit_values,
            method=samples.sampling_backend,
        )


def _ks_uniform(u: np.ndarray) -> float:
    """KS statistic against Uniform(0,1) for sorted samples."""
    n = u.shape[0]
    if n == 0:
        return float("nan")
    s = np.sort(u)
    k = np.arange(1, n + 1, dtype=np.float64) / n
    d = np.max(np.maximum(np.abs(k - s), np.abs(k - 1.0 / n - s)))
    return float(d)


# ---------------------------------------------------------------------------
# M9: Spatial PIT via random projections
# ---------------------------------------------------------------------------


@register_metric
class SpatialPIT(Metric):
    """M9: Spatial PIT via random unit-direction projections.

    Projects the spatial distribution onto 10 random directions, computes
    1D PIT for each, and reports the max KS statistic across directions.
    """

    name = "spatial_pit"
    catalog_id = "M9"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"
    _N_DIRECTIONS = 10

    def compute(self, ctx: "EvalContext") -> MetricResult:
        samples = ctx.samples_predictive
        rng = np.random.default_rng(42)
        directions = rng.standard_normal((self._N_DIRECTIONS, 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12

        pred_locs = samples.next_locs.astype(np.float64)  # (N, K, 2)
        true_locs = samples.true_next_locs.astype(np.float64)  # (N, 2)

        ks_stats: list[float] = []
        success_mask = _sampling_success_mask(samples)
        for v in directions:
            # Project: (N, K) and (N,)
            proj_samples = pred_locs @ v          # (N, K)
            proj_true = true_locs @ v              # (N,)
            # PIT: fraction of samples < true projection
            pit = np.mean(proj_samples < proj_true[:, None], axis=1)
            pit = np.clip(pit, 1e-6, 1.0 - 1e-6)
            ks_stats.append(_ks_uniform(pit[success_mask]))

        return MetricResult(
            value=float(np.max(ks_stats)),
            method=samples.sampling_backend,
        )


# ---------------------------------------------------------------------------
# M10: Top-k hotspot recall
# ---------------------------------------------------------------------------


@register_metric
class HotspotRecall(Metric):
    """M10: Top-α hotspot recall on a 32×32 spatial grid.

    For intensity-queryable models: evaluate λ*(t_i, ·) on the grid.
    For sample-only models: use KDE of spatial samples.
    Reports recall at α ∈ {0.01, 0.05, 0.1, 0.2, 0.5}.
    """

    name = "hotspot_recall"
    catalog_id = "M10"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"

    _ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5]
    _GRID_SIZE = 32

    def compute(self, ctx: "EvalContext") -> MetricResult:
        from scipy.stats import gaussian_kde  # type: ignore[import]

        samples = ctx.samples_predictive
        pred_locs = samples.next_locs.astype(np.float64)  # (N, K, 2)
        true_locs = samples.true_next_locs.astype(np.float64)  # (N, 2)

        G = self._GRID_SIZE
        # Build spatial grid over data range
        lo = true_locs.min(axis=0)
        hi = true_locs.max(axis=0)
        span = np.maximum(hi - lo, 1e-4)
        lo -= 0.05 * span
        hi += 0.05 * span
        gx = np.linspace(lo[0], hi[0], G)
        gy = np.linspace(lo[1], hi[1], G)
        xx, yy = np.meshgrid(gx, gy, indexing="ij")
        grid = np.stack([xx.ravel(), yy.ravel()], axis=0)  # (2, G*G)

        # Cell area for determining which cells contain a true event
        dx = (hi[0] - lo[0]) / G
        dy = (hi[1] - lo[1]) / G

        recall_by_alpha: dict[str, list[float]] = {str(a): [] for a in self._ALPHAS}
        success_mask = _sampling_success_mask(samples)

        for i in range(pred_locs.shape[0]):
            if not bool(success_mask[i]):
                for a in self._ALPHAS:
                    recall_by_alpha[str(a)].append(float("nan"))
                continue
            s_samples = pred_locs[i].T  # (2, K)
            try:
                kde = gaussian_kde(s_samples)
                density = kde(grid).reshape(G, G)
            except Exception:
                for a in self._ALPHAS:
                    recall_by_alpha[str(a)].append(float("nan"))
                continue

            s_true = true_locs[i]  # (2,)
            # Cell index of the true event
            ix = int(np.clip((s_true[0] - lo[0]) / dx, 0, G - 1))
            iy = int(np.clip((s_true[1] - lo[1]) / dy, 0, G - 1))
            rank = int(np.sum(density > density[ix, iy]))

            total_cells = G * G
            for a in self._ALPHAS:
                top_k = max(1, int(a * total_cells))
                recall_by_alpha[str(a)].append(1.0 if rank < top_k else 0.0)

        curve = {k: float(np.nanmean(v)) for k, v in recall_by_alpha.items()}
        # Primary scalar: recall at α=0.1
        value = curve.get("0.1", None)
        return MetricResult(
            value=value,
            curve=curve,
            method=samples.sampling_backend,
        )


# ---------------------------------------------------------------------------
# M11: Coverage at distance r
# ---------------------------------------------------------------------------


@register_metric
class CoverageAtDistance(Metric):
    """M11: Coverage at distance r: fraction of events where min sample distance < r.

    Reports curves over 10 log-spaced r values between the 5th and 95th
    percentile of all pairwise distances.
    """

    name = "coverage_at_distance"
    catalog_id = "M11"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        samples = ctx.samples_predictive
        pred_locs = samples.next_locs.astype(np.float64)  # (N, K, 2)
        true_locs = samples.true_next_locs.astype(np.float64)  # (N, 2)
        success_mask = _sampling_success_mask(samples)

        # Minimum distance from any sample to the true location, per event
        min_dists = np.min(
            np.linalg.norm(pred_locs - true_locs[:, None, :], axis=-1), axis=1
        )  # (N,)
        min_dists = _nan_out_failed_contexts(min_dists, samples)

        # Determine r range from data
        all_dists = np.linalg.norm(
            pred_locs.reshape(-1, 2) - true_locs.repeat(pred_locs.shape[1], axis=0), axis=1
        )
        r_lo = max(float(np.percentile(all_dists, 5)), 1e-6)
        r_hi = float(np.percentile(all_dists, 95))
        if r_hi <= r_lo:
            r_hi = r_lo * 10.0
        rs = np.logspace(math.log10(r_lo), math.log10(r_hi), 10)

        curve = {
            f"{r:.4g}": float(np.nanmean(np.where(np.isfinite(min_dists), min_dists < r, np.nan)))
            for r in rs
        }
        # Primary scalar: coverage at the median r
        mid_r = rs[len(rs) // 2]
        value = (
            float(np.nanmean(np.where(np.isfinite(min_dists), min_dists < mid_r, np.nan)))
            if np.any(success_mask)
            else float("nan")
        )
        return MetricResult(
            value=value,
            curve=curve,
            method=samples.sampling_backend,
        )


# ---------------------------------------------------------------------------
# M12: Temporal MAE
# ---------------------------------------------------------------------------


@register_metric
class TemporalMAE(Metric):
    """M12: Temporal MAE using median of predictive inter-event-time samples."""

    name = "temporal_mae"
    catalog_id = "M12"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        samples = ctx.samples_predictive
        true_iets = (samples.true_next_times - samples.history_end_times).astype(np.float64)
        pred_iets = (samples.next_times - samples.history_end_times[:, None]).astype(np.float64)
        pred_median_iet = np.median(pred_iets, axis=1)  # (N,)
        pred_t = samples.history_end_times.astype(np.float64) + pred_median_iet

        per_event = _nan_out_failed_contexts(
            np.abs(pred_t - samples.true_next_times.astype(np.float64)),
            samples,
        )
        return MetricResult(
            value=float(np.nanmean(per_event)),
            per_event=per_event,
            method=samples.sampling_backend,
        )


# ---------------------------------------------------------------------------
# M13: Spatial MAE and RMSE
# ---------------------------------------------------------------------------


@register_metric
class SpatialMAE(Metric):
    """M13a: Spatial MAE — mean Euclidean distance from predicted mean to true location."""

    name = "spatial_mae"
    catalog_id = "M13"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        samples = ctx.samples_predictive
        pred_mean = samples.next_locs.astype(np.float64).mean(axis=1)  # (N, 2)
        true_locs = samples.true_next_locs.astype(np.float64)          # (N, 2)
        per_event = _nan_out_failed_contexts(
            np.linalg.norm(pred_mean - true_locs, axis=1),
            samples,
        )
        return MetricResult(
            value=float(np.nanmean(per_event)),
            per_event=per_event,
            method=samples.sampling_backend,
        )


@register_metric
class SpatialRMSE(Metric):
    """M13b: Spatial RMSE — root mean squared Euclidean distance."""

    name = "spatial_rmse"
    catalog_id = "M13"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        samples = ctx.samples_predictive
        pred_mean = samples.next_locs.astype(np.float64).mean(axis=1)
        true_locs = samples.true_next_locs.astype(np.float64)
        sq_err = np.sum((pred_mean - true_locs) ** 2, axis=1)  # (N,)
        per_event = _nan_out_failed_contexts(np.sqrt(sq_err), samples)
        sq_err = _nan_out_failed_contexts(sq_err, samples)
        return MetricResult(
            value=float(np.sqrt(np.nanmean(sq_err))),
            per_event=per_event,
            method=samples.sampling_backend,
        )


# ---------------------------------------------------------------------------
# M14: Joint event distance
# ---------------------------------------------------------------------------


@register_metric
class JointDistance(Metric):
    """M14: Normalized joint spatiotemporal distance.

    JD_i = ||(α(t_i - t̂_i), s_i - ŝ_i)||₂  where α = median(||Δs||) / median(Δt).
    """

    name = "joint_distance"
    catalog_id = "M14"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        samples = ctx.samples_predictive
        # Point estimates
        pred_t = (
            samples.history_end_times.astype(np.float64)
            + np.median(
                samples.next_times.astype(np.float64)
                - samples.history_end_times.astype(np.float64)[:, None],
                axis=1,
            )
        )
        pred_s = samples.next_locs.astype(np.float64).mean(axis=1)  # (N, 2)
        true_t = samples.true_next_times.astype(np.float64)
        true_s = samples.true_next_locs.astype(np.float64)

        # Scale factor α
        dt = np.abs(true_t - pred_t)
        ds = np.linalg.norm(true_s - pred_s, axis=1)
        median_dt = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1.0
        median_ds = float(np.median(ds[ds > 0])) if np.any(ds > 0) else 1.0
        alpha = median_ds / max(median_dt, 1e-8)

        per_event = _nan_out_failed_contexts(np.sqrt((alpha * dt) ** 2 + ds ** 2), samples)
        return MetricResult(
            value=float(np.nanmean(per_event)),
            per_event=per_event,
            method=samples.sampling_backend,
        )
