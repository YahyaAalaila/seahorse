"""Generative-quality metrics (M15–M20, M27).

Registers:
  wasserstein           — Wasserstein W₁ distance (catalog M15)
  mmd                   — Maximum Mean Discrepancy (catalog M16)
  temporal_count_chi2   — Temporal event count chi-squared (catalog M17)
  spatial_count_chi2    — Spatial event count chi-squared (catalog M18)
  spatial_ripley_k      — Ripley's K spatial (catalog M19)
  temporal_ripley_k     — Temporal pair correlation (catalog M20)
  rollout_coherence     — Sequential coherence over rollout horizons 1/5/10 (catalog M27)

All metrics share ctx.samples_generative (K=20 fixed-prefix rollouts per test sequence).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from seahorse.evaluation.profiles import GENERATIVE_ROLLOUTS
from seahorse.evaluation.registry import register_metric
from seahorse.evaluation.result import Metric, MetricResult

if TYPE_CHECKING:
    from seahorse.evaluation.context import EvalContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_rollouts(rollout_times, rollout_locs, true_times, true_locs, max_n=500):
    """Return (real_pts, gen_pts) as (n, 3) arrays of (t, x, y), subsampled."""
    real_pts_list: list[np.ndarray] = []
    gen_pts_list: list[np.ndarray] = []

    for seq_i, (rt_seq, rs_seq, tt, tl) in enumerate(
        zip(rollout_times, rollout_locs, true_times, true_locs)
    ):
        if tt.shape[0] > 0 and tl.shape[0] > 0:
            pts = np.column_stack([tt, tl[:, 0], tl[:, 1]])
            real_pts_list.append(pts)

        for r_t, r_s in zip(rt_seq, rs_seq):
            if r_t.shape[0] > 0 and r_s.shape[0] > 0:
                pts = np.column_stack([r_t, r_s[:, 0], r_s[:, 1]])
                gen_pts_list.append(pts)

    real_pts = np.concatenate(real_pts_list, axis=0) if real_pts_list else np.zeros((0, 3))
    gen_pts = np.concatenate(gen_pts_list, axis=0) if gen_pts_list else np.zeros((0, 3))

    # Subsample to max_n
    if real_pts.shape[0] > max_n:
        idx = np.random.choice(real_pts.shape[0], max_n, replace=False)
        real_pts = real_pts[idx]
    if gen_pts.shape[0] > max_n:
        idx = np.random.choice(gen_pts.shape[0], max_n, replace=False)
        gen_pts = gen_pts[idx]

    return real_pts, gen_pts


def _normalize_pts(real: np.ndarray, gen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize both point sets to unit scale using real set stats."""
    if real.shape[0] == 0:
        return real, gen
    std = real.std(axis=0)
    std = np.maximum(std, 1e-8)
    return real / std, gen / std


def _point_cloud(times: np.ndarray, locs: np.ndarray) -> np.ndarray:
    return np.column_stack([times, locs[:, 0], locs[:, 1]]).astype(np.float64, copy=False)


def _context_length(rollouts, seq_i: int, n_events: int) -> int:
    lengths = getattr(rollouts, "context_lengths", None)
    if lengths is not None:
        return int(lengths[seq_i])
    return max(1, n_events // 2)


def _sequence_scale(
    true_times: np.ndarray,
    true_locs: np.ndarray,
    context_len: int,
) -> np.ndarray:
    """Stable scale for horizon point clouds, including the H=1 case."""
    continuation_t = true_times[context_len:]
    continuation_s = true_locs[context_len:]
    if continuation_t.shape[0] >= 2:
        pts = _point_cloud(continuation_t, continuation_s)
    elif true_times.shape[0] >= 2:
        pts = _point_cloud(true_times, true_locs)
    else:
        return np.ones(3, dtype=np.float64)
    scale = pts.std(axis=0)
    return np.maximum(scale, 1e-6)


def _energy_distance(real_pts: np.ndarray, gen_pts: np.ndarray) -> float:
    if real_pts.shape[0] == 0 or gen_pts.shape[0] == 0:
        return float("nan")
    cross = np.linalg.norm(real_pts[:, None, :] - gen_pts[None, :, :], axis=-1).mean()
    rr = np.linalg.norm(real_pts[:, None, :] - real_pts[None, :, :], axis=-1).mean()
    gg = np.linalg.norm(gen_pts[:, None, :] - gen_pts[None, :, :], axis=-1).mean()
    return float(max(2.0 * cross - rr - gg, 0.0))


def _point_cloud_distance(
    real_pts: np.ndarray,
    gen_pts: np.ndarray,
    ot_module,
) -> tuple[float, str]:
    if real_pts.shape[0] == 0 or gen_pts.shape[0] == 0:
        return float("nan"), "energy_fallback"
    if ot_module is not None:
        n_r, n_g = real_pts.shape[0], gen_pts.shape[0]
        a = np.ones(n_r, dtype=np.float64) / n_r
        b = np.ones(n_g, dtype=np.float64) / n_g
        cost = np.linalg.norm(real_pts[:, None, :] - gen_pts[None, :, :], axis=-1)
        try:
            return float(ot_module.emd2(a, b, cost)), "w1_pot"
        except Exception:
            pass
    return _energy_distance(real_pts, gen_pts), "energy_fallback"


def _crps_1d(samples: np.ndarray, target: float) -> float:
    if samples.size == 0:
        return float("nan")
    term1 = float(np.mean(np.abs(samples - target)))
    pairwise = np.abs(samples[:, None] - samples[None, :])
    term2 = float(np.mean(pairwise))
    return term1 - 0.5 * term2


def _spatial_energy_score(samples: np.ndarray, target: np.ndarray) -> float:
    if samples.shape[0] == 0:
        return float("nan")
    term1 = 2.0 * float(np.linalg.norm(samples - target[None, :], axis=-1).mean())
    pairwise = np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1)
    term2 = float(pairwise.mean())
    return term1 - term2


# ---------------------------------------------------------------------------
# M15: Wasserstein W₁
# ---------------------------------------------------------------------------


@register_metric
class Wasserstein(Metric):
    """M15: Wasserstein W₁ between real and generated event point clouds.

    Uses POT `ot.emd2` with Euclidean ground metric on normalized (t, x, y).
    Falls back to Sinkhorn (ε=0.05) on OT solver failure.
    Computed on a random 30% subsample of test sequences if N_seqs > 20.
    """

    name = "wasserstein"
    catalog_id = "M15"
    requires = frozenset({"samples_generative"})
    artifact_families = frozenset({GENERATIVE_ROLLOUTS})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        try:
            import ot  # type: ignore[import]
        except ImportError:
            return MetricResult(
                value=None,
                available=False,
                reason="POT (Python Optimal Transport) not installed",
            )

        rollouts = ctx.samples_generative
        n_seqs = len(rollouts.rollout_times)

        # Subsample sequences if large
        seq_ids = list(range(n_seqs))
        if n_seqs > 20:
            rng = np.random.default_rng(42)
            k = max(1, int(0.3 * n_seqs))
            seq_ids = rng.choice(n_seqs, k, replace=False).tolist()

        w1_vals: list[float] = []

        for i in seq_ids:
            rt = rollouts.rollout_times[i]
            rs = rollouts.rollout_locs[i]
            tt = rollouts.true_times[i]
            tl = rollouts.true_locs[i]

            if tt.shape[0] == 0:
                continue

            # Pool all K rollouts for this sequence
            gen_ts: list[np.ndarray] = []
            gen_ss: list[np.ndarray] = []
            for r_t, r_s in zip(rt, rs):
                if r_t.shape[0] > 0:
                    gen_ts.append(r_t)
                    gen_ss.append(r_s)
            if not gen_ts:
                continue

            gen_t_cat = np.concatenate(gen_ts)
            gen_s_cat = np.concatenate(gen_ss)
            real_pts = np.column_stack([tt, tl[:, 0], tl[:, 1]]).astype(np.float64)
            gen_pts = np.column_stack([gen_t_cat, gen_s_cat[:, 0], gen_s_cat[:, 1]]).astype(np.float64)

            # Subsample
            max_n = 300
            if real_pts.shape[0] > max_n:
                real_pts = real_pts[np.random.choice(real_pts.shape[0], max_n, replace=False)]
            if gen_pts.shape[0] > max_n:
                gen_pts = gen_pts[np.random.choice(gen_pts.shape[0], max_n, replace=False)]

            # Normalize
            real_pts, gen_pts = _normalize_pts(real_pts, gen_pts)

            n, m = real_pts.shape[0], gen_pts.shape[0]
            a = np.ones(n, dtype=np.float64) / n
            b = np.ones(m, dtype=np.float64) / m
            M = np.linalg.norm(real_pts[:, None, :] - gen_pts[None, :, :], axis=-1)

            try:
                w1 = float(ot.emd2(a, b, M))
            except Exception:
                try:
                    w1 = float(ot.sinkhorn2(a, b, M, reg=0.05)[0])
                except Exception:
                    continue
            w1_vals.append(w1)

        if not w1_vals:
            return MetricResult(value=None, available=False, reason="no valid sequences")

        return MetricResult(
            value=float(np.median(w1_vals)),
            method=rollouts.method,
        )


# ---------------------------------------------------------------------------
# M16: Maximum Mean Discrepancy
# ---------------------------------------------------------------------------


@register_metric
class MMD(Metric):
    """M16: Unbiased MMD² with Gaussian kernel (median bandwidth heuristic)."""

    name = "mmd"
    catalog_id = "M16"
    requires = frozenset({"samples_generative"})
    artifact_families = frozenset({GENERATIVE_ROLLOUTS})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        rollouts = ctx.samples_generative
        real_pts, gen_pts = _flatten_rollouts(
            rollouts.rollout_times,
            rollouts.rollout_locs,
            rollouts.true_times,
            rollouts.true_locs,
            max_n=500,
        )
        real_pts, gen_pts = _normalize_pts(real_pts, gen_pts)

        if real_pts.shape[0] < 2 or gen_pts.shape[0] < 2:
            return MetricResult(value=None, available=False, reason="too few points")

        mmd2 = _mmd2_gaussian(real_pts, gen_pts)
        return MetricResult(value=float(mmd2), method=rollouts.method)


def _mmd2_gaussian(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased MMD² estimator with Gaussian kernel, median bandwidth."""
    n, m = X.shape[0], Y.shape[0]

    # Median heuristic bandwidth
    XY = np.concatenate([X, Y], axis=0)
    pairwise = np.linalg.norm(XY[:, None] - XY[None, :], axis=-1)
    median_dist = np.median(pairwise[pairwise > 0])
    sigma2 = max(median_dist ** 2, 1e-8)

    def rbf(A, B):
        dist2 = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-dist2 / (2.0 * sigma2))

    Kxx = rbf(X, X)
    Kyy = rbf(Y, Y)
    Kxy = rbf(X, Y)

    # Unbiased estimators (zero diagonal)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
            - 2.0 * Kxy.mean())
    return float(mmd2)


# ---------------------------------------------------------------------------
# M17: Temporal event count chi-squared
# ---------------------------------------------------------------------------


@register_metric
class TemporalCountChi2(Metric):
    """M17: Chi-squared and KL between temporal event count distributions.

    Partitions [t_min, t_max] into 10 equal bins (merging sparse bins).
    """

    name = "temporal_count_chi2"
    catalog_id = "M17"
    requires = frozenset({"samples_generative"})
    artifact_families = frozenset({GENERATIVE_ROLLOUTS})
    cost_class = "sampling_heavy"

    _N_BINS = 10

    def compute(self, ctx: "EvalContext") -> MetricResult:
        rollouts = ctx.samples_generative

        real_times = np.concatenate(rollouts.true_times) if rollouts.true_times else np.zeros(0)
        gen_times_list: list[np.ndarray] = []
        for seq_rt in rollouts.rollout_times:
            for r_t in seq_rt:
                gen_times_list.append(r_t)
        gen_times = np.concatenate(gen_times_list) if gen_times_list else np.zeros(0)

        if real_times.shape[0] == 0 or gen_times.shape[0] == 0:
            return MetricResult(value=None, available=False, reason="no events")

        t_lo = min(real_times.min(), gen_times.min())
        t_hi = max(real_times.max(), gen_times.max())
        bins = np.linspace(t_lo, t_hi, self._N_BINS + 1)

        real_counts, _ = np.histogram(real_times, bins=bins)
        gen_counts_raw, _ = np.histogram(gen_times, bins=bins)

        # Scale generated counts to same total as real
        scale = real_counts.sum() / max(gen_counts_raw.sum(), 1)
        gen_counts = gen_counts_raw * scale

        # Merge bins with expected count < 5
        real_counts, gen_counts = _merge_sparse_bins(real_counts.astype(float), gen_counts)

        if gen_counts.sum() == 0:
            return MetricResult(value=None, available=False, reason="all bins empty after merging")

        chi2 = float(np.sum((real_counts - gen_counts) ** 2 / np.maximum(gen_counts, 1e-8)))

        # KL: p=real/sum, q=gen/sum
        p = real_counts / max(real_counts.sum(), 1e-8)
        q = gen_counts / max(gen_counts.sum(), 1e-8)
        kl = float(np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(q, 1e-12))))

        return MetricResult(
            value=chi2,
            curve={"chi2": chi2, "kl": kl},
            method=rollouts.method,
        )


def _merge_sparse_bins(obs: np.ndarray, exp: np.ndarray, min_exp: float = 5.0):
    """Merge adjacent bins until all expected counts ≥ min_exp."""
    obs = list(obs)
    exp = list(exp)
    i = 0
    while i < len(exp):
        if exp[i] < min_exp and len(exp) > 1:
            if i == len(exp) - 1:
                i -= 1
            obs[i] += obs[i + 1]
            exp[i] += exp[i + 1]
            del obs[i + 1]
            del exp[i + 1]
        else:
            i += 1
    return np.asarray(obs), np.asarray(exp)


# ---------------------------------------------------------------------------
# M18: Spatial event count chi-squared
# ---------------------------------------------------------------------------


@register_metric
class SpatialCountChi2(Metric):
    """M18: Chi-squared and KL between spatial event count distributions (10×10 bins)."""

    name = "spatial_count_chi2"
    catalog_id = "M18"
    requires = frozenset({"samples_generative"})
    artifact_families = frozenset({GENERATIVE_ROLLOUTS})
    cost_class = "sampling_heavy"

    _N_BINS = 10

    def compute(self, ctx: "EvalContext") -> MetricResult:
        rollouts = ctx.samples_generative

        real_locs = np.concatenate(rollouts.true_locs) if rollouts.true_locs else np.zeros((0, 2))
        gen_locs_list: list[np.ndarray] = []
        for seq_rs in rollouts.rollout_locs:
            for r_s in seq_rs:
                gen_locs_list.append(r_s)
        gen_locs = np.concatenate(gen_locs_list) if gen_locs_list else np.zeros((0, 2))

        if real_locs.shape[0] == 0 or gen_locs.shape[0] == 0:
            return MetricResult(value=None, available=False, reason="no events")

        lo = real_locs.min(axis=0)
        hi = real_locs.max(axis=0)
        span = np.maximum(hi - lo, 1e-4)
        lo -= 0.02 * span
        hi += 0.02 * span
        bins_x = np.linspace(lo[0], hi[0], self._N_BINS + 1)
        bins_y = np.linspace(lo[1], hi[1], self._N_BINS + 1)

        real_counts, _, _ = np.histogram2d(real_locs[:, 0], real_locs[:, 1], bins=[bins_x, bins_y])
        gen_counts_raw, _, _ = np.histogram2d(gen_locs[:, 0], gen_locs[:, 1], bins=[bins_x, bins_y])

        scale = real_counts.sum() / max(gen_counts_raw.sum(), 1)
        gen_counts = gen_counts_raw.ravel() * scale
        real_counts = real_counts.ravel()

        # Flatten and merge sparse bins
        real_counts, gen_counts = _merge_sparse_bins(real_counts.astype(float), gen_counts)

        chi2 = float(np.sum((real_counts - gen_counts) ** 2 / np.maximum(gen_counts, 1e-8)))
        p = real_counts / max(real_counts.sum(), 1e-8)
        q = gen_counts / max(gen_counts.sum(), 1e-8)
        kl = float(np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(q, 1e-12))))

        return MetricResult(
            value=chi2,
            curve={"chi2": chi2, "kl": kl},
            method=rollouts.method,
        )


# ---------------------------------------------------------------------------
# M19: Ripley's K-function (spatial)
# ---------------------------------------------------------------------------


@register_metric
class SpatialRipleyK(Metric):
    """M19: L² distance between real and generated spatial Ripley's K-function curves."""

    name = "spatial_ripley_k"
    catalog_id = "M19"
    requires = frozenset({"samples_generative"})
    artifact_families = frozenset({GENERATIVE_ROLLOUTS})
    cost_class = "sampling_heavy"

    _N_R = 20
    _N_MAX = 500

    def compute(self, ctx: "EvalContext") -> MetricResult:
        rollouts = ctx.samples_generative

        real_locs = np.concatenate(rollouts.true_locs) if rollouts.true_locs else np.zeros((0, 2))
        gen_locs_list: list[np.ndarray] = []
        for seq_rs in rollouts.rollout_locs:
            for r_s in seq_rs:
                gen_locs_list.append(r_s)
        gen_locs = np.concatenate(gen_locs_list) if gen_locs_list else np.zeros((0, 2))

        # Subsample
        if real_locs.shape[0] > self._N_MAX:
            real_locs = real_locs[np.random.choice(real_locs.shape[0], self._N_MAX, replace=False)]
        if gen_locs.shape[0] > self._N_MAX:
            gen_locs = gen_locs[np.random.choice(gen_locs.shape[0], self._N_MAX, replace=False)]

        if real_locs.shape[0] < 5 or gen_locs.shape[0] < 5:
            return MetricResult(value=None, available=False, reason="too few points for Ripley's K")

        # Determine r range from real data
        max_r = float(
            np.percentile(
                np.linalg.norm(real_locs - real_locs.mean(axis=0), axis=1), 80
            )
        )
        rs = np.linspace(max_r * 0.05, max_r, self._N_R)

        k_real = _ripley_k(real_locs, rs)
        k_gen = _ripley_k(gen_locs, rs)

        l2_dist = float(np.sqrt(np.mean((k_real - k_gen) ** 2)))
        return MetricResult(value=l2_dist, method=rollouts.method)


def _ripley_k(pts: np.ndarray, rs: np.ndarray) -> np.ndarray:
    """Estimate Ripley's K(r) for 2D point set via naive O(n²) estimator."""
    n = pts.shape[0]
    area = 1.0  # operate in normalised coordinates
    lam = n / area
    dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)  # (n, n)
    np.fill_diagonal(dists, np.inf)
    k = np.asarray(
        [np.sum(dists < r) / (lam * n) for r in rs], dtype=np.float64
    )
    return k


# ---------------------------------------------------------------------------
# M20: Temporal pair correlation (temporal Ripley's K)
# ---------------------------------------------------------------------------


@register_metric
class TemporalRipleyK(Metric):
    """M20: L² distance between real and generated temporal Ripley's K curves."""

    name = "temporal_ripley_k"
    catalog_id = "M20"
    requires = frozenset({"samples_generative"})
    artifact_families = frozenset({GENERATIVE_ROLLOUTS})
    cost_class = "sampling_heavy"

    _N_R = 20
    _N_MAX = 1000

    def compute(self, ctx: "EvalContext") -> MetricResult:
        rollouts = ctx.samples_generative

        real_times = np.concatenate(rollouts.true_times) if rollouts.true_times else np.zeros(0)
        gen_times_list: list[np.ndarray] = []
        for seq_rt in rollouts.rollout_times:
            for r_t in seq_rt:
                gen_times_list.append(r_t)
        gen_times = np.concatenate(gen_times_list) if gen_times_list else np.zeros(0)

        if real_times.shape[0] > self._N_MAX:
            real_times = real_times[np.random.choice(real_times.shape[0], self._N_MAX, replace=False)]
        if gen_times.shape[0] > self._N_MAX:
            gen_times = gen_times[np.random.choice(gen_times.shape[0], self._N_MAX, replace=False)]

        if real_times.shape[0] < 5 or gen_times.shape[0] < 5:
            return MetricResult(value=None, available=False, reason="too few events for temporal K")

        iet_real = np.abs(real_times[:, None] - real_times[None, :])
        max_r = float(np.percentile(iet_real[iet_real > 0], 80)) if (iet_real > 0).any() else 1.0
        rs = np.linspace(max_r * 0.05, max_r, self._N_R)

        k_real = _ripley_k_1d(real_times, rs)
        k_gen = _ripley_k_1d(gen_times, rs)

        l2_dist = float(np.sqrt(np.mean((k_real - k_gen) ** 2)))
        return MetricResult(value=l2_dist, method=rollouts.method)


def _ripley_k_1d(times: np.ndarray, rs: np.ndarray) -> np.ndarray:
    n = times.shape[0]
    lam = n / max(float(times.max() - times.min()), 1e-8) if n > 1 else 1.0
    dists = np.abs(times[:, None] - times[None, :])
    np.fill_diagonal(dists, np.inf)
    return np.asarray(
        [np.sum(dists < r) / (lam * n) for r in rs], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# M27: Rollout coherence
# ---------------------------------------------------------------------------


@register_metric
class RolloutCoherence(Metric):
    """M27: Sequential coherence — W₁ between generated and real continuations.

    Evaluates at fixed horizon H ∈ {1, 5, 10} events and returns as a curve.
    The scalar value is the H=10 result.
    """

    name = "rollout_coherence"
    catalog_id = "M27"
    requires = frozenset({"samples_generative"})
    artifact_families = frozenset({GENERATIVE_ROLLOUTS})
    cost_class = "sampling_heavy"

    _H_VALUES = (1, 5, 10)

    def compute(self, ctx: "EvalContext") -> MetricResult:
        try:
            import ot  # type: ignore[import]
        except Exception:
            ot = None

        rollouts = ctx.samples_generative
        curve: dict[str, float] = {}
        used_fallback = ot is None

        for H in self._H_VALUES:
            distance_vals: list[float] = []
            n_h = 0
            for seq_i, (rt_seq, rs_seq, tt, tl) in enumerate(
                zip(
                    rollouts.rollout_times,
                    rollouts.rollout_locs,
                    rollouts.true_times,
                    rollouts.true_locs,
                )
            ):
                n = tt.shape[0]
                cond_len = _context_length(rollouts, seq_i, n)
                real_continuation_t = tt[cond_len:cond_len + H]
                real_continuation_s = tl[cond_len:cond_len + H]

                if real_continuation_t.shape[0] < H:
                    continue

                scale = _sequence_scale(tt, tl, cond_len)
                real_pts = _point_cloud(real_continuation_t, real_continuation_s) / scale

                # Pool generated continuations
                gen_pts_list: list[np.ndarray] = []
                for r_t, r_s in zip(rt_seq, rs_seq):
                    if r_t.shape[0] >= H and r_s.shape[0] >= H:
                        gen_pts_list.append(_point_cloud(r_t[:H], r_s[:H]) / scale)

                if not gen_pts_list:
                    continue

                gen_pts = np.concatenate(gen_pts_list, axis=0).astype(np.float64, copy=False)
                distance, method = _point_cloud_distance(real_pts, gen_pts, ot)
                if method != "w1_pot":
                    used_fallback = True
                if not math.isfinite(distance):
                    continue
                distance_vals.append(distance)
                n_h += 1

            if distance_vals:
                curve[str(H)] = float(np.median(distance_vals))
                curve[f"n_h_{H}"] = float(n_h)

        value = curve.get("10")
        if value is None:
            return MetricResult(
                value=None,
                curve=curve or None,
                method="energy_fallback" if used_fallback else "w1_pot",
                available=False,
                reason="no valid h=10 sequences",
            )

        return MetricResult(
            value=float(value),
            curve=curve,
            method="energy_fallback" if used_fallback else "w1_pot",
        )


@register_metric
class ARTemporalCRPSH1(Metric):
    """Single-step temporal CRPS from fixed-prefix autoregressive rollouts."""

    name = "ar_temporal_crps_h1"
    catalog_id = None
    requires = frozenset({"samples_generative"})
    artifact_families = frozenset({GENERATIVE_ROLLOUTS})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        rollouts = ctx.samples_generative
        values: list[float] = []
        for seq_i, (rt_seq, tt) in enumerate(zip(rollouts.rollout_times, rollouts.true_times)):
            n = tt.shape[0]
            cond_len = _context_length(rollouts, seq_i, n)
            if cond_len < 1 or cond_len >= n:
                continue
            hist_end = float(tt[cond_len - 1])
            target = float(tt[cond_len] - hist_end)
            pred = [
                max(float(r_t[0]) - hist_end, 0.0)
                for r_t in rt_seq
                if r_t.shape[0] >= 1 and math.isfinite(float(r_t[0]))
            ]
            if not pred:
                continue
            score = _crps_1d(np.asarray(pred, dtype=np.float64), target)
            if math.isfinite(score):
                values.append(score)
        if not values:
            return MetricResult(value=None, available=False, reason="no valid h=1 rollouts")
        return MetricResult(value=float(np.mean(values)), method=rollouts.method)


@register_metric
class ARSpatialEnergyScoreH1(Metric):
    """Single-step spatial energy score from fixed-prefix autoregressive rollouts."""

    name = "ar_spatial_energy_score_h1"
    catalog_id = None
    requires = frozenset({"samples_generative"})
    artifact_families = frozenset({GENERATIVE_ROLLOUTS})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        rollouts = ctx.samples_generative
        values: list[float] = []
        for seq_i, (rs_seq, tt, tl) in enumerate(
            zip(rollouts.rollout_locs, rollouts.true_times, rollouts.true_locs)
        ):
            n = tt.shape[0]
            cond_len = _context_length(rollouts, seq_i, n)
            if cond_len < 1 or cond_len >= n:
                continue
            target = np.asarray(tl[cond_len], dtype=np.float64)
            pred = [
                np.asarray(r_s[0], dtype=np.float64)
                for r_s in rs_seq
                if r_s.shape[0] >= 1 and np.isfinite(r_s[0]).all()
            ]
            if not pred:
                continue
            score = _spatial_energy_score(np.vstack(pred), target)
            if math.isfinite(score):
                values.append(score)
        if not values:
            return MetricResult(value=None, available=False, reason="no valid h=1 rollouts")
        return MetricResult(value=float(np.mean(values)), method=rollouts.method)
