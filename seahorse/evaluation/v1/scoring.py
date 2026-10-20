"""Pure-numpy scoring primitives for the v1.0 evaluation layer.

Every function here is model-agnostic and unit-tested against closed forms in
``tests/test_eval_v1_scoring.py``. See ``docs/v1.0/evaluation/METRICS_SPEC.md``
for the definitions.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# PIT and uniformity
# ---------------------------------------------------------------------------


def pit_from_compensator(d_lambda: np.ndarray) -> np.ndarray:
    """Time-rescaling PIT ``u = 1 - exp(-dLambda)``.

    For a correctly specified point process, the compensator increments between
    successive events are i.i.d. Exp(1) (time-rescaling theorem), so ``u`` is
    i.i.d. Uniform(0, 1). ``-expm1(-x)`` is used for accuracy at small ``x``.
    """
    d = np.asarray(d_lambda, dtype=np.float64)
    return -np.expm1(-d)


def randomized_rank_pit(
    samples: np.ndarray,
    truth: np.ndarray,
    rng: np.random.Generator,
    *,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Randomized rank PIT ``u = (r + V (e + 1)) / (K + 1)`` per row.

    ``r`` = #samples strictly below the truth, ``e`` = #samples equal to it,
    ``V ~ U(0,1)``. If truth and the K samples are exchangeable (calibrated
    model), ``u`` is *exactly* Uniform(0, 1) for every K, unlike the
    deterministic fraction ``r / K`` whose KS distance has a floor ~1/(K+1).

    ``samples``: (N, K); ``truth``: (N,); ``valid``: optional (N, K) mask, rows
    then use their own effective K.
    """
    s = np.asarray(samples, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)[:, None]
    if valid is None:
        valid = np.isfinite(s)
    valid = np.asarray(valid, dtype=bool)
    k_eff = valid.sum(axis=1)
    below = ((s < y) & valid).sum(axis=1)
    equal = ((s == y) & valid).sum(axis=1)
    v = rng.random(s.shape[0])
    u = (below + v * (equal + 1)) / (k_eff + 1.0)
    u[k_eff == 0] = np.nan
    return u


def deterministic_fraction_pit(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Legacy (paper-era) PIT: fraction of samples strictly below the truth.

    Kept only to reproduce the old numbers; it is *not* uniform under a
    calibrated model for finite K (support {0, 1/K, ..., 1}).
    """
    s = np.asarray(samples, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)[:, None]
    valid = np.isfinite(s)
    return ((s < y) & valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)


def ks_uniform(u: np.ndarray) -> dict:
    """Kolmogorov-Smirnov distance of ``u`` to Uniform(0,1) (+ asymptotic p)."""
    x = np.sort(np.asarray(u, dtype=np.float64)[np.isfinite(u)])
    n = x.shape[0]
    if n == 0:
        return {"ks": float("nan"), "p_value": float("nan"), "n": 0}
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = np.max(i / n - x)
    d_minus = np.max(x - (i - 1) / n)
    d = float(max(d_plus, d_minus))
    try:
        from scipy.stats import kstwo

        p = float(kstwo.sf(d, n))
    except Exception:  # pragma: no cover
        p = float("nan")
    return {"ks": d, "p_value": p, "n": int(n)}


def pit_histogram(u: np.ndarray, bins: int = 20) -> dict:
    x = np.asarray(u, dtype=np.float64)
    x = x[np.isfinite(x)]
    counts, edges = np.histogram(x, bins=bins, range=(0.0, 1.0))
    return {"counts": counts.tolist(), "edges": edges.tolist(), "n": int(x.shape[0])}


def pit_tail_fractions(u: np.ndarray, eps: float = 1.0 / 33.0) -> dict:
    """Fraction of PIT values in the extreme tails (``u < eps`` / ``u > 1-eps``).

    Under calibration each is ``eps``. The default ``eps = 1/33`` matches the
    resolution of the legacy K=32 PIT, so legacy "truth beyond all samples"
    fractions are directly comparable.
    """
    x = np.asarray(u, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"below": float("nan"), "above": float("nan"), "eps": eps}
    return {
        "below": float((x < eps).mean()),
        "above": float((x > 1.0 - eps).mean()),
        "eps": float(eps),
    }


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------


def _seg_int_sq(f0: np.ndarray, f1: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Exact integral of g^2 over a segment where g is linear from f0 to f1."""
    return width * (f0 * f0 + f0 * f1 + f1 * f1) / 3.0


def crps_from_cdf_grid(
    cdf: np.ndarray,
    tau_grid: np.ndarray,
    y: np.ndarray,
    c: float,
) -> np.ndarray:
    """Threshold-weighted CRPS on ``[0, c]`` from CDF values on a grid.

    ``CRPS_c(F, y) = int_0^c (F(z) - 1{z >= y})^2 dz``

    (the CRPS with weight ``w(z) = 1{z <= c}``, a proper scoring rule; equal to
    the ordinary CRPS of the censored variables ``min(X, c)`` vs ``min(y, c)``).
    ``F`` is interpolated linearly between grid points, which must start at 0
    and cover ``c``. The integral of the piecewise-quadratic integrand is exact.

    ``cdf``: (N, G); ``tau_grid``: (G,) increasing with ``tau_grid[0] == 0``;
    ``y``: (N,) observed waiting times.
    """
    F = np.asarray(cdf, dtype=np.float64)
    tg = np.asarray(tau_grid, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if tg[0] != 0.0:
        raise ValueError("tau_grid must start at 0")
    if tg[-1] < c:
        raise ValueError(f"tau_grid must cover c={c}; max={tg[-1]}")
    # Restrict to [0, c] by inserting c as a node.
    j_c = int(np.searchsorted(tg, c, side="left"))
    if tg[j_c] != c:
        w = (c - tg[j_c - 1]) / (tg[j_c] - tg[j_c - 1])
        f_c = F[:, j_c - 1] + w * (F[:, j_c] - F[:, j_c - 1])
        tg_c = np.concatenate([tg[:j_c], [c]])
        F_c = np.concatenate([F[:, :j_c], f_c[:, None]], axis=1)
    else:
        tg_c = tg[: j_c + 1]
        F_c = F[:, : j_c + 1]
    n = F_c.shape[0]
    out = np.empty(n, dtype=np.float64)
    widths = np.diff(tg_c)
    for i in range(n):
        f = F_c[i]
        yi = min(max(yy[i], 0.0), c)
        k = int(np.searchsorted(tg_c, yi, side="right")) - 1  # segment containing y
        k = min(max(k, 0), len(widths) - 1) if len(widths) else 0
        # Full segments left of y: integrand F^2
        left = _seg_int_sq(f[:k], f[1 : k + 1], widths[:k]).sum() if k > 0 else 0.0
        # Full segments right of y: integrand (1-F)^2
        g = 1.0 - f
        right = (
            _seg_int_sq(g[k + 1 : -1], g[k + 2 :], widths[k + 1 :]).sum()
            if k + 1 < len(widths)
            else 0.0
        )
        # Split segment k at y
        a, b = tg_c[k], tg_c[k + 1]
        fa, fb = f[k], f[k + 1]
        fy = fa + (fb - fa) * ((yi - a) / (b - a) if b > a else 0.0)
        mid = _seg_int_sq(fa, fy, yi - a) + _seg_int_sq(1.0 - fy, 1.0 - fb, b - yi)
        out[i] = left + mid + right
    return out


def crps_fair_censored(
    samples: np.ndarray,
    y: np.ndarray,
    c: float,
    *,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fair (unbiased) sample CRPS of the censored variables min(X, c), min(y, c).

    ``CRPS_fair = mean_k |X_k - y| - 1/(2 K (K-1)) sum_{i != j} |X_i - X_j|``

    which is an unbiased estimator of the CRPS of the sampling distribution
    (Ferro 2014). Equals the threshold-weighted CRPS on ``[0, c]`` of
    :func:`crps_from_cdf_grid` in expectation.
    """
    s = np.minimum(np.asarray(samples, dtype=np.float64), c)
    yy = np.minimum(np.asarray(y, dtype=np.float64), c)
    if valid is None:
        valid = np.isfinite(s)
    out = np.full(s.shape[0], np.nan)
    for i in range(s.shape[0]):
        x = np.sort(s[i][valid[i]])
        k = x.shape[0]
        if k < 2:
            continue
        t1 = np.abs(x - yy[i]).mean()
        idx = np.arange(1, k + 1, dtype=np.float64)
        pair_sum = 2.0 * np.sum(x * (2.0 * idx - k - 1.0))  # sum_{i != j} |x_i - x_j|
        out[i] = t1 - pair_sum / (2.0 * k * (k - 1.0))
    return out


def crps_exponential(rate: float, y: float) -> float:
    """Closed-form CRPS of Exp(rate) at y >= 0 (untruncated)."""
    return y + (2.0 * math.exp(-rate * y) - 1.5) / rate


def crps_exponential_truncated(rate: float, y: float, c: float) -> float:
    """Closed-form threshold-weighted CRPS on [0, c] of Exp(rate) at y."""
    y = min(max(y, 0.0), c)
    lam = rate
    # int_0^y (1-e^{-lam z})^2 dz
    left = y - 2.0 * (1.0 - math.exp(-lam * y)) / lam + (1.0 - math.exp(-2.0 * lam * y)) / (2.0 * lam)
    # int_y^c e^{-2 lam z} dz
    right = (math.exp(-2.0 * lam * y) - math.exp(-2.0 * lam * c)) / (2.0 * lam)
    return left + right


# ---------------------------------------------------------------------------
# Multivariate scores
# ---------------------------------------------------------------------------


def energy_score_fair(
    samples: np.ndarray,
    truth: np.ndarray,
    *,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fair energy score per row.

    ``ES = mean_k ||X_k - y|| - 1/(2 K (K-1)) sum_{i != j} ||X_i - X_j||``

    ``samples``: (N, K, D); ``truth``: (N, D). Rows with < 2 valid samples -> nan.
    """
    s = np.asarray(samples, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    n, _, _ = s.shape
    if valid is None:
        valid = np.isfinite(s).all(axis=-1)
    out = np.full(n, np.nan)
    for i in range(n):
        x = s[i][valid[i]]
        k = x.shape[0]
        if k < 2:
            continue
        t1 = np.linalg.norm(x - y[i], axis=-1).mean()
        d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        out[i] = t1 - d.sum() / (2.0 * k * (k - 1.0))
    return out


def hpd_rank_pit(
    logp_samples: np.ndarray,
    logp_truth: np.ndarray,
    rng: np.random.Generator,
    *,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Monte Carlo HPD-PIT ``u = P(p(S) >= p(s_obs))`` via a randomized rank.

    With ``g = -log p`` and samples ``S_k ~ p``, under calibration ``g(s_obs)``
    and ``g(S_k)`` are exchangeable, so the randomized rank of ``g(s_obs)``
    among the samples is exactly Uniform(0,1). ``u -> 1``: the observation fell
    in a low-density region (model too concentrated / misplaced mass);
    ``u -> 0``: observation in the model's mode more often than expected
    (model too dispersed).
    """
    g_s = -np.asarray(logp_samples, dtype=np.float64)
    g_y = -np.asarray(logp_truth, dtype=np.float64)
    return randomized_rank_pit(g_s, g_y, rng, valid=valid)


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return pearson(rankdata(a[m]), rankdata(b[m]))


def tau_grid_quadratic(tau_max: float, n: int) -> np.ndarray:
    """Waiting-time grid ``tau_j = tau_max (j / (n-1))^2`` (dense near 0)."""
    j = np.arange(n, dtype=np.float64) / float(n - 1)
    return tau_max * j * j


def sample_from_cumhazard_grid(
    cum_hazard: np.ndarray,
    tau_grid: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-transform sampling of waiting times from Lambda(tau) on a grid.

    Draws ``E ~ Exp(1)`` and returns ``tau = Lambda^{-1}(E)`` by linear
    interpolation of the (non-decreasing) cumulative hazard. Draws with
    ``E > Lambda(tau_max)`` are returned as ``+inf`` and flagged ``censored``
    (the event falls beyond the grid; for defective distributions this
    includes 'no further event').

    ``cum_hazard``: (N, G). Returns ``(tau (N, K), censored (N, K))``.
    """
    lam = np.maximum.accumulate(np.asarray(cum_hazard, dtype=np.float64), axis=1)
    n = lam.shape[0]
    e = rng.exponential(size=(n, n_samples))
    out = np.empty((n, n_samples), dtype=np.float64)
    cens = e > lam[:, -1:]
    for i in range(n):
        out[i] = np.interp(e[i], lam[i], tau_grid)
    out[cens] = np.inf
    return out, cens
