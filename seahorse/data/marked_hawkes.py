"""Exact simulator for the marked multivariate space-time Hawkes process.

Draws from *precisely* the law implemented by
:class:`seahorse.models.event_models.marked_hawkes.MarkedHawkesEventModel`, so
that fitting simulated data is a well-posed parameter-recovery problem::

    lambda_k(t, y | H) = mu_k * rho_k(y)
                         + sum_{i: t_i < t} alpha[k_i, k] * q e^{-q (t - t_i)}
                                            * f_{k_i, k}(y - x_i)

Sampling uses Ogata thinning on the ground intensity
``Lambda*(t) = sum_k mu_k + sum_j A_{k_j} q e^{-q (t - t_j)}`` (with row sums
``A_j = sum_k alpha[j, k]``), which is piecewise decreasing between events, so the
value at the current time is a valid dominating rate.  Given an accepted time the
mark is drawn from ``lambda_k(t) / sum_k lambda_k(t)`` and the location from the
corresponding background/offspring mixture — this factorization is exact, not an
approximation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

__all__ = ["simulate_marked_hawkes", "simulate_marked_hawkes_sequences"]


def simulate_marked_hawkes(
    *,
    mu: np.ndarray,
    alpha: np.ndarray,
    decay: float,
    T: float,
    rng: np.random.Generator,
    disp_mean: Optional[np.ndarray] = None,
    disp_scale: Optional[np.ndarray] = None,
    bg_mean: Optional[np.ndarray] = None,
    bg_scale: Optional[np.ndarray] = None,
    spatial_dim: int = 2,
    max_events: int = 100_000,
) -> Dict[str, np.ndarray]:
    """Simulate one marked space-time Hawkes sequence on ``[0, T]``.

    Args:
        mu:          ``(M,)`` background rates.
        alpha:       ``(M, M)`` branching matrix; ``alpha[j, k]`` = expected number
                     of direct mark-``k`` offspring of a mark-``j`` event.
        decay:       shared exponential rate ``q`` of the normalized kernel.
        T:           observation window end (window is ``[0, T]``).
        rng:         numpy Generator.
        disp_mean:   ``(M, M, d)`` offspring displacement means (default 0).
        disp_scale:  ``(M, M, d)`` offspring displacement std devs (default 0.5).
        bg_mean:     ``(M, d)`` background location means (default 0).
        bg_scale:    ``(M, d)`` background location std devs (default 1).
        max_events:  hard cap; exceeding it raises (indicates a supercritical
                     ``alpha``, i.e. spectral radius >= 1).

    Returns:
        ``{"times": (N,) float64, "locations": (N, d) float32, "marks": (N,) int64}``
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    alpha = np.asarray(alpha, dtype=np.float64)
    m = mu.shape[0]
    d = int(spatial_dim)
    if alpha.shape != (m, m):
        raise ValueError(f"alpha must be ({m}, {m}), got {alpha.shape}.")
    if np.any(alpha < 0) or np.any(mu < 0):
        raise ValueError("mu and alpha must be non-negative.")
    q = float(decay)
    if q <= 0:
        raise ValueError("decay must be positive.")

    disp_mean = np.zeros((m, m, d)) if disp_mean is None else np.asarray(disp_mean, float)
    disp_scale = np.full((m, m, d), 0.5) if disp_scale is None else np.asarray(disp_scale, float)
    bg_mean = np.zeros((m, d)) if bg_mean is None else np.asarray(bg_mean, float)
    bg_scale = np.ones((m, d)) if bg_scale is None else np.asarray(bg_scale, float)

    row_sums = alpha.sum(axis=1)          # A_j
    mu_total = float(mu.sum())

    times: List[float] = []
    locs: List[np.ndarray] = []
    marks: List[int] = []

    t = 0.0
    while True:
        if times:
            ts = np.asarray(times)
            ks = np.asarray(marks)
            # Dominating rate at the current time (decreasing for s > t).
            lam_bar = mu_total + float(np.sum(row_sums[ks] * q * np.exp(-q * (t - ts))))
        else:
            lam_bar = mu_total
        if lam_bar <= 0.0:
            break

        t = t - np.log(rng.random()) / lam_bar
        if t >= T:
            break

        if times:
            ts = np.asarray(times)
            ks = np.asarray(marks)
            g = q * np.exp(-q * (t - ts))                  # (N,)
            lam_k = mu + (alpha[ks, :] * g[:, None]).sum(axis=0)   # (M,)
        else:
            g = np.zeros(0)
            lam_k = mu.copy()
        lam_total = float(lam_k.sum())

        if rng.random() > lam_total / lam_bar:
            continue  # thinned

        # --- mark ---
        k = int(rng.choice(m, p=lam_k / lam_total))

        # --- location: exact background/offspring mixture for this mark ---
        w_bg = float(mu[k])
        w_par = (alpha[np.asarray(marks, dtype=int), k] * g) if times else np.zeros(0)
        weights = np.concatenate([[w_bg], w_par])
        weights = weights / weights.sum()
        choice = int(rng.choice(weights.shape[0], p=weights))
        if choice == 0:
            x = bg_mean[k] + bg_scale[k] * rng.standard_normal(d)
        else:
            j = choice - 1
            kj = marks[j]
            x = locs[j] + disp_mean[kj, k] + disp_scale[kj, k] * rng.standard_normal(d)

        times.append(float(t))
        locs.append(np.asarray(x, dtype=np.float64))
        marks.append(k)

        if len(times) > max_events:
            raise RuntimeError(
                f"Exceeded max_events={max_events}; alpha is likely supercritical "
                f"(spectral radius = {np.abs(np.linalg.eigvals(alpha)).max():.3f})."
            )

    return {
        "times": np.asarray(times, dtype=np.float64),
        "locations": (
            np.asarray(locs, dtype=np.float32) if locs else np.zeros((0, d), dtype=np.float32)
        ),
        "marks": np.asarray(marks, dtype=np.int64),
    }


def simulate_marked_hawkes_sequences(
    *,
    n_sequences: int,
    T: float,
    mu: np.ndarray,
    alpha: np.ndarray,
    decay: float,
    seed: int = 0,
    min_length: int = 2,
    **kwargs,
) -> List[Dict[str, np.ndarray]]:
    """Simulate ``n_sequences`` independent sequences; drop those shorter than
    ``min_length``.  Returns Seahorse raw-sequence dicts."""
    rng = np.random.default_rng(seed)
    out: List[Dict[str, np.ndarray]] = []
    for _ in range(n_sequences):
        seq = simulate_marked_hawkes(mu=mu, alpha=alpha, decay=decay, T=T, rng=rng, **kwargs)
        if seq["times"].shape[0] >= min_length:
            out.append(seq)
    return out
