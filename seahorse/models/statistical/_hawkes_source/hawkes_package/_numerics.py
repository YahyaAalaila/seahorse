"""Numerical helpers shared across the package.

Two concerns live here, both cross-cutting enough that duplicating them was the
direct cause of shipped bugs:

* **Coercion.** :func:`as_float` and :func:`as_point` are the single place where
  a value coming from — or going to — a user callable is normalised. Ad-hoc
  ``float(...)``, ``np.ravel(x)[0]`` and ``x[0]`` spellings scattered across the
  process classes disagreed with each other, which is how a shape-``(1,)``
  coordinate came to select a different intensity surface from a scalar one.
* **Peak location.** :func:`locate_peak` replaces a bare
  ``scipy.optimize.fmin(..., 0)``. That search is local and starts at zero, so
  on a kernel that is flat near zero it returns zero and reports a peak value of
  zero — silently disabling the bell-shaped thinning bound.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = ["PeakLocation", "as_float", "as_point", "locate_peak"]


def as_float(value: Any) -> float:
    """Coerce a scalar-like value to a Python ``float``.

    Accepts a Python number, a NumPy scalar, a 0-d array, or any array holding
    exactly one element. Plain ``float()`` rejects the last of these: since
    NumPy 2.0 ``float(np.array([1.5]))`` raises :class:`TypeError`, and user
    kernels routinely return shape-``(1,)`` arrays.

    Raises
    ------
    ValueError
        If `value` does not hold exactly one element.
    """
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"expected a scalar or a single-element array, got shape {array.shape}")
    return float(array.reshape(()).item())


def as_point(x: Any, ndim: int) -> np.ndarray:
    """Coerce `x` to a shape-``(ndim,)`` float array.

    This is the contract every user callable sees: `base` and `spatial` are
    always handed a point of the domain's dimension, never a bare Python float
    on one code path and an array on another.

    Accepts a scalar (when ``ndim == 1``), a shape-``(ndim,)`` array, a
    ``(ndim, 1)`` column, or anything else holding exactly `ndim` elements.

    Raises
    ------
    ValueError
        If `x` does not hold exactly `ndim` elements, or holds a non-finite one.
    """
    array = np.asarray(x, dtype=float).reshape(-1)
    if array.size != ndim:
        raise ValueError(
            f"expected a point with {ndim} coordinate(s), got {array.size} (shape {np.shape(x)})"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"coordinates must be finite, got {array.tolist()}")
    return array


class PeakLocation(NamedTuple):
    """Where a kernel attains its maximum, and the value there."""

    lag: float
    value: float


def locate_peak(
    fn: Callable[[float], Any],
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    n_scan: int = 513,
    max_expansions: int = 20,
    refine: bool = True,
    tail_rtol: float = 1e-3,
    name: str = "kernel",
) -> PeakLocation:
    """Locate the global maximum of a non-negative kernel on ``[lower, inf)``.

    A coarse global scan followed by an optional local refinement, rather than a
    local search from a fixed start point.

    Parameters
    ----------
    fn : callable
        The kernel. Called with one Python ``float`` at a time, so a kernel
        written as ``lambda dt: 2 * np.exp(-2 * float(dt))`` works.
    lower, upper : float
        Initial scan window. `upper` is doubled while the window still looks
        unresolved, up to `max_expansions` times.
    n_scan : int
        Number of scan points per window.
    max_expansions : int
        Maximum number of doublings before giving up and warning.
    refine : bool
        Whether to polish the scanned argmax with a bounded local search. The
        refinement is kept only if it strictly improves on the scan.
    tail_rtol : float
        The window counts as resolved once ``fn(upper) <= tail_rtol * best``.
    name : str
        Used in warning and error messages.

    Returns
    -------
    PeakLocation
        The lag and value of the maximum. ``value`` is guaranteed to be at least
        the largest value actually observed — the thinning bound relies on the
        returned value dominating the kernel.

    Raises
    ------
    ValueError
        If the kernel takes a negative value. The thinning bound assumes a
        non-negative temporal kernel, and the scan is the natural place to check.

    Warns
    -----
    UserWarning
        If the kernel has not decayed by the end of the largest window searched,
        so no reliable maximum could be established.

    Notes
    -----
    A peak at ``lag == lower`` is perfectly legitimate — it simply means the
    kernel is monotone decreasing — and is returned without complaint.
    """
    lower = float(lower)
    upper = float(upper)

    def evaluate(grid: np.ndarray) -> np.ndarray:
        return np.array([as_float(fn(float(s))) for s in grid], dtype=float)

    best_lag = lower
    best_value = -np.inf
    grid = np.linspace(lower, upper, n_scan)
    values = evaluate(grid)

    for expansion in range(max_expansions + 1):
        if values.min() < 0:
            raise ValueError(
                f"{name} must be non-negative on [{lower}, {upper}] for the thinning "
                f"bound to be valid, but it reaches {values.min():.6g}"
            )

        index = int(np.argmax(values))
        if values[index] > best_value:
            best_lag, best_value = float(grid[index]), float(values[index])

        argmax_near_the_end = index >= int(0.75 * (n_scan - 1))
        tail_still_high = values[-1] > tail_rtol * max(best_value, np.finfo(float).tiny)
        # A constant window carries no information about where the kernel peaks.
        # This is the signature of a delayed kernel, which is identically zero
        # near the origin and rises only much later. Note this must not key on
        # "the maximum is at `lower`" -- for a monotone kernel that is the
        # correct answer, not a reason to keep searching.
        window_is_flat = bool(np.ptp(values) <= 0.0)

        if not (window_is_flat or argmax_near_the_end or tail_still_high):
            break
        if expansion == max_expansions:
            warnings.warn(
                f"{name} has not decayed by lag {upper:.4g}, so its maximum could not "
                f"be established reliably; the thinning bound may be too small. Pass "
                f"peak_lag= explicitly if you know where the kernel peaks.",
                UserWarning,
                stacklevel=3,
            )
            break

        upper *= 2.0
        grid = np.linspace(lower, upper, n_scan)
        values = evaluate(grid)

    if refine and n_scan > 2:
        step = (grid[-1] - grid[0]) / (n_scan - 1)
        left = max(lower, best_lag - step)
        right = best_lag + step
        if right > left:
            result = minimize_scalar(
                lambda s: -as_float(fn(float(s))), bounds=(left, right), method="bounded"
            )
            if result.success:
                candidate = -float(result.fun)
                # Only ever accept an improvement: the scan maximum is a value we
                # know the kernel actually attains, and the bound depends on the
                # returned value dominating it.
                if candidate > best_value:
                    best_lag, best_value = float(result.x), candidate

    return PeakLocation(lag=best_lag, value=best_value)
