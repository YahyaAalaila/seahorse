r"""A compensator computed without reusing the likelihood's arithmetic.

The likelihood integrates :math:`\lambda` with panelled Gauss-Legendre, placing
panel edges at the event times so the jumps are integrated exactly. That is the
right rule for the job and it is used everywhere a fit needs an integral.

This module deliberately does **not** use it. Composite Simpson on a dense
uniform grid is a different quadrature family with different error behaviour --
:math:`O(h^4)` against a rule that is exact on polynomials of its own order --
and it reaches the intensity through the process hook the simulator thins
against, not through the cached rearrangement the fast backend uses. Two
independent routes to one number.

The grid is uniform on purpose, even though placing nodes at the event times
would be more accurate. An integrator that knew where the jumps were would be
making the same modelling assumption the likelihood makes, and the point here is
to make as few shared assumptions as possible. The cost is paid in node count.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..likelihood import History, LogLikelihood, _bind_history

__all__ = ["compensator_agreement", "independent_compensator"]

#: Grid points per unit of the observation window's *shortest* inter-event gap.
#: Simpson's error is O(h^4 f''''), and the intensity's fourth derivative is
#: largest just after an event, so the gap sets the scale rather than the window
#: length: a window of 100 with events 0.01 apart needs the same h as a window of
#: 1 would.
_POINTS_PER_GAP = 8

#: Floor and ceiling on the grid. The floor keeps a short history honest; the
#: ceiling stops a near-tied pair of events -- which `History` allows down to any
#: positive separation -- from asking for an array that does not fit in memory.
_MIN_POINTS = 2049
_MAX_POINTS = 200_001


def _grid_size(history: History, start: float, end: float) -> int:
    """Choose an odd number of Simpson nodes for ``[start, end]``."""
    inside = history.times[(history.times > start) & (history.times <= end)]
    edges = np.concatenate([[start], inside, [end]])
    gaps = np.diff(edges)
    positive = gaps[gaps > 0.0]
    span = max(float(end - start), float(np.finfo(float).tiny))
    finest = float(np.min(positive)) if positive.size else span

    wanted = int(_POINTS_PER_GAP * span / finest) + 1
    points = min(max(wanted, _MIN_POINTS), _MAX_POINTS)
    # Simpson needs an even number of intervals, so an odd number of points.
    return points + 1 if points % 2 == 0 else points


def _simpson(values: np.ndarray, step: float) -> float:
    """Composite Simpson over an odd number of equally spaced samples."""
    weights = np.ones(values.size, dtype=float)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return float(step / 3.0 * np.dot(weights, values))


def independent_compensator(
    likelihood: LogLikelihood,
    theta: Any,
    history: History,
    times: Any,
    *,
    intensity: Callable[[float], float] | None = None,
) -> np.ndarray:
    r"""Evaluate :math:`\Lambda(t) - \Lambda(start)` without the likelihood's quadrature.

    Parameters
    ----------
    likelihood : LogLikelihood
        Used **only** to build the process at `theta` and condition it on
        `history`. Its :meth:`~LogLikelihood.compensator` is never called; that
        is the entire point of this function.
    theta : array_like
        A single parameter vector.
    history : History
        The observed events.
    times : array_like
        Sorted query times.
    intensity : callable, optional
        Override the integrand, for testing. Given a time, return the intensity
        the compensator should integrate -- space-integrated for a
        spatio-temporal model.

    Returns
    -------
    numpy.ndarray
        One value per query time, non-decreasing.

    Raises
    ------
    ValueError
        If `times` is not sorted, or if the process the likelihood builds
        exposes no intensity hook this can read.

    Notes
    -----
    Composite Simpson on a uniform grid, at eight points per shortest
    inter-event gap and at least 2049 points. That is far more nodes than the
    likelihood needs, and it is meant to be: accuracy here is bought with node
    count rather than with knowledge of where the jumps are, precisely so that
    this integrator shares as little as possible with the one it is checking.

    .. versionadded:: 1.0.0
    """
    query = np.asarray(times, dtype=float).ravel()
    if query.size == 0:
        return np.empty(0, dtype=float)
    if np.any(np.diff(query) < 0.0):
        raise ValueError("times must be sorted for the compensator")

    evaluate = (
        _intensity_of(likelihood, theta, history, float(query[-1]))
        if intensity is None
        else intensity
    )

    out = np.empty(query.size, dtype=float)
    running = 0.0
    left = float(history.start)
    for k, raw in enumerate(query):
        right = float(raw)
        if right > left:
            points = _grid_size(history, left, right)
            grid = np.linspace(left, right, points)
            values = np.array([float(evaluate(float(t))) for t in grid], dtype=float)
            running += _simpson(values, float(grid[1] - grid[0]))
        out[k] = running
        left = right
    return out


def _intensity_of(
    likelihood: LogLikelihood, theta: Any, history: History, upto: float
) -> Callable[[float], float]:
    """Return the process's own intensity hook, conditioned on the history.

    Built from ``likelihood.model`` rather than through the likelihood's own
    ``_process``, and not only because ``ExponentialLogLikelihood`` has no such
    method -- it is the closed form and touches no process at all. Going to the
    model directly means this shares exactly one thing with the likelihood it
    checks: the parameter vector. What gets integrated here is what the
    *simulator* would draw from, floor and all.
    """
    model = getattr(likelihood, "model", None)
    if model is None:  # pragma: no cover - every shipped likelihood carries one
        raise ValueError(
            f"{type(likelihood).__name__} exposes no model to build a process from, "
            "so an independent compensator cannot be built for it."
        )
    process = model(theta, rng=0)
    _bind_history(process, history.upto(upto))

    integrated = getattr(process, "_integrated_intensity", None)
    if integrated is not None:
        return lambda t: float(integrated(t))
    scalar = getattr(process, "_conditional_intensity", None)
    if scalar is None:  # pragma: no cover - guarded by the two-hook contract
        raise ValueError(f"{type(process).__name__} exposes neither intensity hook")
    return lambda t: float(scalar(t))


def compensator_agreement(
    likelihood: LogLikelihood,
    theta: Any,
    history: History,
    *,
    times: Any = None,
) -> float:
    r"""Largest relative gap between the likelihood's compensator and an independent one.

    The number that says whether the goodness-of-fit test below it means
    anything. A fit made with a compensator 20% too small inflates the intensity
    by 25%, and the two errors cancel exactly in the time-rescaled gaps -- so
    the residuals look perfect and this number does not.

    Returns
    -------
    float
        ``max |A - B| / max(|B|, tiny)`` over the query times, with ``B`` the
        independent value. Zero for an empty history.

    Notes
    -----
    Quadrature disagreement of order 1e-6 is expected and means the two rules
    resolved the same integrand. Anything approaching a percent is a finding.

    .. versionadded:: 1.0.0
    """
    query = history.times if times is None else np.asarray(times, dtype=float).ravel()
    if query.size == 0:
        return 0.0

    theirs = np.asarray(likelihood.compensator(theta, history, query), dtype=float)
    ours = independent_compensator(likelihood, theta, history, query)
    scale = np.maximum(np.abs(ours), np.finfo(float).tiny)
    return float(np.max(np.abs(theirs - ours) / scale))
