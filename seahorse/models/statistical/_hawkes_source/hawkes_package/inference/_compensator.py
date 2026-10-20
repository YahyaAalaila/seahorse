r"""Panelled Gauss-Legendre quadrature for the compensator :math:`\int_0^T \lambda`.

The compensator is the half of the log-likelihood that has no closed form for a
general kernel, and getting it slightly too small is the failure mode of the
whole subpackage. Every term it under-counts is subtracted from the penalty on a
high intensity, so ``mu`` and the excitation both drift upward, the fit looks
tighter than it is, and nothing raises. So the integration here is deliberately
conservative about where it puts its panels.

**Where the integrand is smooth.** :math:`\lambda` jumps at every event and is
analytic in between, for an analytic kernel. Panelling at the event times and
applying an order-8 Gauss-Legendre rule inside each panel is then spectrally
accurate -- it converges faster than any power of the panel width, and eight
nodes are already at machine precision for an exponential kernel.

**Where it is not.** A kernel with an interior kink puts one *inside* a panel,
at a fixed lag after each event: the triangular kernel used throughout this
project's tests kinks at ``t_i + 0.5``. Gauss-Legendre across a kink converges
at second order instead, and eight nodes are then nowhere near enough. Hence
`extra_lags`, which breaks the panel at the kink, and
:func:`check_resolution`, which compares order ``P`` against order ``2P`` and
says so when they disagree -- the same idiom, and the same reason, as
:func:`~hawkes_package.spatio_temporal._integration.check_resolution`.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = ["breakpoints", "check_resolution", "integrate", "panels"]

#: Gauss-Legendre nodes per panel. Eight rather than the four
#: :mod:`~hawkes_package.spatio_temporal._integration` uses, because there the
#: panels are fixed and the integrand carries ``max(0, .)`` kinks anywhere,
#: while here the panels are cut at the jumps and the integrand between them is
#: smooth -- which is the regime a higher-order rule wins in.
DEFAULT_ORDER = 8


def breakpoints(
    start: float,
    end: float,
    event_times: Any,
    extra_lags: Sequence[float] = (),
) -> np.ndarray:
    """Panel edges for integrating an intensity over ``[start, end]``.

    Parameters
    ----------
    start, end : float
        The interval. ``end <= start`` gives a two-point degenerate array, which
        :func:`panels` turns into an empty rule.
    event_times : array_like
        Every event time; those outside ``(start, end)`` are dropped, because an
        event at or before `start` causes no jump inside the interval and one at
        or after `end` causes none either.
    extra_lags : sequence of float
        Lags at which the *kernel* has an interior kink. Each event contributes
        a break at ``t_i + lag``, and so does `start` itself -- events before the
        window still excite inside it, and their kinks land inside it too.

    Returns
    -------
    numpy.ndarray
        Sorted, de-duplicated panel edges beginning at `start` and ending at
        `end`.
    """
    start, end = float(start), float(end)
    edges = [np.array([start, end], dtype=float)]

    times = np.asarray(event_times, dtype=float).ravel()
    if times.size:
        interior = times[(times > start) & (times < end)]
        edges.append(interior)
        for lag in extra_lags:
            shifted = times + float(lag)
            edges.append(shifted[(shifted > start) & (shifted < end)])

    merged = np.unique(np.concatenate(edges))
    return np.asarray(merged[(merged >= start) & (merged <= end)], dtype=float)


def panels(edges: Any, order: int = DEFAULT_ORDER) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights for every panel between `edges`.

    Parameters
    ----------
    edges : array_like
        Sorted panel edges, as returned by :func:`breakpoints`.
    order : int
        Nodes per panel.

    Returns
    -------
    nodes, weights : numpy.ndarray
        Flat arrays of length ``order * (len(edges) - 1)``. Weights are all
        strictly positive, so a pointwise-dominating integrand integrates to a
        dominating value -- which is what makes the order-``P``-versus-``2P``
        comparison in :func:`check_resolution` a bound rather than a hint.
    """
    bounds = np.asarray(edges, dtype=float)
    if bounds.size < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    widths = bounds[1:] - bounds[:-1]
    keep = widths > 0.0
    if not np.any(keep):
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    base_nodes, base_weights = np.polynomial.legendre.leggauss(int(order))
    half = 0.5 * widths[keep]
    mid = 0.5 * (bounds[1:][keep] + bounds[:-1][keep])
    nodes = (mid[:, None] + half[:, None] * base_nodes[None, :]).ravel()
    weights = (half[:, None] * base_weights[None, :]).ravel()
    return nodes, weights


def integrate(
    fn: Callable[[float], float],
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Integrate a scalar-valued `fn` against a rule from :func:`panels`.

    `fn` is called one node at a time on purpose. The temporal likelihood
    evaluates it through the process's own ``_conditional_intensity`` hook --
    the function the simulator thins against -- and that hook takes a scalar. A
    vectorized re-implementation would be faster and would be a *second*
    definition of the intensity, which is how this package previously came to
    plot a curve a constant ``mu`` below the one it simulated.
    """
    if nodes.size == 0:
        return 0.0
    values = np.fromiter((fn(float(t)) for t in nodes), dtype=float, count=nodes.size)
    return float(np.dot(weights, values))


def check_resolution(
    fn: Callable[[float], float],
    edges: Any,
    *,
    order: int = DEFAULT_ORDER,
    rtol: float = 1e-3,
    name: str = "compensator",
) -> float:
    """Warn if doubling the panel order moves the integral.

    Returns the order-``2 * order`` value, which is the better of the two and
    the one worth keeping if the caller wants it.

    A disagreement means the integrand is not resolved inside a panel -- almost
    always an interior kink the caller has not declared through `extra_lags`.
    The consequence is not a rough answer but a *biased* one: the error has a
    sign that does not average out over events, so the fitted excitation moves
    with it.

    Parameters
    ----------
    fn : callable
        The integrand, taking one time.
    edges : array_like
        Panel edges.
    order : int
        The order actually in use.
    rtol : float
        Relative disagreement tolerated. Measured on this project's own kernels:
        an undeclared triangular kink comes in at ``6.7e-3``, an exponential or
        an integer-shape gamma kernel at machine precision, and a gamma kernel
        of *fractional* shape between the two -- ``1.8e-4`` at shape 1.5,
        ``6.1e-6`` at 2.5 -- because ``s**(k-1)`` has an algebraic branch point
        at every event, which a Gauss rule resolves algebraically rather than
        spectrally. The default sits in the gap: it catches the kink, which
        biases the fit, and passes the branch point, which does not.
    name : str
        Used in the warning message.

    Warns
    -----
    UserWarning
        If the two orders differ by more than `rtol` relatively.
    """
    coarse = integrate(fn, *panels(edges, order))
    fine = integrate(fn, *panels(edges, 2 * order))
    scale = max(abs(fine), np.finfo(float).tiny)
    error = abs(coarse - fine) / scale
    if error > rtol:
        warnings.warn(
            f"the {name} is not resolved by an order-{order} rule: doubling it changes "
            f"the integral by {100 * error:.3g}%. The intensity has a feature inside a "
            "panel -- a kernel with an interior kink, most likely -- so the "
            "log-likelihood carries an error with a consistent sign and the fitted "
            "excitation moves with it. Declare the kink through extra_lags=, or raise "
            "the order.",
            UserWarning,
            stacklevel=3,
        )
    return fine
