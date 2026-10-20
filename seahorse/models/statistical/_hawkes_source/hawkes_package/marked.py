r"""Hawkes processes whose events carry a mark that scales their excitation.

The ETAS shape: every event has a magnitude, and a larger magnitude produces
more aftershocks. Written

.. math::

    \lambda(t \mid H_t) = \varphi\!\left( \mu
        + \sum_{t_i < t} g(m_i)\, \kappa(t - t_i) \right),
    \qquad g(m) = e^{a (m - m_0)},

with the marks drawn independently from an exponential law on
:math:`[m_0, \infty)` -- the Gutenberg-Richter distribution of magnitudes, whose
rate ``b`` is the *b-value*.

**The stability story here is genuinely new**, and is the reason this module
exists rather than a productivity flag on the temporal classes. The branching
ratio is :math:`E[g(m)] \cdot \int\kappa`, and that expectation

.. math::

    E[g(m)] = \int_{m_0}^{\infty} e^{a(m - m_0)} \, b e^{-b(m - m_0)}\,dm
            = \frac{b}{b - a} \quad (a < b), \qquad +\infty \text{ otherwise}

**diverges while every realised value stays finite**. A parameter with
:math:`a \ge b` produces a perfectly ordinary-looking catalogue on any finite
window and describes a process with infinite expected offspring per event. No
simulation notices, so the constructor does.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ._numerics import as_float, locate_peak
from .base import MarkedTemporalHawkesProcess, SeedLike

__all__ = ["ExponentialMarkedHawkes", "MarkedHawkes", "expected_productivity"]


def expected_productivity(scale: float, b_value: float) -> float:
    r"""Return :math:`E[e^{a(m-m_0)}]` for an exponential mark law of rate `b`.

    ``b / (b - a)`` below the boundary and ``inf`` at or above it. The quantity
    the branching ratio multiplies the kernel mass by, and the one that can be
    infinite while every mark in a realisation is finite.

    .. versionadded:: 1.0.0
    """
    a, b = float(scale), float(b_value)
    if not b > 0:
        raise ValueError(f"the mark rate b must be positive, got {b_value!r}")
    if a >= b:
        return math.inf
    return float(b / (b - a))


class MarkedHawkes(MarkedTemporalHawkesProcess):
    r"""Hawkes process with a general kernel and a mark-dependent productivity.

    The intensity is
    :math:`\varphi(\mu + \sum_{t_i < t} g(m_i)\,\kappa(t - t_i))`, with `g` any
    non-negative productivity and the marks drawn from `mark_sampler`.

    One class covers both bound regimes, as
    :class:`~hawkes_package.multivariate.MultivariateHawkes` does: pass
    ``monotone_temporal_kernel=False`` for a kernel that rises before it decays.

    Parameters
    ----------
    mu : float
        Background rate. Non-negative.
    temporal : callable
        The kernel :math:`\kappa`, taking a non-negative lag, vectorized.
    productivity : callable
        :math:`g`, taking an array of marks and returning one **non-negative**
        value each. Negative values are refused when they appear: the thinning
        bound takes ``sup(g(m) * kappa)`` as ``g(m) * sup(kappa)``, which holds
        only for ``g(m) >= 0``.
    mark_sampler : callable
        Given a :class:`~numpy.random.Generator`, return one mark.
    nonlinearity : callable, optional
        The monotone-increasing :math:`\varphi`. Defaults to the identity.
    monotone_temporal_kernel : bool
        Whether `temporal` decreases everywhere.
    peak_lag, peak_value : float, optional
        Used only when `monotone_temporal_kernel` is ``False``.
    rng : None, int or numpy.random.Generator
        Source of randomness.

    Raises
    ------
    ValueError
        If `mu` is negative or not finite.

    Examples
    --------
    >>> rng_marks = lambda rng: float(rng.exponential(1.0))
    >>> process = MarkedHawkes(
    ...     mu=1.0,
    ...     temporal=lambda s: 0.4 * np.exp(-2.0 * np.asarray(s, dtype=float)),
    ...     productivity=lambda m: np.exp(0.5 * np.asarray(m, dtype=float)),
    ...     mark_sampler=rng_marks,
    ...     rng=0,
    ... )
    >>> process.simulate(30)
    >>> process.events.shape
    (2, 30)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        mu: float,
        temporal: Callable[[Any], Any],
        productivity: Callable[[Any], Any],
        mark_sampler: Callable[[np.random.Generator], float],
        *,
        nonlinearity: Callable[[Any], Any] | None = None,
        monotone_temporal_kernel: bool = True,
        peak_lag: float | None = None,
        peak_value: float | None = None,
        rng: SeedLike = None,
    ) -> None:
        background = float(mu)
        if not math.isfinite(background) or background < 0.0:
            raise ValueError(f"mu must be finite and non-negative, got {mu!r}")

        super().__init__(rng=rng)
        self.mu = background
        self.temporal = temporal
        self.productivity = productivity
        self.mark_sampler = mark_sampler
        self.nonlinearity = nonlinearity
        self.monotone_temporal_kernel = bool(monotone_temporal_kernel)

        self.ext: float | None = None
        self.peak: float | None = None
        if self.monotone_temporal_kernel:

            def suprema(lags: np.ndarray) -> np.ndarray:
                """Return the kernel itself; decreasing, so it is its own supremum."""
                return np.asarray(temporal(lags), dtype=float)

        else:
            if peak_lag is None:
                located = locate_peak(temporal, name="temporal kernel")
                self.ext, self.peak = located.lag, located.value
            else:
                self.ext = float(peak_lag)
                self.peak = as_float(peak_value if peak_value is not None else temporal(self.ext))
            ext, top = self.ext, self.peak

            def suprema(lags: np.ndarray) -> np.ndarray:
                """Return the peak while an event is still rising, its value once past."""
                return np.asarray(np.where(lags < ext, top, temporal(lags)), dtype=float)

        self._suprema: Callable[[np.ndarray], np.ndarray] = suprema

    def _productivity(self, marks: np.ndarray) -> np.ndarray:
        """Evaluate `g` and refuse a negative value, which would break the bound."""
        values = np.asarray(self.productivity(marks), dtype=float)
        if values.size and np.min(values) < 0.0:
            worst = float(np.min(values))
            raise RuntimeError(
                f"the productivity returned {worst!r} for an observed mark. The thinning "
                "bound rewrites sup(g(m) * kappa) as g(m) * sup(kappa), which holds only "
                "for g(m) >= 0 -- a negative productivity would put the bound below the "
                "intensity with nothing raising."
            )
        return values

    def _draw_mark(self) -> float:
        return float(self.mark_sampler(self.rng))

    def _response(self, value: float) -> float:
        """Apply the nonlinearity, or nothing when it is the identity."""
        return value if self.nonlinearity is None else as_float(self.nonlinearity(value))

    def _conditional_intensity(self, t: float) -> float:
        times, marks = self._past(t)
        if times.size == 0:
            return self._response(self.mu)
        factors = np.asarray(self.temporal(t - times), dtype=float)
        return self._response(self.mu + float(np.sum(self._productivity(marks) * factors)))

    def _upper_bound(self, t: float) -> float:
        # Non-strict: at the start of a thinning step `t` *is* the most recent
        # event time, and that event excites the interval a candidate lands in.
        # Every mark summed here has already been drawn -- the supremum of `g`
        # over the mark *distribution* never enters, which is why an unbounded
        # productivity is not a problem for this bound.
        times, marks = self._past(t, inclusive=True)
        if times.size == 0:
            return self._response(self.mu)
        factors = self._suprema(t - times)
        return self._response(self.mu + float(np.sum(self._productivity(marks) * factors)))


class ExponentialMarkedHawkes(MarkedHawkes):
    r"""The ETAS shape: exponential kernel, exponential marks, exponential productivity.

    .. math::

        \lambda(t \mid H_t) = \mu + \sum_{t_i < t}
            e^{a(m_i - m_0)}\, \alpha e^{-\beta (t - t_i)},

    with marks drawn from :math:`m - m_0 \sim \mathrm{Exp}(b)`. The branching
    ratio is :math:`\frac{\alpha}{\beta}\cdot\frac{b}{b-a}`, and the constructor
    refuses a parameter where it is not below one -- including the case where
    the second factor is infinite while every simulated mark is perfectly finite.

    Parameters
    ----------
    mu, alpha, beta : float
        Background rate, excitation size and decay rate, as in
        :class:`~hawkes_package.exponential.ExponentialHawkes`.
    scale : float
        ``a``, how sharply the mark raises productivity. Zero makes every event
        equally productive and reduces this to the unmarked process.
    b_value : float
        ``b``, the rate of the mark law. Must exceed `scale`.
    m0 : float
        The lower end of the mark range, subtracted in both the productivity and
        the mark law so that an event at ``m0`` has productivity one.
    rng : None, int or numpy.random.Generator
        Source of randomness.

    Raises
    ------
    ValueError
        If ``beta <= 0``, ``b_value <= 0``, `mu` or `alpha` is negative, or the
        branching ratio is at or above one -- which includes ``scale >= b_value``,
        where the expected productivity is infinite.

    Examples
    --------
    >>> process = ExponentialMarkedHawkes(mu=1.0, alpha=0.3, beta=2.0, scale=0.5, b_value=1.5)
    >>> process.branching_ratio
    0.225
    >>> process.simulate(40)
    >>> process.events.shape
    (2, 40)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        mu: float,
        alpha: float,
        beta: float,
        scale: float,
        b_value: float,
        *,
        m0: float = 0.0,
        rng: SeedLike = None,
    ) -> None:
        decay, amplitude = float(beta), float(alpha)
        a, b, floor = float(scale), float(b_value), float(m0)
        if not decay > 0:
            raise ValueError(f"beta must be positive, got {beta!r}")
        if amplitude < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha!r}")
        if not math.isfinite(floor):
            raise ValueError(f"m0 must be finite, got {m0!r}")

        expectation = expected_productivity(a, b)
        ratio = amplitude / decay * expectation
        if not ratio < 1.0:
            reason = (
                f"the expected productivity is infinite because scale={a} is at or above "
                f"b_value={b}"
                if math.isinf(expectation)
                else f"the branching ratio is {ratio:.4f}"
            )
            raise ValueError(
                f"Stability condition violated: {reason}. A mark law lighter than the "
                "productivity it feeds gives a process whose every realisation is finite "
                "and whose expected offspring per event is not -- nothing at run time "
                "notices, so it is refused here."
            )

        super().__init__(
            mu=mu,
            temporal=lambda s: amplitude * np.exp(-decay * np.asarray(s, dtype=float)),
            productivity=lambda m: np.exp(a * (np.asarray(m, dtype=float) - floor)),
            mark_sampler=lambda rng_: floor + float(rng_.exponential(1.0 / b)),
            rng=rng,
        )
        self.alpha, self.beta = amplitude, decay
        self.scale, self.b_value, self.m0 = a, b, floor
        self.branching_ratio = float(ratio)
