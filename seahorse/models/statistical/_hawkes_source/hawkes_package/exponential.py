"""Linear Hawkes process with an exponential excitation kernel."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import SeedLike, TemporalHawkesProcess, _IntensityCursor

__all__ = ["ExponentialHawkes"]


class _DecayCursor(_IntensityCursor):
    r"""The thinning loop's intensity, carried forward instead of rebuilt.

    Holds :math:`S(t) = \sum_{t_i \le t} e^{-\beta (t - t_i)}`, from which both
    hooks follow: the bound is :math:`\mu + \alpha S(t)` over every recorded
    event, and the intensity at a candidate is the same expression, because a
    candidate is strictly later than every event and so the strict and
    non-strict sums range over the same set.

    Walking from `t` to `t'` costs one multiply, :math:`S \mapsto S e^{-\beta
    (t' - t)}`, where the direct form costs an exponential per past event. That
    is the difference between a linear and a quadratic simulation: measured at
    8 000 events, 0.042 s against 1.262 s.

    **This is not bit-identical to the direct sum**, and cannot be: a product of
    decays accumulates rounding differently from one exponential of the total
    lag. Measured over five seeds at 2 000 events the realisations agree to
    3.7e-15 relative and no acceptance decision changed, but that is an
    observation, not a guarantee -- see the class docstring of
    :class:`ExponentialHawkes` and ``docs/migration.md``.

    .. versionadded:: 1.0.0
    """

    __slots__ = ("alpha", "beta", "mu", "n_events", "sum")

    def __init__(self, process: ExponentialHawkes, start: float) -> None:
        super().__init__(process, start)
        self.mu = process.mu
        self.alpha = process.alpha
        self.beta = process.beta
        record = process.events
        # The same set `_upper_bound` sums over: every recorded event, none of
        # which can lie past `start`.
        self.sum = float(np.exp(-self.beta * (start - record)).sum()) if record.size else 0.0
        self.n_events = int(record.size)

    def move_to(self, t: float) -> None:
        """Decay the carried sum to `t`."""
        self.sum *= math.exp(-self.beta * (t - self.time))
        self.time = t

    def bound(self) -> float:
        """``mu + alpha * S``, over every event recorded so far."""
        return self.process._require_positive_bound(self.mu + self.alpha * self.sum, self.time)

    def intensity(self) -> float:
        """Return the same expression: at a candidate time the two sums coincide.

        The candidate is strictly later than every recorded event, so
        ``t_i < t`` and ``t_i <= t`` select the same events -- which is why one
        carried sum serves both hooks rather than two.
        """
        return self.mu + self.alpha * self.sum

    def accept(self) -> None:
        """Absorb an event at the current time, whose own kernel value is ``1``."""
        self.sum += 1.0
        self.n_events += 1


class ExponentialHawkes(TemporalHawkesProcess):
    r"""Linear Hawkes process with kernel :math:`\kappa(s) = \alpha e^{-\beta s}`.

    The conditional intensity is

    .. math::

        \lambda(t \mid H_t) = \mu + \sum_{t_i < t} \alpha e^{-\beta (t - t_i)}.

    Parameters
    ----------
    param : array_like of shape (3,)
        ``[mu, alpha, beta]`` — background rate, excitation size and decay rate.
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`~hawkes_package.base.HawkesProcess`.

    Raises
    ------
    ValueError
        If `param` does not have exactly three entries, if ``beta <= 0``, if
        `mu` or `alpha` is negative, or if ``alpha / beta >= 1``. The last is
        the stationarity condition: the branching ratio is ``alpha / beta``, so
        at or above 1 each event spawns at least one offspring on average and
        the simulation would not terminate.

    Examples
    --------
    >>> process = ExponentialHawkes(np.array([2.0, 0.5, 1.0]), rng=42)
    >>> process.simulate(100)
    >>> len(process.events)
    100
    """

    def __init__(self, param: Any, rng: SeedLike = None) -> None:
        param = np.asarray(param, dtype=float).ravel()
        if param.size != 3:
            raise ValueError(
                f"param must have exactly 3 entries [mu, alpha, beta], got {param.size}"
            )
        mu, alpha, beta = (float(v) for v in param)
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if mu < 0 or alpha < 0:
            raise ValueError(f"mu and alpha must be non-negative, got mu={mu}, alpha={alpha}")
        if alpha / beta >= 1:
            raise ValueError(
                f"Stability condition violated: alpha/beta = {alpha / beta:.4f} >= 1. "
                "The process will not be stationary."
            )

        super().__init__(rng=rng)
        self.param = param
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.temporal = lambda s: alpha * np.exp(-beta * np.asarray(s, dtype=float))
        self._saved_cursor: _DecayCursor | None = None

    def _cursor(self, start: float) -> _IntensityCursor:
        """Return the carried-sum cursor, reusing the saved one where it fits.

        Reuse is not an optimisation, it is what keeps
        ``simulate(1); simulate(1)`` equal to ``simulate(2)`` **bit for bit**, as
        `tests/test_base.py` asserts. Rebuilding the sum from the record at the
        start of the second call gives a value a few ulps from the one the
        uninterrupted loop was carrying, and the next event time moves with it.

        The saved cursor is used only when it still describes this record: same
        position, same event count. Anything else -- a record replaced through
        the setter, a `simulate_until` that stopped past the last event -- falls
        back to summing the record directly, which is always correct and only
        sometimes a few ulps from what a continuous run would have held.
        """
        saved = self._saved_cursor
        if saved is not None and saved.time == start and saved.n_events == self.events.size:
            return saved
        return _DecayCursor(self, start)

    def _save_cursor(self, cursor: _IntensityCursor) -> None:
        """Keep the carried sum for a continuation of this realisation."""
        self._saved_cursor = cursor if isinstance(cursor, _DecayCursor) else None

    def _record_replaced(self) -> None:
        """Drop the carried sum: it describes the record that was just replaced.

        Seeding a realisation with ``process.events = history`` is a supported
        move, and a replacement with the same length and the same last time
        would otherwise satisfy the reuse test in :meth:`_cursor` while carrying
        the wrong sum -- a bound and an intensity for a history that is no
        longer there, with nothing raising.
        """
        self._saved_cursor = None

    def _conditional_intensity(self, t: float) -> float:
        record = self.events
        past = record[record < t]
        return float(self.mu + self.alpha * np.exp(-self.beta * (t - past)).sum())

    def _upper_bound(self, t: float) -> float:
        # The kernel decreases and no event can arrive before the next accepted
        # one, so the intensity at `t` dominates the whole interval. Bounding at
        # `t` rather than at the last event is both tighter after a rejection
        # and well defined when no events have occurred yet.
        return float(self.mu + self.alpha * np.exp(-self.beta * (t - self.events)).sum())
