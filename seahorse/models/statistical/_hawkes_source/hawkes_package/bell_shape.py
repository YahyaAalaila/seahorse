"""Hawkes process for kernels with a single interior maximum."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._numerics import as_float, locate_peak
from .base import SeedLike, TemporalHawkesProcess

__all__ = ["BellShapeHawkes"]


class BellShapeHawkes(TemporalHawkesProcess):
    r"""Hawkes process whose kernel rises before it decays.

    The conditional intensity has the same form as
    :class:`~hawkes_package.monotone.MonotoneKernelHawkes`,

    .. math::

        \lambda(t \mid H_t) = \varphi\!\left( \sum_{t_i < t} \kappa(t - t_i) \right),

    but :math:`\kappa` is no longer monotone: excitation ramps up to a peak at
    lag :attr:`ext` before decaying. The bound used by
    :class:`~hawkes_package.monotone.MonotoneKernelHawkes` is therefore invalid
    while the kernel is still rising, so each event is instead bounded by its
    own future supremum -- the peak value if that event has not yet peaked, its
    current value if it has.

    .. versionchanged:: 0.2.0
       Before 0.2.0 the bound added a single peak's worth of headroom to the
       whole intensity. That is not enough when two or more events are rising
       at once, and the thinning invariant ``M >= lambda`` failed in a few
       percent of steps -- silently biasing the simulated process.

    Parameters
    ----------
    temporal : callable
        The kernel :math:`\kappa`, with one global maximum on ``[0, inf)``.
    nonlinearity : callable
        The monotone-increasing :math:`\varphi`. Defaults to ``x + 2``.
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`~hawkes_package.base.HawkesProcess`.

    peak_lag : float, optional
        Lag at which `temporal` peaks. Supply it to skip the numerical search
        -- necessary for a kernel with a spike narrower than the search grid.
    peak_value : float, optional
        The kernel's value at `peak_lag`; defaults to ``temporal(peak_lag)``.

    Attributes
    ----------
    ext : float
        Lag at which `temporal` attains its maximum, located by
        :func:`~hawkes_package._numerics.locate_peak`.
    peak : float
        The kernel's value at :attr:`ext`. Guaranteed to dominate every value
        the search observed -- the thinning bound depends on it.

    Examples
    --------
    >>> def triangular(s):
    ...     s = np.asarray(s, dtype=float)
    ...     return 2 * s * ((s > 0) & (s < 0.5)) + (-2 * s + 2) * ((s >= 0.5) & (s < 1))
    >>> process = BellShapeHawkes(triangular, rng=0)
    >>> process.simulate(20)
    >>> len(process.events)
    20
    """

    def __init__(
        self,
        temporal: Callable[[Any], Any],
        nonlinearity: Callable[[Any], Any] = lambda x: x + 2,
        rng: SeedLike = None,
        *,
        peak_lag: float | None = None,
        peak_value: float | None = None,
    ) -> None:
        super().__init__(rng=rng)
        self.temporal = temporal
        self.nonlinearity = nonlinearity

        if peak_lag is None:
            located = locate_peak(temporal, name="temporal kernel")
            self.ext, self.peak = located.lag, located.value
        else:
            self.ext = float(peak_lag)
            self.peak = as_float(peak_value if peak_value is not None else temporal(self.ext))

    def _conditional_intensity(self, t: float) -> float:
        record = self.events
        past = record[record < t]
        return float(self.nonlinearity(np.sum(self.temporal(t - past))))

    def _upper_bound(self, t: float) -> float:
        # Bound each event's future contribution by its own supremum: an event
        # that has not yet peaked can still climb to the peak, one that has is
        # already decaying. Events at exactly `t` count -- at the start of a
        # thinning step `t` *is* the most recent event time.
        record = self.events
        past = record[record <= t]
        if past.size == 0:
            return float(self.nonlinearity(0.0))
        lags = t - past
        factors = np.where(lags < self.ext, self.peak, self.temporal(lags))
        return float(self.nonlinearity(factors.sum()))
