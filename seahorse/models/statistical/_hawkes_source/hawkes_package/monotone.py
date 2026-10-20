"""Hawkes process with a monotone-decreasing kernel and a nonlinear intensity."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import SeedLike, TemporalHawkesProcess

__all__ = ["MonotoneKernelHawkes"]


class MonotoneKernelHawkes(TemporalHawkesProcess):
    r"""Hawkes process with a monotone-decreasing kernel and nonlinearity.

    The conditional intensity is

    .. math::

        \lambda(t \mid H_t) = \varphi\!\left( \sum_{t_i < t} \kappa(t - t_i) \right)

    where :math:`\kappa` is monotone decreasing and :math:`\varphi` monotone
    increasing. Both conditions matter: they are what make the thinning bound
    below valid.

    Parameters
    ----------
    temporal : callable
        The kernel :math:`\kappa`, taking a non-negative time lag. It must
        accept a NumPy array and return one elementwise.
    nonlinearity : callable
        The monotone-increasing :math:`\varphi`. Defaults to ``x + 2``.
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`~hawkes_package.base.HawkesProcess`.

    Notes
    -----
    The upper bound sums over **all** events including the most recent one.
    Excluding it — as versions before 0.2.0 did — makes ``M < lambda(t + eps)``,
    so every candidate is accepted unconditionally and the result is a Poisson
    process rather than a Hawkes one.

    Examples
    --------
    >>> process = MonotoneKernelHawkes(lambda s: np.exp(-10 * s), rng=0)
    >>> process.simulate(50)
    >>> len(process.events)
    50
    """

    def __init__(
        self,
        temporal: Callable[[Any], Any],
        nonlinearity: Callable[[Any], Any] = lambda x: x + 2,
        rng: SeedLike = None,
    ) -> None:
        super().__init__(rng=rng)
        self.temporal = temporal
        self.nonlinearity = nonlinearity

    def _conditional_intensity(self, t: float) -> float:
        record = self.events
        past = record[record < t]
        return float(self.nonlinearity(np.sum(self.temporal(t - past))))

    def _upper_bound(self, t: float) -> float:
        # Includes the event at t itself: the kernel decreases, so this
        # dominates the intensity for every s >= t.
        return float(self.nonlinearity(np.sum(self.temporal(t - self.events))))
