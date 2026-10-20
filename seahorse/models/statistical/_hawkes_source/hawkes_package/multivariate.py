"""Multivariate Hawkes processes: several event types exciting one another.

One kernel shape and a matrix of scales, rather than one independently
parameterised kernel per ordered pair. That is a modelling restriction --
every cross-excitation decays at the same rate -- and it buys three things
worth the trade: the matrix stays out of every kernel evaluation, the parameter
count is ``d**2 + d`` rather than ``d**2`` kernels, and the likelihood built on
this in :mod:`hawkes_package.inference` keeps a per-type recursion that is
``O(n d)`` rather than ``O(n d**2)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._numerics import as_float, locate_peak
from .base import MultivariateTemporalHawkesProcess, SeedLike

__all__ = ["MultivariateExponentialHawkes", "MultivariateHawkes"]


def spectral_radius(matrix: Any) -> float:
    """Largest absolute eigenvalue of `matrix`.

    The multivariate stationarity condition: with a branching matrix
    ``G[i, j]`` counting the direct offspring of type *i* a type-*j* event
    produces, the process is stationary exactly when this is below one. It
    replaces the scalar ``alpha / beta`` of
    :class:`~hawkes_package.exponential.ExponentialHawkes`, and reduces to it
    at one type.

    The row-sum bound ``rho(G) <= max_i sum_j G[i, j]`` is the cheap majorant,
    exact for a non-negative matrix with equal row sums. It is not used here --
    it would refuse stationary processes near the boundary -- but it is what a
    guard should reach for when an eigensolve is not available or not certified.

    .. versionadded:: 1.0.0
    """
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(matrix, dtype=float)))))


def _square_non_negative(value: Any, *, name: str, size: int | None = None) -> np.ndarray:
    """Validate a non-negative square matrix, returning it as float."""
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {matrix.shape}")
    if size is not None and matrix.shape[0] != size:
        raise ValueError(f"{name} must be {size}x{size} to match mu, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite, got {matrix!r}")
    if np.any(matrix < 0):
        bad = np.argwhere(matrix < 0)[0]
        raise ValueError(
            f"{name} must be non-negative, got {matrix[bad[0], bad[1]]!r} at "
            f"[{bad[0]}, {bad[1]}]. The thinning bound takes the supremum of "
            "each term separately, and sup(a*f) = a*sup(f) only for a >= 0, so "
            "a negative entry would make the bound smaller than the intensity "
            "without anything raising. Inhibitory cross-excitation needs a "
            "different bound, not a different value here."
        )
    return matrix


class MultivariateHawkes(MultivariateTemporalHawkesProcess):
    r"""Mutually exciting Hawkes process over a finite set of event types.

    The conditional intensity of type *i* is

    .. math::

        \lambda_i(t \mid H_t) = \varphi\!\left( \mu_i + \sum_j A_{ij}
            \sum_{t_k < t,\; c_k = j} \kappa(t - t_k) \right),

    with one shared kernel :math:`\kappa`, a non-negative excitation matrix
    :math:`A`, a background rate per type, and a monotone-increasing
    :math:`\varphi` applied elementwise.

    One class covers both bound regimes, as
    :class:`~hawkes_package.spatio_temporal.process.SpatioTemporalHawkesProcess`
    does: pass ``monotone_temporal_kernel=False`` for a kernel that rises before
    it decays, and each event is then bounded by its own future supremum rather
    than by its current value.

    Parameters
    ----------
    mu : array_like of shape (d,)
        Background rate per type. Non-negative.
    excitation : array_like of shape (d, d)
        ``A[i, j]`` scales the excitation type *j* exerts on type *i*.
        **Non-negative**, and refused otherwise: see
        :class:`~hawkes_package.base.MultivariateTemporalHawkesProcess` for why
        that is the bound argument rather than a modelling preference.
    temporal : callable
        The shared kernel :math:`\kappa`, taking a non-negative lag. It must
        accept a NumPy array and return one elementwise.
    nonlinearity : callable, optional
        The monotone-increasing :math:`\varphi`, applied elementwise. Defaults
        to the identity, which is the linear Hawkes process.
    monotone_temporal_kernel : bool
        Whether `temporal` decreases everywhere. ``False`` selects the
        bell-shaped bound.
    peak_lag, peak_value : float, optional
        Lag at which `temporal` peaks and its value there, used only when
        `monotone_temporal_kernel` is ``False``. Supply them to skip the
        numerical search.
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`~hawkes_package.base.HawkesProcess`.

    Raises
    ------
    ValueError
        If `mu` is not one-dimensional or carries a negative entry, if
        `excitation` is not square, not the same size as `mu`, or carries a
        negative entry.

    Examples
    --------
    >>> process = MultivariateHawkes(
    ...     mu=[0.4, 0.2],
    ...     excitation=[[0.3, 0.1], [0.5, 0.2]],
    ...     temporal=lambda s: np.exp(-1.5 * np.asarray(s, dtype=float)),
    ...     rng=0,
    ... )
    >>> process.simulate(50)
    >>> process.events.shape
    (2, 50)
    >>> set(process.types.tolist()) <= {0, 1}
    True

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        mu: Any,
        excitation: Any,
        temporal: Callable[[Any], Any],
        *,
        nonlinearity: Callable[[Any], Any] | None = None,
        monotone_temporal_kernel: bool = True,
        peak_lag: float | None = None,
        peak_value: float | None = None,
        rng: SeedLike = None,
    ) -> None:
        background = np.asarray(mu, dtype=float).ravel()
        if background.size == 0:
            raise ValueError("mu must carry at least one background rate")
        if not np.all(np.isfinite(background)):
            raise ValueError(f"mu must be finite, got {mu!r}")
        if np.any(background < 0):
            raise ValueError(f"mu must be non-negative, got {mu!r}")
        matrix = _square_non_negative(excitation, name="excitation", size=background.size)

        super().__init__(background.size, rng=rng)
        self.mu = background
        self.excitation = matrix
        self.temporal = temporal
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
            # Bound to locals rather than read off `self` per call: the branch is
            # settled at construction, so the loop should not re-decide it, and
            # the attributes stay `float | None` for a reader without making the
            # hot path defend against a `None` this branch has already ruled out.
            ext, top = self.ext, self.peak

            def suprema(lags: np.ndarray) -> np.ndarray:
                """Return the peak while an event is still rising, its value once past."""
                return np.asarray(np.where(lags < ext, top, temporal(lags)), dtype=float)

        self._suprema: Callable[[np.ndarray], np.ndarray] = suprema

    def _response(self, values: np.ndarray) -> np.ndarray:
        """Apply the nonlinearity, or nothing when it is the identity."""
        if self.nonlinearity is None:
            return values
        return np.asarray(self.nonlinearity(values), dtype=float)

    def _components(self, factors: np.ndarray, kinds: np.ndarray) -> np.ndarray:
        """Per-type value from per-event kernel factors and their source types.

        Grouped by *source* and summed before the matrix is applied, rather than
        scaling each event by its own matrix entry. At one type the mask is
        all-true, so the inner sum runs over a same-length, same-order copy and
        returns the identical float the univariate classes compute -- which is
        what makes a one-type process here reproduce them bit for bit.
        """
        by_source = np.array(
            [float(np.sum(factors[kinds == j])) for j in range(self.n_types)], dtype=float
        )
        # Elementwise rather than `@`: a BLAS matrix-vector product is free to
        # differ across builds, and the one-type equality asserted in the tests
        # is exact, not approximate.
        excited = (self.excitation * by_source[None, :]).sum(axis=1)
        return self._response(self.mu + excited)

    def _component_intensities(self, t: float) -> np.ndarray:
        record = self.events
        times, kinds = record[0], record[1]
        keep = times < t
        factors = np.asarray(self.temporal(t - times[keep]), dtype=float)
        return self._components(factors, kinds[keep])

    def _upper_bound(self, t: float) -> float:
        record = self.events
        times, kinds = record[0], record[1]
        # Non-strict: at the start of a thinning step `t` *is* the most recent
        # event time, and that event excites the interval the candidate lands in.
        keep = times <= t
        factors = self._suprema(t - times[keep])
        # Reduced exactly as `_cumulative_intensities` reduces, so an empty
        # history gives M == lambda to the bit rather than to a tolerance.
        return float(np.cumsum(self._components(factors, kinds[keep]))[-1])


class MultivariateExponentialHawkes(MultivariateHawkes):
    r"""Linear multivariate Hawkes process with a shared exponential kernel.

    The conditional intensity of type *i* is

    .. math::

        \lambda_i(t \mid H_t) = \mu_i + \sum_j A_{ij}
            \sum_{t_k < t,\; c_k = j} e^{-\beta (t - t_k)},

    the matrix analogue of
    :class:`~hawkes_package.exponential.ExponentialHawkes`, which it reproduces
    exactly at one type: ``MultivariateExponentialHawkes([mu], [[alpha]], beta)``
    and ``ExponentialHawkes([mu, alpha, beta])`` consume the same stream and
    produce the same events.

    **One decay rate for every pair.** :math:`\beta` is a scalar, so
    cross-excitation between different pairs of types cannot decay at different
    speeds. That is the price of one kernel shape with a matrix of scales, and
    it is what keeps the parameter count at :math:`d^2 + d + 1`.

    Parameters
    ----------
    mu : array_like of shape (d,)
        Background rate per type. Non-negative.
    excitation : array_like of shape (d, d)
        ``A[i, j]`` is the excitation type *j* exerts on type *i*. Non-negative.
    beta : float
        Shared decay rate. Positive.
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`~hawkes_package.base.HawkesProcess`.

    Raises
    ------
    ValueError
        If ``beta <= 0``, if `mu` or `excitation` carries a negative entry, if
        `excitation` is not square or not the size of `mu`, or if the spectral
        radius of ``excitation / beta`` is at or above one. The last is the
        stationarity condition: the kernel has mass ``1 / beta``, so
        ``excitation / beta`` is the branching matrix, and at or above one each
        event spawns at least one offspring on average and the simulation would
        not terminate.

    Examples
    --------
    >>> process = MultivariateExponentialHawkes(
    ...     mu=[0.4, 0.2], excitation=[[0.3, 0.1], [0.5, 0.2]], beta=1.5, rng=0
    ... )
    >>> process.simulate(100)
    >>> process.events.shape
    (2, 100)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        mu: Any,
        excitation: Any,
        beta: float,
        *,
        rng: SeedLike = None,
    ) -> None:
        decay = float(beta)
        if not decay > 0:
            raise ValueError(f"beta must be positive, got {beta!r}")

        def kernel(s: Any) -> np.ndarray:
            return np.asarray(np.exp(-decay * np.asarray(s, dtype=float)), dtype=float)

        super().__init__(mu, excitation, kernel, rng=rng)

        # After super(), so `excitation` has already been checked square, sized
        # against `mu` and non-negative -- an eigensolve on a ragged or NaN-laden
        # matrix reports something useless or raises LinAlgError, and neither is
        # the error this should give. The object is discarded on raise.
        radius = spectral_radius(self.excitation / decay)
        if radius >= 1:
            raise ValueError(
                f"Stability condition violated: the spectral radius of "
                f"excitation/beta = {radius:.4f} >= 1. The process will not be "
                "stationary. This is the matrix form of ExponentialHawkes's "
                "alpha/beta condition and reduces to it at one type."
            )
        self.beta = decay
