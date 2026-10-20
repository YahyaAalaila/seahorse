"""Log-space importance weights, effective sample size and resampling.

Everything here takes **normalised log weights** and nothing takes probabilities.
That is a deliberate narrowing of the interface, because the failure it prevents
is invisible. An effective sample size computed from unnormalised weights is
still a plausible number -- it is just a different, larger one, and a cloud that
has collapsed onto a single particle can be made to report a healthy ESS by
forgetting to normalise. Reading only log weights whose exponentials sum to one
makes that unrepresentable rather than merely discouraged.

Working in logs is not a stylistic preference either. A block of a few hundred
events moves the log-likelihood by hundreds of nats, so ``exp`` of an
un-shifted log weight underflows to exactly zero for every particle, and the
normalisation is then ``0 / 0``.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "effective_sample_size",
    "log_normalise",
    "log_sum_exp",
    "multinomial",
    "systematic",
    "unique_fraction",
]


def log_sum_exp(values: Any) -> float:
    """``log(sum(exp(values)))``, computed by shifting out the maximum.

    Returns ``-inf`` when every entry is ``-inf``, which is the right answer and
    the one a naive implementation turns into ``nan`` by subtracting ``-inf``
    from itself.

    Examples
    --------
    >>> round(log_sum_exp([-10000.0, -10001.0]), 6)
    -9999.686738
    """
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        return -np.inf
    largest = float(np.max(array))
    if not np.isfinite(largest):
        # All -inf (weightless cloud) or an +inf that has already gone wrong.
        return largest
    return largest + float(np.log(np.sum(np.exp(array - largest))))


def log_normalise(log_weights: Any) -> np.ndarray:
    """Shift `log_weights` so their exponentials sum to one.

    Raises
    ------
    ValueError
        If every weight is ``-inf``, or any is ``nan``. Both mean the cloud has
        no probability mass left; normalising would produce ``nan`` everywhere
        and the sampler would carry on for several steps before anything failed.
    """
    array = np.asarray(log_weights, dtype=float).ravel().copy()
    if np.any(np.isnan(array)):
        raise ValueError(
            f"{int(np.sum(np.isnan(array)))} of {array.size} log weights are nan; the "
            "likelihood returned nan for at least one particle rather than -inf"
        )
    total = log_sum_exp(array)
    if not np.isfinite(total):
        raise ValueError(
            "every particle has zero weight: the whole cloud is impossible under the "
            "data. Either the prior misses the truth entirely, or the model cannot "
            "produce this history at any parameter it admits."
        )
    return array - total


def effective_sample_size(log_weights: Any) -> float:
    r"""Kish's effective sample size, :math:`1 / \sum w_i^2`.

    Between 1 and ``N``: ``N`` for equal weights, 1 when one particle carries
    everything. It is the standard trigger for resampling, and also the standard
    thing to be lulled by -- it measures the *weights*, so a cloud of ``N``
    copies of one particle after a resample reports a perfect ESS of ``N``. That
    is why :func:`unique_fraction` is recorded alongside it.

    Parameters
    ----------
    log_weights : array_like
        Normalised log weights.
    """
    array = np.asarray(log_weights, dtype=float).ravel()
    if array.size == 0:
        return 0.0
    return float(np.exp(-log_sum_exp(2.0 * array)))


def unique_fraction(indices: Any, n_particles: int) -> float:
    """Fraction of the cloud's slots filled by distinct particles."""
    array = np.asarray(indices).ravel()
    if n_particles <= 0:
        return 0.0
    return float(np.unique(array).size) / float(n_particles)


def systematic(log_weights: Any, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: one uniform variate, ``N`` evenly spaced positions.

    Unbiased -- particle ``i`` is expected to be drawn ``N w_i`` times -- with
    conditional variance no worse than multinomial's (Douc and Cappe, 2005), and
    ``O(N)``. The property that matters most here is the least glamorous one: it
    consumes **exactly one** random variate whatever ``N`` is, so the number of
    draws a fit makes is a function of how many times it resampled and not of
    how large the cloud is, and two runs that resample at the same steps stay on
    the same stream.

    Parameters
    ----------
    log_weights : array_like
        Normalised log weights, shape ``(N,)``.
    rng : numpy.random.Generator
        Source of the single variate.

    Returns
    -------
    numpy.ndarray
        Ancestor indices, shape ``(N,)``, sorted ascending.
    """
    weights = np.exp(np.asarray(log_weights, dtype=float).ravel())
    n = weights.size
    positions = (rng.uniform() + np.arange(n)) / n
    edges = np.cumsum(weights)
    # The last edge is 1 only up to rounding, and a position past it would index
    # off the end. Pinning it costs nothing and removes the branch.
    edges[-1] = 1.0
    return np.asarray(np.searchsorted(edges, positions, side="right"), dtype=int)


def multinomial(log_weights: Any, rng: np.random.Generator) -> np.ndarray:
    """Multinomial resampling: ``N`` independent draws from the weights.

    Ships for comparison, not for use. It is unbiased like
    :func:`systematic` and strictly noisier, so the only reason to reach for it
    is a test that wants to see the difference in variance.
    """
    weights = np.exp(np.asarray(log_weights, dtype=float).ravel())
    n = weights.size
    edges = np.cumsum(weights)
    edges[-1] = 1.0
    return np.asarray(np.searchsorted(edges, rng.uniform(size=n), side="right"), dtype=int)
