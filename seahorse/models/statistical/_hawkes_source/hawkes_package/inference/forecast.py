"""Forecasting from a fitted posterior, by simulating forward.

A posterior over parameters becomes a forecast by simulating the process ahead
from the observed history, once per parameter draw. Two things about that are
easy to get wrong and neither announces itself.

**The parameter uncertainty has to be carried through.** Simulating many paths
from the posterior *mean* gives a predictive distribution that is too narrow --
it holds the sampling variability of the process but none of the uncertainty
about which process it is. Drawing a fresh particle per path is what makes the
result a posterior predictive rather than a plug-in one, and at a few hundred
events the difference is not subtle.

**The forecast starts at the end of the observation window, not at the last
event.** Those are different times, and the gap between them is data: it says
nothing happened there. Continuing from the last event instead would replay that
gap as if it were unobserved, so the first forecast event would arrive too
early, systematically. This is what
:meth:`~hawkes_package.base.HawkesProcess.simulate_until`'s ``start`` argument
exists for -- and a fixed-count simulation could not express it at all, because
"no events in the whole horizon" is an outcome it cannot produce.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import SeedLike
from .likelihood import History, _bind_history
from .models import ProcessModel
from .smc import ParticleCloud

__all__ = ["posterior_predictive", "predictive_counts", "predictive_interval"]


def posterior_predictive(
    model: ProcessModel,
    cloud: ParticleCloud,
    history: History,
    *,
    horizon: float,
    n_paths: int = 200,
    rng: SeedLike = None,
) -> list[np.ndarray]:
    """Simulate `n_paths` continuations of `history`, one per posterior draw.

    Parameters
    ----------
    model : ProcessModel
        The model the cloud was fitted under.
    cloud : ParticleCloud
        The posterior. Particles are drawn in proportion to their weights, so a
        cloud that has not been resampled is used correctly without one.
    history : History
        The observed data. **Not modified**: every path builds its own process
        and conditions a fresh copy of the record, so the array a caller holds
        is the array it keeps.
    horizon : float
        Length of the forecast window, measured from ``history.end``.
    n_paths : int
        Number of paths. Each is a full simulation, so the cost is `n_paths`
        times whatever one simulation of this model over `horizon` costs -- for
        a spatio-temporal model that includes a Metropolis-Hastings run per
        event and a quadrature rule built per path, which is seconds rather
        than milliseconds.
    rng : None, int or numpy.random.Generator
        Source of randomness.

    Returns
    -------
    list of numpy.ndarray
        One entry per path, holding **only the new events**: shape ``(k,)`` for
        a temporal model and ``(ndim + 1, k)`` for a spatio-temporal one, in the
        layout :attr:`~hawkes_package.base.HawkesProcess.events` uses. An empty
        entry is a genuine outcome, not a failure.

    Examples
    --------
    >>> paths = posterior_predictive(model, cloud, history, horizon=5.0, n_paths=8, rng=0)
    >>> all(path.size == 0 or path.min() > history.end for path in paths)
    True

    .. versionadded:: 0.5.0
    """
    span = float(horizon)
    if not span > 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    paths_wanted = int(n_paths)
    if paths_wanted < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths}")

    generator = np.random.default_rng(rng)
    indices = generator.choice(cloud.n_particles, size=paths_wanted, p=cloud.weights)
    observed = history.n_events
    end = history.end

    paths: list[np.ndarray] = []
    for index in indices:
        process = model(cloud.theta[index], rng=generator)
        _bind_history(process, history)
        process.simulate_until(end + span, start=end)
        record = np.asarray(process.events, dtype=float)
        paths.append(record[observed:] if record.ndim == 1 else record[:, observed:])
    return paths


def predictive_counts(paths: list[np.ndarray]) -> np.ndarray:
    """Count the new events on each path from :func:`posterior_predictive`."""
    return np.array([path.size if path.ndim == 1 else path.shape[1] for path in paths], dtype=int)


def predictive_interval(
    paths: list[np.ndarray],
    times: Any,
    *,
    level: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the predictive band for the cumulative event count at each of `times`.

    Parameters
    ----------
    paths : list of numpy.ndarray
        From :func:`posterior_predictive`.
    times : array_like
        Times to evaluate the running count at.
    level : float
        Coverage of the equal-tailed band.

    Returns
    -------
    lower, median, upper : numpy.ndarray
        Each shape ``(len(times),)``. Counts, so integers in value if not in
        dtype -- the quantiles interpolate between paths.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    grid = np.asarray(times, dtype=float).ravel()
    counts = np.empty((len(paths), grid.size), dtype=float)
    for i, path in enumerate(paths):
        event_times = path if path.ndim == 1 else path[0]
        counts[i] = np.searchsorted(np.sort(event_times), grid, side="right")

    tail = 0.5 * (1.0 - level)
    return (
        np.quantile(counts, tail, axis=0),
        np.quantile(counts, 0.5, axis=0),
        np.quantile(counts, 1.0 - tail, axis=0),
    )
