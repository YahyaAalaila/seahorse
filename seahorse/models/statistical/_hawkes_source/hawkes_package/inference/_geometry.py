r"""Distance tensors that do not depend on the parameters.

The spatio-temporal likelihood is dominated by geometry, not by arithmetic. One
call to ``SpatioTemporalHawkesProcess._integrated_intensity`` costs a spatial
kernel evaluation at every quadrature node for every past event, and every one
of those evaluations begins with a call to ``domain.distance`` -- 6.9 us on a
``Torus2D`` and 79 us on a ``FundamentalDomain`` with a warm window cache, both
of them Python-level loops. Multiplied out over a fit, that is the difference
between minutes and days.

But none of those distances depend on ``theta``. The domain is fixed, the
quadrature nodes are fixed, and the observed event locations are data. Only the
kernel evaluated *on* the distances moves. So they are computed once:

.. math::

    D[j, i, g] = d(\text{node}_j,\ g \cdot x_i), \qquad
    E[i, j, g] = d(x_i,\ g \cdot x_j)

with :math:`g` running over the images the periodised kernel sums over, which is
a single identity when the kernel is not periodised. Afterwards a full
log-likelihood is a handful of vectorized array operations.

**The images have to match the simulator's exactly.** Not approximately: a
periodised kernel summed over a different set of images is a different kernel,
and a likelihood built on the wrong one produces a posterior that is wrong
without being obviously wrong. That is why the image set comes from
:func:`~hawkes_package.spatio_temporal.kernels.image_distance_fn`, the same
function :func:`~hawkes_package.make_periodic` builds the simulator's kernel
from, rather than from a second copy of the rule.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..spatio_temporal import _integration
from ..spatio_temporal.kernels import image_distance_fn
from .models import SpatialComponents

__all__ = ["GeometryCache", "build_geometry", "extend_geometry"]

#: Default ceiling on one cache, in bytes. Reached at roughly 1000 nodes, 200
#: events and 49 images -- a ``Torus2D`` fit of ordinary size -- so it is a real
#: limit rather than a formality, and it is better met with a message naming the
#: three numbers than with the machine swapping.
DEFAULT_MAX_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class GeometryCache:
    """Precomputed, parameter-free distances for one history on one domain.

    Attributes
    ----------
    nodes : numpy.ndarray
        Quadrature nodes, shape ``(m, ndim)`` -- the same restricted, reweighted
        rule the process builds, so the cached integral is the process's
        integral and not a second opinion about it.
    weights : numpy.ndarray
        Quadrature weights, shape ``(m,)``, strictly positive.
    node_event : numpy.ndarray
        ``D``, shape ``(m, n, n_images_total)``.
    event_event : numpy.ndarray
        ``E``, shape ``(n, n, n_images_total)``.
    times : numpy.ndarray
        The event times the tensors were built for, shape ``(n,)``. Kept so a
        mismatched history is caught rather than silently mixed in.
    points : numpy.ndarray
        The event locations, shape ``(ndim, n)``.
    """

    nodes: np.ndarray
    weights: np.ndarray
    node_event: np.ndarray
    event_event: np.ndarray
    times: np.ndarray
    points: np.ndarray

    @property
    def n_events(self) -> int:
        """Number of events the cache covers."""
        return int(self.times.size)

    @property
    def n_nodes(self) -> int:
        """Number of quadrature nodes."""
        return int(self.weights.size)

    @property
    def nbytes(self) -> int:
        """Total size of the two distance tensors."""
        return int(self.node_event.nbytes + self.event_event.nbytes)

    def matches(self, times: Any) -> bool:
        """Whether `times` begins with exactly the events this cache holds."""
        wanted = np.asarray(times, dtype=float).ravel()
        return bool(
            wanted.size >= self.n_events and np.array_equal(wanted[: self.n_events], self.times)
        )


def quadrature_for(components: SpatialComponents) -> _integration.TensorQuadrature:
    """Build the same spatial rule the process would build.

    Deliberately a call into
    :mod:`~hawkes_package.spatio_temporal._integration` rather than a rule
    assembled here: the masking by ``contains`` and the reweighting by
    ``volume_element`` are what let a domain be a proper subset of its bounding
    box, and a likelihood integrating over the box while the simulator
    integrates over the domain would differ by the ratio of their areas -- on a
    hexagon, a factor of 1.30.
    """
    domain = components.domain
    return _integration.restrict(
        _integration.build(domain.bounds, components.nodes_per_axis),
        domain.contains,
        domain.volume_element,
    )


def _distance_block(
    distances: Any,
    left: np.ndarray,
    right: np.ndarray,
    n_images: int,
) -> np.ndarray:
    """Distances from every row of `left` to every image of every row of `right`."""
    block = np.empty((left.shape[0], right.shape[0], n_images), dtype=float)
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            block[i, j] = distances(x, y)
    return block


def _distance_rule(components: SpatialComponents) -> Any:
    """Return the map from a pair of points to the distances the kernel sees.

    One place, so that the cache and the extension cannot disagree, and so that
    "periodised" is decided once rather than at every call site.
    """
    if components.n_images is None:
        domain = components.domain

        def geodesic(x: Any, y: Any) -> np.ndarray:
            return np.array([domain.distance(x, y)], dtype=float)

        return geodesic
    return image_distance_fn(components.domain, components.n_images)


def _check_budget(shape: tuple[int, ...], extra: int, max_bytes: int, what: str) -> None:
    """Refuse a tensor larger than the caller allowed, saying which axis is big."""
    needed = int(np.prod(shape)) * 8 + extra
    if needed > max_bytes:
        raise MemoryError(
            f"the {what} tensor would need {needed / 1024**2:.0f} MiB for shape {shape} "
            f"(nodes x events x images), above the {max_bytes / 1024**2:.0f} MiB budget. "
            "Lower n_quad, fit a shorter history, drop n_images, or raise "
            "max_cache_bytes if the memory is really there."
        )


def build_geometry(
    components: SpatialComponents,
    times: Any,
    points: Any,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> GeometryCache:
    """Compute the distance tensors for one history.

    Parameters
    ----------
    components : SpatialComponents
        The domain, and whether the kernel is periodised.
    times : array_like of shape (n,)
        Event times, only carried so :meth:`GeometryCache.matches` can check
        them.
    points : array_like of shape (ndim, n)
        Event locations.
    max_bytes : int
        Ceiling on the two tensors together.

    Returns
    -------
    GeometryCache

    Raises
    ------
    MemoryError
        If the tensors would exceed `max_bytes`.
    """
    rule = quadrature_for(components)
    event_times = np.asarray(times, dtype=float).ravel()
    locations = np.asarray(points, dtype=float).reshape(-1, event_times.size)

    distances = _distance_rule(components)
    # One probe rather than a formula: the image count is 2k+1 on a circle,
    # (2k+1)^2 on a torus and whatever the deck group happens to enumerate on a
    # fundamental domain, and hard-coding any of those would be wrong for the
    # other two.
    probe = np.asarray(components.domain.interior_point, dtype=float)
    n_terms = int(np.asarray(distances(probe, probe), dtype=float).size)

    m, n = rule.nodes.shape[0], event_times.size
    _check_budget((m, n, n_terms), n * n * n_terms * 8, max_bytes, "node-event distance")

    coordinates = locations.T
    node_event = _distance_block(distances, rule.nodes, coordinates, n_terms)
    event_event = _distance_block(distances, coordinates, coordinates, n_terms)

    return GeometryCache(
        nodes=rule.nodes,
        weights=rule.weights,
        node_event=node_event,
        event_event=event_event,
        times=event_times,
        points=locations,
    )


def extend_geometry(
    cache: GeometryCache,
    components: SpatialComponents,
    times: Any,
    points: Any,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> GeometryCache:
    """Grow `cache` to cover a longer history sharing its prefix.

    The online use is a sequence of blocks over one growing history, so
    rebuilding from scratch each time would make the geometry cost quadratic in
    the number of blocks. Only the new columns are computed.

    Raises
    ------
    ValueError
        If `times` does not extend the cache's own event times. Silently
        rebuilding on a mismatch would hide a caller that changed history
        underneath a running fit.
    """
    wanted = np.asarray(times, dtype=float).ravel()
    if not cache.matches(wanted):
        raise ValueError(
            "the history does not extend the one this geometry cache was built for: "
            f"it holds {cache.n_events} event(s) that are not a prefix of the "
            f"{wanted.size} given. Build a fresh cache."
        )
    if wanted.size == cache.n_events:
        return cache

    locations = np.asarray(points, dtype=float).reshape(-1, wanted.size)
    old_n, new_n = cache.n_events, wanted.size
    n_terms = cache.node_event.shape[2]
    _check_budget(
        (cache.n_nodes, new_n, n_terms),
        new_n * new_n * n_terms * 8,
        max_bytes,
        "node-event distance",
    )

    distances = _distance_rule(components)
    fresh = locations.T[old_n:]
    node_event = np.concatenate(
        [cache.node_event, _distance_block(distances, cache.nodes, fresh, n_terms)], axis=1
    )

    # The event-event tensor grows on both axes: the new events see the old
    # ones, and the old ones see the new. Only the new *rows* are ever read by
    # the log-sum -- an event's intensity depends on its predecessors -- but the
    # block is kept square so `event_event[i, :i]` stays the natural slice.
    event_event = np.empty((new_n, new_n, n_terms), dtype=float)
    event_event[:old_n, :old_n] = cache.event_event
    old_points = locations.T[:old_n]
    event_event[:old_n, old_n:] = _distance_block(distances, old_points, fresh, n_terms)
    event_event[old_n:, :] = _distance_block(distances, fresh, locations.T, n_terms)

    return GeometryCache(
        nodes=cache.nodes,
        weights=cache.weights,
        node_event=node_event,
        event_event=event_event,
        times=wanted,
        points=locations,
    )
