"""Spatial kernel utilities for spatio-temporal Hawkes processes.

Provides :func:`make_periodic`, which wraps an isotropic spatial kernel so it
respects the periodic structure of a :class:`SpatialDomain` by summing the
contributions of image points.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from .._numerics import as_float, as_point
from .domains import SpatialDomain

__all__ = ["PairwiseKernel", "check_image_sum", "image_distance_fn", "make_periodic"]


class PairwiseKernel:
    """A spatial kernel that consumes both endpoints rather than a distance.

    :class:`~hawkes_package.spatio_temporal.process.SpatioTemporalHawkesProcess`
    normally reduces a pair of points to a geodesic distance before calling
    `spatial`. An image sum cannot be written that way -- on a torus the
    geodesic distance does not determine the lattice sum -- so such a kernel
    declares itself by carrying ``pairwise = True`` and is then called with the
    two points.

    Any callable may opt in the same way; it need not be an instance of this
    class.
    """

    pairwise = True

    def __init__(self, fn: Callable[[Any, Any], Any], /) -> None:
        self._fn = fn

    def __call__(self, x: Any, y: Any) -> float:
        """Evaluate the kernel between two domain points."""
        return as_float(self._fn(x, y))


KernelFn = Callable[[Any], Any]


def make_periodic(kernel_fn: KernelFn, domain: SpatialDomain, n_images: int = 3) -> PairwiseKernel:
    r"""Return a periodised version of `kernel_fn` that sums over image points.

    For a domain with discrete translational symmetry (:class:`Circle`,
    :class:`Torus2D`) the true kernel between ``x`` and ``y`` is

    .. math::

        K(x, y) = \sum_{\text{offset} \in \text{images}}
                  \mathrm{kernel\_fn}\!\left( \| x - y - \text{offset} \| \right).

    With ``n_images=3`` the sum covers offsets up to three periods in each
    direction, which is ample for a rapidly decaying kernel.

    Parameters
    ----------
    kernel_fn : callable
        Isotropic kernel evaluated at a non-negative scalar distance.
    domain : SpatialDomain
        The domain whose periodicity structure to respect. :class:`Circle` and
        :class:`Torus2D` carry hand-written lattices; any other domain that
        declares a deck group through :meth:`~hawkes_package.SpatialDomain.orbit`
        — as :class:`~hawkes_package.FundamentalDomain` does — is periodised by
        summing over that orbit. A domain with no deck group falls back to a single evaluation at
        ``domain.distance(x, y)``.
    n_images : int
        Number of image periods in each direction.

    Returns
    -------
    callable
        The periodised kernel, taking two domain points.

    Notes
    -----
    .. versionchanged:: 0.3.0
       The ``orbit`` branch was added. Before 0.3.0 only :class:`Circle` and
       :class:`Torus2D` were recognised and every other domain received the
       unperiodised kernel — which, wherever the kernel has not decayed by the
       boundary, is a different process rather than a coarser one.

    Examples
    --------
    >>> from hawkes_package import Circle
    >>> kernel = make_periodic(lambda d: np.exp(-(d**2)), Circle())
    >>> round(kernel(0.3, -1.2), 6)
    0.105399
    """
    distances = image_distance_fn(domain, n_images)
    if getattr(distances, "sums_over_an_orbit", False):
        check_image_sum(kernel_fn, domain, n_images)

    def periodised(x: Any, y: Any) -> float:
        # Built-in `sum`, not a `+=` loop: since CPython 3.12 it sums floats
        # with Neumaier compensation, and the image sum is exactly the shape
        # that benefits -- a few dozen terms spanning several orders of
        # magnitude, the far images being the small ones. Two of the four
        # branches below already summed this way; using it for all of them
        # leaves `Circle` bit-identical and moves `Torus2D` -- 49 terms rather
        # than 7 -- by one unit in the last place, which `CHANGELOG.md` and
        # `docs/migration.md` record.
        return float(sum(as_float(kernel_fn(distance)) for distance in distances(x, y)))

    return PairwiseKernel(periodised)


def image_distance_fn(domain: SpatialDomain, n_images: int = 3) -> Callable[[Any, Any], np.ndarray]:
    r"""Return the map from a pair of points to the distances a periodised kernel sums.

    Factored out of :func:`make_periodic` because two callers need to agree on
    it exactly. The simulator reaches the image sum by *evaluating* the
    periodised kernel; :mod:`hawkes_package.inference._geometry` reaches it by
    caching the distances once and evaluating a parameterised kernel on them
    thousands of times. If those two ever computed a different set of images,
    the cached likelihood would be fitting a different process from the one the
    simulator draws -- and the difference would show up as a plausible but wrong
    posterior rather than as an error.

    The returned callable carries ``sums_over_an_orbit`` when its images come
    from :meth:`~hawkes_package.SpatialDomain.orbit`, which is the branch whose
    truncation :func:`check_image_sum` has something to say about.

    Parameters
    ----------
    domain : SpatialDomain
        :class:`~hawkes_package.Circle` and :class:`~hawkes_package.Torus2D`
        carry hand-written lattices; anything else declaring a deck group
        through :meth:`~hawkes_package.SpatialDomain.orbit` is handled by that.
        A domain with neither gives the single geodesic distance, which is the
        unperiodised kernel.
    n_images : int
        Images per direction.

    Returns
    -------
    callable
        Takes two domain points and returns a 1-D array of distances.

    .. versionadded:: 0.5.0
    """
    # Imported here rather than at module scope: domains.py is imported by
    # process.py, which also imports this module.
    from .domains import Circle, Torus2D

    if isinstance(domain, Circle):
        period = domain.volume

        def circle_distances(x: Any, y: Any) -> np.ndarray:
            # Reduce to the canonical offset first. Summing images about a raw
            # difference leaves the nearest image outside the window once
            # |x - y| exceeds n_images * period, so the kernel would decay to
            # zero instead of staying periodic.
            offset = (as_point(x, 1)[0] - as_point(y, 1)[0] + period / 2) % period - period / 2
            return np.array(
                [abs(offset - n * period) for n in range(-n_images, n_images + 1)],
                dtype=float,
            )

        return circle_distances

    if isinstance(domain, Torus2D):
        length_1, length_2 = domain.width, domain.height
        periods = np.array([length_1, length_2])
        offsets = [
            np.array([n1 * length_1, n2 * length_2])
            for n1 in range(-n_images, n_images + 1)
            for n2 in range(-n_images, n_images + 1)
        ]

        def torus_distances(x: Any, y: Any) -> np.ndarray:
            # Canonical offset per axis, for the same reason as on the circle.
            delta = as_point(x, 2) - as_point(y, 2)
            delta = (delta + periods / 2) % periods - periods / 2
            # A norm per offset rather than one batched `norm(..., axis=1)`
            # call: the batched form reduces along a different code path and
            # can differ in the last bit, which is enough to flip a thinning
            # acceptance and change a realisation that was published.
            return np.array(
                [float(np.linalg.norm(delta - offset)) for offset in offsets], dtype=float
            )

        return torus_distances

    # Any other domain that declares a deck group -- `FundamentalDomain` does --
    # gets the same image sum, driven by `orbit` instead of a hand-written
    # lattice. Reached only after the two branches above, so neither of them
    # changes behaviour.
    ndim = np.asarray(domain.bounds, dtype=float).shape[0]
    if domain.orbit(np.asarray(domain.interior_point, dtype=float), n_images) is not None:

        def orbit_distances(x: Any, y: Any) -> np.ndarray:
            # Canonical representatives first, for the same reason as on the
            # circle: measured from unreduced lifts the nearest image can fall
            # outside the window, and the kernel would decay to zero instead of
            # staying periodic.
            here = as_point(domain.wrap(x), ndim)
            images = domain.orbit(domain.wrap(y), n_images)
            if images is None:  # pragma: no cover - the branch above proves otherwise
                return np.array([domain.distance(x, y)], dtype=float)
            # `lift_distance`, not a chart norm: the sum runs over images in the
            # universal cover, and only where the covering map is a local
            # isometry -- every flat domain, no curved one -- is the chart norm
            # the distance between them.
            return np.array([domain.lift_distance(here, im) for im in images], dtype=float)

        orbit_distances.sums_over_an_orbit = True  # type: ignore[attr-defined]
        return orbit_distances

    def geodesic_distance(x: Any, y: Any) -> np.ndarray:
        return np.array([domain.distance(x, y)], dtype=float)

    return geodesic_distance


def check_image_sum(
    kernel_fn: KernelFn,
    domain: SpatialDomain,
    n_images: int,
    *,
    rtol: float = 1e-2,
) -> None:
    r"""Warn if the image sum is truncated somewhere the kernel has not decayed.

    Periodising by summing over a truncated orbit is only an approximation to
    the true sum, and how good an approximation depends on a race between the
    kernel's decay and the deck group's growth. In a flat geometry the group
    grows polynomially and any ordinary kernel wins the race comfortably. In a
    **hyperbolic** one the number of deck elements at distance :math:`R` grows
    like :math:`e^{R}`, so the tail beyond the window is of order
    :math:`e^{R} \sup_{d > R} \kappa_s(d)` and converges only for a kernel
    decaying faster than :math:`e^{-d}`. A Gaussian qualifies; an exponential
    with rate above one qualifies; a power law does not, and nothing about a
    power law announces that it does not.

    So the diagnostic is the same shape as
    :func:`~hawkes_package.spatio_temporal._integration.check_resolution`:
    compare what one more ring of images adds against what the sum already
    holds. A last ring worth more than `rtol` of the total means the sum has
    been cut where the kernel is still contributing, and the simulated
    excitation is smaller than the model asks for -- which, since the location
    sampler and the thinning bound both use it, is not a visible error but a
    quietly different process.

    Deterministic, and evaluated once at construction rather than on every one
    of the hundreds of thousands of kernel evaluations a simulation runs.

    .. versionadded:: 0.4.0
    """
    here = np.asarray(domain.interior_point, dtype=float)

    def ring_total(depth: int) -> float:
        images = domain.orbit(here, depth)
        if images is None:  # pragma: no cover - only called where orbit exists
            return 0.0
        return float(sum(as_float(kernel_fn(domain.lift_distance(here, im))) for im in images))

    total = ring_total(n_images)
    wider = ring_total(n_images + 1)
    added = abs(wider - total)
    scale = max(abs(wider), np.finfo(float).tiny)
    if added / scale > rtol:
        warnings.warn(
            f"the image sum over {type(domain).__name__} is truncated where the spatial kernel "
            f"has not decayed: one more ring of images adds {100 * added / scale:.1f}% to it. "
            f"The periodised kernel is smaller than the true one by at least that much, and by "
            f"more if the deck group grows exponentially. Raise n_images above {n_images}, or "
            "use a faster-decaying kernel.",
            UserWarning,
            stacklevel=3,
        )
