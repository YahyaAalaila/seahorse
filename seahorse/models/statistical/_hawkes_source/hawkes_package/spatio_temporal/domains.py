#!/usr/bin/env python3
r"""
Spatial domain abstractions for spatio-temporal Hawkes processes.

Each domain defines how distances are measured, how coordinates are wrapped to
stay within the domain, and how to sample uniformly from the domain.

:class:`Circle` and :class:`Torus2D` write out one quotient each by hand, and
:class:`Sphere` writes out the one closed surface that is *not* a quotient --
it is simply connected, so a deck group is the wrong tool for it.
:class:`FundamentalDomain` is the general construction: a convex geodesic
polygon in a constant-curvature model space, glued to itself along its
boundary. Together with the three model spaces in
:mod:`~hawkes_package.spatio_temporal._model` it reaches **every closed
surface** -- orientable or not, of any genus.

Which model space a surface needs is not a choice. By uniformisation the sign
of the Euler characteristic fixes it:

=================  =========================  ==========  ==========================
:math:`\chi`       surface                    geometry    built by
=================  =========================  ==========  ==========================
:math:`> 0`        :math:`S^2`                spherical   :class:`Sphere`
:math:`> 0`        :math:`\mathbb{RP}^2`      spherical   ``projective_plane()``
:math:`= 0`        torus                      flat        ``rectangle()``, ``hexagon()``
:math:`= 0`        Klein bottle               flat        ``klein_bottle()``
:math:`< 0`        genus :math:`\ge 2`        hyperbolic  ``genus(g)``
:math:`< 0`        :math:`N_k`, :math:`k > 2`  hyperbolic  ``crosscaps(k)``
=================  =========================  ==========  ==========================

The last four are classmethods of :class:`FundamentalDomain`.

Gauss-Bonnet, :math:`\int K\,dA = 2\pi\chi`, is what ties the three columns
together, and every :class:`FundamentalDomain` checks it at construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .._numerics import as_point
from . import _gluing, _integration
from ._model import EuclideanPlane, HyperbolicPlane, ModelSpace, SphericalPlane, isometry_between

__all__ = [
    "Circle",
    "FundamentalDomain",
    "Polygon",
    "Rectangle",
    "SpatialDomain",
    "Sphere",
    "Torus2D",
]


class SpatialDomain(ABC):
    """Abstract base class for spatial domains.

    The spatial integral is a quadrature rule over :attr:`bounds`, restricted to
    the nodes :meth:`contains` admits and weighted by :meth:`volume_element`.
    Subclasses must therefore satisfy

    ``volume == integral of volume_element over {x in bounds : contains(x)}``.

    The optional hooks default to "the domain fills its bounding box, with the
    flat chart measure", which reduces that requirement to the older
    ``volume == prod(bounds widths)`` and leaves an existing subclass — which
    overrides none of them — behaving exactly as before.

    .. versionchanged:: 0.3.0
       Before 0.3.0 the contract was ``volume == prod(bounds widths)`` exactly,
       enforced by a :class:`ValueError` raised when a process was built on the
       domain. A domain may now be a proper subset of its bounding box, and that
       check is now made against the summed quadrature weights — see
       :class:`~hawkes_package.SpatioTemporalHawkesProcess`.

    Attributes
    ----------
    periodic : bool
        Whether :meth:`wrap` folds coordinates through a translation symmetry,
        as opposed to clipping them. Only a genuinely periodic domain may have
        MCMC proposals folded rather than rejected: folding is a symmetric move
        on the quotient and therefore reversible, whereas folding with a
        clipping map would pile every draw onto the boundary. Defaults to
        ``False``, so a third-party domain is treated conservatively.
    has_boundary : bool
        Whether the domain has an edge that excitation can spill over. A closed
        surface does not: the spatial kernel's mass over the domain is the same
        wherever an event lands, so an event's realised productivity does not
        depend on its position. A domain with a boundary breaks that, and
        :class:`~hawkes_package.SpatioTemporalHawkesProcess` renormalises the
        spatial kernel per event to compensate.

        **This is not the negation of** :attr:`periodic`. ``Sphere`` and
        ``FundamentalDomain`` are both ``periodic = False`` -- the first is not
        a quotient at all, the second is glued by isometries that are not
        translations -- and neither has a boundary. Keying the correction off
        ``periodic`` would rescale their intensity for no reason.

        Defaults to ``False``, so every domain that predates 1.0.0 takes the
        identical path and no previously produced number moves.

        .. versionadded:: 1.0.0
    """

    periodic: bool = False
    has_boundary: bool = False

    @abstractmethod
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Geodesic distance between points x and y in the domain."""

    @abstractmethod
    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Map x to its canonical representative inside the domain."""

    @abstractmethod
    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a single point uniformly at random from the domain."""

    @property
    @abstractmethod
    def volume(self) -> float:
        """Measure (length/area/volume) of the domain."""

    @property
    @abstractmethod
    def bounds(self) -> np.ndarray:
        """Bounding box as array of shape (ndim, 2) — used for MCMC initialisation."""

    # ------------------------------------------------------------------
    # Optional hooks. Concrete, not abstract: a domain that fills its
    # bounding box with the flat measure and has no deck group -- which is
    # every domain that existed before 0.3.0 -- inherits all of them.
    # ------------------------------------------------------------------

    def contains(self, x: np.ndarray) -> bool:  # noqa: ARG002 - the interface, not this default
        """Whether `x` lies in the domain.

        Defaults to ``True``: the domain *is* its bounding box, so every node
        of the quadrature rule is admitted.

        .. versionadded:: 0.3.0
        """
        return True

    def volume_element(self, x: np.ndarray) -> float:  # noqa: ARG002 - as above
        """Return the measure density ``sqrt(det g)`` at `x`.

        Relates the chart measure the quadrature nodes live in to the true
        measure of the domain. Defaults to ``1.0``, the flat chart.

        Must be **strictly positive**: the Ogata thinning bound is valid because
        a pointwise-dominating integrand meets strictly positive weights on a
        shared node set, and a zero or negative weight breaks that.

        .. versionadded:: 0.3.0
        """
        return 1.0

    def orbit(self, y: np.ndarray, n_images: int = 3) -> list[np.ndarray] | None:  # noqa: ARG002
        """Return the images of `y` under the deck group, or ``None`` if there is no group.

        :func:`~hawkes_package.spatio_temporal.kernels.make_periodic` sums a
        kernel over this set. ``None`` — the default — means the domain declares
        no translational structure, and periodisation falls back to a single
        evaluation at :meth:`distance`.

        .. versionadded:: 0.3.0
        """
        return None

    def lift_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Distance between two *lifts*, in the universal cover rather than the quotient.

        :meth:`distance` minimises over the deck group; this does not. It is
        what an image sum needs, because the whole point of summing over
        :meth:`orbit` is to reach images the quotient distance has already
        minimised away.

        The default is the Euclidean norm in the chart, which is correct exactly
        when the chart *is* the cover and the covering map is a local isometry —
        true for every flat domain, and false for a curved one, where the chart
        norm is not a distance at all.

        .. versionadded:: 0.4.0
        """
        ndim = np.asarray(self.bounds, dtype=float).shape[0]
        return float(np.linalg.norm(as_point(x, ndim) - as_point(y, ndim)))

    @property
    def interior_point(self) -> np.ndarray:
        """A point known to lie inside the domain.

        The centre of :attr:`bounds` by default, which is correct whenever the
        domain fills its box. A domain that is a proper subset must override
        this: the box centre may well lie outside it, and this point is used to
        probe whether the spatial kernel is resolved by the quadrature rule.

        .. versionadded:: 0.3.0
        """
        return np.asarray(self.bounds, dtype=float).mean(axis=1)

    @property
    def max_distance(self) -> float:
        """An upper bound on :meth:`distance` between any two points of the domain.

        The half-diagonal of :attr:`bounds` by default, which is the diameter of
        a flat box-filling domain with all axes periodic. A curved domain must
        override it: on a curved surface the chart box says nothing about
        geodesic distance, and the Poincare disc chart of a genus-2 surface is
        barely a unit across while the surface itself is several units wide.

        .. versionadded:: 0.4.0
        """
        widths = np.asarray(self.bounds, dtype=float)
        widths = widths[:, 1] - widths[:, 0]
        return float(np.sqrt(np.sum((widths / 2) ** 2)))

    @property
    def nodes_per_axis(self) -> int:
        """Quadrature nodes per axis this domain needs to be measured correctly.

        The default is the dimension-based one every domain used before 0.4.0.
        A domain overrides it when its own geometry makes that default too
        coarse — which is not a matter of taste: a rule that mismeasures the
        domain's area scales the simulated event rate by exactly that factor,
        and the only symptom is a warning from the process constructor.

        A caller's explicit ``n_quad=`` still wins.

        .. versionadded:: 0.4.0
        """
        ndim = np.asarray(self.bounds, dtype=float).shape[0]
        return _integration.default_nodes_per_axis(ndim)


class Circle(SpatialDomain):
    """1-D circular domain [0, 2π·radius) with periodic boundary.

    Points are represented as arc lengths in [-π·radius, π·radius);
    :attr:`bounds` reports the closed bounding box used for quadrature.
    """

    periodic = True

    def __init__(self, radius: float = 1.0):
        self.radius = radius
        self._period = 2 * np.pi * radius

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Arc length between x and y, taking the shorter way round."""
        # as_point rather than .flat[0]: the latter silently accepted a
        # two-vector and measured only its first component.
        diff = abs(as_point(x, 1)[0] - as_point(y, 1)[0]) % self._period
        return float(min(diff, self._period - diff))

    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Fold x into the canonical half-open interval."""
        half = self._period / 2
        return (as_point(x, 1) + half) % self._period - half

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one point uniformly on the circle."""
        half = self._period / 2
        return rng.uniform(-half, half, size=(1,))

    @property
    def volume(self) -> float:
        """Circumference of the circle."""
        return self._period

    @property
    def bounds(self) -> np.ndarray:
        """Bounding interval, shape (1, 2)."""
        half = self._period / 2
        return np.array([[-half, half]])

    def orbit(self, y: np.ndarray, n_images: int = 3) -> list[np.ndarray]:
        """Images of `y` under translation by whole periods."""
        y = as_point(y, 1)
        return [y + n * self._period for n in range(-n_images, n_images + 1)]


class Torus2D(SpatialDomain):
    """Flat 2-D torus with periodic boundaries in both dimensions.

    The two side lengths are the periods of the lattice: points are represented
    in ``[-width/2, width/2) x [-height/2, height/2)``.

    .. versionchanged:: 0.4.0
       ``L1`` and ``L2`` are now ``width`` and ``height``, as arguments and as
       attributes.

    .. versionchanged:: 0.5.0
       The ``L1`` and ``L2`` spellings are gone. An unknown keyword now raises
       Python's own ``TypeError`` rather than one this class composed.
    """

    periodic = True

    def __init__(self, width: float = 2 * np.pi, height: float = 2 * np.pi) -> None:
        # Assigned as given, which is what the removed `_sides` resolver did --
        # the removal takes out the deprecation path and nothing else.
        self.width, self.height = width, height

    def _wrap_1d(self, x: float, period: float) -> float:
        """Fold a single coordinate into (-period/2, period/2]."""
        half = period / 2
        return (x + half) % period - half

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Euclidean distance on the flat torus, wrapping both axes."""
        x, y = as_point(x, 2), as_point(y, 2)
        dx = abs(float(x[0] - y[0])) % self.width
        dy = abs(float(x[1] - y[1])) % self.height
        dx = min(dx, self.width - dx)
        dy = min(dy, self.height - dy)
        return float(np.sqrt(dx**2 + dy**2))

    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Fold both coordinates into the canonical rectangle."""
        x = as_point(x, 2)
        return np.array(
            [
                self._wrap_1d(x[0], self.width),
                self._wrap_1d(x[1], self.height),
            ]
        )

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one point uniformly on the torus."""
        return np.array(
            [
                rng.uniform(-self.width / 2, self.width / 2),
                rng.uniform(-self.height / 2, self.height / 2),
            ]
        )

    @property
    def volume(self) -> float:
        """Surface area of the torus."""
        return self.width * self.height

    @property
    def bounds(self) -> np.ndarray:
        """Bounding rectangle, shape (2, 2)."""
        return np.array([[-self.width / 2, self.width / 2], [-self.height / 2, self.height / 2]])

    def orbit(self, y: np.ndarray, n_images: int = 3) -> list[np.ndarray]:
        """Lattice images of `y`, out to `n_images` periods on each axis."""
        y = as_point(y, 2)
        periods = np.array([self.width, self.height])
        return [
            y + np.array([n1, n2]) * periods
            for n1 in range(-n_images, n_images + 1)
            for n2 in range(-n_images, n_images + 1)
        ]


class Rectangle(SpatialDomain):
    """A bounded rectangular region with a real boundary.

    The first domain here that is **not** a closed surface. Every other one is
    boundaryless -- periodic, like :class:`Circle` and :class:`Torus2D`, or a
    quotient, like :class:`Sphere` and :class:`FundamentalDomain` -- so the
    spatial kernel's mass over the domain is the same wherever an event lands.
    Here it is not: an event near an edge spreads offspring into a region that
    is partly outside the domain, so its realised productivity is lower than an
    event in the middle.

    That is what :attr:`~SpatialDomain.has_boundary` announces, and what
    :class:`~hawkes_package.SpatioTemporalHawkesProcess` renormalises for. It is
    not a property this class can fix on its own: the correction needs the
    kernel and the quadrature rule, and a domain knows neither.

    Not to be confused with :meth:`FundamentalDomain.rectangle`, which glues the
    opposite sides together and is a **torus**. Same box, no boundary, different
    process.

    Parameters
    ----------
    width, height, ... : float
        Side lengths, one per axis. Any number of axes; two is the usual case,
        and one gives a bounded interval, the non-periodic counterpart of
        :class:`Circle`.
    origin : array_like, optional
        The lower corner. Defaults to centring the box on the origin.

    Examples
    --------
    >>> domain = Rectangle(4.0, 3.0)
    >>> domain.volume
    12.0
    >>> domain.has_boundary
    True
    >>> bool(domain.contains(np.array([1.0, 1.0])))
    True
    >>> bool(domain.contains(np.array([9.0, 1.0])))
    False

    .. versionadded:: 1.0.0
    """

    periodic = False
    has_boundary = True

    def __init__(self, *sides: float, origin: Any = None) -> None:
        lengths = np.asarray(sides, dtype=float).ravel()
        if lengths.size == 0:
            raise ValueError("a Rectangle needs at least one side length")
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError(f"every side length must be finite and positive, got {sides!r}")

        if origin is None:
            lower = -lengths / 2.0
        else:
            lower = np.asarray(origin, dtype=float).ravel()
            if lower.size != lengths.size:
                raise ValueError(
                    f"origin must have one entry per side: got {lower.size} for "
                    f"{lengths.size} side(s)"
                )
            if not np.all(np.isfinite(lower)):
                raise ValueError(f"origin must be finite, got {origin!r}")

        self.lengths = lengths
        self.lower = lower
        self.upper = lower + lengths
        self._ndim = int(lengths.size)

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Plain Euclidean distance -- nothing wraps."""
        x, y = as_point(x, self._ndim), as_point(y, self._ndim)
        return float(np.linalg.norm(x - y))

    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Clip `x` into the box.

        Clipping, not folding, which is why :attr:`~SpatialDomain.periodic` is
        ``False``: a clipping map is not reversible, so folding an MCMC proposal
        through it would pile every out-of-bounds draw onto the boundary instead
        of rejecting it. The location sampler rejects here, and that is correct
        for any domain rather than only for a translation quotient.
        """
        return np.asarray(np.clip(as_point(x, self._ndim), self.lower, self.upper), dtype=float)

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one point uniformly from the box."""
        return np.asarray(rng.uniform(self.lower, self.upper), dtype=float)

    def contains(self, x: np.ndarray) -> bool:
        """Whether `x` lies in the box.

        The inherited default returns ``True`` unconditionally, which is sound
        for the quadrature -- it only ever asks about nodes drawn from
        :attr:`bounds`, and this domain fills its box -- but it is not what the
        method says. A bounded domain is the first one here where a caller might
        reasonably ask about a point *outside* the box, so answer honestly.
        """
        point = as_point(x, self._ndim)
        return bool(np.all(point >= self.lower) and np.all(point <= self.upper))

    @property
    def volume(self) -> float:
        """Product of the side lengths."""
        return float(np.prod(self.lengths))

    @property
    def bounds(self) -> np.ndarray:
        """The box itself, shape ``(ndim, 2)``."""
        return np.stack([self.lower, self.upper], axis=1)

    @property
    def max_distance(self) -> float:
        """The full diagonal.

        Not the *half*-diagonal the base class defaults to. That default is the
        diameter of a flat domain with every axis periodic, where the farthest
        two points are half a period apart. Nothing wraps here, so the two
        opposite corners are the farthest pair and the diameter is twice as
        large -- and this value sizes the window an image sum searches.
        """
        return float(np.linalg.norm(self.lengths))


class Polygon(SpatialDomain):
    """A bounded convex polygon in the plane.

    The general case :class:`Rectangle` is the axis-aligned special case of, and
    the first domain here that is a **proper subset of its bounding box** *and*
    has a boundary. Those are separate properties and both matter: masking makes
    the quadrature measure the polygon rather than the box, and
    :attr:`~SpatialDomain.has_boundary` makes
    :class:`~hawkes_package.SpatioTemporalHawkesProcess` renormalise the
    excitation that would otherwise leak out of it.

    Convex only. :meth:`contains` is then a conjunction of half-planes, which is
    cheap, vectorises, and -- the point -- is decided entirely at construction.
    A predicate that recomputed anything per call could depend on the order
    callers happened to ask in, which is the failure that once cost this project
    five of ten CI jobs on identical code.

    Parameters
    ----------
    vertices : array_like of shape (k, 2)
        Corners in order, either winding. At least three, no three collinear
        enough to be degenerate, and convex.

    Raises
    ------
    ValueError
        If fewer than three vertices are given, if they are not finite, if the
        polygon has zero area, or if it is not convex.

    Examples
    --------
    >>> triangle = Polygon([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    >>> triangle.volume
    6.0
    >>> bool(triangle.contains(np.array([0.5, 0.5])))
    True
    >>> bool(triangle.contains(np.array([3.0, 3.0])))
    False

    .. versionadded:: 1.0.0
    """

    periodic = False
    has_boundary = True

    def __init__(self, vertices: Any) -> None:
        corners = np.asarray(vertices, dtype=float)
        if corners.ndim != 2 or corners.shape[1] != 2 or corners.shape[0] < 3:
            raise ValueError(
                f"a Polygon needs at least three (x, y) vertices, got shape {corners.shape}"
            )
        if not np.all(np.isfinite(corners)):
            raise ValueError("every vertex must be finite")

        # Shoelace, signed: its sign is the winding and its magnitude the area.
        rolled = np.roll(corners, -1, axis=0)
        twice_area = float(np.sum(corners[:, 0] * rolled[:, 1] - rolled[:, 0] * corners[:, 1]))
        if abs(twice_area) < 1e-12:
            raise ValueError("the polygon has zero area; its vertices are collinear")
        if twice_area < 0:
            corners = corners[::-1]
            rolled = np.roll(corners, -1, axis=0)
            twice_area = -twice_area

        edges = rolled - corners
        crosses = (
            edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
            - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        )
        if np.any(crosses < -1e-12):
            raise ValueError(
                "the polygon is not convex. `contains` is a conjunction of half-planes, "
                "which is only the interior for a convex polygon; a re-entrant corner "
                "would silently admit points outside it."
            )

        self.vertices = corners
        self._area = 0.5 * twice_area
        # Inward normals and offsets, fixed here and never recomputed: a point is
        # inside exactly when `normals @ x <= offsets` on every edge.
        self._normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        self._offsets = np.sum(self._normals * corners, axis=1)
        self._lower = corners.min(axis=0)
        self._upper = corners.max(axis=0)

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Plain Euclidean distance -- nothing wraps."""
        return float(np.linalg.norm(as_point(x, 2) - as_point(y, 2)))

    def contains(self, x: np.ndarray) -> bool:
        """Whether `x` satisfies every edge's half-plane.

        The tolerance is absolute and one-sided: a quadrature node sitting
        exactly on an edge is admitted rather than dropped, so a rule whose
        nodes happen to land on the boundary measures the polygon rather than
        an arbitrary subset of it.
        """
        point = as_point(x, 2)
        return bool(np.all(self._normals @ point <= self._offsets + 1e-12))

    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Return `x` if inside, else the nearest point of the polygon.

        Clipping rather than folding, as on :class:`Rectangle`, which is why
        :attr:`~SpatialDomain.periodic` stays ``False``.
        """
        point = as_point(x, 2)
        if self.contains(point):
            return point

        best, best_distance = point, np.inf
        rolled = np.roll(self.vertices, -1, axis=0)
        for start, end in zip(self.vertices, rolled, strict=True):
            edge = end - start
            length_squared = float(edge @ edge)
            t = (
                0.0
                if length_squared == 0.0
                else float(np.clip((point - start) @ edge, 0.0, length_squared) / length_squared)
            )
            candidate = start + t * edge
            gap = float(np.linalg.norm(point - candidate))
            if gap < best_distance:
                best, best_distance = candidate, gap
        return np.asarray(best, dtype=float)

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one point uniformly, by rejection from the bounding box.

        The acceptance rate is the polygon's share of its box, which for a
        convex polygon is bounded below by 1/2 -- so the expected number of
        draws is at most two and the loop cannot be slow. It is still bounded,
        because an unbounded loop that "cannot" run long is how a simulation
        hangs instead of raising.
        """
        box = self._upper - self._lower
        for _ in range(1000):
            candidate = self._lower + rng.uniform(size=2) * box
            if self.contains(candidate):
                return np.asarray(candidate, dtype=float)
        raise RuntimeError(  # pragma: no cover - unreachable for a convex polygon
            "rejection sampling failed 1000 times on a convex polygon, which should be "
            "impossible: its area is at least half its bounding box's."
        )

    @property
    def volume(self) -> float:
        """Area, by the shoelace formula."""
        return self._area

    @property
    def bounds(self) -> np.ndarray:
        """Bounding box of the vertices, shape ``(2, 2)``."""
        return np.stack([self._lower, self._upper], axis=1)

    @property
    def interior_point(self) -> np.ndarray:
        """The vertex centroid, which convexity puts inside by construction.

        The base class returns the centre of the bounding box. That is a
        *guarantee* only for a domain that fills its box, and this point is what
        probes whether the quadrature resolves the spatial kernel -- probing
        from outside the domain would measure the wrong thing.

        In practice the box centre lands inside a convex polygon too: it did on
        all 2024 random convex polygons tried, and no counterexample was found.
        The centroid is used anyway, because a mean of the vertices is inside by
        the definition of convexity and needs no such search to believe.
        """
        return np.asarray(self.vertices.mean(axis=0), dtype=float)

    @property
    def max_distance(self) -> float:
        """The diameter, which for a convex polygon is attained at two vertices."""
        diffs = self.vertices[:, None, :] - self.vertices[None, :, :]
        return float(np.max(np.linalg.norm(diffs, axis=2)))

    @property
    def nodes_per_axis(self) -> int:
        """Four times the flat default, and not a matter of taste.

        A tensor Gauss rule resolves the indicator jump at a boundary only to
        the width of a panel, and an axis-aligned edge costs nothing because it
        lands between panels. A *diagonal* one cuts every panel it crosses, so
        the area error falls only like ``1/n``. Measured on the 3-4-5 triangle:
        3.4% at the flat default of 32, 1.7% at 64, 0.85% at 128 -- and the
        error in the measured area is exactly the factor by which the simulated
        event rate is wrong.

        128 is the first value under the 1% the process warns at. It costs 16
        times the nodes of the default, which is real: a polygon is an expensive
        domain to simulate on, in the same way a hyperbolic one is. Pass
        ``n_quad`` to trade accuracy back for speed -- the process will say how
        much area, and so how much rate, that costs.
        """
        return 128


class Sphere(SpatialDomain):
    r"""The round 2-sphere of radius `radius`, charted by colatitude and longitude.

    The one closed surface here that is *not* presented as a quotient: the
    sphere is simply connected, so it has no deck group and
    :class:`FundamentalDomain` would be the wrong tool. It needs no new
    machinery either -- only the curved measure
    :math:`\mathrm{d}A = R^2 \sin\theta\, \mathrm{d}\theta\, \mathrm{d}\varphi`,
    which the quadrature and the location sampler both already honour through
    :meth:`~SpatialDomain.volume_element`.

    Points are ``(theta, phi)`` with colatitude in :math:`[0, \pi]` and
    longitude in :math:`(-\pi, \pi]`.

    Parameters
    ----------
    radius : float
        Sphere radius. The area is :math:`4\pi R^2` and the curvature
        :math:`1/R^2`.

    Notes
    -----
    Two properties of the chart are worth knowing rather than fixing.

    The chart is **singular at the poles**, where the area element vanishes and
    all longitudes name the same point. That costs nothing: the poles are a
    measure-zero set, Gauss-Legendre nodes are strictly interior to their panels
    so none of them lands on one, and every geometric predicate runs in ambient
    coordinates.

    The chart is **seamed at** :math:`\varphi = \pm\pi`, and :attr:`periodic` is
    ``False``, so a Metropolis proposal across the seam is rejected rather than
    folded. The chain therefore mixes slowly near the seam. Folding would be
    reversible here — longitude really is periodic — but it is not reversible in
    colatitude, and a half-folded proposal is not a symmetric move; correctness
    first, mixing second, as for :class:`FundamentalDomain`.

    .. versionadded:: 0.4.0

    Examples
    --------
    >>> sphere = Sphere()
    >>> round(sphere.volume / np.pi, 6)
    4.0
    >>> round(sphere.distance([0.0, 0.0], [np.pi, 0.0]) / np.pi, 6)
    1.0
    """

    periodic = False

    def __init__(self, radius: float = 1.0) -> None:
        self.radius = float(radius)
        self.model = SphericalPlane(radius)

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Great-circle distance."""
        return self.model.distance(as_point(x, 2), as_point(y, 2))

    def wrap(self, x: np.ndarray) -> np.ndarray:
        r"""Fold to the canonical chart, through the sphere itself.

        Lifting and charting back is the whole implementation: any real pair
        names a point of the sphere, and :meth:`~ModelSpace.chart` returns the
        one representative with ``theta`` in :math:`[0, \pi]`.
        """
        return self.model.chart(self.model.lift(as_point(x, 2)))

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        r"""Draw one point uniformly on the sphere.

        By inverse transform on ``cos(theta)``, not by drawing ``theta``
        uniformly: the area element carries the :math:`\sin\theta`, and
        ignoring it clusters the draws at the poles.
        """
        colatitude = float(np.arccos(rng.uniform(-1.0, 1.0)))
        return np.array([colatitude, float(rng.uniform(-np.pi, np.pi))])

    @property
    def volume(self) -> float:
        r"""Surface area, :math:`4\pi R^2`."""
        return float(4.0 * np.pi * self.radius**2)

    @property
    def bounds(self) -> np.ndarray:
        """Chart box, ``[[0, pi], [-pi, pi]]``."""
        return np.array([[0.0, np.pi], [-np.pi, np.pi]])

    def volume_element(self, x: np.ndarray) -> float:
        r"""Return the spherical area element :math:`R^2 \sin\theta`."""
        return self.model.volume_element(as_point(x, 2))

    @property
    def interior_point(self) -> np.ndarray:
        """A point on the equator, away from both chart singularities."""
        return np.array([np.pi / 2, 0.0])

    @property
    def max_distance(self) -> float:
        """Half the circumference: antipodal points."""
        return float(np.pi * self.radius)

    def __repr__(self) -> str:
        return f"Sphere(radius={self.radius:g})"


# ---------------------------------------------------------------------------
# Fundamental domains
# ---------------------------------------------------------------------------

#: How many group elements the certified distance search evaluates at a time.
#: Small enough that a typical call touches one or two blocks of a hyperbolic
#: window that may hold thousands, large enough that the vectorised distance
#: still dominates the Python overhead.
_DISTANCE_BLOCK = 64


def _isometry(model: ModelSpace, matrix: Any) -> np.ndarray:
    """Coerce a side pairing to a ``3x3`` ambient matrix and check it is admissible.

    Two separate tests, with separate messages, because they fail for separate
    reasons.

    **Is it an isometry?** A shear or a scaling does not preserve distances, so
    the quotient it defines is not a metric quotient at all.

    **Does it act freely?** An isometry with a fixed point quotients to an
    *orbifold* -- a cone point, a mirror line -- rather than to a surface, and
    nothing downstream can tell the difference. A rotation is the case that
    matters in practice: it is an isometry, its determinant is ``+1``, and
    before 0.4.0 it passed every check the package made. Orientation is
    deliberately *not* tested: an orientation-reversing pairing quotients to a
    non-orientable surface, which is a perfectly good closed surface, and the
    Klein bottle is exactly one glide reflection away from the torus.
    """
    a = np.asarray(matrix, dtype=float)
    if a.shape == (2, 3):
        a = np.vstack([a, [0.0, 0.0, 1.0]])
    if a.shape != (3, 3):
        raise ValueError(f"a side pairing must be a 3x3 (or 2x3) affine matrix, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("a side pairing must be finite")
    if isinstance(model, EuclideanPlane) and not np.allclose(a[2], [0.0, 0.0, 1.0]):
        raise ValueError(f"the last row of an affine matrix must be [0, 0, 1], got {a[2].tolist()}")
    if not model.is_isometry(a):
        raise ValueError(
            f"a side pairing must be an isometry of the {model.name}; this one is not. "
            "Its linear part must preserve the form the model space is defined by."
        )

    motion = model.classify(a)
    if not motion.free:
        raise ValueError(
            f"a side pairing must act freely, and this one is a {motion.kind}, which fixes at "
            "least one point of the model space. A pairing with a fixed point quotients to an "
            "orbifold -- a cone point or a mirror line -- not to a surface, and nothing "
            "downstream of here can tell the two apart."
        )
    return a


class FundamentalDomain(SpatialDomain):
    r"""A convex geodesic polygon with side pairings, presenting a closed surface.

    The domain is the polygon `vertices` in a constant-curvature `model` space;
    `pairings` are the side-pairing isometries that generate the deck group, and
    points differing by a group element are the same point of the quotient
    surface.

    Which surface that is comes out of the construction rather than in: the side
    correspondence is inferred, the corner cycles are walked, and
    :attr:`topology` reports the orientability, Euler characteristic and genus
    that result. :class:`Torus2D` is the rectangle case, available here as
    :meth:`rectangle`; :meth:`hexagon`, :meth:`klein_bottle`,
    :meth:`projective_plane`, :meth:`genus` and :meth:`crosscaps` between them
    reach every closed surface.

    Parameters
    ----------
    vertices : array_like, shape (n, 2)
        Polygon corners in order, either winding, in the model space's chart
        coordinates. Must be convex.
    pairings : sequence of array_like
        Side-pairing isometries as ``3x3`` (or ``2x3``) ambient matrices. Each
        must be an isometry of `model` and must act freely. They need not be the
        individual side pairings — they need only *generate* the group, as the
        hexagonal torus's two translations generate its three side pairings.
    model : ModelSpace, optional
        The geometry the polygon lives in. Defaults to the Euclidean plane.
    interior : array_like, optional
        A chart point strictly inside the polygon, used to orient the edge
        half-spaces and as the centre of the Dirichlet reduction. Defaults to
        the geodesic centroid of the vertices, which is inside any convex
        polygon that does not surround the whole model space. The hemisphere
        presenting :math:`\mathbb{RP}^2` is the case that needs it explicitly:
        its four corners lie on one great circle and average to the centre of
        the sphere.

    Attributes
    ----------
    topology : hawkes_package.spatio_temporal._gluing.Topology
        Orientability, Euler characteristic, genus and name of the quotient.

    Raises
    ------
    ValueError
        If the polygon is not convex, if a pairing is not a free isometry, if
        some side is left unpaired, if a corner cycle does not close up with an
        angle sum of :math:`2\pi`, or if Gauss-Bonnet is violated.

    Notes
    -----
    :attr:`periodic` is ``False``, so MCMC proposals leaving the polygon are
    *rejected* rather than folded back through :meth:`wrap`. Folding is
    reversible only when the group acts by translations, which leave a Gaussian
    proposal invariant; that holds for the flat orientable presentations and for
    nothing else, and this package has a history of bounds and samplers that
    were wrong silently rather than loudly.

    .. versionadded:: 0.3.0

    .. versionchanged:: 0.4.0
       Orientation-reversing pairings are accepted, so the presentation reaches
       non-orientable surfaces. In exchange, three conditions that were
       previously unchecked are enforced at construction: freeness of every
       pairing, a complete side correspondence, and Poincare's angle-sum
       condition on every corner cycle.

    .. versionchanged:: 0.4.0
       Takes a `model`, so the polygon may be spherical or hyperbolic rather
       than only flat.

    Examples
    --------
    >>> hexagon = FundamentalDomain.hexagon(1.0)
    >>> round(hexagon.volume, 6)
    2.598076
    >>> bool(hexagon.contains(hexagon.wrap([5.0, -4.0])))
    True
    >>> hexagon.topology.name
    'torus'
    >>> FundamentalDomain.klein_bottle().topology.euler_characteristic
    0
    """

    periodic = False

    def __init__(
        self,
        vertices: Any,
        pairings: Any,
        *,
        model: ModelSpace | None = None,
        interior: Any = None,
    ) -> None:
        self.model = model if model is not None else EuclideanPlane()
        corners = np.asarray(vertices, dtype=float)
        if corners.ndim != 2 or corners.shape[1] != 2 or corners.shape[0] < 3:
            raise ValueError(f"vertices must have shape (n >= 3, 2), got {corners.shape}")
        if not np.isfinite(corners).all():
            raise ValueError("vertices must be finite")
        self.vertices = corners
        count = len(corners)

        self._centre = (
            self.model.centroid(corners)
            if interior is None
            else as_point(np.asarray(interior, dtype=float), 2)
        )
        self._scale = float(max(self.model.distance(self._centre, v) for v in corners))
        if not self._scale > 0:
            raise ValueError("vertices are degenerate: the polygon has zero area")
        self._eps = 1e-12 * self._scale  # the provisional width of "on the boundary"
        self._tol = 1e-9 * self._scale  # when the reduction has stopped making progress

        self._build_halfspaces(corners)
        self._area = self.model.polygon_area(corners)
        if not self._area > 0:
            raise ValueError("vertices are degenerate: the polygon has zero area")
        self._diameter = float(
            max(
                self.model.distance(corners[i], corners[j])
                for i in range(count)
                for j in range(i + 1, count)
            )
        )

        pairing_list = [_isometry(self.model, p) for p in pairings]
        if not pairing_list:
            raise ValueError("at least one side pairing is required")
        self.pairings = pairing_list

        # Kept as the default word length `orbit` reads, which is the only
        # thing it was ever used for once `distance` began certifying its own
        # truncation by displacement radius. `orbit(y, n_images=...)` overrides
        # it per call and is not deprecated.
        self.n_images = 3

        self._pairing = _gluing.infer_pairing(
            self.model, corners, pairing_list, self._outward, self._centre, self._tol
        )
        self._check_poincare(corners)

        orientable = all(self.model.classify(p).orientation > 0 for p in pairing_list)
        self.topology = _gluing.topology(len(self._cycles), count, orientable)
        self._check_gauss_bonnet()

        self._bounds = self.model.chart_bounds(corners, self._centre)
        self._density_sup = self.model.hull_volume_element_sup(corners)
        self._nodes_per_axis: int | None = None
        self._window_cache: dict[int, np.ndarray] = {}
        self._radius_window: np.ndarray = np.empty((0, 3, 3))
        self._boundary_window: np.ndarray = np.empty((0, 3, 3))
        self._radius_displacements: np.ndarray = np.empty(0)
        self._window_norms: np.ndarray = np.empty(0)
        self._window_radius = 0.0
        self._band = self._eps
        # Twice the circumradius, and the margin is not decoration: an element
        # carrying a boundary point of the polygon to another boundary point
        # moves the centre by *exactly* that in the symmetric presentations, and
        # the antipodal map of the projective plane misses the cut by four ulp
        # without it -- which leaves the boundary with no representative at all.
        self._grow_window(2.0 * self._scale + self._tol)
        # Snapshot it. `_grow_window` replaces the window whenever a distance
        # needs a wider search, and the boundary convention must not move when
        # it does: `contains` is a property of the domain, not of its search
        # history. Sharing one domain between two callers -- as a pytest fixture
        # does -- would otherwise make the answer depend on which of them ran
        # first, and the answer it changes is which of a corner cycle's images
        # is *the* representative.
        self._boundary_window = self._radius_window
        self._band = self._boundary_band(corners)

    # -- construction -------------------------------------------------

    def _build_halfspaces(self, corners: np.ndarray) -> None:
        """Orient one geodesic half-space per side, and check the polygon is convex.

        Each side's supporting geodesic is oriented so the interior point sits
        strictly on the negative side; the polygon is then convex exactly when
        every *other* corner does too. That test replaces the signed turn angles
        the flat implementation used, and unlike them it is written in the model
        space rather than in the chart, so it holds for a spherical or
        hyperbolic polygon unchanged.
        """
        count = len(corners)
        centre = self.model.lift(self._centre)
        planes, outward = [], []
        for i in range(count):
            plane = self.model.halfspace(corners[i], corners[(i + 1) % count])
            inside = float(plane @ centre)
            if abs(inside) <= self._eps:
                raise ValueError(
                    f"the interior point {self._centre.tolist()} lies on the geodesic through "
                    f"side {i}, so it does not orient it. Pass a different `interior=`."
                )
            if inside > 0:
                plane = -plane
            planes.append(plane)
            midpoint = self.model.midpoint(corners[i], corners[(i + 1) % count])
            outward.append(self.model.normal_at(plane, midpoint))
        self._planes = np.asarray(planes, dtype=float)
        self._outward = np.asarray(outward, dtype=float)

        signed = self._planes @ self.model.lift_many(corners).T
        if np.any(signed > self._eps + self._tol):
            raise ValueError(
                "vertices must describe a convex polygon; `contains` is a half-plane test "
                "against every edge, which only characterises a convex region."
            )

    def _check_poincare(self, corners: np.ndarray) -> None:
        r"""Walk the corner cycles and require each to close up exactly once.

        Poincare's polygon theorem: a convex polygon whose side pairings carry
        each side onto another and whose corner cycles have angle sum
        :math:`2\pi` with trivial cycle transformation generates a discrete
        group acting freely, with the polygon as a fundamental domain. Checking
        it here replaces the older behaviour, where a presentation that did not
        tile surfaced -- if it surfaced at all -- as a failure inside
        :meth:`wrap`, mid-simulation and far from the mistake.
        """
        count = len(corners)
        self._angles = np.array(
            [
                self.model.angle(corners[i - 1], corners[i], corners[(i + 1) % count])
                for i in range(count)
            ]
        )
        self._cycles = _gluing.vertex_cycles(
            self.model, corners, self._pairing, self._angles, self._tol
        )
        for cycle in self._cycles:
            if abs(cycle.angle_sum - 2 * np.pi) > 1e-7:
                order = 2 * np.pi / cycle.angle_sum if cycle.angle_sum > 0 else float("inf")
                raise ValueError(
                    f"the corners {list(cycle.corners)} form a cycle whose interior angles sum "
                    f"to {cycle.angle_sum:.6f}, not 2*pi = {2 * np.pi:.6f}. Poincare's polygon "
                    f"theorem needs exactly one turn: {order:.3f} of them means the copies of "
                    "the polygon meet at a cone point and the quotient is an orbifold, not a "
                    "surface."
                )
            if not np.allclose(cycle.transformation, np.eye(3), atol=1e-7):
                raise ValueError(
                    f"the corners {list(cycle.corners)} close up geometrically but the deck "
                    "element that closes them is not the identity, so the quotient has a cone "
                    "point there. The side pairings do not satisfy Poincare's theorem."
                )

    def _check_gauss_bonnet(self) -> None:
        r"""Cross-check area, curvature and topology against each other.

        :math:`\int K\,dA = 2\pi\chi` over the closed surface, and the
        surface's area is the polygon's. Three quantities computed by three
        unrelated routes -- the angle excess, the model's curvature, and the
        corner-cycle count -- have to agree, which makes this the cheapest
        strong check available on a presentation and the one most likely to
        catch a silently wrong one.
        """
        target = 2 * np.pi * self.topology.euler_characteristic
        actual = self.model.curvature * self._area
        if abs(actual - target) > 1e-6 * max(1.0, abs(target)):
            raise ValueError(
                f"Gauss-Bonnet fails: the {self.topology.name} this presentation glues to has "
                f"Euler characteristic {self.topology.euler_characteristic}, so the integral of "
                f"the curvature over it must be {target:.6f}, but the polygon has area "
                f"{self._area:.6f} at curvature {self.model.curvature:.6f}, giving "
                f"{actual:.6f}. The polygon, the model space and the gluing do not describe the "
                "same surface."
            )

    # -- presentations -------------------------------------------------

    @classmethod
    def rectangle(
        cls,
        width: float = 2 * np.pi,
        height: float = 2 * np.pi,
    ) -> FundamentalDomain:
        """Build the centred rectangle of these sides, opposite sides paired.

        Presents the same flat torus as :class:`Torus2D`, and exists chiefly so
        that the general machinery can be checked against that hand-written
        implementation.

        .. versionchanged:: 0.4.0
           ``L1`` and ``L2`` are now ``width`` and ``height``.

        .. versionchanged:: 0.5.0
           The ``L1`` and ``L2`` spellings are gone.
        """
        corners = _rectangle_corners(width, height)
        return cls(corners, [_translation(width, 0.0), _translation(0.0, height)])

    @classmethod
    def hexagon(cls, side: float = 1.0) -> FundamentalDomain:
        """Build the regular hexagon of circumradius `side`, opposite sides paired.

        This is the Dirichlet domain of the triangular lattice of spacing
        ``side * sqrt(3)``, and presents the hexagonal torus.
        """
        if not side > 0:
            raise ValueError(f"side must be positive, got {side}")
        angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
        corners = side * np.column_stack([np.cos(angles), np.sin(angles)])
        spacing = side * np.sqrt(3.0)
        return cls(
            corners,
            [
                _translation(spacing, 0.0),
                _translation(spacing / 2, spacing * np.sqrt(3.0) / 2),
            ],
        )

    @classmethod
    def klein_bottle(
        cls,
        width: float = 2 * np.pi,
        height: float = 2 * np.pi,
    ) -> FundamentalDomain:
        r"""Build the flat Klein bottle: a rectangle, one side pair glued with a flip.

        One translation and one **glide reflection**, ``(x, y) -> (-x, y + height)``.
        Exactly one sign away from :meth:`rectangle`, and the whole difference
        between the two closed flat surfaces: the glide reverses orientation, so
        the quotient is non-orientable, while the corner cycle still sums to
        :math:`2\pi` and the Euler characteristic is still zero.

        A glide reflection is fixed-point-free -- that is what makes it a legal
        side pairing, where the pure reflection ``(x, y) -> (-x, y)`` is not.

        .. versionadded:: 0.4.0

        Examples
        --------
        >>> bottle = FundamentalDomain.klein_bottle(3.0, 5.0)
        >>> bottle.topology.orientable, bottle.topology.name
        (False, 'Klein bottle')
        """
        corners = _rectangle_corners(width, height)
        glide = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, height], [0.0, 0.0, 1.0]])
        return cls(corners, [_translation(width, 0.0), glide])

    @classmethod
    def projective_plane(cls, radius: float = 1.0) -> FundamentalDomain:
        r"""Build :math:`\mathbb{RP}^2` as a hemisphere with antipodal boundary points glued.

        The deck group is :math:`\{\pm I\}` and the fundamental domain is a
        closed hemisphere. Presented here as a **four-sided** polygon whose
        corners are equally spaced round the equator, rather than as the
        two-sided one the textbook word ``aa`` suggests: a bigon's two corners
        are antipodal, so the geodesic between them is not unique and neither
        the side-membership test nor the corner angles are well defined. Four
        quarter-arcs fix that and change nothing about the quotient, since
        subdividing a side is a subdivision of the cell structure. Both corner
        cycles then sum to :math:`2\pi` -- the corners sit on a geodesic, so
        each interior angle is :math:`\pi` -- and
        :math:`\chi = 2 - 2 + 1 = 1`.

        The antipodal map reverses orientation on :math:`S^2`, which is
        :math:`\mathbb{RP}^2` being non-orientable; before 0.4.0 this
        presentation was rejected on that ground alone.

        .. versionadded:: 0.4.0

        Examples
        --------
        >>> plane = FundamentalDomain.projective_plane()
        >>> plane.topology.name, plane.topology.euler_characteristic
        ('projective plane', 1)
        >>> round(plane.volume / np.pi, 9)
        2.0
        """
        model = SphericalPlane(radius)
        equator = np.array([[np.pi / 2, k * np.pi / 2] for k in (0, 1, 2, -1)])
        return cls(
            equator,
            [-np.eye(3)],
            model=model,
            interior=np.array([0.0, 0.0]),
        )

    @classmethod
    def genus(cls, handles: int) -> FundamentalDomain:
        r"""Build the orientable surface of genus `handles`.

        For ``handles >= 2`` this is the regular hyperbolic :math:`4g`-gon with
        the classical word :math:`a_1 b_1 a_1^{-1} b_1^{-1} \cdots`, whose
        single corner cycle needs every interior angle to be
        :math:`2\pi / 4g`. A regular hyperbolic polygon's angle shrinks as it
        grows, so exactly one size works, and it is
        :math:`\cosh r = \cot^2(\pi/4g)`. Gauss-Bonnet then fixes the area at
        :math:`4\pi(g - 1)`, and the constructor checks that it comes out.

        ``handles == 1`` is the flat torus, on a square; its size is arbitrary,
        and :meth:`rectangle` is the way to choose one. ``handles == 0`` is the
        sphere, which is simply connected and has no such presentation --
        use :class:`Sphere`.

        .. versionadded:: 0.4.0

        Examples
        --------
        >>> surface = FundamentalDomain.genus(2)
        >>> surface.topology.euler_characteristic
        -2
        >>> round(surface.volume / np.pi, 6)
        4.0
        """
        if handles < 1:
            raise ValueError(
                f"genus must be at least 1, got {handles}. The sphere is simply connected: it "
                "has no deck group and no fundamental domain. Use Sphere()."
            )
        _check_sides(4 * handles, f"a genus-{handles} surface")
        word = []
        for handle in range(handles):
            word += [(handle, 1), (handles + handle, 1), (handle, -1), (handles + handle, -1)]
        return cls._from_word(word)

    @classmethod
    def crosscaps(cls, count: int) -> FundamentalDomain:
        r"""Build the non-orientable surface :math:`N_k` of `count` crosscaps.

        The classical word :math:`a_1 a_1 a_2 a_2 \cdots a_k a_k` on a
        :math:`2k`-gon: each side is glued to its neighbour *without* reversal,
        which is what a crosscap is and what makes each pairing a glide
        reflection.

        ``count == 1`` is the projective plane, which is spherical and is built
        by :meth:`projective_plane`. ``count == 2`` is the Klein bottle, flat,
        on a square. From three crosscaps on the surface is hyperbolic, on the
        regular :math:`2k`-gon of interior angle :math:`\pi/k`, and has area
        :math:`2\pi(k - 2)`.

        .. versionadded:: 0.4.0

        Examples
        --------
        >>> surface = FundamentalDomain.crosscaps(3)
        >>> surface.topology.orientable, surface.topology.euler_characteristic
        (False, -1)
        """
        if count < 1:
            raise ValueError(f"the number of crosscaps must be at least 1, got {count}")
        if count == 1:
            return cls.projective_plane()
        _check_sides(2 * count, f"a surface with {count} crosscaps")
        word = [(crosscap, 1) for crosscap in range(count) for _ in range(2)]
        return cls._from_word(word)

    @classmethod
    def _from_word(cls, word: list[tuple[int, int]]) -> FundamentalDomain:
        r"""Build the regular polygon carrying `word`, in whichever geometry it needs.

        The number of sides decides the geometry, through the angle the corner
        cycle demands. A single cycle through all :math:`n` corners needs each
        interior angle to be :math:`2\pi/n`; a Euclidean regular :math:`n`-gon
        has :math:`(n-2)\pi/n`, and the two agree only at ``n == 4``. So four
        sides is flat and anything larger is hyperbolic -- which is Gauss-Bonnet
        again, read off the polygon instead of the surface.
        """
        sides = len(word)
        angles = 2 * np.pi * np.arange(sides) / sides
        if sides == 4:
            model: ModelSpace = EuclideanPlane()
            corners = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            model = HyperbolicPlane()
            interior_half = np.pi / sides  # half of the required 2*pi/sides
            radius = float(np.arccosh(1.0 / np.tan(interior_half) ** 2))
            euclidean = float(np.tanh(radius / 2))  # the disc chart is radial
            corners = euclidean * np.column_stack([np.cos(angles), np.sin(angles)])
        return cls(corners, _pairings_from_word(model, corners, word), model=model)

    # -- the deck group ------------------------------------------------

    def _window(self, depth: int) -> np.ndarray:
        """Group elements as words of length at most `depth`, identity first."""
        cached = self._window_cache.get(depth)
        if cached is None:
            cached = _gluing.group_window(self.model, self.pairings, self._centre, depth)
            self._window_cache[depth] = cached
        return cached

    def _boundary_band(self, corners: np.ndarray) -> float:
        """How far off a side a point may sit and still count as being on it.

        Sized against the *arithmetic*, not against the geometry, and **measured
        rather than assumed**. A point of the boundary is normally arrived at
        through a deck-group product, and the rounding such a product leaves
        behind grows with the size of the matrix and of the point. Estimating
        that from first principles is what went wrong: a coefficient chosen as
        thirty machine epsilons gave a genus-3 surface a band of ``1.5e-10``
        against a residue that reached ``1.0e-10`` on one platform and
        ``1.1e-11`` on another -- the same computation, straddling the
        threshold, so ``wrap`` returned a representative that ``contains``
        rejected on Linux and accepted on Windows.

        So the residue is measured on the corners themselves. Every image of a
        corner that lands on a corner is, exactly, on the boundary; whatever the
        half-space test then reports for it is pure accumulated rounding, and
        the largest such value is what the band has to clear. A hundredfold
        margin over the worst observed still leaves twelve orders of magnitude
        between the band and the smallest signed distance of an image that is
        genuinely outside -- on the genus-3 surface those are ``1e-8`` and
        ``13.4`` -- so there is no tension between the two sides of the test.

        Measured once, over the window the polygon is built with rather than
        whatever it may later have grown to, so that widening the search cannot
        quietly widen the domain along with it.
        """
        observed = 0.0
        for corner in corners:
            images = np.einsum("kij,j->ki", self._boundary_window, self.model.lift(corner))
            landed = self.model.chart_many(images)
            for image, charted in zip(images, landed, strict=True):
                # A corner carried onto a corner is on the boundary by
                # construction, whatever the arithmetic says about it.
                if np.min(np.linalg.norm(corners - charted, axis=1)) <= self._tol:
                    observed = max(observed, float(np.max(self._planes @ image)))
        return max(self._eps, 100.0 * observed)

    def _grow_window(self, radius: float) -> None:
        """Rebuild the displacement-truncated window out to `radius`, sorted by displacement.

        The slack the search expands through is the polygon's circumradius, and
        that is exactly what completeness needs: the copies crossed by the
        geodesic from the centre to ``g.centre`` each contain a point within
        `radius` of the centre, and each copy's own centre is within a
        circumradius of any of its points.
        """
        self._radius_window, self._radius_displacements = _gluing.window_by_radius(
            self.model, self.pairings, self._centre, radius, self._scale
        )
        self._window_norms = np.linalg.norm(self._radius_window, axis=(1, 2))
        self._window_radius = radius

    def _images(self, y: np.ndarray, depth: int) -> np.ndarray:
        """Every image of `y` under the word-truncated group, shape ``(m, 2)``."""
        return self.model.apply_many(self._window(depth), y)

    def orbit(self, y: np.ndarray, n_images: int = 3) -> list[np.ndarray]:
        """Images of `y` under the deck group, out to word length `n_images`."""
        return list(self._images(as_point(y, 2), n_images))

    # -- the SpatialDomain contract ------------------------------------

    def _inside(self, x: np.ndarray, slack: float = 0.0) -> bool:
        """Strictly inside every side, or on a *closed* one.

        The comparison runs against a band rather than exact zero, and it has to.
        A side's midpoint and its partner under the side pairing both evaluate to
        a signed distance of *about* zero on their respective sides, with a sign
        set by rounding; testing ``signed == 0`` therefore admits both -- and the
        quotient double-counts the side.

        A point inside the band is then admitted only if it is the *canonical*
        member of its orbit; see :meth:`_canonical_image`.
        """
        if not self.model.in_chart(x):
            return False
        signed = self._planes @ self.model.lift(x)
        edge = self._band + slack
        if np.all(signed < -edge):
            return True
        if np.any(signed > edge):
            return False

        best = self._canonical_image(x, edge)
        return best is not None and bool(np.all(np.abs(best - x) <= self._tol))

    def _closure_images(self, x: np.ndarray, edge: float) -> np.ndarray:
        """Every image of `x` under the deck group that lands in the closed polygon.

        Searched over the window as it stood when the polygon was built, which
        is exactly the right set and exactly the right *time*: an element
        carrying a point of the polygon back into the polygon moves the centre
        by at most twice the circumradius, and the initial window is built to
        that radius. Using the current window instead would make the boundary
        convention depend on whether some earlier call to :meth:`distance` had
        widened the search.
        """
        images = np.einsum("kij,j->ki", self._boundary_window, self.model.lift(x))
        inside = images[np.all(images @ self._planes.T <= edge, axis=1)]
        return self.model.chart_many(inside) if len(inside) else np.empty((0, 2))

    def _canonical_image(self, x: np.ndarray, edge: float) -> np.ndarray | None:
        """Return the one representative of `x`'s orbit that the closed polygon keeps.

        A point of the boundary has several images in the closed polygon -- two
        for a side, a whole corner cycle for a corner -- and exactly one must be
        admitted, or the quotient counts that point twice and every integral
        over the domain double-counts the set the boundary sits on.

        The choice is the lexicographically smallest image. Doing it that way
        rather than by flagging each side "closed" or "open" is not
        fastidiousness. The flag rule generalises the ``[-L/2, L/2)`` convention
        of :class:`Circle` and :class:`Torus2D`, and reproduces it exactly on
        the rectangle and the hexagon, but it is wrong in general: on the
        hemisphere presenting the projective plane, four sides paired two and
        two, *every* assignment of flags leaves one corner cycle with two
        representatives and the other with none. No assignment fixes it, because
        a corner's membership is decided by both of its sides at once.

        Returns ``None`` when no image lands in the closure, which means the
        polygon is not a fundamental domain for its own pairings.
        """
        candidates = self._closure_images(x, edge)
        if len(candidates) == 0:
            return None
        # Lexicographic, with a tolerance on the first coordinate: two images
        # agreeing there to within rounding are separated by the second.
        leftmost = candidates[candidates[:, 0] <= float(np.min(candidates[:, 0])) + self._tol]
        return np.asarray(leftmost[int(np.argmin(leftmost[:, 1]))])

    def contains(self, x: np.ndarray) -> bool:
        """Whether `x` lies in the polygon, with one representative per boundary orbit."""
        return self._inside(as_point(x, 2), 0.0)

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Distance on the quotient: the shortest hop to any image of `y`.

        Both points are reduced into the polygon before the search, for the same
        reason :func:`~hawkes_package.spatio_temporal.kernels.make_periodic`
        reduces to the canonical offset first: measured from an unreduced lift,
        the nearest image can lie outside the truncated window.

        The truncation is then **certified rather than assumed**. Any deck
        element ``g`` beating the best distance found so far must satisfy

        ``d(c, g.c) <= d(c, x) + best + d(c, y)``

        by the triangle inequality, so once the window has been searched out to
        that displacement no unexamined element can improve on it, and the
        answer is the exact minimum. Word length -- what this used to truncate by
        -- carries no such guarantee, and on a hyperbolic surface, where the
        element count grows like :math:`e^R`, the difference between a bound and
        a heuristic is the difference between a kernel that stays periodic and
        one that quietly decays to zero.
        """
        model = self.model
        here = model.lift(self.wrap(x))
        there = model.lift(self.wrap(y))
        reach = model.ambient_distance(model.lift(self._centre), here) + model.ambient_distance(
            model.lift(self._centre), there
        )

        best = float("inf")
        for _ in range(8):
            best = self._search_window(here, there, reach, best)
            required = reach + best
            if self._window_radius + self._tol >= required:
                return best
            # Grow to what the certificate asks for and no further. Overshooting
            # is not free insurance: in a hyperbolic geometry the element count
            # grows like exp(R), so a growth *factor* turns a window of hundreds
            # into one of millions in two steps. The cap is what the triangle
            # inequality allows -- both points have been reduced into the
            # polygon, so neither `reach` nor `best` can exceed twice the
            # circumradius.
            self._grow_window(min(4.0 * self._scale, 1.05 * required) + self._tol)
        raise RuntimeError(  # pragma: no cover - needs a group that is not discrete
            "the deck-group window could not be grown far enough to certify a distance"
        )

    def _search_window(
        self, here: np.ndarray, there: np.ndarray, reach: float, best: float
    ) -> float:
        """Minimise the distance from `here` over the images of `there`, block by block.

        The window is sorted by displacement, so once an element moves the
        centre further than ``reach + best`` no later one can beat `best`
        either, and the scan stops. A typical call therefore touches one or two
        blocks of a window that may hold thousands.
        """
        window = self._radius_window
        displacements = self._radius_displacements
        total = len(window)
        start = 0
        while start < total:
            if displacements[start] > reach + best:
                break
            stop = min(start + _DISTANCE_BLOCK, total)
            images: np.ndarray = np.einsum("kij,j->ki", window[start:stop], there)
            best = min(best, float(np.min(self.model.ambient_distances(here, images))))
            start = stop
        return best

    def lift_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Distance between two lifts, in the model space rather than the quotient."""
        return self.model.distance(as_point(x, 2), as_point(y, 2))

    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Map `x` to its representative inside the polygon.

        Raises :class:`ValueError` when `x` is not a point of the model space at
        all -- only the Poincare disc chart can fail that way, and there it is
        the honest answer: a chart point of modulus one is at infinity.
        """
        # Canonicalise the chart first. A colatitude of 5.05 names a perfectly
        # good point of the sphere, and every predicate here would agree that it
        # is inside the polygon, but it is not the representative in `bounds` --
        # so `wrap` would return a point outside the box the quadrature and the
        # sampler both work in. Exact and free in the two charts that are
        # already canonical.
        point = self.model.chart(self.model.lift(as_point(x, 2)))
        point = self._reduce(point)
        if self._inside(point, 0.0):
            return point

        # On the boundary but on a non-canonical image of it: hand back the
        # canonical one directly rather than searching for an image that passes
        # `contains`. The two are then the same computation by construction, and
        # cannot drift apart -- which they did when this swept the window with a
        # widened tolerance, returning for a genus-3 corner an image that
        # `contains` went on to reject.
        representative = self._canonical_image(point, self._band)
        if representative is not None:
            return representative

        # Not on the boundary and not inside: `_reduce` lands in the Dirichlet
        # cell about the centre, which is the polygon itself for the lattice
        # presentations but need not be for a caller's own.
        images = self.model.apply_many(self._radius_window, point)
        order = np.argsort(self.model.distances(self._centre, images))
        for index in order:
            candidate: np.ndarray = images[index]
            if self._inside(candidate, self._tol):
                return candidate
        raise ValueError(
            f"no image of {np.asarray(x).tolist()} under the deck group lies in the polygon; "
            "the side pairings do not tile the model space by it."
        )

    def _reduce(self, x: np.ndarray, max_steps: int = 1000) -> np.ndarray:
        """Walk `x` towards the centre one generator at a time.

        Terminates because the distance to the centre strictly decreases at
        every step; the cap only guards a caller's non-discrete "group".
        """
        current = self.model.distance(self._centre, x)
        for _ in range(max_steps):
            images = self._images(x, 1)
            distances = self.model.distances(self._centre, images)
            best = int(np.argmin(distances))
            if best == 0 or distances[best] >= current - self._tol:
                return x
            x, current = np.asarray(images[best]), float(distances[best])
        raise ValueError(
            "reducing a point into the polygon did not converge; the side pairings "
            "do not generate a discrete group."
        )

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one point uniformly on the polygon, by rejection in its box.

        Uniform in the *surface* measure, not in the chart: a proposal is kept
        with probability ``volume_element(x) / sup volume_element``, which is
        what makes the marginal flat on a curved domain. The supremum is the
        exact one over the polygon's geodesic hull, never a sampled maximum --
        rejection against a sampled maximum silently truncates the tail of the
        target.
        """
        box = self.bounds
        ceiling = self._density_sup
        for _ in range(100_000):
            candidate: np.ndarray = rng.uniform(box[:, 0], box[:, 1])
            if not self._inside(candidate, 0.0):
                continue
            if rng.uniform() * ceiling <= self.model.volume_element(candidate):
                return candidate
        raise RuntimeError(  # pragma: no cover - needs a pathological polygon
            "100000 draws over the bounding box all fell outside the polygon or were rejected "
            "by the measure; the polygon occupies a negligible part of its own bounding box."
        )

    def volume_element(self, x: np.ndarray) -> float:
        """Return the model space's measure density at `x`, in chart coordinates."""
        return self.model.volume_element(as_point(x, 2))

    @property
    def volume(self) -> float:
        """Area of the polygon, and so of the surface it presents."""
        return self._area

    @property
    def bounds(self) -> np.ndarray:
        """Bounding rectangle of the polygon in chart coordinates, shape (2, 2)."""
        return self._bounds

    @property
    def interior_point(self) -> np.ndarray:
        """A point known to lie inside the polygon.

        The geodesic centroid of the vertices unless the caller named one. Also
        the point :meth:`wrap` reduces towards, and for a domain that is the
        Dirichlet cell of its lattice -- the rectangle and the hexagon -- it is
        the lattice site that cell belongs to.
        """
        return self._centre

    @property
    def max_distance(self) -> float:
        """The polygon's diameter, which bounds the quotient's."""
        return self._diameter

    @property
    def nodes_per_axis(self) -> int:
        """The coarsest tensor rule that measures this polygon's own area.

        Calibrated rather than declared, and calibrated against a number the
        quadrature had no hand in: :attr:`volume` comes from the angle excess,
        by Gauss-Bonnet, so agreeing with it is evidence and not a tautology.

        The default of 32 nodes per axis is ample for a flat polygon -- the
        rectangle and the hexagon are measured exactly -- and is not remotely
        enough for a hyperbolic one, where it misses the genus-2 octagon's area
        by 5%. Two effects compound there: the mask cuts the boundary at panel
        resolution, and the disc chart's area element varies by a factor of
        forty across the polygon, with the variation concentrated exactly where
        the mask error is.

        Computed on first use and cached, so a domain that is never simulated on
        never pays for it.

        .. versionadded:: 0.4.0
        """
        if self._nodes_per_axis is None:
            self._nodes_per_axis = self._calibrate_nodes()
        return self._nodes_per_axis

    def _calibrate_nodes(self) -> int:
        """Climb the node ladder until the rule reproduces :attr:`volume` to 0.5%."""
        coarsest = _integration.default_nodes_per_axis(2)
        candidate = coarsest
        for _ in range(4):
            rule = _integration.restrict(
                _integration.build(self.bounds, candidate), self.contains, self.volume_element
            )
            if abs(float(rule.weights.sum()) - self._area) <= 5e-3 * self._area:
                return candidate
            candidate *= 2
        # Out of ladder: hand back the finest tried and let the process
        # constructor's volume check say how far off it still is.
        return candidate // 2

    @property
    def cycles(self) -> list[_gluing.VertexCycle]:
        """The corner cycles: one per vertex of the glued surface.

        .. versionadded:: 0.4.0
        """
        return list(self._cycles)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.topology.name}, {len(self.vertices)} sides, "
            f"{self.model.name}, area {self._area:.6g})"
        )


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------


#: Most sides a hyperbolic presentation may have. The polygon grows with the
#: side count, and so does the deck-group window a certified distance needs --
#: exponentially, because the element count in negative curvature grows like
#: exp(R). Twelve sides is where that stops being affordable: the widest window
#: a genus-3 surface's own distances ask for takes a few seconds to build, and
#: the next size up cannot be built at all.
_MAX_HYPERBOLIC_SIDES = 12


def _check_sides(sides: int, what: str) -> None:
    """Refuse a presentation whose deck group is out of computational reach.

    At construction, and with the reason, rather than from inside
    :meth:`~FundamentalDomain.distance` on whichever pair of points first needs
    a wide enough search -- which is the same failure arriving later, less
    clearly, and only sometimes.
    """
    if sides > _MAX_HYPERBOLIC_SIDES:
        raise ValueError(
            f"{what} needs a {sides}-sided polygon, and this implementation stops at "
            f"{_MAX_HYPERBOLIC_SIDES}. Its deck group grows like exp(R) in the radius a "
            "certified distance has to search, and beyond twelve sides that radius cannot be "
            "enumerated. Nothing about the geometry forbids it; the search does."
        )


def _translation(dx: float, dy: float) -> np.ndarray:
    """Return the affine matrix translating by ``(dx, dy)``."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _rectangle_corners(width: float, height: float) -> list[list[float]]:
    """Corners of the centred rectangle, counter-clockwise from the bottom left."""
    half_1, half_2 = width / 2, height / 2
    return [
        [-half_1, -half_2],
        [half_1, -half_2],
        [half_1, half_2],
        [-half_1, half_2],
    ]


def _pairings_from_word(
    model: ModelSpace, corners: np.ndarray, word: list[tuple[int, int]]
) -> list[np.ndarray]:
    r"""Build the side pairings a boundary word asks for.

    `word` gives one ``(label, sign)`` per side, in order, and each label appears
    twice. Two sides with **opposite** signs are glued head to tail -- the
    orientable case, :math:`a \ldots a^{-1}` -- and two with the **same** sign
    are glued head to head, which is a crosscap and reverses orientation.

    Each correspondence is realised by exactly two isometries, one of each
    orientation, and only one of them is a side pairing: the other maps the
    polygon onto *itself* rather than onto the neighbouring copy, and it is
    always the rotation or reflection that would quotient to an orbifold. They
    are told apart by where the outward normal goes -- a side pairing must send
    the outward normal of its source side to the *inward* normal of its target,
    since the image of the polygon lies on the far side.
    """
    count = len(corners)
    positions: dict[int, list[tuple[int, int]]] = {}
    for side, (label, sign) in enumerate(word):
        positions.setdefault(label, []).append((side, sign))

    centre = model.centroid(corners)
    planes, outward = [], []
    for i in range(count):
        plane = model.halfspace(corners[i], corners[(i + 1) % count])
        if float(plane @ model.lift(centre)) > 0:
            plane = -plane
        planes.append(plane)
        outward.append(model.normal_at(plane, model.midpoint(corners[i], corners[(i + 1) % count])))

    pairings = []
    for label, occurrences in sorted(positions.items()):
        if len(occurrences) != 2:
            raise ValueError(f"label {label} appears {len(occurrences)} times in the word, not 2")
        (source, source_sign), (target, target_sign) = occurrences
        if source_sign == target_sign:
            images = (corners[target], corners[(target + 1) % count])
        else:
            images = (corners[(target + 1) % count], corners[target])
        pair = (corners[source], corners[(source + 1) % count])

        for orientation in (1.0, -1.0):
            candidate = isometry_between(model, pair, images, orientation)
            moved = candidate @ outward[source]
            if np.allclose(moved, -np.asarray(outward[target]), atol=1e-7):
                pairings.append(candidate)
                break
        else:  # pragma: no cover - a regular polygon always admits one
            raise ValueError(
                f"neither isometry realising the gluing of sides {source} and {target} carries "
                "the polygon onto the copy across the target side."
            )
    return pairings
