#!/usr/bin/env python3
r"""Constant-curvature model spaces, in the one form that makes them the same code.

By uniformisation every closed surface is a quotient of exactly one of three
model spaces -- :class:`EuclideanPlane` (:math:`K = 0`),
:class:`SphericalPlane` (:math:`K > 0`) or :class:`HyperbolicPlane`
(:math:`K < 0`) -- by a discrete group acting freely. What lets one
implementation serve all three is that each has a **linear** model, in which
isometries are :math:`3 \times 3` matrices and geodesic half-spaces are linear
half-spaces of the ambient :math:`\mathbb{R}^3`:

===============  ===================  ==================  =============================
model space      ambient point        preserved form      isometries
===============  ===================  ==================  =============================
:math:`E^2`      ``(x, y, 1)``        affine              affine, orthogonal linear part
:math:`S^2_R`    ``|p| = R``          ``diag(1, 1, 1)``   ``O(3)``
:math:`H^2`      ``<p,p> = -1``       ``diag(1, 1, -1)``  ``O(2,1)+``
===============  ===================  ==================  =============================

So the deck-group breadth-first search, the convexity test, the half-space
membership predicate and the Dirichlet reduction are *identical* in all three;
only the bilinear form, the chart and the measure differ. That is what this
module abstracts, and it is why adding a geometry is not a rewrite.

Two conventions are load-bearing and worth stating once:

**Half-spaces are stored as covectors on the ambient coordinates**, so the
membership test is a plain dot product ``c @ lift(p)`` in every geometry -- no
form matrix at the call site. Each is normalised so that ``c @ lift(p)`` equals
the signed geodesic distance to the bounding geodesic *to first order*
(:math:`R\sin(d/R)` on the sphere, :math:`\sinh d` in :math:`H^2`, exactly
:math:`d` in the plane). One absolute tolerance therefore means "on the
boundary" in all three, which is what
:meth:`~hawkes_package.FundamentalDomain.contains` needs.

**The chart is only ever a coordinate system**, used for
:attr:`~hawkes_package.SpatialDomain.bounds`, for the quadrature nodes and for
the Metropolis proposal. Every geometric predicate runs in ambient coordinates,
where it is exact and free of the chart's singularities -- the poles of
:math:`(\theta, \varphi)`, the seam at :math:`\varphi = \pm\pi`, the boundary
circle of the Poincare disc. The one thing the chart owes the rest of the
package is :meth:`ModelSpace.volume_element`, the Jacobian relating chart
measure to surface measure, without which the quadrature weights and the
location sampler both integrate the wrong thing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

__all__ = [
    "EuclideanPlane",
    "HyperbolicPlane",
    "ModelSpace",
    "Motion",
    "SphericalPlane",
    "isometry_between",
]

#: Minkowski form of the hyperboloid model, ``diag(1, 1, -1)``.
_MINKOWSKI = np.diag([1.0, 1.0, -1.0])

#: How far from an exact matrix identity a caller's isometry may sit. Generous
#: enough for a pairing built by composing a dozen matrices, tight enough that a
#: shear or a scaling is still rejected.
_ISOMETRY_TOL = 1e-9


class Motion(NamedTuple):
    """How an isometry of a model space moves it.

    Attributes
    ----------
    kind : str
        Human-readable classification, used verbatim in error messages.
    orientation : float
        ``+1.0`` if the isometry preserves orientation, ``-1.0`` if it reverses
        it.
    free : bool
        Whether the isometry acts **without fixed points** on the model space.
        This is the property that separates a surface from an orbifold, and
        before 0.4.0 nothing in the package checked it: a rotation pairing has
        an orthogonal linear part and determinant ``+1``, so it passed every
        other test, quotiented to a cone point, and produced a domain whose
        geometry silently disagreed with the one the caller meant. The identity
        is reported as *not* free; callers must exclude it explicitly, since it
        is in every group.
    """

    kind: str
    orientation: float
    free: bool


class ModelSpace(ABC):
    r"""A simply connected surface of constant curvature, in its linear model.

    Concrete subclasses supply the ambient embedding, the chart, the measure and
    the classification of isometries. Everything a quotient needs beyond that --
    the group window, the polygon, the vertex cycles -- is written once against
    this interface.

    Attributes
    ----------
    name : str
        Used in error messages and in the topology report.
    curvature : float
        The constant Gaussian curvature :math:`K`. Gauss-Bonnet,
        :math:`\int K\,dA = 2\pi\chi`, ties it to the area and the Euler
        characteristic, which is the cheapest and strongest consistency check
        available on a presentation.
    """

    name: str = "model space"
    curvature: float = 0.0

    # -- chart <-> ambient ---------------------------------------------

    @abstractmethod
    def lift(self, x: np.ndarray) -> np.ndarray:
        """Map a chart point, shape ``(2,)``, to ambient coordinates, shape ``(3,)``."""

    @abstractmethod
    def chart(self, p: np.ndarray) -> np.ndarray:
        """Map an ambient point, shape ``(3,)``, back to the chart, shape ``(2,)``."""

    def lift_many(self, xs: np.ndarray) -> np.ndarray:
        """Lift a stack of chart points, shape ``(m, 2)``, to shape ``(m, 3)``."""
        return np.array([self.lift(x) for x in np.atleast_2d(xs)])

    def chart_many(self, ps: np.ndarray) -> np.ndarray:
        """Chart a stack of ambient points, shape ``(m, 3)``, to shape ``(m, 2)``."""
        return np.array([self.chart(p) for p in np.atleast_2d(ps)])

    # -- the metric ----------------------------------------------------

    @abstractmethod
    def inner(self, u: np.ndarray, v: np.ndarray) -> float:
        """Ambient bilinear form, evaluated on two ambient vectors."""

    @abstractmethod
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Geodesic distance between two chart points."""

    @abstractmethod
    def distances(self, x: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Geodesic distance from one chart point to a stack of them, shape ``(m,)``.

        Separate from :meth:`distance` because the deck-group search evaluates
        it once per group element per call, and a Python loop there dominates
        the cost of a simulation.
        """

    @abstractmethod
    def ambient_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Geodesic distance between two *ambient* points.

        The ambient form is what the deck-group search works in, because acting
        by a group element is a single matrix product there. Charting each image
        back only to measure a distance would be both slower and less accurate.
        """

    @abstractmethod
    def ambient_distances(self, p: np.ndarray, qs: np.ndarray) -> np.ndarray:
        """Geodesic distances from one ambient point to a stack of them, shape ``(m,)``."""

    def in_chart(self, x: np.ndarray) -> bool:  # noqa: ARG002 - the interface, not this default
        """Whether the chart coordinates `x` name a point of the model space.

        Defaults to ``True``. Only the Poincare disc has a chart that does not
        cover the plane it charts, and there the answer matters: the bounding
        box of a hyperbolic polygon has corners outside the unit disc, so the
        quadrature grid contains chart points that are not points of
        :math:`H^2` at all.
        """
        return True

    @abstractmethod
    def volume_element(self, x: np.ndarray) -> float:
        """Return ``sqrt(det g)`` at a chart point: chart measure to surface measure."""

    @abstractmethod
    def hull_volume_element_sup(self, vertices: np.ndarray) -> float:
        """Supremum of :meth:`volume_element` over the geodesic hull of `vertices`.

        Exact, or at worst a true bound -- never a sampled maximum. Rejection
        sampling against a sampled maximum silently truncates the tail of the
        target, which is the same class of defect as a Monte Carlo thinning
        bound and just as quiet.

        The *hull*, not the bounding box: the box of a hyperbolic polygon
        reaches outside the Poincare disc, where the area element is infinite,
        and a supremum taken there would reject everything.
        """

    # -- isometries ----------------------------------------------------

    @abstractmethod
    def is_isometry(self, matrix: np.ndarray) -> bool:
        """Whether a ``3x3`` ambient matrix is an isometry of this model space."""

    @abstractmethod
    def classify(self, matrix: np.ndarray) -> Motion:
        """Classify an isometry: what kind of motion it is, and whether it is free."""

    def apply(self, matrix: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Act by `matrix` on a chart point."""
        return self.chart(matrix @ self.lift(x))

    def apply_many(self, matrices: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Act by a stack of matrices, shape ``(m, 3, 3)``, on one chart point."""
        images: np.ndarray = np.einsum("kij,j->ki", matrices, self.lift(x))
        return self.chart_many(images)

    @abstractmethod
    def frame(self, p: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Ambient matrix carrying the base frame to `p` with the bearing to `direction`.

        `direction` is another chart point: the frame's first axis points along
        the geodesic from `p` towards it. The returned matrix is an isometry, so
        composing one frame with the inverse of another gives the unique
        isometry matching two point-and-bearing pairs.
        """

    @abstractmethod
    def tangent(self, p: np.ndarray, towards: np.ndarray) -> np.ndarray:
        """Return the unit ambient tangent at `p` along the geodesic to `towards`."""

    @abstractmethod
    def halfspace(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Covector of the geodesic through chart points `a` and `b`.

        Normalised so that ``c @ lift(p)`` is the signed geodesic distance from
        `p` to that geodesic, to first order. The sign is arbitrary; the caller
        orients it.
        """

    @abstractmethod
    def normal_at(self, covector: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Return the unit ambient tangent at `p` normal to `covector`'s geodesic.

        Points towards increasing ``covector @ lift(.)``, so it is the *outward*
        normal whenever the covector is oriented to make the domain ``<= 0``.
        """

    # -- polygons ------------------------------------------------------

    def angle(self, previous: np.ndarray, vertex: np.ndarray, following: np.ndarray) -> float:
        """Interior angle at `vertex` of the geodesic path through the three points."""
        first = self.tangent(vertex, previous)
        second = self.tangent(vertex, following)
        cosine = self.inner(first, second)
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def polygon_area(self, vertices: np.ndarray) -> float:
        """Area of the geodesic polygon with these chart vertices.

        Everywhere except the plane this is Gauss-Bonnet applied to a single
        cell: the area is the angle excess over the Euclidean answer, divided by
        the curvature. So it is exact, and needs no quadrature.
        """
        count = len(vertices)
        total = sum(
            self.angle(vertices[i - 1], vertices[i], vertices[(i + 1) % count])
            for i in range(count)
        )
        excess = total - (count - 2) * np.pi
        return float(excess / self.curvature)

    @abstractmethod
    def centroid(self, vertices: np.ndarray) -> np.ndarray:
        """Return a point inside the convex hull of these chart vertices."""

    def midpoint(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Geodesic midpoint of two chart points."""
        span = self.distance(a, b)
        return self.chart(self._advance(self.lift(a), self.tangent(a, b), span / 2.0))

    def chart_bounds(
        self, vertices: np.ndarray, interior: np.ndarray, samples: int = 64
    ) -> np.ndarray:
        r"""Bounding box, in chart coordinates, of the geodesic polygon.

        Sampled rather than read off the corners: a geodesic is a straight line
        in the chart only in the Euclidean case, and taking the corner extremes
        elsewhere clips the domain out of its own bounding box -- which the
        quadrature then measures as smaller than the declared volume, scaling
        the simulated event rate by exactly that factor and reporting nothing.

        The *interior* is sampled too, not just the boundary. In the plane that
        would be redundant, since a bounded region and its boundary have the
        same bounding box. It is not redundant on a sphere: the chart image of a
        hemisphere is a full rectangle in :math:`(\theta, \varphi)` whose
        boundary is mostly not the image of the polygon's boundary -- one edge
        of it is the pole, a single point, and another is the seam, which is
        interior to the region. Fanning geodesics out from `interior` covers the
        polygon, because a convex region is star-shaped about any of its points.
        """
        count = len(vertices)
        steps = np.linspace(0.0, 1.0, samples)
        boundary: list[np.ndarray] = []
        for i in range(count):
            boundary.extend(self._interpolate(vertices[i], vertices[(i + 1) % count], steps))
        points = list(boundary)
        for target in boundary:
            points.extend(self._interpolate(interior, target, steps))
        stack = np.asarray(points, dtype=float)
        return np.column_stack([stack.min(axis=0), stack.max(axis=0)])

    def _interpolate(self, a: np.ndarray, b: np.ndarray, steps: np.ndarray) -> list[np.ndarray]:
        """Chart points along the geodesic from `a` to `b` at the given fractions."""
        span = self.distance(a, b)
        if span == 0.0:  # pragma: no cover - a degenerate edge is rejected earlier
            return [np.asarray(a, dtype=float)]
        base, direction = self.lift(a), self.tangent(a, b)
        return [self.chart(self._advance(base, direction, float(s) * span)) for s in steps]

    @abstractmethod
    def _advance(self, p: np.ndarray, tangent: np.ndarray, arc: float) -> np.ndarray:
        """Ambient point reached from `p` along `tangent` after arc length `arc`."""


# ---------------------------------------------------------------------------
# K = 0
# ---------------------------------------------------------------------------


class EuclideanPlane(ModelSpace):
    """The flat plane, charted by itself and lifted to ``(x, y, 1)``.

    The chart is the identity, so every operation reduces to the plain vector
    arithmetic :class:`~hawkes_package.FundamentalDomain` used before this
    abstraction existed -- deliberately, and to the last bit: the rectangle
    presentation has to keep reproducing :class:`~hawkes_package.Torus2D`
    exactly, and that agreement is what makes the general machinery
    trustworthy.
    """

    name = "Euclidean plane"
    curvature = 0.0

    def lift(self, x: np.ndarray) -> np.ndarray:
        """Homogeneous coordinates ``(x, y, 1)``."""
        return np.array([float(x[0]), float(x[1]), 1.0])

    def chart(self, p: np.ndarray) -> np.ndarray:
        """Drop the homogeneous coordinate, dividing through by it."""
        scale = float(p[2])
        return np.array([float(p[0]) / scale, float(p[1]) / scale])

    def lift_many(self, xs: np.ndarray) -> np.ndarray:
        """Append a column of ones."""
        stack = np.atleast_2d(np.asarray(xs, dtype=float))
        return np.column_stack([stack, np.ones(len(stack))])

    def chart_many(self, ps: np.ndarray) -> np.ndarray:
        """Drop the homogeneous column."""
        stack = np.atleast_2d(np.asarray(ps, dtype=float))
        return np.asarray(stack[:, :2] / stack[:, 2:3])

    def inner(self, u: np.ndarray, v: np.ndarray) -> float:
        """Dot product of the tangent components; the lift's last slot is not a direction."""
        return float(u[0] * v[0] + u[1] * v[1])

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Euclidean norm of the difference."""
        return float(np.linalg.norm(np.asarray(x, dtype=float) - np.asarray(y, dtype=float)))

    def distances(self, x: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Euclidean norms, row by row."""
        return np.asarray(np.linalg.norm(np.atleast_2d(ys) - np.asarray(x, dtype=float), axis=1))

    def ambient_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Euclidean norm, after dividing out the homogeneous coordinate."""
        return self.distance(self.chart(p), self.chart(q))

    def ambient_distances(self, p: np.ndarray, qs: np.ndarray) -> np.ndarray:
        """Euclidean norms, after dividing out the homogeneous coordinate."""
        return self.distances(self.chart(p), self.chart_many(qs))

    def volume_element(self, x: np.ndarray) -> float:  # noqa: ARG002 - the flat chart
        """Return one: the flat chart carries the flat measure."""
        return 1.0

    def hull_volume_element_sup(self, vertices: np.ndarray) -> float:  # noqa: ARG002 - as above
        """Return one: the flat chart carries the flat measure."""
        return 1.0

    def apply_many(self, matrices: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Affine action, written out rather than routed through the lift.

        Identical arithmetic to the pre-abstraction implementation, which is the
        point: the ``abs=1e-12`` agreement with :class:`~hawkes_package.Torus2D`
        is a regression gate, and a reassociated dot product would move it.
        """
        point = np.asarray(x, dtype=float)
        images: np.ndarray = np.einsum("kij,j->ki", matrices[:, :2, :2], point)
        return np.asarray(images + matrices[:, :2, 2])

    def is_isometry(self, matrix: np.ndarray) -> bool:
        """Affine with an orthogonal linear part."""
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=_ISOMETRY_TOL):
            return False
        linear = matrix[:2, :2]
        return bool(np.allclose(linear @ linear.T, np.eye(2), atol=_ISOMETRY_TOL))

    def classify(self, matrix: np.ndarray) -> Motion:
        """Classify a plane isometry, exactly and in closed form.

        Two dimensions need no eigen-decomposition:

        * ``det +1`` with a non-zero rotation angle fixes exactly one point.
        * ``det +1`` with angle zero is a translation, free unless it is the
          identity.
        * ``det -1`` reflects in a line of direction ``v``. It is a *glide*
          reflection -- free -- when the translation has a component along
          ``v``, and a *pure* reflection -- fixing that whole line -- when it
          does not.
        """
        linear = matrix[:2, :2]
        shift = matrix[:2, 2]
        determinant = float(np.linalg.det(linear))
        moved = float(np.linalg.norm(shift))

        if determinant > 0:
            angle = float(np.arctan2(linear[1, 0], linear[0, 0]))
            if abs(angle) > _ISOMETRY_TOL:
                return Motion("rotation", 1.0, free=False)
            if moved <= _ISOMETRY_TOL:
                return Motion("identity", 1.0, free=False)
            return Motion("translation", 1.0, free=True)

        # Reflection in the line at angle `axis / 2`; its +1 eigenvector is that
        # line's direction, and only the component of the shift along it
        # survives the reflection.
        axis = float(np.arctan2(linear[0, 1], linear[0, 0]))
        direction = np.array([np.cos(axis / 2), np.sin(axis / 2)])
        along = float(abs(shift @ direction))
        if along > _ISOMETRY_TOL:
            return Motion("glide reflection", -1.0, free=True)
        return Motion("reflection", -1.0, free=False)

    def frame(self, p: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Affine matrix taking the origin and the ``x`` axis to `p` and the bearing."""
        first = self.tangent(p, direction)[:2]
        second = np.array([-first[1], first[0]])
        matrix = np.eye(3)
        matrix[:2, 0] = first
        matrix[:2, 1] = second
        matrix[:2, 2] = np.asarray(p, dtype=float)
        return matrix

    def tangent(self, p: np.ndarray, towards: np.ndarray) -> np.ndarray:
        """Return the unit direction from `p` to `towards`, with a zero last slot."""
        delta = np.asarray(towards, dtype=float) - np.asarray(p, dtype=float)
        norm = float(np.linalg.norm(delta))
        if norm == 0.0:
            raise ValueError("cannot take a bearing from a point to itself")
        return np.array([delta[0] / norm, delta[1] / norm, 0.0])

    def halfspace(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Covector ``(n_x, n_y, -offset)`` of the line through `a` and `b`."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        edge = b - a
        normal = np.array([edge[1], -edge[0]])
        length = float(np.linalg.norm(normal))
        if length == 0.0:
            raise ValueError("two coincident vertices do not name a unique geodesic")
        normal = normal / length
        return np.array([normal[0], normal[1], -float(normal @ a)])

    def normal_at(self, covector: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return the covector's own normal; in the plane it does not depend on `p`."""
        return np.array([covector[0], covector[1], 0.0])

    def polygon_area(self, vertices: np.ndarray) -> float:
        """Shoelace area; the curvature is zero, so the Gauss-Bonnet form is unavailable."""
        corners = np.asarray(vertices, dtype=float)
        shoelace = np.sum(
            corners[:, 0] * np.roll(corners[:, 1], -1) - np.roll(corners[:, 0], -1) * corners[:, 1]
        )
        return float(abs(0.5 * shoelace))

    def centroid(self, vertices: np.ndarray) -> np.ndarray:
        """Mean of the vertices, which a convex polygon contains."""
        return np.asarray(np.asarray(vertices, dtype=float).mean(axis=0))

    def chart_bounds(
        self,
        vertices: np.ndarray,
        interior: np.ndarray,  # noqa: ARG002 - a straight chart needs neither
        samples: int = 64,  # noqa: ARG002 - of these
    ) -> np.ndarray:
        """Return the corner extremes, exactly: a Euclidean geodesic is straight in this chart.

        Overridden rather than inherited so the quadrature nodes stay the same
        floating-point values they were before the model space existed.
        """
        corners = np.asarray(vertices, dtype=float)
        return np.column_stack([corners.min(axis=0), corners.max(axis=0)])

    def _advance(self, p: np.ndarray, tangent: np.ndarray, arc: float) -> np.ndarray:
        """Straight-line travel."""
        return np.asarray(p + arc * tangent)


# ---------------------------------------------------------------------------
# K > 0
# ---------------------------------------------------------------------------


class SphericalPlane(ModelSpace):
    r"""The round sphere of radius `radius`, charted by colatitude and longitude.

    Ambient points are the vectors of norm :math:`R` in :math:`\mathbb{R}^3` and
    the isometries are :math:`O(3)`, so the linear algebra is the most familiar
    of the three. The chart :math:`(\theta, \varphi)` is singular at the poles
    and seamed at :math:`\varphi = \pm\pi`; neither costs correctness, because
    every predicate runs in ambient coordinates and the two defects form a
    measure-zero set the quadrature never samples. What they do cost is mixing:
    a Metropolis chain confined to the chart box cannot step across the seam.

    Parameters
    ----------
    radius : float
        Sphere radius :math:`R`. The curvature is :math:`1 / R^2`.
    """

    name = "sphere"

    def __init__(self, radius: float = 1.0) -> None:
        if not radius > 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.radius = float(radius)
        self.curvature = 1.0 / self.radius**2

    def lift(self, x: np.ndarray) -> np.ndarray:
        """Colatitude and longitude to a point of norm ``radius``."""
        colatitude, longitude = float(x[0]), float(x[1])
        sine = np.sin(colatitude)
        return self.radius * np.array(
            [sine * np.cos(longitude), sine * np.sin(longitude), np.cos(colatitude)]
        )

    def chart(self, p: np.ndarray) -> np.ndarray:
        """Ambient point to colatitude and longitude, renormalising onto the sphere."""
        point = np.asarray(p, dtype=float)
        norm = float(np.linalg.norm(point))
        if norm == 0.0:
            raise ValueError("the centre of the sphere has no chart coordinates")
        point = point / norm
        return np.array(
            [
                float(np.arccos(np.clip(point[2], -1.0, 1.0))),
                float(np.arctan2(point[1], point[0])),
            ]
        )

    def inner(self, u: np.ndarray, v: np.ndarray) -> float:
        """Euclidean dot product of the ambient vectors."""
        return float(np.dot(u, v))

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Great-circle distance."""
        return self.ambient_distance(self.lift(x), self.lift(y))

    def distances(self, x: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Great-circle distances from one point to a stack of them."""
        return self.ambient_distances(self.lift(x), self.lift_many(ys))

    def ambient_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Great-circle distance from the chord, which stays accurate near zero.

        ``R * arccos(<p, q> / R^2)`` loses half its significant digits as the
        two points approach each other, because ``arccos`` is vertical at one.
        The chord form is exact there, and a distance that is *nearly* zero --
        a point against its own image under a deck element -- is precisely the
        comparison the quotient metric makes most often.
        """
        chord = float(np.linalg.norm(np.asarray(p, dtype=float) - np.asarray(q, dtype=float)))
        return float(2.0 * self.radius * np.arcsin(np.clip(chord / (2.0 * self.radius), 0.0, 1.0)))

    def ambient_distances(self, p: np.ndarray, qs: np.ndarray) -> np.ndarray:
        """Great-circle distances from the chords."""
        chords = np.linalg.norm(np.atleast_2d(qs) - np.asarray(p, dtype=float), axis=1)
        return np.asarray(
            2.0 * self.radius * np.arcsin(np.clip(chords / (2.0 * self.radius), 0.0, 1.0))
        )

    def volume_element(self, x: np.ndarray) -> float:
        r"""Return the area element :math:`R^2 \sin\theta` of the spherical chart.

        Zero at the two poles, which is why the poles must not be quadrature
        nodes: :func:`~hawkes_package.spatio_temporal._integration.restrict`
        rejects a non-positive weight outright, because a zero weight breaks the
        domination argument the whole thinning bound rests on. Gauss-Legendre
        nodes are strictly interior to their panels, so a rule built over the
        closed box never lands on one.
        """
        return float(self.radius**2 * np.sin(float(x[0])))

    def hull_volume_element_sup(self, vertices: np.ndarray) -> float:  # noqa: ARG002 - global
        r"""Return the global maximum :math:`R^2`, attained on the equator.

        Deliberately not tightened to the hull. Whether a spherical polygon's
        hull reaches the equator is not something the vertices alone answer, and
        a bound that is loose only costs rejection efficiency, where a bound
        that is *wrong* costs a biased sample. The hemisphere presenting the
        projective plane touches the equator anyway, so the loss is nil there.
        """
        return float(self.radius**2)

    def is_isometry(self, matrix: np.ndarray) -> bool:
        """Orthogonal."""
        return bool(np.allclose(matrix @ matrix.T, np.eye(3), atol=_ISOMETRY_TOL))

    def classify(self, matrix: np.ndarray) -> Motion:
        """Classify an orthogonal matrix by whether ``1`` is one of its eigenvalues.

        An isometry of the sphere with eigenvalue ``1`` fixes the corresponding
        axis, and one without moves every point. So freeness is exactly the
        invertibility of ``M - I``, and the only free motions are the antipodal
        map and the rotary reflections.
        """
        determinant = float(np.linalg.det(matrix))
        orientation = 1.0 if determinant > 0 else -1.0
        if np.allclose(matrix, np.eye(3), atol=_ISOMETRY_TOL):
            return Motion("identity", 1.0, free=False)
        if np.allclose(matrix, -np.eye(3), atol=_ISOMETRY_TOL):
            return Motion("antipodal map", -1.0, free=True)
        smallest = float(np.linalg.svd(matrix - np.eye(3), compute_uv=False)[-1])
        if smallest <= 1e-7:
            kind = "rotation" if orientation > 0 else "reflection"
            return Motion(kind, orientation, free=False)
        return Motion("rotary reflection", orientation, free=True)

    def frame(self, p: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Orthogonal matrix whose columns are the tangent frame at `p` and `p` itself."""
        first = self.tangent(p, direction)
        base = self.lift(p) / self.radius
        second = np.cross(base, first)
        return np.column_stack([first, second, base])

    def tangent(self, p: np.ndarray, towards: np.ndarray) -> np.ndarray:
        """Return the unit tangent at `p` along the great circle towards `towards`."""
        base, target = self.lift(p) / self.radius, self.lift(towards) / self.radius
        direction = target - float(np.dot(target, base)) * base
        norm = float(np.linalg.norm(direction))
        if norm <= _ISOMETRY_TOL:
            raise ValueError("cannot take a bearing between coincident or antipodal points")
        return np.asarray(direction / norm)

    def halfspace(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the unit pole of the great circle through `a` and `b`."""
        pole = np.cross(self.lift(a), self.lift(b))
        norm = float(np.linalg.norm(pole))
        if norm <= _ISOMETRY_TOL * self.radius**2:
            raise ValueError("two coincident or antipodal vertices do not name a unique geodesic")
        return np.asarray(pole / norm)

    def normal_at(self, covector: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Component of the pole tangent at `p`, normalised."""
        base = self.lift(p) / self.radius
        direction = covector - float(np.dot(covector, base)) * base
        return np.asarray(direction / np.linalg.norm(direction))

    def centroid(self, vertices: np.ndarray) -> np.ndarray:
        """Radial projection of the mean of the lifted vertices."""
        mean = self.lift_many(np.asarray(vertices, dtype=float)).mean(axis=0)
        if float(np.linalg.norm(mean)) <= _ISOMETRY_TOL * self.radius:
            raise ValueError(
                "the vertices average to the centre of the sphere, so they do not determine "
                "an interior point; pass `interior=` explicitly."
            )
        return self.chart(mean)

    def chart_bounds(
        self, vertices: np.ndarray, interior: np.ndarray, samples: int = 64
    ) -> np.ndarray:
        r"""Return the sampled box, widened to the whole circle where the seam is crossed.

        Longitude is *periodic*, so a region straddling :math:`\varphi = \pm\pi`
        has a sampled range that reports the two ends of the chart and misses
        the middle -- and clipping a few thousandths of a radian off a domain is
        exactly the kind of few-per-mille error in the event rate that nothing
        else here would announce. Detecting the crossing and taking the whole
        circle costs only rejected quadrature nodes, and for the hemisphere that
        presents the projective plane the whole circle is the right answer
        anyway.

        Colatitude is not periodic and is clamped rather than widened: the area
        element is negative outside :math:`[0, \pi]`.
        """
        box = super().chart_bounds(vertices, interior, samples)
        count = len(vertices)
        steps = np.linspace(0.0, 1.0, samples)
        for i in range(count):
            longitudes = np.array(
                [p[1] for p in self._interpolate(vertices[i], vertices[(i + 1) % count], steps)]
            )
            if np.any(np.abs(np.diff(longitudes)) > np.pi):
                box[1] = [-np.pi, np.pi]
                break
        box[0] = np.clip(box[0], 0.0, np.pi)
        return box

    def _advance(self, p: np.ndarray, tangent: np.ndarray, arc: float) -> np.ndarray:
        """Travel along a great circle."""
        angle = arc / self.radius
        return np.asarray(np.cos(angle) * p + self.radius * np.sin(angle) * tangent)


# ---------------------------------------------------------------------------
# K < 0
# ---------------------------------------------------------------------------


class HyperbolicPlane(ModelSpace):
    r"""Curvature :math:`-1`, computed on the hyperboloid and charted to the disc.

    The hyperboloid is what makes the code shared: isometries are
    :math:`O(2,1)^+` matrices and geodesic half-spaces are Minkowski-linear, so
    the group search, the convexity test and the Dirichlet reduction are the
    same routines the plane and the sphere use. The Poincare disc appears only
    as the *chart* -- for :attr:`~hawkes_package.SpatialDomain.bounds`, for the
    quadrature grid and for the Metropolis proposal -- where the area element is
    :math:`4 / (1 - |z|^2)^2`. That diverges at the unit circle, but a compact
    quotient has its polygon compactly contained in the disc, so over the
    polygon's own bounding box the element is bounded.
    """

    name = "hyperbolic plane"
    curvature = -1.0

    def lift(self, x: np.ndarray) -> np.ndarray:
        """Poincare disc coordinates to the upper sheet of the hyperboloid."""
        u, v = float(x[0]), float(x[1])
        squared = u * u + v * v
        if squared >= 1.0:
            raise ValueError(
                f"the Poincare disc chart covers |z| < 1; got |z| = {np.sqrt(squared):.6g}"
            )
        scale = 1.0 - squared
        return np.array([2.0 * u / scale, 2.0 * v / scale, (1.0 + squared) / scale])

    def chart(self, p: np.ndarray) -> np.ndarray:
        """Hyperboloid point to Poincare disc coordinates, renormalising onto the sheet."""
        point = self._normalise(np.asarray(p, dtype=float))
        return np.array([point[0] / (1.0 + point[2]), point[1] / (1.0 + point[2])])

    def _normalise(self, p: np.ndarray) -> np.ndarray:
        r"""Project an ambient vector back onto the upper sheet of the hyperboloid.

        A word of a dozen matrices drifts off the sheet by a few ulp, and the
        drift compounds along a Dirichlet reduction that may take dozens of
        steps. Renormalising on every trip back to the chart is what keeps a
        long reduction on the sheet it started on.

        Factored as :math:`(t - r)(t + r)` rather than evaluated as
        :math:`t^2 - r^2`. The two are the same number and not the same
        computation: on the sheet :math:`t^2` and :math:`r^2` agree to within
        one, so subtracting them cancels away every significant digit once
        :math:`t` is large, while the factored form cancels only in the first
        factor and keeps the second intact. That buys roughly eight more
        significant digits, and with them the difference between a deck element
        at displacement 9 and one at 18.

        Past that the representation genuinely runs out -- at
        :math:`t \approx 5 \times 10^7` the spacing of doubles at :math:`t`
        exceeds :math:`1/2t`, which is the whole gap between the sheet and its
        asymptotic cone -- and this says so rather than returning a point on the
        wrong sheet.
        """
        point = np.asarray(p, dtype=float)
        time_like = abs(float(point[2]))
        space_like = float(np.hypot(point[0], point[1]))
        norm = (time_like - space_like) * (time_like + space_like)
        if norm <= 0.0:
            raise ValueError(
                f"a hyperboloid coordinate has reached {time_like:.3g}, where double precision "
                "can no longer tell the sheet from its asymptotic cone. The deck-group window "
                "has been asked for elements at a displacement beyond about 18, which the "
                "hyperboloid model cannot represent; ask for a smaller radius or a shorter word."
            )
        normalised = point / np.sqrt(norm)
        return np.asarray(normalised if normalised[2] > 0 else -normalised)

    def inner(self, u: np.ndarray, v: np.ndarray) -> float:
        """Minkowski form ``u0 v0 + u1 v1 - u2 v2``."""
        return float(u[0] * v[0] + u[1] * v[1] - u[2] * v[2])

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Hyperbolic distance between two chart points."""
        return self.ambient_distance(self.lift(x), self.lift(y))

    def distances(self, x: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Hyperbolic distances from one point to a stack of them."""
        return self.ambient_distances(self.lift(x), self.lift_many(ys))

    def ambient_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        r"""Hyperbolic distance from the Minkowski chord, accurate near zero.

        ``arccosh(-<p, q>)`` loses half its significant digits as the two points
        approach each other. The identity
        :math:`\langle p - q, p - q\rangle = 2(\cosh d - 1) = 4\sinh^2(d/2)`
        turns that into an ``arcsinh``, which is not steep at zero -- and the
        near-zero case is the common one, since the quotient metric spends its
        time comparing a point against images of itself.
        """
        delta = np.asarray(p, dtype=float) - np.asarray(q, dtype=float)
        return float(2.0 * np.arcsinh(np.sqrt(max(0.0, self.inner(delta, delta))) / 2.0))

    def ambient_distances(self, p: np.ndarray, qs: np.ndarray) -> np.ndarray:
        """Hyperbolic distances from the Minkowski chords."""
        deltas = np.atleast_2d(qs) - np.asarray(p, dtype=float)
        squared = np.einsum("ij,jk,ik->i", deltas, _MINKOWSKI, deltas)
        return np.asarray(2.0 * np.arcsinh(np.sqrt(np.maximum(0.0, squared)) / 2.0))

    def in_chart(self, x: np.ndarray) -> bool:
        r"""Whether the chart point lies inside the open unit disc.

        Not a formality: the bounding box of a hyperbolic polygon has corners
        outside the disc -- for the genus-2 octagon the vertices sit at
        :math:`|z| \approx 0.84` and the box corners at :math:`1.19` -- so a
        tensor quadrature grid over that box contains nodes that are not points
        of the plane at all.
        """
        return bool(float(x[0]) ** 2 + float(x[1]) ** 2 < 1.0)

    def volume_element(self, x: np.ndarray) -> float:
        r"""Return the Poincare-disc area element :math:`4 / (1 - |z|^2)^2`."""
        squared = float(x[0]) ** 2 + float(x[1]) ** 2
        return float(4.0 / (1.0 - squared) ** 2)

    def hull_volume_element_sup(self, vertices: np.ndarray) -> float:
        """Attained at whichever vertex is furthest from the disc's centre.

        Exactly, and for a reason particular to this chart: the element depends
        only on :math:`|z|`, and a hyperbolic disc centred at the origin is a
        Euclidean disc centred at the origin, so the geodesic hull of the
        vertices cannot reach further out than the furthest of them.
        """
        radius = float(np.max(np.linalg.norm(np.asarray(vertices, dtype=float), axis=1)))
        if radius >= 1.0:
            raise ValueError(
                f"a vertex sits at |z| = {radius:.6g}, outside the Poincare disc; the polygon "
                "is not compactly contained in the disc."
            )
        return float(4.0 / (1.0 - radius**2) ** 2)

    def is_isometry(self, matrix: np.ndarray) -> bool:
        """Preserves the Minkowski form and the upper sheet."""
        preserved = np.allclose(matrix.T @ _MINKOWSKI @ matrix, _MINKOWSKI, atol=1e-8)
        return bool(preserved and matrix[2, 2] > 0)

    def classify(self, matrix: np.ndarray) -> Motion:
        """Classify by whether the fixed subspace of `matrix` contains a timelike vector.

        A point of :math:`H^2` *is* a timelike ambient direction, so an isometry
        fixes a point of the plane exactly when the eigenspace of ``1`` meets
        the interior of the light cone. That one test separates the two motions
        that are not free -- elliptic rotations and reflections in a geodesic --
        from the three that are: translations along a geodesic, parabolics and
        glide reflections.

        The eigenspace is taken as the null space of ``M - I`` via a singular
        value decomposition rather than an eigenvector solve. It is
        two-dimensional for a reflection, and can be returned spanned by two
        *lightlike* vectors whose span nonetheless contains timelike ones;
        restricting the form to the subspace and looking for a negative
        eigenvalue tests the space, not a particular basis of it.
        """
        determinant = float(np.linalg.det(matrix))
        orientation = 1.0 if determinant > 0 else -1.0
        if np.allclose(matrix, np.eye(3), atol=_ISOMETRY_TOL):
            return Motion("identity", 1.0, free=False)

        _, singular, right = np.linalg.svd(matrix - np.eye(3))
        cutoff = 1e-7 * max(1.0, float(singular[0]))
        fixed = right[singular <= cutoff]
        if len(fixed) > 0:
            restricted = fixed @ _MINKOWSKI @ fixed.T
            if float(np.min(np.linalg.eigvalsh(restricted))) < -1e-9:
                kind = "elliptic rotation" if orientation > 0 else "reflection"
                return Motion(kind, orientation, free=False)

        if orientation < 0:
            return Motion("glide reflection", -1.0, free=True)
        kind = "parabolic" if abs(float(np.trace(matrix)) - 3.0) <= 1e-7 else "translation"
        return Motion(kind, 1.0, free=True)

    def frame(self, p: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Minkowski-orthonormal frame at `p`, with `p` itself as the timelike column."""
        first = self.tangent(p, direction)
        base = self.lift(p)
        second = _MINKOWSKI @ np.cross(base, first)
        return np.column_stack([first, second, base])

    def tangent(self, p: np.ndarray, towards: np.ndarray) -> np.ndarray:
        """Return the unit Minkowski tangent at `p` towards `towards`."""
        base, target = self.lift(p), self.lift(towards)
        direction = target + self.inner(target, base) * base
        norm = self.inner(direction, direction)
        if norm <= _ISOMETRY_TOL:
            raise ValueError("cannot take a bearing from a point to itself")
        return np.asarray(direction / np.sqrt(norm))

    def halfspace(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Covector of the geodesic through `a` and `b`, of unit spacelike norm.

        The Euclidean cross product of the two lifts is already the right
        covector: ``cross(u, v) @ p`` is ``det[u, v, p]``, which vanishes
        exactly on the plane through the origin that cuts out the geodesic.
        """
        covector = np.cross(self.lift(a), self.lift(b))
        norm = float(covector @ _MINKOWSKI @ covector)
        if norm <= _ISOMETRY_TOL:
            raise ValueError("two coincident vertices do not name a unique geodesic")
        return np.asarray(covector / np.sqrt(norm))

    def normal_at(self, covector: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Minkowski-tangential component at `p` of the covector's dual."""
        base = self.lift(p)
        dual = _MINKOWSKI @ covector
        direction = dual + self.inner(dual, base) * base
        return np.asarray(direction / np.sqrt(self.inner(direction, direction)))

    def centroid(self, vertices: np.ndarray) -> np.ndarray:
        """Hyperboloid projection of the mean of the lifted vertices."""
        mean = self.lift_many(np.asarray(vertices, dtype=float)).mean(axis=0)
        return self.chart(mean)

    def _advance(self, p: np.ndarray, tangent: np.ndarray, arc: float) -> np.ndarray:
        """Travel along a geodesic of the hyperboloid."""
        return np.asarray(np.cosh(arc) * p + np.sinh(arc) * tangent)


def isometry_between(
    model: ModelSpace,
    source: tuple[np.ndarray, np.ndarray],
    target: tuple[np.ndarray, np.ndarray],
    orientation: float,
) -> np.ndarray:
    """Build the isometry carrying one point-and-bearing pair onto another.

    An isometry of any of the three model spaces is determined by the image of
    one point, one tangent direction there, and a choice of orientation -- three
    real numbers and a sign, which is exactly the dimension of the isometry
    group. So a side pairing can be *constructed* from the correspondence a
    presentation asks for, rather than written out by hand for each surface.

    Parameters
    ----------
    model : ModelSpace
        The geometry to build in.
    source, target : tuple of array_like
        ``(point, towards)`` chart pairs. The isometry maps the first point to
        the second, and the bearing at the first to the bearing at the second.
    orientation : float
        ``+1`` to preserve orientation, ``-1`` to reverse it. Both are
        isometries; which one a presentation needs is decided by the caller.

    Returns
    -------
    numpy.ndarray
        The ``3x3`` ambient matrix.
    """
    from_frame = model.frame(source[0], source[1])
    to_frame = model.frame(target[0], target[1])
    if orientation < 0:
        to_frame = to_frame.copy()
        to_frame[:, 1] *= -1.0
    return np.asarray(to_frame @ np.linalg.inv(from_frame))
