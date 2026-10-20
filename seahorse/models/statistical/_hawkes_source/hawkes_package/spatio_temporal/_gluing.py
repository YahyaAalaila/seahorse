#!/usr/bin/env python3
r"""Side pairings, vertex cycles and the topology they determine.

A polygon plus a set of side-pairing isometries is a *presentation*, and a
presentation is a surface only if it satisfies Poincare's polygon theorem: the
pairings must carry each side onto another side, and walking the corners of the
polygon through those pairings must close up with an angle sum of exactly
:math:`2\pi` and a trivial cycle transformation. Anything else glues to
something that is not a closed surface -- most commonly an *orbifold*, where the
angle sum is :math:`2\pi/m` and the quotient has a cone point.

Nothing downstream notices the difference. The Ogata loop, the quadrature and
the location sampler all run happily on an orbifold presentation and produce a
process on the wrong space, so the checks here are made **at construction**,
where the caller can still act on them. That replaces the older behaviour, in
which a bad presentation surfaced -- if at all -- as a
:class:`ValueError` from :meth:`~hawkes_package.FundamentalDomain.wrap`, deep
inside a simulation and long after the mistake.

Three things live here:

* :func:`group_window` and :func:`window_by_radius`, the two truncations of the
  deck group. Word length is a heuristic; displacement radius is a *bound*, and
  :meth:`~hawkes_package.FundamentalDomain.distance` uses it for that reason.
* :func:`infer_pairing`, which recovers which side goes to which. The pairings
  a caller supplies generate the group but need not be the individual side
  pairings -- the hexagonal torus has three pairs of sides and only two
  generators -- so the search runs over a group window rather than over the
  generators alone.
* :func:`vertex_cycles` and :func:`topology`, the corner walk and the
  Euler characteristic, orientability and genus read off from it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._model import ModelSpace

__all__ = [
    "Pairing",
    "Topology",
    "VertexCycle",
    "group_window",
    "infer_pairing",
    "topology",
    "vertex_cycles",
    "window_by_radius",
]

#: Hard cap on the *answer*: how many elements a radius search may return. A
#: discrete group puts finitely many in any ball, so only a non-discrete one
#: reaches this -- and it would otherwise show up as a hang. Legitimate windows
#: are tiny by comparison: a genus-3 surface needs 193 elements for the widest
#: radius its own distances ever ask for.
_MAX_KEPT = 20_000

#: Budget for the *search*, which is a different quantity and much larger. The
#: slack that makes the result complete is a circumradius wide, and in negative
#: curvature the element count grows like exp(R), so certifying an answer of a
#: couple of hundred elements legitimately touches tens of thousands. Counting
#: the two together -- as this did until 0.4.0 -- rejects a genus-3 surface for
#: a cost its answer never incurs, and does it from inside `distance`, so
#: whether a run survives depends on which pairs it happened to compare first.
_MAX_EXPLORED = 400_000


#: Separation, on the unit sphere of ambient directions, below which two deck
#: elements count as the same one. Real neighbouring copies stay far above it
#: over every radius these quotients are searched at, and accumulated rounding
#: over a word of a dozen matrices stays far below.
_DEDUP_TOL = 1e-9


class _ElementIndex:
    r"""Deduplicate deck elements by where they send one reference point.

    Two group elements agreeing on a single point differ by something fixing
    that point, and a side-pairing group acts freely, so agreeing on one point
    means being the same element. That makes the image of the polygon's centre
    a complete fingerprint -- and a far better behaved one than the matrix.

    Two other keys were tried, and both fail the same way, for the same reason,
    at different depths.

    *The matrix*, rounded to fixed decimals, deduplicates up to about ten
    letters. Beyond that a hyperbolic word's entries have grown to
    :math:`\cosh R`, its rounding error has grown with them, two spellings of
    one element round to different keys, and the search stops converging: it ran
    to convergence at word length 8 and re-exploded at 11, growing without bound
    over a group with a few thousand elements in the ball being searched.

    *The chart* fails sooner and more quietly. The Poincare disc crowds the
    whole of infinity into a unit circle, so the images of distinct elements at
    radius :math:`R` sit :math:`O(e^{-R})` apart in it, and by word length five
    on a genus-2 surface that is already comparable to any fixed tolerance --
    the two keys disagree on the size of that ball. A key that is losing
    resolution can merge distinct elements as easily as split one, and merging
    is the direction that fails silently: the window comes back short and the
    quotient metric returns a distance that is too large, which reads as a
    kernel that has already decayed.

    The *ambient direction* -- the image point divided by its Euclidean norm --
    has neither problem. Distinct copies stay a bounded Minkowski chord apart
    however far out they go, so dividing by a norm that grows with them leaves
    the separation comfortably above rounding.

    Lookup is by grid bucket rather than by exact key, so a point sitting on a
    rounding boundary still matches its neighbour.
    """

    def __init__(self, tol: float = _DEDUP_TOL) -> None:
        self.tol = tol
        self._buckets: dict[tuple[int, ...], list[np.ndarray]] = {}
        self.size = 0

    def add(self, image: np.ndarray) -> bool:
        """Record an ambient image point and return whether it was new."""
        point = np.asarray(image, dtype=float)
        point = point / np.linalg.norm(point)
        cell = np.floor(point / self.tol).astype(np.int64)
        for offsets in _NEIGHBOURS:
            for seen in self._buckets.get(tuple(cell + offsets), ()):
                if np.all(np.abs(seen - point) <= self.tol):
                    return False
        self._buckets.setdefault(tuple(cell), []).append(point)
        self.size += 1
        return True


#: The 27 cells a point could have a within-tolerance neighbour in.
_NEIGHBOURS = np.array(
    [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
)


def group_window(
    model: ModelSpace, generators: list[np.ndarray], centre: np.ndarray, depth: int
) -> np.ndarray:
    """Every word of length at most `depth` in `generators` and their inverses.

    Returned as a ``(m, 3, 3)`` stack with the identity first, so a caller can
    treat index 0 as "no move". Deduplicated by where each word sends `centre`,
    which matters twice over: for an abelian group the word count grows
    exponentially while the element count does not, and for a hyperbolic one a
    naive matrix key stops deduplicating at all once the entries are large
    enough to lose their last digits.
    """
    identity = np.eye(3)
    alphabet = [*generators, *(np.linalg.inv(g) for g in generators)]
    base = model.lift(centre)

    index = _ElementIndex()
    index.add(base)
    elements = [identity]
    frontier = [identity]
    for _ in range(max(0, depth)):
        next_frontier = []
        for word in frontier:
            for letter in alphabet:
                candidate = letter @ word
                if index.add(candidate @ base):
                    elements.append(candidate)
                    next_frontier.append(candidate)
        frontier = next_frontier
    return np.array(elements)


def window_by_radius(
    model: ModelSpace,
    generators: list[np.ndarray],
    centre: np.ndarray,
    radius: float,
    slack: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Every deck element moving `centre` no further than `radius`, identity first.

    Word length truncates the group by how a presentation was *written*;
    displacement truncates it by geometry, which is the quantity every consumer
    actually cares about. For
    :meth:`~hawkes_package.FundamentalDomain.distance` that turns a heuristic
    into a theorem — see the note there for the radius that suffices.

    The search expands every element within ``radius + slack`` and keeps every
    element within `radius`. The slack is what makes it complete rather than
    merely plausible. The copies of the polygon tile the model space, so the
    geodesic from ``centre`` to ``g.centre`` crosses a chain of copies, each
    adjacent to the next and therefore differing from it by a single generator.
    Every copy in that chain contains a point of the geodesic, so within
    `radius` of ``centre``, and its own centre is within a circumradius of any
    of its points. Passing the polygon's circumradius as `slack` therefore
    reaches every such ``g``, and stopping the expansion outside it drops
    nothing.

    In a hyperbolic geometry that margin is not a formality: the element count
    grows like :math:`e^R`, so an over-generous slack is the difference between
    a few hundred elements and a few million.

    Parameters
    ----------
    model : ModelSpace
        Geometry in which the displacement is measured.
    generators : list of numpy.ndarray
        Side pairings, as ambient ``3x3`` matrices.
    centre : numpy.ndarray
        Chart point the displacement is measured at.
    radius : float
        Elements moving `centre` further than this are not returned.
    slack : float
        Extra radius the breadth-first search expands through. Must be at least
        the polygon's circumradius for the result to be complete.

    Returns
    -------
    elements : numpy.ndarray
        Shape ``(m, 3, 3)``, identity first, sorted by displacement.
    displacements : numpy.ndarray
        Shape ``(m,)``: how far each element moves `centre`. Returned rather
        than recomputed, so the caller can binary-search or early-exit on it.

    Raises
    ------
    RuntimeError
        If the answer exceeds ``_MAX_KEPT``, which only a non-discrete group can
        do, or the search exceeds ``_MAX_EXPLORED``, which means the radius is
        beyond what this curvature affords.
    """
    identity = np.eye(3)
    alphabet = np.array([*generators, *(np.linalg.inv(g) for g in generators)])
    base = model.lift(centre)

    index = _ElementIndex()
    index.add(base)
    kept = [identity]
    moved_kept = [0.0]
    frontier = identity[None]
    while len(frontier):
        # Batched, because the frontier of a hyperbolic group is hundreds of
        # elements wide and a Python loop over the products dominates.
        products = np.einsum("aij,fjk->afik", alphabet, frontier).reshape(-1, 3, 3)
        images = np.einsum("kij,j->ki", products, base)
        fresh = [i for i in range(len(products)) if index.add(images[i])]
        if index.size > _MAX_EXPLORED:
            raise RuntimeError(
                f"the deck-group search visited more than {_MAX_EXPLORED} elements while "
                f"certifying a window of radius {radius:.6g}. In negative curvature the "
                "element count grows like exp(R), so this radius is out of reach; a smaller "
                "polygon, or fewer handles, is."
            )
        if not fresh:
            break
        candidates = products[fresh]
        displaced = model.ambient_distances(base, images[fresh])
        kept.extend(candidates[displaced <= radius])
        moved_kept.extend(displaced[displaced <= radius])
        if len(kept) > _MAX_KEPT:
            raise RuntimeError(
                f"more than {_MAX_KEPT} deck elements move the centre by at most "
                f"{radius:.6g}. A discrete group puts finitely many in any ball, so the side "
                "pairings do not generate one."
            )
        frontier = candidates[displaced <= radius + slack]

    elements = np.array(kept)
    displacements = np.array(moved_kept)
    order = np.argsort(displacements, kind="stable")
    return elements[order], displacements[order]


class Pairing(NamedTuple):
    """Which side of the polygon goes to which, and by what.

    Attributes
    ----------
    partner : numpy.ndarray
        ``partner[i]`` is the index of the side that side ``i`` is glued to. A
        fixed-point-free involution on the side indices.
    forward : numpy.ndarray
        Shape ``(n, 3, 3)``. ``forward[i]`` carries side ``i`` onto side
        ``partner[i]``, and carries the polygon onto the copy on the *far* side
        of it. ``forward[partner[i]]`` is its inverse.
    reverses : numpy.ndarray
        Whether ``forward[i]`` reverses the direction of the side it carries.
        Recorded for reporting rather than for the algorithm, which reads the
        vertex correspondence directly.
    """

    partner: np.ndarray
    forward: np.ndarray
    reverses: np.ndarray


def _vertex_index(model: ModelSpace, vertices: np.ndarray, point: np.ndarray, tol: float) -> int:
    """Index of the vertex `point` coincides with, or ``-1``."""
    for index, vertex in enumerate(vertices):
        if model.distance(vertex, point) <= tol:
            return index
    return -1


def infer_pairing(
    model: ModelSpace,
    vertices: np.ndarray,
    generators: list[np.ndarray],
    outward: np.ndarray,
    centre: np.ndarray,
    tol: float,
    max_depth: int = 4,
) -> Pairing:
    """Recover the side-pairing correspondence from the generators.

    Poincare's theorem needs to know *which* side each pairing carries to which,
    and a caller supplies only a generating set. Those two are not the same
    thing: the hexagonal torus has three pairs of sides and is generated by two
    translations, the third pairing being their difference. So the search runs
    over words of increasing length until every side is matched.

    A candidate ``g`` pairs side ``i`` to side ``j`` when it carries the two
    endpoints of ``i`` onto the two endpoints of ``j`` *and* carries the outward
    normal of ``i`` to minus the outward normal of ``j``. That second condition
    is not decoration: there are always two isometries realising a given vertex
    correspondence, and only one of them puts the image of the polygon on the
    far side of the target side. The other maps the polygon onto itself, and it
    is exactly the rotation or reflection that would quotient to an orbifold.

    Parameters
    ----------
    model : ModelSpace
        The geometry.
    vertices : numpy.ndarray
        Polygon corners, shape ``(n, 2)``, in chart coordinates.
    generators : list of numpy.ndarray
        Side pairings supplied by the caller.
    outward : numpy.ndarray
        Unit outward ambient normals at the side midpoints, shape ``(n, 3)``.
    centre : numpy.ndarray
        A chart point inside the polygon, used to deduplicate group elements.
    tol : float
        Distance below which two points count as the same.
    max_depth : int
        Longest word searched before giving up.

    Returns
    -------
    Pairing

    Raises
    ------
    ValueError
        If some side is left unpaired, or if a side is paired with itself.
    """
    count = len(vertices)
    midpoints = _midpoints(model, vertices)
    partner = np.full(count, -1, dtype=int)
    forward = np.zeros((count, 3, 3))
    reverses = np.zeros(count, dtype=bool)

    for depth in range(1, max_depth + 1):
        window = group_window(model, generators, centre, depth)
        for side in range(count):
            if partner[side] >= 0:
                continue
            match = _match_side(model, vertices, midpoints, outward, window, side, tol)
            if match is None:
                continue
            element, target, flipped = match
            if target == side:  # pragma: no cover - freeness rules this out first
                # Unreachable while every pairing is checked for freeness, and
                # kept because that argument is not local: an isometry carrying
                # a geodesic segment onto itself either fixes it pointwise or
                # reverses it about its midpoint, and both have a fixed point.
                # If freeness ever moves or weakens, this is what would
                # otherwise let half a side be glued to the other half.
                raise ValueError(
                    f"side {side} of the polygon is paired with itself. Half of it would have "
                    "to be identified with the other half, which no free action does: the "
                    "pairing fixes the side's midpoint, so the quotient is an orbifold."
                )
            partner[side], partner[target] = target, side
            forward[side], forward[target] = element, np.linalg.inv(element)
            reverses[side] = reverses[target] = flipped
        if np.all(partner >= 0):
            return Pairing(partner, forward, reverses)

    unpaired = np.flatnonzero(partner < 0).tolist()
    raise ValueError(
        f"sides {unpaired} of the polygon are not carried onto another side by any word of "
        f"length up to {max_depth} in the pairings. Every side of a fundamental domain must "
        "be glued to exactly one other, or the quotient has a boundary rather than being a "
        "closed surface."
    )


def _midpoints(model: ModelSpace, vertices: np.ndarray) -> np.ndarray:
    """Geodesic midpoint of every side, in chart coordinates."""
    count = len(vertices)
    points = [model.midpoint(vertices[i], vertices[(i + 1) % count]) for i in range(count)]
    return np.asarray(points, dtype=float)


def _match_side(
    model: ModelSpace,
    vertices: np.ndarray,
    midpoints: np.ndarray,
    outward: np.ndarray,
    window: np.ndarray,
    side: int,
    tol: float,
) -> tuple[np.ndarray, int, bool] | None:
    """Find the group element carrying `side` onto another side, if there is one."""
    count = len(vertices)
    start, end = side, (side + 1) % count
    for element in window[1:]:
        first = _vertex_index(model, vertices, model.apply(element, vertices[start]), tol)
        second = _vertex_index(model, vertices, model.apply(element, vertices[end]), tol)
        if first < 0 or second < 0:
            continue
        if second == (first + 1) % count:
            target, flipped = first, False
        elif first == (second + 1) % count:
            target, flipped = second, True
        else:
            continue
        # The polygon must land on the *far* side of the target: the outward
        # normal has to arrive pointing inwards. Without this an element mapping
        # the polygon onto *itself* would be accepted as a side pairing, and
        # such an element is exactly the rotation or reflection that quotients
        # to an orbifold.
        moved = element @ outward[side]
        if not np.allclose(moved, -outward[target], atol=1e-6):
            continue
        if model.distance(model.apply(element, midpoints[side]), midpoints[target]) > tol:
            continue  # pragma: no cover - the normal test above already excludes it
        return element, target, flipped
    return None


class VertexCycle(NamedTuple):
    r"""One orbit of polygon corners under the side pairings.

    Attributes
    ----------
    corners : tuple of int
        Vertex indices visited, in walk order. Each becomes the same point of
        the quotient surface.
    angle_sum : float
        Total interior angle around that point. Poincare's theorem requires
        exactly :math:`2\pi` for a free action; :math:`2\pi/m` gives a cone
        point of order ``m``, and the quotient is an orbifold.
    transformation : numpy.ndarray
        The accumulated isometry after one full turn. The identity for a free
        action.
    """

    corners: tuple[int, ...]
    angle_sum: float
    transformation: np.ndarray


def vertex_cycles(
    model: ModelSpace,
    vertices: np.ndarray,
    pairing: Pairing,
    angles: np.ndarray,
    tol: float,
) -> list[VertexCycle]:
    r"""Walk the corners of the polygon through the side pairings.

    The walk is over *flags* -- a corner together with which of its two sides
    the walk is about to cross -- rather than over corners, because a corner is
    visited once for each of its sides and the two visits belong to different
    cycles.

    One step crosses side ``e`` of the current copy. The copy on the other side
    of ``e`` is ``forward[e]**-1`` applied to the polygon, so the accumulated
    isometry right-multiplies by that inverse; the corner the walk is turning
    about is unmoved, so in the new copy's own coordinates it is the vertex
    ``forward[e]`` sends the old one to; and the next side to cross is the other
    of the two sides meeting there. The cycle closes when the walk returns to
    the flag it started from.

    Two quantities come out. The **angle sum** must be :math:`2\pi`: the copies
    around the point must close up exactly once. The **cycle transformation**
    must be the identity: closing up geometrically is not enough if the group
    element that does it is a non-trivial rotation about the point, which is
    precisely a cone point of the quotient.

    Flags come in pairs -- turning the other way round the same point walks the
    same corners backwards -- so each vertex of the quotient produces *two* flag
    cycles, and only one of them is reported. Counting both would double every
    ``V`` and put the Euler characteristic out by exactly the number of
    vertices, which is the kind of error a torus reporting :math:`\chi = 1`
    announces immediately and a subtler presentation would not.

    Returns
    -------
    list of VertexCycle
        One per vertex of the glued surface, in order of first appearance.
    """
    count = len(vertices)
    unvisited = {(j, e) for j in range(count) for e in ((j - 1) % count, j)}
    cycles: list[VertexCycle] = []

    def reverse(flag: tuple[int, int]) -> tuple[int, int]:
        """Return the same corner, approached from its other side."""
        corner, side = flag
        return (corner, corner if side == (corner - 1) % count else (corner - 1) % count)

    while unvisited:
        start = min(unvisited)
        corner, side = start
        accumulated = np.eye(3)
        corners: list[int] = []
        walked: list[tuple[int, int]] = []
        total = 0.0
        for _ in range(2 * count):
            walked.append((corner, side))
            corners.append(corner)
            total += float(angles[corner])
            element = pairing.forward[side]
            accumulated = accumulated @ np.linalg.inv(element)
            landed = _vertex_index(model, vertices, model.apply(element, vertices[corner]), tol)
            if landed < 0:  # pragma: no cover - the pairing search proves otherwise
                raise ValueError("a side pairing does not carry a corner onto a corner")
            crossed = pairing.partner[side]
            side = (landed - 1) % count if crossed == landed else landed
            corner = landed
            if (corner, side) == start:
                break
        else:  # pragma: no cover - a flag walk cannot exceed the flag count
            raise ValueError("the corner walk did not close; the side pairings are inconsistent")
        for flag in walked:
            unvisited.discard(flag)
            unvisited.discard(reverse(flag))
        cycles.append(VertexCycle(tuple(corners), total, accumulated))

    return cycles


class Topology(NamedTuple):
    """What surface a presentation glues to.

    Attributes
    ----------
    orientable : bool
        False as soon as one pairing reverses orientation.
    euler_characteristic : int
        ``V - E + F`` for the cell structure the presentation defines: one face,
        one edge per pair of sides, one vertex per corner cycle.
    genus : int
        Handles for an orientable surface; crosscaps -- the *non-orientable*
        genus -- otherwise. A sphere and a torus are genus 0 and 1; the
        projective plane and the Klein bottle are crosscap 1 and 2.
    name : str
        Plain-language name of the surface, for error messages and ``repr``.
    """

    orientable: bool
    euler_characteristic: int
    genus: int
    name: str


#: The four surfaces common enough to deserve their own name.
_NAMED = {
    (True, 0): "torus",
    (True, 2): "sphere",
    (False, 1): "projective plane",
    (False, 0): "Klein bottle",
}


def topology(n_vertex_cycles: int, n_sides: int, orientable: bool) -> Topology:
    r"""Euler characteristic, genus and name of the glued surface.

    The presentation is a CW structure with one face, ``n_sides / 2`` edges --
    the sides are glued in pairs -- and one vertex per corner cycle, so
    :math:`\chi = V - E + F` is immediate. Genus follows from
    :math:`\chi = 2 - 2g` when orientable and :math:`\chi = 2 - k` when not.
    """
    characteristic = n_vertex_cycles - n_sides // 2 + 1
    genus = (2 - characteristic) // 2 if orientable else 2 - characteristic
    if (orientable, characteristic) in _NAMED:
        name = _NAMED[(orientable, characteristic)]
    elif orientable:
        name = f"genus-{genus} surface"
    else:
        name = f"non-orientable surface of genus {genus}"
    return Topology(orientable, characteristic, genus, name)
