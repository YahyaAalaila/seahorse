r"""Immersions of the closed surfaces this package simulates on into :math:`\mathbb{R}^3`.

A domain is a chart plus a gluing. To draw one, the chart rectangle has to be
carried into ambient :math:`\mathbb{R}^3` by a map that identifies *exactly* the
points the gluing identifies -- no more and no fewer. Two of the four surfaces in
scope already have that map: :class:`~hawkes_package.spatio_temporal.Sphere` and
the projective plane live on a
:class:`~hawkes_package.spatio_temporal._model.SphericalPlane`, whose ``lift`` is
the isometric embedding, so they reuse it rather than carrying a second formula
for the same sphere. The two flat surfaces have no isometric embedding at all and
get an immersion supplied here.

What fails silently
-------------------

**A wrong axis on the Klein bottle.** Its two pairings are a translation and a
glide reflection, and the figure-8 immersion flips its fibre exactly once per
turn of its base circle. Line the base circle up with the translated axis instead
of the glided one and the picture still renders, still looks like a Klein bottle,
and still animates -- but the intensity field is mirrored across one seam and
torn across the other, which reads as an artefact of the simulation rather than
of the drawing. :func:`embed` therefore *reads the pairing matrices* rather than
trusting the constructor that produced them, and
``tests/viz/test_embedding.py`` closes every gluing numerically.

**A tube that swallows the hole.** The donut's tube-to-centre ratio follows the
flat torus's aspect ratio, and once it reaches 1 the surface self-intersects and
renders as an opaque blob with the intensity hidden inside it. Clamped at
:data:`_MAX_TUBE_RATIO`.

What the pictures are not
-------------------------

The flat torus and the Klein bottle admit no isometric :math:`C^2` embedding in
:math:`\mathbb{R}^3`, so the donut and the figure-8 distort distance: the
geodesic distances driving the intensity are **not** the distances a viewer
measures on the screen. :attr:`SurfaceEmbedding.isometric` says which is which
and :attr:`SurfaceEmbedding.note` says what the distortion is, so the caption can
carry it.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..spatio_temporal._model import EuclideanPlane, SphericalPlane
from ..spatio_temporal.domains import FundamentalDomain, SpatialDomain, Sphere, Torus2D

__all__ = ["SurfaceEmbedding", "embed"]

#: Largest tube-to-centre radius the donut is drawn with. At 1 the tube exactly
#: swallows the hole; past it the torus self-intersects, and a self-intersecting
#: surface renders as an opaque blob with the interesting part inside it. A tall
#: thin flat torus is drawn out of proportion rather than drawn wrong.
_MAX_TUBE_RATIO = 0.75

#: Radius of the circle the figure-8 fibre is swept around. The fibre reaches
#: about 1.5 from its own centre, so anything below ~1.6 pulls the immersion
#: through its own axis.
_FIGURE_EIGHT_RADIUS = 2.0

_AmbientFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SurfaceEmbedding:
    """An immersion of a domain's chart rectangle into ambient three-space.

    Parameters
    ----------
    name : str
        The surface's own name, for captions and error messages.
    urange, vrange : tuple of float
        Span of chart coordinate 0 and coordinate 1. These are the coordinates
        the domain itself uses, in the domain's own order, so a chart point may
        be handed straight to
        :meth:`~hawkes_package.SpatialDomain.distance`. They are **not** always
        ``domain.bounds``: the projective plane is drawn on the whole sphere
        while its polygon is only a hemisphere.
    ambient : callable
        Maps a stack of chart points, shape ``(m, 2)``, to ambient points, shape
        ``(m, 3)``. Vectorised rather than pointwise because it is called once
        per grid node and a Python-level loop there is the whole render budget
        on the cheaper domains.
    isometric : bool
        Whether the picture preserves distance. ``False`` means the geodesic
        distances the intensity is built from are not the ones on screen.
    note : str
        What the immersion does to the geometry, in one sentence a caption can
        carry.
    """

    name: str
    urange: tuple[float, float]
    vrange: tuple[float, float]
    ambient: _AmbientFn
    isometric: bool
    note: str

    def point(self, x: np.ndarray) -> np.ndarray:
        """Map a single chart point, shape ``(2,)``, to an ambient point, shape ``(3,)``."""
        return np.asarray(self.ambient(np.asarray(x, dtype=float).reshape(1, 2))[0], dtype=float)


def embed(
    domain: SpatialDomain,
    *,
    outer_radius: float = _FIGURE_EIGHT_RADIUS,
    tube_ratio: float | None = None,
) -> SurfaceEmbedding:
    r"""Return the immersion this package draws `domain` with.

    Parameters
    ----------
    domain : SpatialDomain
        One of :class:`~hawkes_package.spatio_temporal.Sphere`,
        :class:`~hawkes_package.spatio_temporal.Torus2D`, or a
        :class:`~hawkes_package.spatio_temporal.FundamentalDomain` presenting the
        projective plane, the flat torus or the Klein bottle.
    outer_radius : float, optional
        Radius of the circle the tube or the figure-8 fibre is swept around.
        Ignored for the two spherical surfaces, which are drawn at their own
        radius.
    tube_ratio : float, optional
        Tube-to-centre radius of the donut. Defaults to the flat torus's own
        aspect ratio, clamped at :data:`_MAX_TUBE_RATIO` so the surface cannot
        self-intersect.

    Returns
    -------
    SurfaceEmbedding
        The immersion, its chart ranges and what it does to the geometry.

    Raises
    ------
    ValueError
        If `domain` is a surface no immersion is implemented for. Hyperbolic
        surfaces are refused rather than approximated: by Hilbert's theorem no
        complete surface of constant negative curvature embeds isometrically in
        :math:`\mathbb{R}^3`, and a picture that pretended otherwise would be
        making a claim about the geometry that is false.

    Examples
    --------
    >>> from hawkes_package.spatio_temporal import Torus2D
    >>> from hawkes_package.viz import embed
    >>> embed(Torus2D(4.0, 2.0)).isometric
    False
    """
    if not outer_radius > 0:
        raise ValueError(f"outer_radius must be positive, got {outer_radius}")

    if isinstance(domain, Torus2D):
        return _donut(domain.width, domain.height, outer_radius, tube_ratio)

    if isinstance(domain, Sphere):
        return _round_sphere(domain.model, name="sphere", note=_SPHERE_NOTE)

    if isinstance(domain, FundamentalDomain):
        return _embed_quotient(domain, outer_radius, tube_ratio)

    raise ValueError(
        f"no immersion is implemented for {type(domain).__name__}; the supported surfaces are "
        "Sphere, Torus2D, and a FundamentalDomain presenting the projective plane, the flat "
        "torus or the Klein bottle"
    )


# ---------------------------------------------------------------------------
# Dispatch on the presentation, not on the constructor that produced it
# ---------------------------------------------------------------------------


def _embed_quotient(
    domain: FundamentalDomain,
    outer_radius: float,
    tube_ratio: float | None,
) -> SurfaceEmbedding:
    """Choose an immersion by reading `domain`'s model and pairing matrices."""
    model = domain.model

    if isinstance(model, SphericalPlane):
        # The only spherical quotient with a free deck group is the projective
        # plane, and its group is {+-I}. Checked rather than assumed, because
        # `topology.name` is derived from a cell count and would still read
        # "projective plane" for a presentation whose pairings we cannot draw.
        if len(domain.pairings) == 1 and np.allclose(domain.pairings[0], -np.eye(3)):
            return _round_sphere(model, name="projective plane", note=_PROJECTIVE_NOTE)
        raise ValueError(
            f"the only spherical quotient this can draw is the projective plane, whose deck "
            f"group is {{+-I}}; got {len(domain.pairings)} pairing(s) on {domain.topology.name}"
        )

    if not isinstance(model, EuclideanPlane):
        raise ValueError(
            f"no immersion is implemented for a {model.name} quotient ({domain.topology.name}). "
            "By Hilbert's theorem a hyperbolic surface has no isometric embedding in R^3 at "
            "all, and drawing one anyway would misstate its geometry."
        )

    extents = _rectangle_extents(domain.vertices)
    if extents is None:
        raise ValueError(
            f"only a centred axis-aligned rectangle can be drawn as a donut or a figure-8; "
            f"{domain.topology.name} is presented on a {len(domain.vertices)}-sided polygon "
            "that is not one (the hexagonal torus, for instance)"
        )
    width, height = extents

    flip_axis = _flip_axis(domain.pairings, width, height)
    if flip_axis is None:
        return _donut(width, height, outer_radius, tube_ratio)
    return _figure_eight(width, height, outer_radius, flip_axis)


def _rectangle_extents(vertices: np.ndarray) -> tuple[float, float] | None:
    """Width and height of `vertices`, or ``None`` if it is not a centred rectangle."""
    corners = np.asarray(vertices, dtype=float)
    if corners.shape != (4, 2):
        return None
    width = float(corners[:, 0].max() - corners[:, 0].min())
    height = float(corners[:, 1].max() - corners[:, 1].min())
    if not (width > 0 and height > 0):
        return None
    # Every corner of the centred axis-aligned rectangle, and no other
    # quadrilateral, has |x| = width/2 and |y| = height/2 simultaneously.
    if not np.allclose(np.abs(corners), [width / 2, height / 2]):
        return None
    return width, height


def _flip_axis(pairings: list[np.ndarray], width: float, height: float) -> int | None:
    """Which chart axis a glide reflection negates, or ``None`` when both sides translate.

    Returns the axis of the *fibre* -- the coordinate the glide reverses. The
    other axis is then the base circle you traverse to pick the flip up. Reading
    this off the matrices is what keeps the two flat surfaces apart: they differ
    by exactly one sign, and the whole topology is in it.
    """
    periods = (width, height)
    flip: int | None = None

    for matrix in pairings:
        linear = np.asarray(matrix, dtype=float)[:2, :2]
        shift = np.asarray(matrix, dtype=float)[:2, 2]

        for axis in (0, 1):
            other = 1 - axis
            translation = np.zeros(2)
            translation[axis] = periods[axis]
            if np.allclose(linear, np.eye(2)) and np.allclose(np.abs(shift), np.abs(translation)):
                break  # a pure translation along `axis`; no flip here
            reflect = np.eye(2)
            reflect[other, other] = -1.0
            if np.allclose(linear, reflect) and np.allclose(np.abs(shift), np.abs(translation)):
                if flip is not None and flip != other:
                    raise ValueError(
                        "two pairings glide on different axes; that is not one of the two "
                        "closed flat surfaces this can draw"
                    )
                flip = other
                break
        else:
            raise ValueError(
                "a side pairing is neither an axis-aligned translation by a side length nor a "
                f"glide reflection across one:\n{np.asarray(matrix, dtype=float)}"
            )

    return flip


# ---------------------------------------------------------------------------
# The immersions
# ---------------------------------------------------------------------------

_SPHERE_NOTE = "Isometric: the round sphere is drawn as itself, at its own radius."

_PROJECTIVE_NOTE = (
    "Drawn as its double cover, the sphere. The covering map is a local isometry, so distances "
    "on screen are true; the antipodal symmetry of the colouring is the identification itself."
)

_TORUS_NOTE = (
    "An immersion, not an isometry: the flat torus admits no isometric C^2 embedding in R^3, so "
    "the geodesic distances driving the intensity are not the distances you see. The outer rim "
    "is stretched and the inner rim compressed."
)

_KLEIN_NOTE = (
    "The figure-8 immersion, which self-intersects along a circle -- the Klein bottle embeds in "
    "no three-space at all. Distances are not preserved, and the flipped fibre means the "
    "intensity appears mirrored after one turn around the base circle. That is the surface, not "
    "an artefact."
)


def _round_sphere(model: SphericalPlane, *, name: str, note: str) -> SurfaceEmbedding:
    r"""Draw the sphere through the model space's own ``lift``.

    The whole sphere, including the projective plane's case: its polygon is only
    a hemisphere, but its deck group identifies antipodes, so the double cover is
    the honest closed picture and needs no immersion of :math:`\mathbb{RP}^2`
    itself. Callers fold each grid point with
    :meth:`~hawkes_package.SpatialDomain.wrap` before evaluating anything on it.

    Reuses ``lift_many`` rather than re-deriving the spherical coordinates,
    because a second copy of that formula could drift from the one every
    geometric predicate in the package runs through.
    """
    return SurfaceEmbedding(
        name=name,
        urange=(0.0, np.pi),  # colatitude, poles included: the chart is degenerate there,
        vrange=(-np.pi, np.pi),  # the ambient is not. Both endpoints, so the seam closes.
        ambient=model.lift_many,
        isometric=True,
        note=note,
    )


def _donut(
    width: float,
    height: float,
    outer_radius: float,
    tube_ratio: float | None,
) -> SurfaceEmbedding:
    r"""Draw the flat torus as a surface of revolution.

    .. math::

        \alpha = 2\pi x / w, \quad \beta = 2\pi y / h, \quad
        (X, Y, Z) = ((R + r\cos\beta)\cos\alpha,\ (R + r\cos\beta)\sin\alpha,\ r\sin\beta)

    Both seams close exactly, by :math:`2\pi`-periodicity in each of
    :math:`\alpha` and :math:`\beta`.
    """
    ratio = min(height / width, _MAX_TUBE_RATIO) if tube_ratio is None else float(tube_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"tube_ratio must lie strictly between 0 and 1, got {ratio}")
    tube = ratio * outer_radius

    def ambient(xs: np.ndarray) -> np.ndarray:
        points = np.asarray(xs, dtype=float).reshape(-1, 2)
        alpha = 2.0 * np.pi * points[:, 0] / width
        beta = 2.0 * np.pi * points[:, 1] / height
        radial = outer_radius + tube * np.cos(beta)
        return np.stack(
            [radial * np.cos(alpha), radial * np.sin(alpha), tube * np.sin(beta)], axis=-1
        )

    return SurfaceEmbedding(
        name="flat torus",
        urange=(-width / 2, width / 2),
        vrange=(-height / 2, height / 2),
        ambient=ambient,
        isometric=False,
        note=_TORUS_NOTE,
    )


def _figure_eight(
    width: float,
    height: float,
    outer_radius: float,
    flip_axis: int,
) -> SurfaceEmbedding:
    r"""Draw the flat Klein bottle as the figure-8 immersion.

    With :math:`u` the base circle and :math:`v` the fibre,

    .. math::

        \rho = R + \cos(u/2)\sin v - \sin(u/2)\sin 2v, \quad
        (X, Y, Z) = (\rho\cos u,\ \rho\sin u,\ \sin(u/2)\sin v + \cos(u/2)\sin 2v)

    which satisfies :math:`F(u + 2\pi, v) = F(u, -v)`: one turn of the base
    circle reverses the fibre. That is the glide reflection, so `u` must be the
    axis the glide *translates* along and `v` the axis it *negates* --
    `flip_axis` names the latter.

    The fibre parameter carries no offset. The glide negates its coordinate about
    the chart's centre and the immersion negates :math:`v` about :math:`v = 0`,
    so the two centres have to be the same point; the chart rectangle is centred
    on the origin, which makes :math:`v = 2\pi x_{\text{flip}} / w` the map that
    lines them up. The base parameter's offset is free -- it only rotates the
    finished picture -- and is chosen to put :math:`u = 0` at the rectangle's
    lower edge.
    """
    base_axis = 1 - flip_axis
    extents = (width, height)
    flip_period, base_period = extents[flip_axis], extents[base_axis]

    def ambient(xs: np.ndarray) -> np.ndarray:
        points = np.asarray(xs, dtype=float).reshape(-1, 2)
        u = 2.0 * np.pi * (points[:, base_axis] + base_period / 2) / base_period
        v = 2.0 * np.pi * points[:, flip_axis] / flip_period
        fibre = np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)
        height_ = np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v)
        radial = outer_radius + fibre
        return np.stack([radial * np.cos(u), radial * np.sin(u), height_], axis=-1)

    return SurfaceEmbedding(
        name="Klein bottle",
        urange=(-width / 2, width / 2),
        vrange=(-height / 2, height / 2),
        ambient=ambient,
        isometric=False,
        note=_KLEIN_NOTE,
    )
