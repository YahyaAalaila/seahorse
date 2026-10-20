r"""The intensity sampled on a chart grid, one array per animation frame.

Every value here comes from the hooks the simulator thins against. The slow path
*is*
:meth:`~hawkes_package.spatio_temporal.process.SpatioTemporalHawkesProcess.intensity`,
called once per grid node per frame. The fast path is that same sum with its
time-independent factors hoisted out of the frame loop:

.. math::

    \lambda(t_k, x_j) = \max\Big(0,\ \mu(x_j)
        + \sum_{t_i < t_k} \kappa_t(t_k - t_i)\,\kappa_s\big(x_j, x_i\big)\Big)

Neither :math:`\mu(x_j)` nor the spatial factor depends on :math:`k`, so both are
computed once and the frame loop reduces to one weighted sum per node. That is a
rearrangement, not a second expression -- and the spatial factor is built by
calling ``process._spatial_at``, the process's *own* kernel dispatch, so the
pairwise and the distance protocols are handled by the code that handles them
during a simulation rather than by a copy of it.

Bit-exactness, and why the reduction is shaped the way it is
------------------------------------------------------------

``_full_intensity`` reduces with ``np.multiply(temporal, spatial).sum()`` over a
one-dimensional array holding **only the past events**. Reproducing the value bit
for bit means reproducing that shape, because floating-point addition is not
associative and numpy's pairwise summation blocks differently for different
lengths and layouts. Measured over 5 625 (frame, node) pairs on a
:class:`~hawkes_package.spatio_temporal.Torus2D`, against ``process.intensity``:

===========================================================  ================
reduction                                                    exact mismatches
===========================================================  ================
``np.multiply(past_temporal, spatial[j, past]).sum()``        **0**
the same over a zero-padded full-length vector                       175
``spatial @ temporal`` (BLAS matrix-vector product)                  454
===========================================================  ================

So the loop over nodes is deliberate and **the matvec is forbidden**: it is
faster and it is not the same number. The differences are one ulp and would never
show in a picture, but an exact test cannot drift and an approximate one can, so
the stronger property is the one worth keeping. ``tests/viz/test_field.py`` pins
all three rows of that table.

What fails silently
-------------------

* **The strict inequality.** The intensity at an event time is the *left* limit,
  so the sum runs over ``t_i < t_k`` strictly. Admitting the event at ``t_k``
  would add a full self-excitation term the simulator never applied.
* **An impure kernel, or an event record that moves.** The hoist is valid only
  while ``base``, ``spatial`` and ``temporal`` are functions of their arguments
  and ``process.events`` holds still. Neither is checkable statically, so it is
  checked at runtime: :data:`_SELF_CHECK_SAMPLES` finished values are compared
  against ``process.intensity`` and a disagreement raises.
* **A per-frame colour scale.** Rescaling each frame to its own maximum makes
  every frame equally bright and destroys the one thing an animation of a
  self-exciting process is for. :attr:`IntensityFrames.vmax` is global.
* **An unfolded grid.** See the note on :func:`intensity_frames`.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._numerics import as_float
from ._embedding import SurfaceEmbedding, embed

if TYPE_CHECKING:  # pragma: no cover - runtime import would be circular
    from ..spatio_temporal.process import SpatioTemporalHawkesProcess

__all__ = ["IntensityFrames", "event_opacities", "intensity_frames"]

#: Default chart grid. Chosen against the *output* size rather than the compute:
#: one three.js page carries a position and a colour per vertex per frame, which
#: is roughly ``nu * nv * n_frames * 150`` bytes -- 14 MB at 48x48 over 40
#: frames, 25 MB at 64x64. The page size binds sooner than the render does.
DEFAULT_RESOLUTION = (48, 48)

#: How many finished values are checked back against ``process.intensity``.
#: Enough to catch a callable that is not a function of its arguments, cheap
#: enough to be unconditional -- well under a second even on the projective
#: plane, whose ``distance`` costs 327 us.
_SELF_CHECK_SAMPLES = 16


@dataclass(frozen=True)
class IntensityFrames:
    """The conditional intensity on a chart grid, at each of several times.

    Parameters
    ----------
    times : numpy.ndarray, shape (n_t,)
        The frame times, as given.
    chart : numpy.ndarray, shape (nu, nv, 2)
        The grid in the domain's own chart coordinates, already folded by
        :meth:`~hawkes_package.SpatialDomain.wrap`. These are the points the
        intensity was actually evaluated at, so a caller checking a value
        against ``process.intensity`` must use them and not the unfolded grid.
    values : numpy.ndarray, shape (n_t, nu, nv)
        The intensity, floored at zero exactly as the simulator floors it.
    vmin, vmax : float
        The colour range, global across every frame.
    factorised : bool
        Whether the hoisted path ran. ``False`` means every value came from a
        direct call to ``process.intensity``.
    """

    times: np.ndarray
    chart: np.ndarray
    values: np.ndarray
    vmin: float
    vmax: float
    factorised: bool

    def normalised(self, index: int) -> np.ndarray:
        """Return frame `index` rescaled to ``[0, 1]``, the range a colour map needs.

        Parameters
        ----------
        index : int
            Which frame.

        Returns
        -------
        numpy.ndarray, shape (nu, nv)
            The frame mapped through the *global* range, then clipped. Clipping
            bites only when a caller has narrowed `vmax` by hand to bring out
            detail near the background.
        """
        span = self.vmax - self.vmin
        if not span > 0:
            # A flat field. Half way up the map reads as "nothing to see"; zero
            # would read as the map's darkest value and imply a measured minimum.
            return np.full(self.values[index].shape, 0.5)
        unit: np.ndarray = np.clip((self.values[index] - self.vmin) / span, 0.0, 1.0)
        return unit

    def summary(self) -> str:
        """One line naming the grid, the frame count and the colour range."""
        nu, nv = self.chart.shape[:2]
        path = "factorised" if self.factorised else "direct"
        return (
            f"{len(self.times)} frames on a {nu}x{nv} grid, "
            f"lambda in [{self.vmin:.4g}, {self.vmax:.4g}], {path}"
        )


def intensity_frames(
    process: SpatioTemporalHawkesProcess,
    times: Any,
    *,
    embedding: SurfaceEmbedding | None = None,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    fast: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> IntensityFrames:
    r"""Evaluate :math:`\lambda(t, x \mid H_t)` on a chart grid, once per frame time.

    Parameters
    ----------
    process : SpatioTemporalHawkesProcess
        A process carrying a realisation, simulated or assigned to
        ``process.events``. The intensity is conditional on that record.
    times : array_like, shape (n_t,)
        Frame times, used exactly as given. Unlike ``intensity_over_interval``
        the realised event times are **not** merged in, because an animation
        needs a fixed cadence and merging would silently change the frame rate.
    embedding : SurfaceEmbedding, optional
        The immersion whose chart ranges the grid spans. Defaults to
        :func:`~hawkes_package.viz.embed` of the process's domain. It is the
        embedding and not ``domain.bounds`` that sets the ranges: the projective
        plane is drawn on the whole sphere while its polygon is a hemisphere.
    resolution : tuple of int, optional
        Nodes along chart axis 0 and axis 1.
    fast : bool, optional
        Whether to hoist the time-independent factors out of the frame loop. The
        result is the same number either way, bit for bit; ``False`` calls
        ``process.intensity`` directly and is what the equivalence test compares
        against.
    vmin, vmax : float, optional
        Override the colour range. By default `vmin` is zero -- the intensity is
        floored there, so it is a true minimum and not a sampled one -- and
        `vmax` is the largest value over every frame.

    Returns
    -------
    IntensityFrames
        The grid, the values, and the range they are to be coloured through.

    Raises
    ------
    ValueError
        If `resolution` is not two integers of at least two, or if the colour
        range is empty.
    RuntimeError
        If a finished value disagrees with ``process.intensity``, which means
        one of the three user callables is not a function of its arguments, or
        the event record changed while the frames were being built.

    Notes
    -----
    Each grid node is folded by :meth:`~hawkes_package.SpatialDomain.wrap`
    before anything is evaluated on it. For the two flat surfaces that is very
    nearly a no-op, since the grid already spans the fundamental rectangle. For
    the projective plane it is not: the grid covers the whole sphere while the
    polygon is one hemisphere. ``distance`` would cope unaided -- it reduces
    both of its arguments into the polygon -- but ``base`` is handed the raw
    chart point, so a chart-dependent background would be sampled outside the
    domain it was written for and the picture would tear along the equator.

    Why the colour ceiling is measured rather than derived: ``_upper_bound(t)``
    is the space *integral* of a dominating field, so dividing it by the volume
    gives a spatial average. For a sharp spatial kernel on a large domain that
    average sits well below the peak, and using it would clip the very bursts
    the animation exists to show.

    .. versionadded:: 0.5.0
    """
    nu, nv = _checked_resolution(resolution)
    surface = embed(process.domain) if embedding is None else embedding

    grid = _chart_grid(surface, nu, nv)
    folded = np.array([process.domain.wrap(point) for point in grid.reshape(-1, 2)])

    frame_times = np.asarray(times, dtype=float).ravel()
    events = np.array(process.events, dtype=float)  # a view onto a growable buffer
    event_times, event_points = events[0, :], events[1:, :].T

    if fast:
        values = _factorised(process, folded, frame_times, event_times, event_points)
        _self_check(process, folded, frame_times, values)
    else:
        values = _direct(process, folded, frame_times)

    low = 0.0 if vmin is None else float(vmin)
    high = float(values.max()) if vmax is None else float(vmax)
    if not high >= low:
        raise ValueError(f"the colour range is empty: vmax={high} is below vmin={low}")

    return IntensityFrames(
        times=frame_times,
        chart=folded.reshape(nu, nv, 2),
        values=values.reshape(len(frame_times), nu, nv),
        vmin=low,
        vmax=high,
        factorised=fast,
    )


# ---------------------------------------------------------------------------
# The two evaluation paths
# ---------------------------------------------------------------------------


def _direct(
    process: SpatioTemporalHawkesProcess,
    points: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Call the process's own intensity accessor, once per node per frame."""
    return np.array(
        [[process.intensity(float(t), point) for point in points] for t in times],
        dtype=float,
    )


def _factorised(
    process: SpatioTemporalHawkesProcess,
    points: np.ndarray,
    times: np.ndarray,
    event_times: np.ndarray,
    event_points: np.ndarray,
) -> np.ndarray:
    """Evaluate the same sum, with everything independent of `t` computed once.

    ``base`` and the spatial factor are evaluated one scalar at a time. The
    kernel protocol promises a user callable a single non-negative distance, or
    a single pair of points when it is pairwise; handing it a vector would work
    for the kernels that happen to be written in numpy and fail for the ones
    that are not, and the failure would land inside a user's own function.

    ``_spatial_at`` rather than ``spatial(domain.distance(...))`` because it *is*
    the dispatch: a kernel carrying ``pairwise = True`` -- everything
    :func:`~hawkes_package.spatio_temporal.kernels.make_periodic` returns -- gets
    both endpoints, and neither branch depends on time. Choosing between them
    again here would be the second expression the package forbids, and would
    have shut image-sum kernels out of the fast path for no reason.
    """
    background = np.array([as_float(process.base(point)) for point in points], dtype=float)

    n_points, n_events = len(points), len(event_times)
    if n_events == 0:
        return np.tile(np.maximum(0.0, background), (len(times), 1))

    spatial = np.array(
        [[process._spatial_at(point, event) for event in event_points] for point in points],
        dtype=float,
    )

    values = np.empty((len(times), n_points), dtype=float)
    for k, t in enumerate(times):
        # Strictly earlier, and *sliced* rather than masked to zero: the sum in
        # `_full_intensity` runs over a one-dimensional array holding only the
        # past events, and reproducing its value bit for bit means reproducing
        # that length. See the table in the module docstring.
        past = event_times < float(t)
        if not past.any():
            values[k] = np.maximum(0.0, background)
            continue
        temporal = np.array(
            [as_float(process.temporal(float(t - s))) for s in event_times[past]], dtype=float
        )
        near = np.ascontiguousarray(spatial[:, past])
        for j in range(n_points):
            values[k, j] = max(0.0, background[j] + float(np.multiply(temporal, near[j]).sum()))
    return values


def _self_check(
    process: SpatioTemporalHawkesProcess,
    points: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
) -> None:
    """Compare a sample of finished values against the hook they claim to reproduce.

    Exact equality, not a tolerance. The factorisation is a rearrangement of the
    same sum, in the same order, over an array of the same length, so it agrees
    to the last bit; anything else means an assumption failed rather than that
    rounding accumulated, and a tolerance would hide exactly the case worth
    catching.
    """
    if values.size == 0:
        return
    rng = np.random.default_rng(0)  # a fixed probe, so a failure is reproducible
    n_probe = min(_SELF_CHECK_SAMPLES, values.size)
    for index in rng.choice(values.size, size=n_probe, replace=False):
        k, j = divmod(int(index), len(points))
        expected = process.intensity(float(times[k]), points[j])
        if values[k, j] != expected:
            raise RuntimeError(
                f"the hoisted intensity disagrees with process.intensity at "
                f"t={times[k]!r}, x={points[j]!r}: {values[k, j]!r} against {expected!r}. "
                "Either base, spatial or temporal is not a function of its arguments, or "
                "process.events changed while the frames were being built. Pass fast=False "
                "to evaluate through the hook directly."
            )


# ---------------------------------------------------------------------------
# Event markers
# ---------------------------------------------------------------------------

#: Faintest a spent event is drawn. Zero would delete the history of the
#: realisation from the picture; the excitation is gone but the event happened,
#: and a viewer scrubbing the slider needs to see where the bursts came from.
_OPACITY_FLOOR = 0.15


def event_opacities(
    process: SpatioTemporalHawkesProcess,
    t: float,
    *,
    floor: float = _OPACITY_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Which events have happened by `t`, and how brightly to draw each.

    Parameters
    ----------
    process : SpatioTemporalHawkesProcess
        The process whose record and temporal kernel are read.
    t : float
        The frame time.
    floor : float, optional
        Faintest a fully decayed event is drawn.

    Returns
    -------
    points : numpy.ndarray, shape (n, ndim)
        Locations of the events that have occurred, in chart coordinates and in
        record order.
    opacities : numpy.ndarray, shape (n,)
        One opacity per point, in ``[floor, 1]``.

    Notes
    -----
    Inclusive in time, ``t_i <= t``, deliberately unlike the strict ``t_i < t``
    the intensity uses. The strict inequality is the left-limit convention for a
    *rate*; a marker answers a different question -- whether the event has
    happened -- and an event marker that appeared one frame late would read as a
    lag in the simulation.

    The fade is the temporal kernel's own decay, normalised by its peak, so a
    marker dims at exactly the rate its excitation does rather than at a rate
    invented for the picture.

    .. versionadded:: 0.5.0
    """
    events = np.array(process.events, dtype=float)
    if events.size == 0:
        return np.empty((0, events.shape[0] - 1)), np.empty(0)

    happened = events[0, :] <= float(t)
    points = events[1:, happened].T
    if not happened.any():
        return points, np.empty(0)

    peak = as_float(process.temporal(0.0)) if process.monotone_temporal_kernel else None
    if peak is None or not peak > 0:
        peak = float(getattr(process, "temporal_peak", 0.0))
    if not peak > 0:
        # Nothing to normalise against -- a kernel that is zero at its own peak
        # carries no excitation, so every marker is drawn at full strength
        # rather than dividing by zero.
        return points, np.ones(len(points))

    decay = np.array(
        [as_float(process.temporal(float(t - s))) for s in events[0, happened]], dtype=float
    )
    return points, np.clip(decay / peak, floor, 1.0)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def _checked_resolution(resolution: tuple[int, int]) -> tuple[int, int]:
    """Validate the node counts, which index into the returned arrays."""
    nu, nv = (int(n) for n in resolution)
    if nu < 2 or nv < 2:
        raise ValueError(f"resolution must be at least 2 along each axis, got {resolution}")
    return nu, nv


def _chart_grid(surface: SurfaceEmbedding, nu: int, nv: int) -> np.ndarray:
    """Build the tensor grid over the embedding's chart ranges, shape ``(nu, nv, 2)``.

    Both endpoints of both ranges are included, on every surface. That is what
    closes the seam: the last column names the same points of the surface as the
    first, so the mesh joins up instead of showing a crack. On the Klein bottle
    it closes *with the flip*, because each node was folded through the domain's
    own ``wrap`` and the wrap knows the gluing.
    """
    u = np.linspace(surface.urange[0], surface.urange[1], nu)
    v = np.linspace(surface.vrange[0], surface.vrange[1], nv)
    mesh_u, mesh_v = np.meshgrid(u, v, indexing="ij")
    return np.stack([mesh_u, mesh_v], axis=-1)
