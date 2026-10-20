r"""The rendering backend: chart grid plus intensity frames to one interactive page.

The only module here that names a plotting library, and it names it inside a
function body. Everything else in :mod:`hawkes_package.viz` imports with numpy
and scipy alone, which ``tests/test_api_surface.py`` asserts -- so the frames can
be built, inspected and tested without the extra, and only writing the page
needs it.

Two choices worth stating
-------------------------

**The geometry is sent once, not once per frame.** A frame carries only the new
``surfacecolor``; the vertex positions live in the base trace. The mesh is
static -- it is the *colour* that animates -- so repeating it would multiply the
page size by the frame count for nothing. Measured on a 48x48 grid over 40
frames: 2.8 MB with partial frames against roughly 90 MB without. That is the
difference between a page a browser opens and one it does not.

**The colour scale is in the intensity's own units.** The raw
:math:`\lambda` values go to the surface with ``cmin``/``cmax`` fixed to the
frames' global range, rather than a unit-interval ramp, so the colour bar is
labelled in intensity and not in an arbitrary 0-to-1. The range is global across
every frame; see :mod:`hawkes_package.viz._field` for why a per-frame rescale
would destroy the thing being shown.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from ._embedding import SurfaceEmbedding, embed
from ._field import DEFAULT_RESOLUTION, IntensityFrames, event_opacities, intensity_frames

if TYPE_CHECKING:  # pragma: no cover - runtime import would be circular
    from ..spatio_temporal.process import SpatioTemporalHawkesProcess

__all__ = ["animate_intensity", "build_figure"]

#: Milliseconds a frame is held during playback.
DEFAULT_DURATION = 80

#: Marker colour for events. Deliberately outside the sequential colour maps a
#: field is drawn with -- Inferno and Viridis both run dark-to-bright through
#: purple, orange and yellow, so a black or a white marker disappears at one end
#: of the scale. Cyan does not sit anywhere on either ramp.
_EVENT_COLOUR = (0, 229, 255)


def _backend() -> Any:
    """Import plotly, or say what to install.

    Imported here rather than at module scope so that
    :mod:`hawkes_package.viz` stays importable with numpy and scipy alone and
    the extra stays genuinely optional.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - exercised by monkeypatching
        raise ImportError(
            "rendering an intensity surface needs plotly, which is an optional extra:\n"
            '    pip install "the-hawkes-package[viz]"\n'
            "Building the frames themselves needs nothing beyond numpy: see "
            "hawkes_package.viz.intensity_frames."
        ) from exc
    return go


def animate_intensity(
    process: SpatioTemporalHawkesProcess,
    times: Any,
    path: str | os.PathLike[str],
    *,
    embedding: SurfaceEmbedding | None = None,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    colorscale: str = "Inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    events: bool = True,
    duration: int = DEFAULT_DURATION,
    fast: bool = True,
    inline_js: bool = True,
) -> IntensityFrames:
    r"""Render :math:`\lambda(t, x \mid H_t)` on `process`'s surface and write one HTML page.

    Parameters
    ----------
    process : SpatioTemporalHawkesProcess
        A process carrying a realisation.
    times : array_like, shape (n_t,)
        Frame times, used exactly as given.
    path : str or path-like
        Where to write the page.
    embedding : SurfaceEmbedding, optional
        The immersion to draw. Defaults to :func:`~hawkes_package.viz.embed` of
        the process's domain.
    resolution : tuple of int, optional
        Chart grid, nodes along each axis.
    colorscale : str, optional
        Any named plotly colour scale.
    vmin, vmax : float, optional
        Override the colour range, which is otherwise zero to the largest value
        over every frame.
    events : bool, optional
        Whether to mark the realised events, fading each at its own kernel's
        decay rate.
    duration : int, optional
        Milliseconds per frame during playback.
    fast : bool, optional
        Passed to :func:`~hawkes_package.viz.intensity_frames`.
    inline_js : bool, optional
        Whether to embed the plotly library in the page. ``True`` gives one
        self-contained file that opens with no network; ``False`` links a CDN
        and saves about 3 MB, at the cost of a page that is blank offline.

    Returns
    -------
    IntensityFrames
        The frames that were drawn -- returned rather than discarded because the
        realised ``vmin``/``vmax`` is not otherwise recoverable, and a caller
        comparing two runs needs it to put them on one scale.

    Examples
    --------
    >>> frames = animate_intensity(process, np.linspace(0, 10, 40), "out.html")  # doctest: +SKIP
    >>> print(frames.summary())  # doctest: +SKIP

    .. versionadded:: 0.5.0
    """
    surface = embed(process.domain) if embedding is None else embedding
    frames = intensity_frames(
        process,
        times,
        embedding=surface,
        resolution=resolution,
        fast=fast,
        vmin=vmin,
        vmax=vmax,
    )
    figure = build_figure(
        process,
        frames,
        embedding=surface,
        colorscale=colorscale,
        events=events,
        duration=duration,
    )
    figure.write_html(os.fspath(path), include_plotlyjs="inline" if inline_js else "cdn")
    return frames


def build_figure(
    process: SpatioTemporalHawkesProcess,
    frames: IntensityFrames,
    *,
    embedding: SurfaceEmbedding | None = None,
    colorscale: str = "Inferno",
    events: bool = True,
    duration: int = DEFAULT_DURATION,
) -> Any:
    """Assemble the plotly figure for `frames`, without writing anything.

    Parameters
    ----------
    process : SpatioTemporalHawkesProcess
        Read only for its event record and temporal kernel, to place and fade
        the markers.
    frames : IntensityFrames
        The field to colour by.
    embedding : SurfaceEmbedding, optional
        The immersion. Defaults to that of the process's domain.
    colorscale : str, optional
        Any named plotly colour scale.
    events : bool, optional
        Whether to mark the realised events.
    duration : int, optional
        Milliseconds per frame during playback.

    Returns
    -------
    plotly.graph_objects.Figure
        Typed as :class:`~typing.Any` because plotly ships no type information.

    .. versionadded:: 0.5.0
    """
    go = _backend()
    surface = embed(process.domain) if embedding is None else embedding

    nu, nv = frames.chart.shape[:2]
    ambient = surface.ambient(frames.chart.reshape(-1, 2)).reshape(nu, nv, 3)
    x, y, z = ambient[..., 0], ambient[..., 1], ambient[..., 2]

    marker_size = 0.025 * float(np.ptp(ambient.reshape(-1, 3), axis=0).max())

    def field_trace(index: int) -> Any:
        return go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=frames.values[index],
            cmin=frames.vmin,
            cmax=frames.vmax,
            colorscale=colorscale,
            colorbar={"title": {"text": "λ(t, x)"}},
            showscale=True,
            lighting={"ambient": 0.75, "diffuse": 0.5, "specular": 0.1},
        )

    def event_trace(index: int) -> Any:
        points, opacity = (
            event_opacities(process, float(frames.times[index]))
            if events
            else (np.empty((0, 2)), np.empty(0))
        )
        if len(points) == 0:
            return go.Scatter3d(x=[], y=[], z=[], mode="markers", showlegend=False)
        # Markers sit *on* the surface, with no offset along a normal: the Klein
        # bottle is non-orientable, so there is no consistent outward direction
        # to offset along and a nudge that read as "outside" on one side of the
        # bottle would read as "inside" after one turn.
        seats = surface.ambient(np.array([process.domain.wrap(p) for p in points]))
        red, green, blue = _EVENT_COLOUR
        return go.Scatter3d(
            x=seats[:, 0],
            y=seats[:, 1],
            z=seats[:, 2],
            mode="markers",
            marker={
                "size": marker_size,
                "sizemode": "diameter",
                "color": [f"rgba({red},{green},{blue},{a:.3f})" for a in opacity],
            },
            name="events",
            showlegend=False,
        )

    # Only the two data properties that change are repeated per frame. The
    # vertex positions are in the base traces and stay there; see the module
    # docstring for what including them costs.
    animation = [
        go.Frame(
            name=f"{t:.6g}",
            traces=[0, 1],
            data=[
                go.Surface(surfacecolor=frames.values[k]),
                event_trace(k),
            ],
        )
        for k, t in enumerate(frames.times)
    ]

    figure = go.Figure(data=[field_trace(0), event_trace(0)], frames=animation)
    figure.update_layout(
        title={"text": _caption(surface, frames)},
        scene={
            "aspectmode": "data",  # never rescale the axes: it would distort the surface
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
        margin={"l": 0, "r": 0, "t": 90, "b": 0},
        updatemenus=[_controls(go, duration)],
        sliders=[_slider(frames, duration)],
    )
    return figure


def _caption(surface: SurfaceEmbedding, frames: IntensityFrames) -> str:
    """Name the surface, the colour range, and what the immersion does to distance.

    The range goes in the title as well as on the colour bar because a reader
    comparing two saved pages has only the titles side by side, and two
    animations on different scales look alike.
    """
    honesty = "isometric" if surface.isometric else "immersed, distances distorted"
    return (
        f"{surface.name} — λ in [{frames.vmin:.4g}, {frames.vmax:.4g}] "
        f"({honesty})<br><sub>{surface.note}</sub>"
    )


def _controls(go: Any, duration: int) -> Any:
    """Play and pause. ``redraw`` must stay true: a 3-D surface is not redrawn without it."""
    play = {"frame": {"duration": duration, "redraw": True}, "fromcurrent": True}
    pause = {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}
    return go.layout.Updatemenu(
        type="buttons",
        direction="left",
        x=0.05,
        y=0.02,
        showactive=False,
        buttons=[
            {"label": "Play", "method": "animate", "args": [None, play]},
            {"label": "Pause", "method": "animate", "args": [[None], pause]},
        ],
    )


def _slider(frames: IntensityFrames, duration: int) -> dict[str, Any]:
    """Build a slider step per frame, labelled with the time it shows."""
    step = {"frame": {"duration": duration, "redraw": True}, "mode": "immediate"}
    return {
        "active": 0,
        "x": 0.2,
        "len": 0.75,
        "currentvalue": {"prefix": "t = "},
        "steps": [
            {
                "label": f"{t:.4g}",
                "method": "animate",
                "args": [[f"{t:.6g}"], step],
            }
            for t in frames.times
        ],
    }
