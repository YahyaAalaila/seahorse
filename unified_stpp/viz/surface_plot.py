"""Plotting utilities for SurfaceResult objects.

Intentionally model-agnostic: all display metadata comes from the SurfaceResult
itself (label, unit, comparable, surface_type). No model-specific logic here.

render_mode
-----------
"3d"  (default) — 3D surface plot via mpl_toolkits.mplot3d.
"2d"            — 2D pcolormesh heatmap (previous behaviour).

For 1-D spatial models (result.ys is empty), both modes fall back to a line plot.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from unified_stpp.evaluation.surface import SurfaceResult


def plot_surface(
    result: SurfaceResult,
    ax=None,
    cmap: str = "viridis",
    history_locs: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    render_mode: str = "3d",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
):
    """Plot a SurfaceResult as a 3D surface or 2D heatmap.

    Parameters
    ----------
    result      : SurfaceResult to plot
    ax          : matplotlib Axes (or Axes3D for 3d mode); created if None
    cmap        : colormap name
    history_locs: (L, 2) event locations to overlay as scatter; optional
    title       : figure title; defaults to result.label
    render_mode : ``"3d"`` (default) or ``"2d"``
    vmin, vmax  : colorscale limits; auto-computed from result if None

    Returns
    -------
    ax : matplotlib Axes or Axes3D
    """
    import matplotlib.pyplot as plt

    is_2d = result.ys.size > 0 and result.values.ndim == 2

    if is_2d and render_mode == "3d":
        ax = _plot_surface_3d(result, ax, cmap, history_locs, vmin, vmax, show_colorbar)
    elif is_2d:
        ax = _plot_surface_2d(result, ax, cmap, history_locs, vmin, vmax, show_colorbar)
    else:
        # 1-D spatial: line plot regardless of render_mode
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(result.xs, result.values)
        ax.set_xlabel("x")
        ax.set_ylabel(result.unit)

    if not result.comparable:
        ax.set_facecolor("#f8f0e3")

    ax.set_title(title if title is not None else result.label, fontsize=10)
    return ax


# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------

def _plot_surface_3d(result, ax, cmap, history_locs, vmin, vmax, show_colorbar=True):
    """3D surface (plot_surface) renderer."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import mpl_toolkits.mplot3d  # noqa: F401 — registers "3d" projection

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")

    v_lo = float(result.values.min()) if vmin is None else vmin
    v_hi = float(result.values.max()) if vmax is None else vmax

    X, Y = np.meshgrid(result.xs, result.ys, indexing="ij")
    norm = Normalize(vmin=v_lo, vmax=v_hi)
    ax.plot_surface(X, Y, result.values, cmap=cmap, norm=norm, alpha=0.9)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if show_colorbar:
        # Colorbar via ScalarMappable (required for 3D axes)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
        cb.set_label(result.unit, fontsize=9)
        ax.set_zlabel(result.unit)

    if history_locs is not None:
        locs = np.asarray(history_locs)
        if locs.ndim == 2 and locs.shape[1] >= 2:
            ax.scatter(
                locs[:, 0], locs[:, 1], zs=v_lo, zdir="z",
                c="white", s=10, zorder=5, alpha=0.7,
                linewidths=0.5, edgecolors="gray",
            )
    return ax


def _plot_surface_2d(result, ax, cmap, history_locs, vmin, vmax, show_colorbar=True):
    """2D pcolormesh heatmap renderer."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    v_lo = result.values.min() if vmin is None else vmin
    v_hi = result.values.max() if vmax is None else vmax

    X, Y = np.meshgrid(result.xs, result.ys, indexing="ij")
    im = ax.pcolormesh(X, Y, result.values, cmap=cmap, shading="auto",
                       vmin=v_lo, vmax=v_hi)
    if show_colorbar:
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(result.unit, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if history_locs is not None:
        locs = np.asarray(history_locs)
        if locs.ndim == 2 and locs.shape[1] >= 2:
            ax.scatter(
                locs[:, 0], locs[:, 1],
                c="white", s=10, zorder=5, alpha=0.7,
                linewidths=0.5, edgecolors="gray",
            )
    return ax
