"""Multi-panel surface plotting utilities.

Public API
----------
plot_surface_panel(surfaces, references=None, render_mode="3d", ...)
    Single model, multiple time steps (≤ 5).
    Layout: 1 row × N columns (model), or 2 rows × N columns with reference row.

plot_model_comparison(surfaces_by_model, references=None, render_mode="3d", ...)
    Multiple models × multiple time steps (≤ 5 per model).
    Layout: rows = models [+ optional reference row], columns = time steps.

Both wrappers delegate to ``_plot_grid``, the single generic core.

render_mode
-----------
``"3d"`` (default) — 3D surface panels via mpl_toolkits.mplot3d.
``"2d"``            — 2D pcolormesh heatmap panels.

Shared colorscale is only applied in 2D mode (3D panels self-normalise
individually because mpl 3D doesn't share axes norms cleanly).

Scientific honesty rules (enforced)
------------------------------------
- All labels/units come from ``SurfaceResult`` — no hardcoded strings.
- ``comparable=False`` → beige panel background.
- Mixing comparable and non-comparable in one figure → asterisk note.
- Shared colorscale only within the same ``surface_type`` (2D only).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from seahorse.evaluation.surface import SurfaceResult
from seahorse.viz.surface_plot import plot_surface


# ---------------------------------------------------------------------------
# Internal generic grid core
# ---------------------------------------------------------------------------

def _plot_grid(
    surfaces_grid: list[list[Optional[SurfaceResult]]],
    row_labels: Optional[list[str]] = None,
    col_labels: Optional[list[str]] = None,
    history_locs: Optional[list[list[Optional[np.ndarray]]]] = None,
    cmap: str = "viridis",
    share_colorscale_within_type: bool = True,
    suptitle: Optional[str] = None,
    render_mode: str = "3d",
):
    """Generic surface grid renderer (internal).

    Parameters
    ----------
    surfaces_grid : list[list[Optional[SurfaceResult]]]
        surfaces_grid[row][col]; None → empty panel.
    row_labels : optional row label list (len = n_rows)
    col_labels : optional column label list (len = n_cols)
    history_locs : history_locs[row][col] → (L, 2) or None; scatter overlay
    cmap : colormap name
    share_colorscale_within_type : share vmin/vmax within same surface_type (2D only)
    suptitle : optional figure super-title
    render_mode : ``"3d"`` or ``"2d"``

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    n_rows = len(surfaces_grid)
    n_cols = max((len(row) for row in surfaces_grid), default=0)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("surfaces_grid must not be empty.")

    # ---- Create figure + axes (3D requires add_subplot, not plt.subplots) ---
    figsize = (4.5 * n_cols + (0.8 if row_labels else 0), 4 * n_rows)
    if render_mode == "3d":
        import mpl_toolkits.mplot3d  # noqa: F401 — registers "3d" projection
        fig = plt.figure(figsize=figsize)
        axes = [
            [fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
             for c in range(n_cols)]
            for r in range(n_rows)
        ]
    else:
        fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = [list(row) for row in axes_arr]

    # ---- Compute shared colorscale per surface_type (2D and 3D) ------------
    type_vranges: dict[str, tuple[float, float]] = {}
    if share_colorscale_within_type:
        for row in surfaces_grid:
            for s in row:
                if s is None:
                    continue
                vmin = float(s.values.min())
                vmax = float(s.values.max())
                if s.surface_type in type_vranges:
                    prev_min, prev_max = type_vranges[s.surface_type]
                    type_vranges[s.surface_type] = (min(prev_min, vmin), max(prev_max, vmax))
                else:
                    type_vranges[s.surface_type] = (vmin, vmax)

    # ---- Check for comparable / non-comparable mix -------------------------
    has_comparable    = any(s is not None and s.comparable for row in surfaces_grid for s in row)
    has_noncomparable = any(s is not None and not s.comparable for row in surfaces_grid for s in row)
    mixed = has_comparable and has_noncomparable

    # ---- Draw each cell (no per-cell colorbars; one shared bar per row) ----
    for r, row in enumerate(surfaces_grid):
        for c, s in enumerate(row):
            ax = axes[r][c]
            if s is None:
                ax.set_visible(False)
                continue

            hlocs = None
            if history_locs is not None and r < len(history_locs) and c < len(history_locs[r]):
                hlocs = history_locs[r][c]

            vmin_cell = vmax_cell = None
            if s.surface_type in type_vranges:
                vmin_cell, vmax_cell = type_vranges[s.surface_type]

            plot_surface(
                s, ax=ax, cmap=cmap, history_locs=hlocs,
                render_mode=render_mode, vmin=vmin_cell, vmax=vmax_cell,
                show_colorbar=False,   # colorbar is added per-row below
                title="",              # col/row labels set below
            )

            # Row labels (left border panels only)
            if c == 0 and row_labels is not None and r < len(row_labels):
                if render_mode == "2d":
                    ax.set_ylabel(row_labels[r], fontsize=9)
                else:
                    ax.set_title(f"[{row_labels[r]}]", fontsize=9)

            # Column labels (top row only)
            if r == 0 and col_labels is not None and c < len(col_labels):
                existing = ax.get_title()
                ax.set_title(
                    f"{col_labels[c]}\n{existing}" if existing else col_labels[c],
                    fontsize=9,
                )

    # ---- One shared colorbar per row ---------------------------------------
    for r, row in enumerate(surfaces_grid):
        sample = next((s for s in row if s is not None), None)
        if sample is None:
            continue
        stype = sample.surface_type
        if stype in type_vranges:
            vmin_r, vmax_r = type_vranges[stype]
        else:
            row_vals = [s.values for s in row if s is not None]
            vmin_r = float(min(v.min() for v in row_vals))
            vmax_r = float(max(v.max() for v in row_vals))
        sm_r = plt.cm.ScalarMappable(
            cmap=cmap, norm=Normalize(vmin=vmin_r, vmax=vmax_r)
        )
        sm_r.set_array([])
        row_axes = [axes[r][c] for c in range(n_cols)
                    if c < len(row) and row[c] is not None]
        cb = fig.colorbar(sm_r, ax=row_axes, shrink=0.6, pad=0.05)
        cb.set_label(sample.unit, fontsize=9)

    # ---- Figure-level annotations -----------------------------------------
    title = suptitle or ""
    if mixed:
        title = (title + " *") if title else "*"
        fig.text(
            0.5, -0.01,
            "* Panel background indicates non-comparable proxy surface (KDE-based).",
            ha="center", fontsize=8, color="gray",
        )

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    if render_mode == "2d":
        fig.tight_layout()
    else:
        fig.subplots_adjust(wspace=0.5, hspace=0.4)
    return fig


# ---------------------------------------------------------------------------
# Public wrapper: single model, multiple time steps
# ---------------------------------------------------------------------------

def plot_surface_panel(
    surfaces: list[SurfaceResult],
    references: Optional[list[SurfaceResult]] = None,
    cmap: str = "viridis",
    history_locs: Optional[list[Optional[np.ndarray]]] = None,
    share_colorscale: bool = True,
    suptitle: Optional[str] = None,
    render_mode: str = "3d",
):
    """Single model, multiple time steps.

    Parameters
    ----------
    surfaces    : list of SurfaceResult, one per time step (max 5)
    references  : optional list of reference SurfaceResult, same length as surfaces
    cmap        : colormap name
    history_locs: per-time-step history locations for scatter overlay (list of (L,2) or None)
    share_colorscale : share colorscale within the same surface_type (2D only)
    suptitle    : figure super-title
    render_mode : ``"3d"`` (default) or ``"2d"``

    Layout
    ------
    - No references : 1 row × N columns
    - With references: 2 rows × N columns (top=model, bottom=reference)

    Raises
    ------
    ValueError if len(surfaces) > 5 or references length does not match.
    """
    if len(surfaces) > 5:
        raise ValueError(
            f"plot_surface_panel supports at most 5 time steps; got {len(surfaces)}."
        )
    if references is not None and len(references) != len(surfaces):
        raise ValueError(
            f"references length ({len(references)}) must match surfaces length ({len(surfaces)})."
        )

    n = len(surfaces)
    col_labels = [f"t = {s.t_query:.3f}" for s in surfaces]

    if references is None:
        grid = [list(surfaces)]
        hlocs_grid = [list(history_locs)] if history_locs is not None else None
        row_labels = None
    else:
        grid = [list(surfaces), list(references)]
        row_labels = ["Model", "Reference"]
        if history_locs is not None:
            hlocs_grid = [list(history_locs), [None] * n]
        else:
            hlocs_grid = None

    return _plot_grid(
        grid,
        row_labels=row_labels,
        col_labels=col_labels,
        history_locs=hlocs_grid,
        cmap=cmap,
        share_colorscale_within_type=share_colorscale,
        suptitle=suptitle,
        render_mode=render_mode,
    )


# ---------------------------------------------------------------------------
# Public wrapper: multiple models × multiple time steps
# ---------------------------------------------------------------------------

def plot_model_comparison(
    surfaces_by_model: dict[str, list[SurfaceResult]],
    references: Optional[list[SurfaceResult]] = None,
    cmap: str = "viridis",
    history_locs: Optional[list[Optional[np.ndarray]]] = None,
    share_colorscale_within_type: bool = True,
    suptitle: Optional[str] = None,
    render_mode: str = "3d",
):
    """Multiple models × multiple time steps.

    Parameters
    ----------
    surfaces_by_model : {model_label: [SurfaceResult per time step]}
    references        : optional reference list (one per time step; shown as last row)
    cmap              : colormap name
    history_locs      : per-time-step scatter overlay (list of (L,2) or None)
    share_colorscale_within_type : share colorscale within same surface_type (2D only)
    suptitle          : figure super-title
    render_mode       : ``"3d"`` (default) or ``"2d"``

    Layout
    ------
    Rows = models [+ optional reference row], columns = time steps.

    Raises
    ------
    ValueError if any model list has len > 5 or lists differ in length.
    """
    if not surfaces_by_model:
        raise ValueError("surfaces_by_model must not be empty.")

    lengths = [len(v) for v in surfaces_by_model.values()]
    if any(l > 5 for l in lengths):
        raise ValueError(f"Each model may have at most 5 time steps; got lengths {lengths}.")
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All models must have the same number of time steps; got {lengths}."
        )

    n = lengths[0]
    model_names = list(surfaces_by_model.keys())
    col_labels = [f"t = {surfaces_by_model[model_names[0]][i].t_query:.3f}" for i in range(n)]

    grid: list[list[Optional[SurfaceResult]]] = [
        list(surfaces_by_model[name]) for name in model_names
    ]
    row_labels = model_names

    if references is not None:
        if len(references) != n:
            raise ValueError(
                f"references length ({len(references)}) must match n_time_steps ({n})."
            )
        grid.append(list(references))
        row_labels = row_labels + ["Reference"]

    n_rows = len(grid)
    if history_locs is not None:
        hlocs_grid = [list(history_locs)] + [[None] * n] * (n_rows - 1)
    else:
        hlocs_grid = None

    return _plot_grid(
        grid,
        row_labels=row_labels,
        col_labels=col_labels,
        history_locs=hlocs_grid,
        cmap=cmap,
        share_colorscale_within_type=share_colorscale_within_type,
        suptitle=suptitle,
        render_mode=render_mode,
    )
