"""Temporal animation of surface sequences.

Single entry point: ``animate_surface_sequence``.

Supports two modes:
  - Single-model: ``surfaces`` is a ``list[SurfaceResult]``; one panel per frame.
  - Multi-model:  ``surfaces`` is a ``dict[str, list[SurfaceResult]]``; side-by-side
    panels per frame (one column per model, plus optional reference column).

render_mode
-----------
``"3d"`` (default) — 3D surface animation.
``"2d"``            — 2D pcolormesh animation.

Output format is determined by ``output_path`` extension:
  - ``.gif``  — uses ``PillowWriter`` (requires Pillow)
  - ``.mp4``  — uses ``FFMpegWriter`` (requires ffmpeg); falls back to .gif with warning

Frame N shows the surface(s) evaluated at time step N.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from unified_stpp.evaluation.surface import SurfaceResult
from unified_stpp.viz.surface_plot import plot_surface


def animate_surface_sequence(
    surfaces: Union[list[SurfaceResult], dict[str, list[SurfaceResult]]],
    output_path: Union[str, Path],
    references: Optional[list[SurfaceResult]] = None,
    history_locs: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    fps: int = 4,
    figsize: Optional[tuple] = None,
    render_mode: str = "3d",
    reference_first: bool = False,
    share_colorscale: bool = True,
) -> Path:
    """Animate a surface sequence as a GIF or MP4.

    Parameters
    ----------
    surfaces         : list[SurfaceResult] (single-model) or dict[str, list[SurfaceResult]]
                       (multi-model, {label: [t0, t1, ...]}).
    output_path      : destination file; ``.gif`` or ``.mp4``
    references       : optional reference surface per time step (one entry per frame)
    history_locs     : (L, 2) array or list[Optional[np.ndarray]] — scatter overlay on first
                       column per frame; if a list, entry i is used for frame i (rolling mode)
    cmap             : colormap name
    fps              : frames per second
    figsize          : (width, height) in inches; auto-computed if None
    render_mode      : ``"3d"`` (default) or ``"2d"``
    share_colorscale : if True (default), compute a single global vmin/vmax per
                       ``surface_type`` across all frames so the colormap range is
                       fixed for the entire animation. Surfaces of different types
                       (e.g. ``"intensity"`` vs ``"proxy_kde"``) always get
                       independent ranges.

    Returns
    -------
    Path
        The path actually written (may differ from output_path if fallback was used).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # ---- Normalise inputs --------------------------------------------------
    if isinstance(surfaces, list):
        model_names: list[str] = ["Model"]
        surfaces_by_model: dict[str, list[SurfaceResult]] = {"Model": surfaces}
    else:
        model_names = list(surfaces.keys())
        surfaces_by_model = dict(surfaces)

    n_frames = len(next(iter(surfaces_by_model.values())))
    if references is not None and len(references) != n_frames:
        raise ValueError(
            f"references length ({len(references)}) must match n_frames ({n_frames})."
        )

    ref_col = ["Reference"] if references is not None else []
    col_names = (ref_col + model_names) if reference_first else (model_names + ref_col)
    n_cols = len(col_names)
    if figsize is None:
        figsize = (5.5 * n_cols, 6.5) if render_mode == "3d" else (4.5 * n_cols, 4.5)

    # ---- Compute global vmin/vmax per surface_type (fixed colorscale) ------
    _type_vranges: dict[str, tuple[float, float]] = {}
    if share_colorscale:
        all_surfs: list[SurfaceResult] = []
        for surf_list in surfaces_by_model.values():
            all_surfs.extend(surf_list)
        if references is not None:
            all_surfs.extend(references)
        for s in all_surfs:
            lo, hi = float(s.values.min()), float(s.values.max())
            if s.surface_type in _type_vranges:
                plo, phi = _type_vranges[s.surface_type]
                _type_vranges[s.surface_type] = (min(plo, lo), max(phi, hi))
            else:
                _type_vranges[s.surface_type] = (lo, hi)

    # ---- Create figure with correct projection per column ------------------
    # fig.clf() is called each frame to wipe all axes (main + colorbar children)
    # and reset SubplotSpec layout state — the only approach that prevents
    # plt.colorbar(ax=ax) from shrinking axes across frames.
    if render_mode == "3d":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)

    def _update(frame_idx: int):
        fig.clf()
        if render_mode == "3d":
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            cur_axes = [fig.add_subplot(1, n_cols, i + 1, projection="3d")
                        for i in range(n_cols)]
        else:
            cur_axes = [fig.add_subplot(1, n_cols, i + 1) for i in range(n_cols)]

        fig.subplots_adjust(top=0.88)  # headroom for suptitle (both 2d and 3d)

        h_locs = history_locs[frame_idx] if isinstance(history_locs, list) else history_locs
        first_model_col = next(i for i, n in enumerate(col_names) if n != "Reference")
        for col_i, col_name in enumerate(col_names):
            ax = cur_axes[col_i]
            s = (references[frame_idx]
                 if col_name == "Reference" and references is not None
                 else surfaces_by_model[col_name][frame_idx])
            _vr = _type_vranges.get(s.surface_type, (None, None))
            plot_surface(s, ax=ax, cmap=cmap,
                         history_locs=h_locs if col_i == first_model_col else None,
                         render_mode=render_mode,
                         vmin=_vr[0], vmax=_vr[1])
            # Increase z-label padding to prevent overlap with tick labels in 3D.
            if render_mode == "3d" and hasattr(ax, "zaxis"):
                ax.zaxis.labelpad = 15

        primary = surfaces_by_model[model_names[0]][frame_idx]
        fig.suptitle(
            f"Frame {frame_idx + 1} / {n_frames}  |  t = {primary.t_query:.4f}",
            fontsize=11,
        )

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000 // fps, blit=False
    )

    # ---- Save --------------------------------------------------------------
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    def _try_save(path: Path, writer_name: str):
        ani.save(str(path), writer=writer_name, fps=fps)

    if suffix == ".mp4":
        try:
            _try_save(output_path, "ffmpeg")
        except Exception:
            warnings.warn(
                "ffmpeg not available; falling back to .gif output.",
                UserWarning, stacklevel=2,
            )
            output_path = output_path.with_suffix(".gif")
            _try_save(output_path, "pillow")
    else:
        if suffix != ".gif":
            output_path = output_path.with_suffix(".gif")
        _try_save(output_path, "pillow")

    plt.close(fig)
    return output_path
