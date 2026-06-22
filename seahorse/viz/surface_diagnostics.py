"""Rendering for single-run exact and factorized surface diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seahorse.evaluation.surface import SurfaceDiagnosticResult
from seahorse.evaluation.surface.diagnostics import (
    history_overlay_z_level,
    representative_indices,
)
from . import plotly_intensity as _plotly_intensity


@dataclass(frozen=True)
class SurfaceRenderConfig:
    interactive: bool = True


def _history_until_t(
    history_times: np.ndarray,
    history_locs: np.ndarray,
    t_query: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(history_times, dtype=np.float32) <= float(t_query)
    return history_times[mask], history_locs[mask]


def _save_static_plots(
    out_dir: Path,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    frame_values: np.ndarray,
    t_query: float,
    history_times: np.ndarray,
    history_locs: np.ndarray,
    title_prefix: str,
    value_name: str,
    value_label: str,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    heatmap_path = out_dir / f"{value_name}_heatmap.png"
    heatmap_hist_path = out_dir / f"{value_name}_heatmap_with_history.png"
    surface3d_path = out_dir / f"{value_name}_surface_3d.png"
    surface3d_hist_path = out_dir / f"{value_name}_surface_3d_with_history.png"
    surface3d_contour_path = out_dir / f"{value_name}_surface_3d_with_contours.png"
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    _hist_t, hist_s = _history_until_t(history_times, history_locs, t_query)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(frame_values.T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto", cmap="magma")
    ax.set_title(f"{title_prefix} {value_label} @ t={t_query:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label=value_label)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(frame_values.T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto", cmap="magma")
    if hist_s.size > 0:
        ax.plot(hist_s[:, 0], hist_s[:, 1], "-o", color="white", lw=1.0, ms=2.5)
    ax.set_title(f"{title_prefix} {value_label} + history @ t={t_query:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label=value_label)
    fig.tight_layout()
    fig.savefig(heatmap_hist_path, dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, frame_values.T, cmap=cm.magma, linewidth=0, antialiased=True, alpha=0.96)
    ax.set_title(f"{title_prefix} {value_label} surface @ t={t_query:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(value_label)
    ax.view_init(elev=34, azim=-58)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label=value_label)
    fig.tight_layout()
    fig.savefig(surface3d_path, dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, frame_values.T, cmap=cm.magma, linewidth=0, antialiased=True, alpha=0.82)
    if hist_s.size > 0:
        z_hist_level = history_overlay_z_level(frame_values)
        z_hist = np.full(hist_s.shape[0], z_hist_level, dtype=np.float32)
        ax.plot(hist_s[:, 0], hist_s[:, 1], z_hist, "-o", color="black", lw=1.0, ms=3.0)
    ax.set_title(f"{title_prefix} {value_label} + history @ t={t_query:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(value_label)
    ax.view_init(elev=30, azim=-60)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label=value_label)
    fig.tight_layout()
    fig.savefig(surface3d_hist_path, dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, frame_values.T, cmap=cm.magma, linewidth=0, antialiased=True, alpha=0.78)
    ax.contour(xx, yy, frame_values.T, zdir="z", offset=0.0, cmap=cm.magma, levels=12, linewidths=0.9)
    ax.set_zlim(bottom=0.0)
    ax.set_title(f"{title_prefix} {value_label} + contours @ t={t_query:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(value_label)
    ax.view_init(elev=30, azim=-60)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label=value_label)
    fig.tight_layout()
    fig.savefig(surface3d_contour_path, dpi=180)
    plt.close(fig)

    return {
        f"{value_name}_heatmap": str(heatmap_path),
        f"{value_name}_heatmap_with_history": str(heatmap_hist_path),
        f"{value_name}_surface_3d": str(surface3d_path),
        f"{value_name}_surface_3d_with_history": str(surface3d_hist_path),
        f"{value_name}_surface_3d_with_contours": str(surface3d_contour_path),
    }


def _save_temporal_curve_plot(out_dir: Path, *, t_grid: np.ndarray, lambda_t: np.ndarray, title: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_dir / "temporal_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_grid, lambda_t, color="tab:blue", lw=2.0)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("lambda_T(t | H)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def render_surface_bundle(
    result: SurfaceDiagnosticResult,
    out_dir: Path,
    config: SurfaceRenderConfig | None = None,
) -> dict[str, Path]:
    cfg = config or SurfaceRenderConfig()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    title_prefix = f"{result.preset} seq={result.seq_idx}"
    if result.profile == "history_frame":
        frame_index = int(result.extra_metadata.get("frame_index", len(result.t_grid) // 2))
        frame_index = max(0, min(frame_index, len(result.t_grid) - 1))
        frame_values = result.primary_cube[frame_index]
        t_query = float(result.t_grid[frame_index])
        paths = _save_static_plots(
            out_dir,
            xs=result.x_grid,
            ys=result.y_grid,
            frame_values=frame_values,
            t_query=t_query,
            history_times=result.history_times,
            history_locs=result.history_locs,
            title_prefix=title_prefix,
            value_name=result.primary_value_name,
            value_label=result.primary_value_label,
        )
        for key, value in paths.items():
            artifacts[key] = Path(value)
        if cfg.interactive:
            fig = _plotly_intensity.plot_lambst_interactive(
                result.primary_cube,
                result.x_grid,
                result.y_grid,
                result.t_grid,
                show=False,
                master_title=f"{title_prefix} history-frame intensity",
            )
            html_path = out_dir / "intensity_interactive.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            artifacts["interactive_html"] = html_path
    else:
        lambda_t = result.extra_arrays.get("lambda_t")
        if lambda_t is not None:
            path = _save_temporal_curve_plot(
                out_dir,
                t_grid=result.t_grid,
                lambda_t=lambda_t,
                title=f"{title_prefix} temporal intensity",
            )
            artifacts["temporal_curve"] = Path(path)
        for label, idx in representative_indices(len(result.t_grid)):
            subdir = out_dir / f"{result.primary_value_name}_{label}"
            subdir.mkdir(parents=True, exist_ok=True)
            paths = _save_static_plots(
                subdir,
                xs=result.x_grid,
                ys=result.y_grid,
                frame_values=result.primary_cube[idx],
                t_query=float(result.t_grid[idx]),
                history_times=result.history_times,
                history_locs=result.history_locs,
                title_prefix=title_prefix,
                value_name=result.primary_value_name,
                value_label=result.primary_value_label,
            )
            for key, value in paths.items():
                artifacts[f"{label}_{key}"] = Path(value)
        if cfg.interactive:
            fig = _plotly_intensity.plot_lambst_interactive(
                result.primary_cube,
                result.x_grid,
                result.y_grid,
                result.t_grid,
                show=False,
                master_title=f"{title_prefix} future exact surface",
            )
            html_path = out_dir / f"{result.primary_value_name}_interactive.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            artifacts["interactive_html"] = html_path
    return artifacts
