"""Rendering for sample-based predictive comparison bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unified_stpp.evaluation.predictive import PredictiveComparisonResult
from unified_stpp.viz.render_common import plot_styles, write_gif_if_requested


@dataclass(frozen=True)
class PredictiveRenderConfig:
    plot_style: str = "2d"
    fps: float = 2.0
    write_gif: bool = False


def _render_heatmap(
    ax,
    *,
    values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
    history_locs: np.ndarray | None,
    true_window_locs: np.ndarray | None,
):
    im = ax.imshow(
        values.T,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        aspect="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    if history_locs is not None and history_locs.size > 0:
        ax.plot(history_locs[:, 0], history_locs[:, 1], "-o", color="white", lw=1.0, ms=2.5)
    if true_window_locs is not None and true_window_locs.size > 0:
        ax.scatter(
            true_window_locs[:, 0],
            true_window_locs[:, 1],
            s=18,
            marker="x",
            color="cyan",
            linewidths=0.9,
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return im


def _render_surface3d(
    ax,
    *,
    values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
    history_locs: np.ndarray | None,
    true_window_locs: np.ndarray | None,
):
    from matplotlib import cm

    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    surf = ax.plot_surface(
        xx,
        yy,
        values.T,
        cmap=cm.magma,
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )
    z_hist = max(float(values.max()) * 0.88, float(vmax) * 0.7, 1e-6)
    z_true = max(float(values.max()) * 0.95, float(vmax) * 0.8, 1e-6)
    if history_locs is not None and history_locs.size > 0:
        ax.plot(
            history_locs[:, 0],
            history_locs[:, 1],
            np.full(history_locs.shape[0], z_hist, dtype=np.float32),
            "-o",
            color="black",
            lw=1.0,
            ms=2.5,
        )
    if true_window_locs is not None and true_window_locs.size > 0:
        ax.scatter(
            true_window_locs[:, 0],
            true_window_locs[:, 1],
            np.full(true_window_locs.shape[0], z_true, dtype=np.float32),
            s=18,
            marker="x",
            color="cyan",
            linewidths=0.9,
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("proxy")
    ax.view_init(elev=32, azim=-58)
    return surf


def _render_frame(
    out_dir: Path,
    *,
    model_label: str,
    frame_idx: int,
    values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
    history_locs: np.ndarray | None,
    true_window_locs: np.ndarray | None,
    plot_style: str,
    filename_suffix: str = "",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plot_style == "3d":
        fig = plt.figure(figsize=(8.5, 5.8))
        ax = fig.add_subplot(111, projection="3d")
        mappable = _render_surface3d(
            ax,
            values=values,
            xs=xs,
            ys=ys,
            vmin=vmin,
            vmax=vmax,
            title=title,
            history_locs=history_locs,
            true_window_locs=true_window_locs,
        )
        # Use explicit cax so the 3D axes is not resized by colorbar
        cax = fig.add_axes([0.87, 0.18, 0.025, 0.58])
        fig.colorbar(mappable, cax=cax, label="predictive spatial rate proxy")
    else:
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        mappable = _render_heatmap(
            ax,
            values=values,
            xs=xs,
            ys=ys,
            vmin=vmin,
            vmax=vmax,
            title=title,
            history_locs=history_locs,
            true_window_locs=true_window_locs,
        )
        fig.colorbar(mappable, ax=ax, label="sample-based predictive spatial rate proxy")
        fig.tight_layout()
    model_dir = out_dir / "models" / model_label
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"frame_{frame_idx:03d}{filename_suffix}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _render_panel(
    out_dir: Path,
    *,
    frame_idx: int,
    panels: list[dict[str, Any]],
    xs: np.ndarray,
    ys: np.ndarray,
    vmin: float,
    vmax: float,
    rollout_mode: str,
    frame_window,
    plot_style: str,
    filename_suffix: str = "",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(panels)
    if plot_style == "3d":
        fig = plt.figure(figsize=(5.8 * n_models + 1.2, 6.4))
        axes_row = [fig.add_subplot(1, n_models, idx + 1, projection="3d") for idx in range(n_models)]
    else:
        fig, axes = plt.subplots(1, n_models, figsize=(5.4 * n_models, 5.4), squeeze=False)
        axes_row = list(axes[0])
    mappable = None
    for ax, panel in zip(axes_row, panels):
        if plot_style == "3d":
            mappable = _render_surface3d(
                ax,
                values=panel["surface"],
                xs=xs,
                ys=ys,
                vmin=vmin,
                vmax=vmax,
                title=panel["title"],
                history_locs=panel["history_locs"],
                true_window_locs=panel["true_window_locs"],
            )
        else:
            mappable = _render_heatmap(
                ax,
                values=panel["surface"],
                xs=xs,
                ys=ys,
                vmin=vmin,
                vmax=vmax,
                title=panel["title"],
                history_locs=panel["history_locs"],
                true_window_locs=panel["true_window_locs"],
            )
    fig.suptitle(
        (
            f"{rollout_mode} | frame={frame_idx} | "
            f"window=[{frame_window.start:.3f}, {frame_window.end:.3f})\n"
            "quantity=sample-based predictive spatial rate proxy"
        ),
        y=0.99,
        fontsize=9,
    )
    if mappable is not None:
        if plot_style == "3d":
            # Explicit cax keeps 3D axes from being resized by the colorbar
            cax = fig.add_axes([0.91, 0.18, 0.018, 0.55])
            fig.colorbar(mappable, cax=cax, label="predictive spatial rate proxy")
        else:
            fig.colorbar(
                mappable,
                ax=axes_row,
                shrink=0.85,
                pad=0.02,
                label="sample-based predictive spatial rate proxy",
            )
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    path = panel_dir / f"frame_{frame_idx:03d}{filename_suffix}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def render_predictive_bundle(
    result: PredictiveComparisonResult,
    out_dir: Path,
    config: PredictiveRenderConfig | None = None,
) -> dict[str, Path]:
    cfg = config or PredictiveRenderConfig()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    styles = plot_styles(cfg.plot_style)
    artifacts: dict[str, Path] = {}
    panel_paths_by_style: dict[str, list[str]] = {style: [] for style in styles}
    gif_paths_by_style: dict[str, str | None] = {style: None for style in styles}
    vmin = float(result.color_scale["vmin"])
    vmax = float(result.color_scale["vmax"])
    for model in result.models:
        for frame in model.frames:
            title = (
                f"{model.label}\n"
                f"mean count/rollout={frame.mean_events_per_rollout:.2f} | "
                f"pooled={frame.pooled_event_locs.shape[0]}"
            )
            for style in styles:
                suffix = "" if style == styles[0] else f"_{style}"
                png = _render_frame(
                    out_dir,
                    model_label=model.safe_label,
                    frame_idx=frame.window.index,
                    values=frame.derived_kde_rate_surface,
                    xs=result.xs,
                    ys=result.ys,
                    vmin=vmin,
                    vmax=vmax,
                    title=title,
                    history_locs=frame.history_locs,
                    true_window_locs=frame.true_event_locs,
                    plot_style=style,
                    filename_suffix=suffix,
                )
                artifacts[f"predictive_model_{model.safe_label}_frame_{frame.window.index:03d}{suffix}"] = Path(png)

    for frame_idx, frame_window in enumerate(result.frame_schedule):
        panels = []
        for model in result.models:
            frame = model.frames[frame_idx]
            panels.append(
                {
                    "surface": frame.derived_kde_rate_surface,
                    "title": (
                        f"{model.label}\n"
                        f"mean count/rollout={frame.mean_events_per_rollout:.2f} | "
                        f"pooled={frame.pooled_event_locs.shape[0]}"
                    ),
                    "history_locs": frame.history_locs,
                    "true_window_locs": frame.true_event_locs,
                }
            )
        for style in styles:
            suffix = "" if style == styles[0] else f"_{style}"
            path = _render_panel(
                out_dir,
                frame_idx=frame_idx,
                panels=panels,
                xs=result.xs,
                ys=result.ys,
                vmin=vmin,
                vmax=vmax,
                rollout_mode=result.spec.rollout_mode,
                frame_window=frame_window,
                plot_style=style,
                filename_suffix=suffix,
            )
            panel_paths_by_style[style].append(path)
            artifacts[f"predictive_panel_frame_{frame_idx:03d}{suffix}"] = Path(path)

    if cfg.write_gif:
        for style in styles:
            suffix = "" if style == styles[0] else f"_{style}"
            out_path = out_dir / "panels" / f"animation{suffix}.gif"
            gif_path = write_gif_if_requested(panel_paths_by_style[style], out_path=out_path, fps=cfg.fps)
            if gif_path is not None:
                gif_paths_by_style[style] = gif_path
                artifacts[f"predictive_animation{suffix}"] = Path(gif_path)
    return artifacts
