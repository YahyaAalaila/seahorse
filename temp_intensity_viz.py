#!/usr/bin/env python3
"""Temporary intensity visualizer for local model-family checks.

This script is intentionally outside the main benchmark/evaluation API.
It loads a trained run, conditions on a user-provided sequence JSONL, computes
the selected surface/intensity object for the chosen sequence, and saves both
static plots and optional interactive HTML output.

Current support:
- deep_stpp
- auto_stpp_faithful
- neural_stpp_shared_cond_gmm
- neural_stpp_shared_jumpcnf
- neural_stpp_shared_attncnf

Important:
- The run directory alone is not enough for meaningful conditional intensity
  plots. You must also provide the original sequence JSONL via --history.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unified_stpp.evaluation.intensity import calc_lamb_from_runner
from unified_stpp.runner.runner import STPPRunner
from unified_stpp.utils import load_jsonl
from unified_stpp.viz import plot_lambst_interactive

DEFAULT_X_NSTEP = 81
DEFAULT_Y_NSTEP = 81
DEFAULT_T_NSTEP = 41


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Path to a trained run directory.")
    p.add_argument(
        "--history",
        required=True,
        help=(
            "JSONL file containing candidate sequences. "
            "Required because run artifacts do not persist original histories."
        ),
    )
    p.add_argument(
        "--split",
        default="test",
        choices=("val", "test"),
        help="Label used for metadata and calc_lamb split semantics (default: test).",
    )
    p.add_argument("--seq-idx", type=int, default=0, help="Sequence index inside --history.")
    p.add_argument(
        "--history-length",
        type=int,
        default=0,
        help="If > 0, keep only the last N events from the chosen sequence before plotting.",
    )
    p.add_argument("--x-nstep", type=int, default=81, help="Number of x-grid points.")
    p.add_argument("--y-nstep", type=int, default=81, help="Number of y-grid points.")
    p.add_argument("--t-nstep", type=int, default=41, help="Number of time frames.")
    p.add_argument(
        "--future-horizon",
        type=float,
        default=None,
        help=(
            "Original-time horizon beyond the last history event for factorized "
            "Neural-STPP surfaces. If omitted, infer a small local horizon."
        ),
    )
    p.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Static-frame index to render. Negative means the middle frame.",
    )
    p.add_argument(
        "--no-round-time",
        action="store_true",
        help="Disable upstream notebook-style rounded time-grid start/end behavior.",
    )
    p.add_argument(
        "--trunc",
        action="store_true",
        help="Force truncation to the model's max_history during calc_lamb.",
    )
    p.add_argument("--xmin", type=float, default=None, help="Optional original-space xmin override.")
    p.add_argument("--xmax", type=float, default=None, help="Optional original-space xmax override.")
    p.add_argument("--ymin", type=float, default=None, help="Optional original-space ymin override.")
    p.add_argument("--ymax", type=float, default=None, help="Optional original-space ymax override.")
    p.add_argument(
        "--spatial-chunk-size",
        type=int,
        default=None,
        help=(
            "Chunk size for Neural-STPP spatial conditional-density queries. "
            "If omitted, use preset-aware conservative defaults."
        ),
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device for loading/evaluation: auto, cpu, cuda, cuda:0, mps, ...",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <run>/temp_intensity_viz.",
    )
    return p.parse_args()


def _resolve_neural_stpp_viz_profile(
    *,
    preset: str,
    x_nstep: int,
    y_nstep: int,
    t_nstep: int,
    spatial_chunk_size: int | None,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "x_nstep": int(x_nstep),
        "y_nstep": int(y_nstep),
        "t_nstep": int(t_nstep),
        "spatial_chunk_size": int(spatial_chunk_size) if spatial_chunk_size is not None else None,
        "auto_coarsened_grid": False,
        "warnings": [],
    }
    defaults_unchanged = (
        int(x_nstep) == DEFAULT_X_NSTEP
        and int(y_nstep) == DEFAULT_Y_NSTEP
        and int(t_nstep) == DEFAULT_T_NSTEP
    )

    if preset == "neural_stpp_shared_cond_gmm":
        if profile["spatial_chunk_size"] is None:
            profile["spatial_chunk_size"] = 4096
        profile["warnings"].append(
            "Neural-STPP CondGMM uses closed-form conditional-mixture queries and is the cheapest "
            "family member for dense temporary surface visualization."
        )
        return profile

    if preset == "neural_stpp_shared_jumpcnf":
        if defaults_unchanged:
            profile.update(
                {
                    "x_nstep": 49,
                    "y_nstep": 49,
                    "t_nstep": 21,
                    "auto_coarsened_grid": True,
                }
            )
        if profile["spatial_chunk_size"] is None:
            profile["spatial_chunk_size"] = 1024
        profile["warnings"].append(
            "Neural-STPP JumpCNF is expensive: each time slice evaluates a CNF conditional log-density "
            "over spatial chunks. Coarse defaults are used when the dense script defaults are left unchanged."
        )
        return profile

    if preset == "neural_stpp_shared_attncnf":
        if defaults_unchanged:
            profile.update(
                {
                    "x_nstep": 33,
                    "y_nstep": 33,
                    "t_nstep": 11,
                    "auto_coarsened_grid": True,
                }
            )
        if profile["spatial_chunk_size"] is None:
            profile["spatial_chunk_size"] = 512
        profile["warnings"].append(
            "Neural-STPP AttnCNF is the heaviest variant: each time slice pays for attentive CNF spatial "
            "queries. Expect slow runs and prefer coarse grids unless you explicitly override them."
        )
        return profile

    raise ValueError(f"Unsupported Neural-STPP preset for factorized viz profile: {preset}")


def _resolve_device(spec: str, *, preset: str | None = None) -> torch.device:
    if (
        spec == "auto"
        and preset is not None
        and preset.startswith("neural_stpp_shared_")
    ):
        try:
            import torch as _torch

            mps_available = getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available()
        except AttributeError:
            mps_available = False
        if mps_available:
            return torch.device("cpu")
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def _load_sequence(path: Path, seq_idx: int, history_length: int) -> dict[str, Any]:
    seqs = load_jsonl(path)
    if not seqs:
        raise SystemExit(f"No sequences found in {path}")
    if seq_idx < 0 or seq_idx >= len(seqs):
        raise SystemExit(f"--seq-idx {seq_idx} out of range for {path} (n={len(seqs)})")

    seq = dict(seqs[seq_idx])
    full_times = np.asarray(seq["times"], dtype=np.float32)
    full_locs = np.asarray(seq["locations"], dtype=np.float32)
    times = full_times
    locs = full_locs
    if history_length > 0 and len(times) > history_length:
        times = times[-history_length:]
        locs = locs[-history_length:]
    return {
        "times": times,
        "locations": locs,
        "full_times": full_times,
        "full_locations": full_locs,
    }


def _resolve_frame_index(n_frames: int, frame_index: int) -> int:
    if n_frames <= 0:
        raise SystemExit("Intensity cube has no frames.")
    if frame_index < 0:
        return n_frames // 2
    if frame_index >= n_frames:
        raise SystemExit(f"--frame-index {frame_index} out of range for n_frames={n_frames}")
    return frame_index


def _build_future_query_grid(
    *,
    last_t: float,
    horizon: float,
    n_steps: int,
) -> np.ndarray:
    """Build a strictly-future query grid for Neural-STPP visualization.

    Neural-STPP surfaces are meant to visualize the future conditional object
    given a history, so the first frame should sit *after* the last observed
    event rather than exactly on it.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    t_end = float(last_t) + float(horizon)
    return np.linspace(float(last_t), t_end, int(n_steps) + 1, dtype=np.float32)[1:]


def _history_until_t(
    history_times: np.ndarray,
    history_locs: np.ndarray,
    t_query: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(history_times, dtype=np.float32) <= float(t_query)
    return history_times[mask], history_locs[mask]


def _history_overlay_z_level(frame_values: np.ndarray) -> float:
    """Place the history overlay inside the actual surface range.

    This avoids floating the path above an all-zero surface with an arbitrary
    epsilon floor.
    """
    values = np.asarray(frame_values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        return vmax
    return vmin + 0.88 * (vmax - vmin)


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
    value_name: str = "intensity",
    value_label: str = "intensity",
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
    hist_t, hist_s = _history_until_t(history_times, history_locs, t_query)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        frame_values.T,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        aspect="auto",
        cmap="magma",
    )
    ax.set_title(f"{title_prefix} {value_label} @ t={t_query:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label=value_label)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        frame_values.T,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        aspect="auto",
        cmap="magma",
    )
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
    surf = ax.plot_surface(
        xx,
        yy,
        frame_values.T,
        cmap=cm.magma,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )
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
    surf = ax.plot_surface(
        xx,
        yy,
        frame_values.T,
        cmap=cm.magma,
        linewidth=0,
        antialiased=True,
        alpha=0.82,
    )
    if hist_s.size > 0:
        z_hist_level = _history_overlay_z_level(frame_values)
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
    surf = ax.plot_surface(
        xx,
        yy,
        frame_values.T,
        cmap=cm.magma,
        linewidth=0,
        antialiased=True,
        alpha=0.78,
    )
    ax.contour(
        xx,
        yy,
        frame_values.T,
        zdir="z",
        offset=0.0,
        cmap=cm.magma,
        levels=12,
        linewidths=0.9,
    )
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


def _save_temporal_curve_plot(
    out_dir: Path,
    *,
    t_grid: np.ndarray,
    lambda_t: np.ndarray,
    title: str,
) -> str:
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


def _normalize_history(
    *,
    history_times: np.ndarray,
    history_locs: np.ndarray,
    norm_stats: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    time_mean = float(norm_stats["time_mean"])
    time_std = max(float(norm_stats["time_std"]), 1e-8)
    loc_mean = np.asarray(norm_stats["loc_mean"], dtype=np.float32)
    loc_std = np.maximum(np.asarray(norm_stats["loc_std"], dtype=np.float32), 1e-8)
    times_norm = (np.asarray(history_times, dtype=np.float32) - time_mean) / time_std
    locs_norm = (np.asarray(history_locs, dtype=np.float32) - loc_mean) / loc_std
    return times_norm.astype(np.float32), locs_norm.astype(np.float32)


def _infer_future_horizon(
    *,
    history_times: np.ndarray,
    full_times: np.ndarray,
    future_horizon_arg: float | None,
) -> float:
    if future_horizon_arg is not None:
        return max(float(future_horizon_arg), 1e-3)

    last_t = float(history_times[-1])
    future = full_times[full_times > last_t]
    if future.size > 0:
        return max(float(future[-1] - last_t), 1e-3)

    if history_times.size > 1:
        dts = np.diff(history_times.astype(np.float32))
        dts = dts[dts > 0]
        if dts.size > 0:
            return max(float(np.median(dts) * 3.0), 1e-3)
        span = float(history_times[-1] - history_times[0])
        if span > 0.0:
            return max(span * 0.25, 1e-3)
    return 1.0


def _resolve_spatial_bounds(
    *,
    full_locs: np.ndarray,
    norm_stats: dict[str, Any],
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
) -> tuple[float, float, float, float]:
    loc_std = np.maximum(np.asarray(norm_stats["loc_std"], dtype=np.float32), 1e-8)
    x_vals = np.asarray(full_locs[:, 0], dtype=np.float32)
    y_vals = np.asarray(full_locs[:, 1], dtype=np.float32)

    def _bounds(vals: np.ndarray, *, lo: float | None, hi: float | None, pad_scale: float) -> tuple[float, float]:
        if lo is not None and hi is not None:
            return float(lo), float(hi)
        v_lo = float(vals.min())
        v_hi = float(vals.max())
        span = max(v_hi - v_lo, 1e-6)
        pad = max(0.15 * span, float(pad_scale))
        return (
            float(lo) if lo is not None else v_lo - pad,
            float(hi) if hi is not None else v_hi + pad,
        )

    x_lo, x_hi = _bounds(x_vals, lo=xmin, hi=xmax, pad_scale=0.5 * float(loc_std[0]))
    y_lo, y_hi = _bounds(y_vals, lo=ymin, hi=ymax, pad_scale=0.5 * float(loc_std[1]))
    return x_lo, x_hi, y_lo, y_hi


def _representative_indices(n_frames: int) -> list[tuple[str, int]]:
    labels = [("start", 0), ("mid", n_frames // 2), ("end", n_frames - 1)]
    seen: set[int] = set()
    out: list[tuple[str, int]] = []
    for label, idx in labels:
        if idx not in seen:
            seen.add(idx)
            out.append((label, idx))
    return out


def _run_neural_stpp_factorized_viz(
    *,
    runner: STPPRunner,
    run_dir: Path,
    out_dir: Path,
    seq: dict[str, Any],
    seq_idx: int,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    model = runner.model.to(device)
    model.eval()
    preset = runner.config.model.preset
    profile = _resolve_neural_stpp_viz_profile(
        preset=preset,
        x_nstep=args.x_nstep,
        y_nstep=args.y_nstep,
        t_nstep=args.t_nstep,
        spatial_chunk_size=args.spatial_chunk_size,
    )

    history_times = np.asarray(seq["times"], dtype=np.float32)
    history_locs = np.asarray(seq["locations"], dtype=np.float32)
    full_times = np.asarray(seq["full_times"], dtype=np.float32)
    full_locs = np.asarray(seq["full_locations"], dtype=np.float32)

    times_norm, locs_norm = _normalize_history(
        history_times=history_times,
        history_locs=history_locs,
        norm_stats=runner.norm_stats,
    )
    lengths = torch.tensor([len(history_times)], dtype=torch.long, device=device)
    times_t = torch.as_tensor(times_norm, dtype=torch.float32, device=device).unsqueeze(0)
    locs_t = torch.as_tensor(locs_norm, dtype=torch.float32, device=device).unsqueeze(0)

    state_ctx = model.state_model.encode_history(
        times=times_t,
        locations=locs_t,
        lengths=lengths,
    )

    horizon = _infer_future_horizon(
        history_times=history_times,
        full_times=full_times,
        future_horizon_arg=args.future_horizon,
    )
    t_start = float(history_times[-1])
    t_grid = _build_future_query_grid(
        last_t=t_start,
        horizon=float(horizon),
        n_steps=int(profile["t_nstep"]),
    )

    x_lo, x_hi, y_lo, y_hi = _resolve_spatial_bounds(
        full_locs=full_locs,
        norm_stats=runner.norm_stats,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
    )
    x_grid = np.linspace(x_lo, x_hi, int(profile["x_nstep"]), dtype=np.float32)
    y_grid = np.linspace(y_lo, y_hi, int(profile["y_nstep"]), dtype=np.float32)

    xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")
    s_grid_orig = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    loc_mean = np.asarray(runner.norm_stats["loc_mean"], dtype=np.float32)
    loc_std = np.maximum(np.asarray(runner.norm_stats["loc_std"], dtype=np.float32), 1e-8)
    s_grid_norm = (s_grid_orig - loc_mean) / loc_std
    s_grid_norm_t = torch.as_tensor(s_grid_norm, dtype=torch.float32, device=device)
    jacobian_scale = float(np.prod(loc_std))

    time_mean = float(runner.norm_stats["time_mean"])
    time_std = max(float(runner.norm_stats["time_std"]), 1e-8)
    lambda_t = np.zeros(len(t_grid), dtype=np.float32)
    spatial_density = np.zeros((len(t_grid), len(x_grid), len(y_grid)), dtype=np.float32)
    joint_intensity = np.zeros_like(spatial_density)
    chunk_size = int(profile["spatial_chunk_size"])

    total_spatial_points = int(s_grid_norm_t.shape[0])
    chunk_calls_per_frame = int(math.ceil(total_spatial_points / max(chunk_size, 1)))
    print(
        f"[temp_intensity_viz] {preset}: evaluating Neural-STPP factorized surface "
        f"with t_nstep={len(t_grid)}, x_nstep={len(x_grid)}, y_nstep={len(y_grid)}, "
        f"spatial_chunk_size={chunk_size}."
    )
    if profile["auto_coarsened_grid"]:
        print(
            "[temp_intensity_viz] Dense script defaults were left unchanged, so this heavy "
            f"{preset} query was auto-coarsened to {len(t_grid)} time slices and "
            f"{len(x_grid)}x{len(y_grid)} spatial points."
        )
    for warning in profile["warnings"]:
        print(f"[temp_intensity_viz] note: {warning}")
    print(
        "[temp_intensity_viz] query complexity: "
        f"{total_spatial_points} spatial points per frame, about {chunk_calls_per_frame} "
        f"chunked spatial decoder calls per frame, {chunk_calls_per_frame * len(t_grid)} total."
    )

    for i, t_query_raw in enumerate(t_grid):
        t_query_norm = (float(t_query_raw) - time_mean) / time_std
        terms = model.event_model.fixed_time_query_terms(
            state=state_ctx,
            query_time=t_query_norm,
            device=device,
        )
        lambda_i = float(torch.as_tensor(terms["lambda_t"]).detach().cpu().item())
        lambda_t[i] = lambda_i

        logprob_chunks: list[np.ndarray] = []
        logprob_fn = terms["logprob_fn"]
        for j in range(0, s_grid_norm_t.shape[0], chunk_size):
            chunk = s_grid_norm_t[j : j + chunk_size]
            logprob = logprob_fn(chunk).detach().cpu().numpy().astype(np.float32)
            logprob_chunks.append(logprob)
        logprob_grid = np.concatenate(logprob_chunks, axis=0).reshape(len(x_grid), len(y_grid))
        density_orig = np.exp(logprob_grid).astype(np.float32) / max(jacobian_scale, 1e-8)
        spatial_density[i] = density_orig
        joint_intensity[i] = density_orig * lambda_i

    temporal_curve_path = _save_temporal_curve_plot(
        out_dir,
        t_grid=t_grid,
        lambda_t=lambda_t,
        title=f"{runner.config.model.preset} seq={seq_idx} temporal intensity",
    )

    spatial_slice_paths: dict[str, dict[str, str]] = {}
    joint_slice_paths: dict[str, dict[str, str]] = {}
    for label, idx in _representative_indices(len(t_grid)):
        spatial_dir = out_dir / f"spatial_density_{label}"
        spatial_dir.mkdir(parents=True, exist_ok=True)
        spatial_slice_paths[label] = _save_static_plots(
            spatial_dir,
            xs=x_grid,
            ys=y_grid,
            frame_values=spatial_density[idx],
            t_query=float(t_grid[idx]),
            history_times=history_times,
            history_locs=history_locs,
            title_prefix=f"{runner.config.model.preset} seq={seq_idx}",
            value_name="spatial_density",
            value_label="spatial density",
        )

        joint_dir = out_dir / f"joint_intensity_{label}"
        joint_dir.mkdir(parents=True, exist_ok=True)
        joint_slice_paths[label] = _save_static_plots(
            joint_dir,
            xs=x_grid,
            ys=y_grid,
            frame_values=joint_intensity[idx],
            t_query=float(t_grid[idx]),
            history_times=history_times,
            history_locs=history_locs,
            title_prefix=f"{runner.config.model.preset} seq={seq_idx}",
            value_name="joint_intensity",
            value_label="joint intensity",
        )

    interactive_html_path = out_dir / "joint_intensity_interactive.html"
    interactive_error: str | None = None
    try:
        fig = plot_lambst_interactive(
            joint_intensity,
            x_grid,
            y_grid,
            t_grid,
            show=False,
            master_title=(
                f"{runner.config.model.preset} seq={seq_idx} "
                "Neural-STPP joint conditional intensity"
            ),
        )
        fig.write_html(str(interactive_html_path), include_plotlyjs="cdn")
    except Exception as exc:  # keep static plots even if Plotly fails
        interactive_error = str(exc)

    npz_path = out_dir / "neural_stpp_surface_slices.npz"
    np.savez_compressed(
        npz_path,
        t_grid=t_grid.astype(np.float32),
        x_grid=x_grid.astype(np.float32),
        y_grid=y_grid.astype(np.float32),
        lambda_t=lambda_t.astype(np.float32),
        spatial_density=spatial_density.astype(np.float32),
        joint_intensity=joint_intensity.astype(np.float32),
        history_times=history_times.astype(np.float32),
        history_locations=history_locs.astype(np.float32),
    )

    summary = {
        "run_dir": str(run_dir),
        "history_path": str(Path(args.history).resolve()),
        "split": args.split,
        "seq_idx": int(seq_idx),
        "history_length": int(args.history_length),
        "preset": preset,
        "device": str(device),
        "x_nstep": int(profile["x_nstep"]),
        "y_nstep": int(profile["y_nstep"]),
        "t_nstep": int(profile["t_nstep"]),
        "requested_x_nstep": int(args.x_nstep),
        "requested_y_nstep": int(args.y_nstep),
        "requested_t_nstep": int(args.t_nstep),
        "future_horizon": float(horizon),
        "spatial_chunk_size": int(chunk_size),
        "auto_coarsened_grid": bool(profile["auto_coarsened_grid"]),
        "notes": list(profile["warnings"]),
        "query_complexity": {
            "spatial_points_per_frame": total_spatial_points,
            "chunk_calls_per_frame": chunk_calls_per_frame,
            "total_chunk_calls": chunk_calls_per_frame * len(t_grid),
        },
        "plots": {
            "temporal_curve": temporal_curve_path,
            "spatial_density_slices": spatial_slice_paths,
            "joint_intensity_slices": joint_slice_paths,
        },
        "interactive_html": str(interactive_html_path) if interactive_error is None else None,
        "interactive_error": interactive_error,
        "surface_npz": str(npz_path),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved to: {out_dir}")
    print(f"  temporal_curve: {temporal_curve_path}")
    for label, paths in spatial_slice_paths.items():
        print(f"  spatial_density_{label}: {paths['spatial_density_heatmap']}")
    for label, paths in joint_slice_paths.items():
        print(f"  joint_intensity_{label}: {paths['joint_intensity_heatmap']}")
    if interactive_error is None:
        print(f"  joint_intensity_interactive_html: {interactive_html_path}")
    else:
        print(f"  joint_intensity_interactive_html: skipped ({interactive_error})")
    print(f"  surface_npz: {npz_path}")


def _run_notebook_intensity_viz(
    *,
    runner: STPPRunner,
    run_dir: Path,
    out_dir: Path,
    seq: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    cube = calc_lamb_from_runner(
        runner=runner,
        sequences=[{"times": seq["times"], "locations": seq["locations"]}],
        seq_idx=0,
        split=args.split,
        x_nstep=args.x_nstep,
        y_nstep=args.y_nstep,
        t_nstep=args.t_nstep,
        round_time=not args.no_round_time,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        trunc=bool(args.trunc),
        device=device,
    )

    frame_index = _resolve_frame_index(len(cube.t_range), args.frame_index)
    t_query = float(cube.t_range[frame_index])
    frame_values = cube.lambs[frame_index]
    title_prefix = f"{runner.config.model.preset} seq={args.seq_idx}"

    static_plot_paths = _save_static_plots(
        out_dir,
        xs=cube.x_range,
        ys=cube.y_range,
        frame_values=frame_values,
        t_query=t_query,
        history_times=cube.history_times,
        history_locs=cube.history_locs,
        title_prefix=title_prefix,
    )

    interactive_html_path = out_dir / "intensity_interactive.html"
    interactive_error: str | None = None
    try:
        fig = plot_lambst_interactive(
            cube.lambs,
            cube.x_range,
            cube.y_range,
            cube.t_range,
            show=False,
            master_title=f"{title_prefix} notebook-faithful intensity",
        )
        fig.write_html(str(interactive_html_path), include_plotlyjs="cdn")
    except Exception as exc:  # temporary utility: keep static plots even if Plotly fails
        interactive_error = str(exc)

    np.savez_compressed(
        out_dir / "intensity_cube.npz",
        lambs=cube.lambs.astype(np.float32),
        x_range=cube.x_range.astype(np.float32),
        y_range=cube.y_range.astype(np.float32),
        t_range=cube.t_range.astype(np.float32),
        history_times=cube.history_times.astype(np.float32),
        history_locations=cube.history_locs.astype(np.float32),
        frame_index=np.asarray(frame_index, dtype=np.int64),
        frame_time=np.asarray(t_query, dtype=np.float32),
    )

    summary = {
        "run_dir": str(run_dir),
        "history_path": str(Path(args.history).resolve()),
        "split": args.split,
        "seq_idx": int(args.seq_idx),
        "history_length": int(args.history_length),
        "preset": runner.config.model.preset,
        "device": str(device),
        "x_nstep": int(args.x_nstep),
        "y_nstep": int(args.y_nstep),
        "t_nstep": int(args.t_nstep),
        "frame_index": int(frame_index),
        "frame_time": float(t_query),
        "plots": static_plot_paths,
        "interactive_html": str(interactive_html_path) if interactive_error is None else None,
        "interactive_error": interactive_error,
        "intensity_cube_npz": str(out_dir / "intensity_cube.npz"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved to: {out_dir}")
    for key, value in static_plot_paths.items():
        print(f"  {key}: {value}")
    if interactive_error is None:
        print(f"  intensity_interactive_html: {interactive_html_path}")
    else:
        print(f"  intensity_interactive_html: skipped ({interactive_error})")
    print(f"  intensity_cube_npz: {out_dir / 'intensity_cube.npz'}")


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run).resolve()
    history_path = Path(args.history).resolve()
    if not history_path.exists():
        raise SystemExit(f"History JSONL not found: {history_path}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "temp_intensity_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = STPPRunner.load(run_dir)
    preset = runner.config.model.preset
    if preset not in {
        "deep_stpp",
        "auto_stpp_faithful",
        "neural_stpp_shared_cond_gmm",
        "neural_stpp_shared_jumpcnf",
        "neural_stpp_shared_attncnf",
    }:
        raise SystemExit(
            f"Unsupported preset for this temporary script: {preset}. "
            "Supported: deep_stpp, auto_stpp_faithful, "
            "neural_stpp_shared_cond_gmm, neural_stpp_shared_jumpcnf, "
            "neural_stpp_shared_attncnf."
        )

    device = _resolve_device(args.device, preset=preset)
    runner.model.to(device)
    runner.model.eval()

    seq = _load_sequence(history_path, args.seq_idx, args.history_length)
    if preset in {
        "neural_stpp_shared_cond_gmm",
        "neural_stpp_shared_jumpcnf",
        "neural_stpp_shared_attncnf",
    }:
        _run_neural_stpp_factorized_viz(
            runner=runner,
            run_dir=run_dir,
            out_dir=out_dir,
            seq=seq,
            seq_idx=args.seq_idx,
            device=device,
            args=args,
        )
    else:
        _run_notebook_intensity_viz(
            runner=runner,
            run_dir=run_dir,
            out_dir=out_dir,
            seq=seq,
            device=device,
            args=args,
        )


if __name__ == "__main__":
    main()
