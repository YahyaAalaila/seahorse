"""
Intensity visualization for benchmark runs.

plot_bench_intensities()
    Given a BenchmarkTable + the raw train/test sequences, loads the best
    checkpoint for each preset and renders a side-by-side animated GIF (or
    MP4) of lambda*(t, s) evolving over time.

Layout  (one output file per dataset):
    columns = presets  |  rows = one row of 3-D surface plots
    animated: time sweeps from T_start -> T_end of the test sequence

Called from BenchmarkTable.plot_intensities() or directly from cmd_bench().
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from archive.registry import build_model
from unified_stpp.config import STPPConfig
from unified_stpp.models import IntensityEvaluator
from unified_stpp.runner.results import RunResult


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(run: RunResult, device: str = "cpu") -> nn.Module:
    """Reconstruct and load a model from a RunResult."""
    cfg = STPPConfig(**run.effective_config)
    mc = cfg.model
    model = build_model(
        mc.build_overrides,
        preset=mc.preset,
        hidden_dim=mc.hidden_dim,
        spatial_dim=mc.spatial_dim,
        n_marks=mc.n_marks,
        event_cov_dim=mc.event_cov_dim,
        field_cov_dim=mc.field_cov_dim,
    )
    _load_ckpt(run.checkpoint_path, model)
    return model.to(device).eval()


def _load_ckpt(ckpt_path: Path, model: nn.Module) -> None:
    """Lightning checkpoint loader with compatibility shims."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", {})
    stripped: dict = {}
    for k, v in sd.items():
        if not k.startswith("model."):
            continue
        key = k[6:]
        # Old checkpoints used velocity.net; new code uses velocity.mlp
        key = key.replace("decoder.spatial.velocity.net.", "decoder.spatial.velocity.mlp.")
        stripped[key] = v
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    # intensity_module is an alias for decoder.temporal — already loaded via that path
    real_missing = [k for k in missing
                    if not k.startswith("dynamics.aug_func.intensity_module.")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {ckpt_path.name}:\n"
            f"  missing={real_missing[:5]}\n  unexpected={unexpected[:5]}"
        )


# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------

def _norm_stats(train_seqs: List[dict]) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Z-score stats (mean, std) — used for normalize=True models."""
    all_t = np.concatenate([np.asarray(s["times"]).reshape(-1) for s in train_seqs])
    all_s = np.concatenate([np.asarray(s["locations"]).reshape(-1, 2) for s in train_seqs])
    time_mean = float(all_t.mean())
    time_std = max(float(all_t.std()), 1e-8)
    loc_mean = all_s.mean(axis=0)
    loc_std = np.where(np.abs(all_s.std(axis=0)) > 1e-8, all_s.std(axis=0), 1.0)
    return time_mean, time_std, loc_mean, loc_std


def _minmax_stats(train_seqs: List[dict]) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """MinMax stats (min, range) — used for normalize=False / paper_autostpp_sthp models.

    Returns (t_min, t_range, loc_min, loc_range) so that the model receives
    (x - loc_min) / loc_range which maps coordinates to [0, 1].
    """
    all_t = np.concatenate([np.asarray(s["times"]).reshape(-1) for s in train_seqs])
    all_s = np.concatenate([np.asarray(s["locations"]).reshape(-1, 2) for s in train_seqs])
    t_min = float(all_t.min())
    t_range = max(float(all_t.max() - t_min), 1e-8)
    loc_min = all_s.min(axis=0)
    loc_range = np.maximum(all_s.max(axis=0) - loc_min, 1e-8)
    return t_min, t_range, loc_min, loc_range


# ---------------------------------------------------------------------------
# Empirical KDE panel (data reference)
# ---------------------------------------------------------------------------

def _kde_lam(
    seq_t: np.ndarray,
    seq_s: np.ndarray,
    t_query: float,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """Spatial KDE of events up to t_query — model-free data reference."""
    t_arr = np.asarray(seq_t, dtype=np.float64).reshape(-1)
    s_arr = np.asarray(seq_s, dtype=np.float64).reshape(-1, 2)
    n_hist = max(1, int(np.searchsorted(t_arr, t_query, "right")))
    pts = s_arr[:n_hist]
    n = len(pts)
    if n < 2:
        return np.zeros_like(xx)
    # Scott's rule bandwidth, per axis
    bw_x = max(pts[:, 0].std(), 1e-6) * n ** (-1.0 / 6.0)
    bw_y = max(pts[:, 1].std(), 1e-6) * n ** (-1.0 / 6.0)
    dx = (xx[:, :, np.newaxis] - pts[:, 0]) / bw_x   # (G, G, n)
    dy = (yy[:, :, np.newaxis] - pts[:, 1]) / bw_y
    lam = np.exp(-0.5 * (dx ** 2 + dy ** 2)).sum(axis=-1)
    lam /= n * 2.0 * np.pi * bw_x * bw_y
    return lam


# ---------------------------------------------------------------------------
# Intensity on mesh
# ---------------------------------------------------------------------------

@torch.no_grad()
def _model_lam(
    model: nn.Module,
    times: np.ndarray,
    locs: np.ndarray,
    t_query: float,
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    loc_mean: np.ndarray,
    loc_std: np.ndarray,
    time_mean: float,
    time_std: float,
    device: str,
) -> np.ndarray:
    """Evaluate model lambda*(t_query, ·) on a meshgrid. Returns (n_grid, n_grid)."""
    t_arr = np.asarray(times, dtype=np.float64).reshape(-1)
    s_arr = np.asarray(locs, dtype=np.float64).reshape(-1, 2)
    n_hist = max(1, min(int(np.searchsorted(t_arr, t_query, "right")), len(t_arr)))

    h_times = ((t_arr[:n_hist] - time_mean) / time_std).astype(np.float32)
    h_locs = ((s_arr[:n_hist] - loc_mean) / loc_std).astype(np.float32)

    dev = torch.device(device)
    h_t = torch.tensor(h_times, device=dev).unsqueeze(0)     # (1, N)
    h_s = torch.tensor(h_locs, device=dev).unsqueeze(0)      # (1, N, 2)
    h_len = torch.tensor([n_hist], dtype=torch.long, device=dev)

    s_min_n = torch.tensor(
        (np.array([float(xx.min()), float(yy.min())]) - loc_mean) / loc_std,
        dtype=torch.float32, device=dev,
    )
    s_max_n = torch.tensor(
        (np.array([float(xx.max()), float(yy.max())]) - loc_mean) / loc_std,
        dtype=torch.float32, device=dev,
    )
    t_n = float((t_query - time_mean) / time_std)
    n_g = int(xx.shape[0])
    jac = max(float(time_std * np.prod(loc_std)), 1e-12)

    ev = IntensityEvaluator(
        model,
        history_times=h_t,
        history_locations=h_s,
        history_lengths=h_len,
    )

    _, _, lam_n = ev.intensity_grid(t=t_n, s_min=s_min_n, s_max=s_max_n, n_grid=n_g)
    return np.clip(lam_n.detach().cpu().numpy() / jac, 0.0, None)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_bench_intensities(
    table,  # BenchmarkTable
    splits: Dict[str, Tuple[List[dict], List[dict], Optional[List[dict]]]],
    out_dir: str | Path,
    fmt: str = "gif",       # "gif" | "mp4" | "png"
    n_frames: int = 24,
    n_grid: int = 40,
    fps: int = 8,
    device: str = "cpu",
) -> List[Path]:
    """Render animated intensity plots for each dataset.

    Parameters
    ----------
    table   : BenchmarkTable returned by Benchmark.run()
    splits  : same dict passed to Benchmark ({dataset_id: (train, val, test)})
    out_dir : output directory (created if necessary)
    fmt     : ``"gif"`` | ``"mp4"`` | ``"png"`` (static grid, no animation)
    n_frames: number of animation frames (ignored for png)
    n_grid  : spatial grid resolution per axis
    fps     : frames per second for gif/mp4
    device  : torch device for inference (``"cpu"`` recommended)

    Returns
    -------
    List of paths to produced files, one per dataset.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
        if fmt in ("gif", "mp4"):
            from matplotlib import animation
    except ImportError as exc:
        print(f"[intensity_plot] matplotlib unavailable — skipping: {exc}")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []

    # Best run per (preset, dataset): lowest val_nll across seeds
    best: Dict[Tuple[str, str], RunResult] = {}
    for run in table.runs:
        key = (run.preset, run.dataset_id)
        if key not in best or run.val_nll < best[key].val_nll:
            best[key] = run

    presets = sorted({run.preset for run in table.runs})
    dataset_ids = sorted({run.dataset_id for run in table.runs})

    for ds_id in dataset_ids:
        if ds_id not in splits:
            print(f"[intensity_plot] Dataset '{ds_id}' not in splits — skipping.")
            continue
        train_seqs, val_seqs, test_seqs = splits[ds_id]
        if not test_seqs:
            test_seqs = val_seqs      # fall back to val if no test

        # Z-score stats for normalize=True models (default)
        time_mean, time_std, loc_mean, loc_std = _norm_stats(train_seqs)
        # MinMax stats for normalize=False / paper_autostpp_sthp models
        t_min, t_range, loc_min, loc_range = _minmax_stats(train_seqs)

        # Pick longest test sequence for richest history
        seq = max(test_seqs, key=lambda s: len(np.asarray(s["times"]).reshape(-1)))
        seq_t = np.asarray(seq["times"], dtype=np.float64).reshape(-1)
        seq_s = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, 2)

        # Spatial extent from all sequences (1st/99th percentiles)
        all_locs = np.concatenate(
            [np.asarray(s["locations"]).reshape(-1, 2)
             for s in train_seqs + (test_seqs or [])],
            axis=0,
        )
        xq = np.percentile(all_locs[:, 0], [1, 99])
        yq = np.percentile(all_locs[:, 1], [1, 99])
        xs = max(float(xq[1] - xq[0]), 1e-3)
        ys = max(float(yq[1] - yq[0]), 1e-3)
        xb = (float(xq[0] - 0.1 * xs), float(xq[1] + 0.1 * xs))
        yb = (float(yq[0] - 0.1 * ys), float(yq[1] + 0.1 * ys))
        x = np.linspace(xb[0], xb[1], n_grid)
        y = np.linspace(yb[0], yb[1], n_grid)
        xx, yy = np.meshgrid(x, y, indexing="ij")

        # Time grid — skip the very first event (no history)
        t0 = float(seq_t[max(1, len(seq_t) // 10)])  # start after 10% of events
        t1 = float(seq_t[-1])
        if fmt == "png":
            t_snaps = np.linspace(t0, t1, 4, endpoint=False)
        else:
            t_snaps = np.linspace(t0, t1, n_frames, endpoint=False)

        # Load models
        models: Dict[str, nn.Module] = {}
        for preset in presets:
            run = best.get((preset, ds_id))
            if run is None or run.checkpoint_path is None:
                print(f"[intensity_plot] No checkpoint for ({preset}, {ds_id}) — skipping.")
                continue
            if not Path(run.checkpoint_path).exists():
                print(f"[intensity_plot] Checkpoint missing: {run.checkpoint_path}")
                continue
            print(f"  Loading {preset} from {Path(run.checkpoint_path).name}...")
            try:
                models[preset] = _load_model(run, device=device)
            except Exception as exc:
                print(f"  [WARN] Could not load {preset}: {exc}")

        if not models:
            print(f"[intensity_plot] No models loaded for dataset '{ds_id}' — skipping.")
            continue

        loaded_presets = [p for p in presets if p in models]
        # First column is always the empirical KDE reference panel
        display_names = ["Actual (KDE)"] + loaded_presets
        n_col = len(display_names)

        print(f"  Evaluating {len(t_snaps)} frames × {len(loaded_presets)} presets on '{ds_id}'...")

        # Pre-compute all frames — column 0 = KDE, columns 1..n = model predictions
        grids: List[List[np.ndarray]] = []   # [frame_idx][col_idx] -> (n_grid, n_grid)
        # Per-preset normalization stats used to normalize inputs to the model.
        # Prefer the exact stats stored in run.norm_stats (recorded at training
        # time) so the intensity evaluator uses the identical coordinate system
        # the model was trained in. Fall back to stats recomputed from raw
        # sequences when norm_stats is absent (old checkpoints).
        preset_norm: dict = {}
        for preset in loaded_presets:
            run = best.get((preset, ds_id))
            ns = getattr(run, "norm_stats", None) or {}
            if ns.get("time_std"):
                preset_norm[preset] = (
                    ns["time_mean"], ns["time_std"],
                    np.asarray(ns["loc_mean"]), np.asarray(ns["loc_std"]),
                )
            else:
                # Legacy fallback: infer from effective_config protocol
                data_cfg = run.effective_config.get("data", {}) if run else {}
                use_minmax = (
                    data_cfg.get("protocol") == "paper_autostpp_sthp"
                    and not data_cfg.get("normalize", True)
                )
                if use_minmax:
                    preset_norm[preset] = (t_min, t_range, loc_min, loc_range)
                else:
                    preset_norm[preset] = (time_mean, time_std, loc_mean, loc_std)

        for ti in t_snaps:
            frame = [_kde_lam(seq_t, seq_s, float(ti), xx, yy)]  # col 0: data KDE
            for preset in loaded_presets:
                tm, ts, lm, ls = preset_norm[preset]
                try:
                    lam = _model_lam(
                        models[preset], seq_t, seq_s, float(ti), xx, yy,
                        loc_mean=lm, loc_std=ls,
                        time_mean=tm, time_std=ts,
                        device=device,
                    )
                except Exception as exc:
                    print(f"  [WARN] {preset} @ t={ti:.2f}: {exc}")
                    lam = np.zeros_like(xx)
                frame.append(lam)
            grids.append(frame)

        # Per-column zmax: each panel is scaled to its own peak so that models
        # trained with different normalisations (MinMax vs z-score) all show
        # their spatial structure clearly. The y-axis label carries the peak
        # value so absolute comparisons can still be made.
        col_zmaxes = [
            max(1e-8, max(float(grids[f][c].max()) for f in range(len(grids))))
            for c in range(n_col)
        ]

        # ---- Static PNG (4 time snapshots) --------------------------------
        if fmt == "png":
            n_row = 1
            fig = plt.figure(figsize=(4 * n_col, 4 * n_row + 0.6))
            gs = fig.add_gridspec(n_row, n_col, top=0.90, bottom=0.06,
                                  left=0.04, right=0.995, wspace=0.1, hspace=0.1)
            mid = len(grids) // 2
            for c_idx, name in enumerate(display_names):
                zm = col_zmaxes[c_idx]
                ax = fig.add_subplot(gs[0, c_idx], projection="3d")
                ax.plot_surface(xx, yy, grids[mid][c_idx],
                                cmap=cm.viridis, vmin=0, vmax=zm,
                                linewidth=0, antialiased=False)
                ax.set_zlim(0, zm * 1.05)
                ax.set_title(f"{name}\n(peak {zm:.3g})", fontsize=9)
                ax.tick_params(labelsize=6)
            fig.suptitle(f"Intensity snapshot — {ds_id}", fontsize=10, y=0.97)
            out_path = out_dir / f"intensity_{ds_id}.png"
            fig.savefig(out_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            produced.append(out_path)
            print(f"  Saved PNG -> {out_path}")
            continue

        # ---- Animated GIF / MP4 -------------------------------------------
        fig, axes = plt.subplots(
            1, n_col,
            figsize=(4.5 * n_col, 4.5),
            subplot_kw={"projection": "3d"},
        )
        if n_col == 1:
            axes = [axes]
        fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.04, wspace=0.05)
        title_obj = fig.suptitle("", fontsize=10)

        # Initialise surfaces with frame 0
        surfs = []
        for col, (ax, name) in enumerate(zip(axes, display_names)):
            zm = col_zmaxes[col]
            surf = ax.plot_surface(xx, yy, grids[0][col],
                                   cmap=cm.viridis, vmin=0, vmax=zm,
                                   linewidth=0, antialiased=False)
            ax.set_zlim(0, zm * 1.05)
            ax.set_title(f"{name}\n(peak {zm:.3g})", fontsize=9, pad=2)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("x", fontsize=7, labelpad=1)
            ax.set_ylabel("y", fontsize=7, labelpad=1)
            surfs.append(surf)

        def _update(frame_idx: int):
            for col, ax in enumerate(axes):
                zm = col_zmaxes[col]
                for coll in list(ax.collections):
                    coll.remove()
                ax.plot_surface(xx, yy, grids[frame_idx][col],
                                cmap=cm.viridis, vmin=0, vmax=zm,
                                linewidth=0, antialiased=False)
            title_obj.set_text(
                f"lambda*(t, s | H)  —  dataset: {ds_id}  |  t = {t_snaps[frame_idx]:.2f}"
            )
            return []

        ani = animation.FuncAnimation(
            fig, _update, frames=len(t_snaps), interval=1000 // fps, blit=False
        )

        if fmt == "mp4":
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=1200)
                out_path = out_dir / f"intensity_{ds_id}.mp4"
                ani.save(str(out_path), writer=writer)
                produced.append(out_path)
                print(f"  Saved MP4 -> {out_path}")
            except Exception as exc:
                print(f"  [WARN] MP4 save failed ({exc}); falling back to GIF.")
                fmt = "gif"

        if fmt == "gif":
            # PillowWriter + Axes3D is broken in many matplotlib builds — capture frames manually.
            from io import BytesIO as _BytesIO
            try:
                from PIL import Image as _PIL
            except ImportError:
                print("  [WARN] Pillow not installed — cannot save GIF.")
                plt.close(fig)
                continue
            gif_frames = []
            for fi in range(len(t_snaps)):
                _update(fi)
                buf = _BytesIO()
                fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
                buf.seek(0)
                gif_frames.append(_PIL.open(buf).copy())
            out_path = out_dir / f"intensity_{ds_id}.gif"
            if gif_frames:
                gif_frames[0].save(
                    str(out_path), save_all=True, append_images=gif_frames[1:],
                    duration=1000 // fps, loop=0,
                )
                produced.append(out_path)
                print(f"  Saved GIF -> {out_path}")

        plt.close(fig)

    return produced


__all__ = ["plot_bench_intensities"]
