"""Intensity evaluation helpers.

Two layers live here:

1. Generic helpers
   - ``eval_intensity``: evaluate pointwise intensity at one query time over a
     spatial grid from a provided history.
   - ``calc_lamb``: build a full intensity cube from explicit x/y/t ranges.

2. Notebook-faithful helpers for DeepSTPP / AutoSTPP
   - ``calc_lamb_sequence``: mimic the upstream Visualizer-style ``calc_lamb``
     semantics on one full sequence.
   - ``calc_lamb_from_runner``: thin wrapper that pulls normalization stats and
     paper-space bounds from a loaded ``STPPRunner``.

The notebook-faithful path deliberately preserves the active upstream history
selection convention (including the ``<=`` prefix rule) while avoiding upstream
notebook-state coupling and brittle DataLoader dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from unified_stpp.evaluation.context import GenerativeRollouts, GroundTruth, IntensityGrid

if TYPE_CHECKING:
    from unified_stpp.runner.runner import STPPRunner


@dataclass
class IntensityCubeResult:
    """Notebook-style intensity cube plus the chosen sequence history."""

    lambs: np.ndarray
    x_range: np.ndarray
    y_range: np.ndarray
    t_range: np.ndarray
    history_locs: np.ndarray
    history_times: np.ndarray


def _resolve_device(model, device):
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _as_float_array(value, *, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"Expected array with ndim={ndim}, got shape {arr.shape}.")
    return arr


def _intensity_scale_factor(surface_type: str, *, t_scale: float, s_scale: np.ndarray) -> float:
    s_prod = float(np.prod(np.maximum(np.asarray(s_scale, dtype=np.float32), 1e-8)))
    t_scale = float(max(float(t_scale), 1e-8))
    if surface_type == "intensity":
        return s_prod * t_scale
    if surface_type == "density":
        return s_prod
    return 1.0


def _paper_stats_from_model(model) -> tuple[np.ndarray, np.ndarray, float] | None:
    state_model = getattr(model, "state_model", None)
    loc_min = getattr(state_model, "paper_loc_min", None)
    loc_range = getattr(state_model, "paper_loc_range", None)
    dt_range = getattr(state_model, "paper_dt_range", None)
    if loc_min is None or loc_range is None or dt_range is None:
        return None
    loc_min_np = np.asarray(loc_min.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    loc_range_np = np.asarray(loc_range.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    dt_range_f = float(np.asarray(dt_range.detach().cpu().numpy(), dtype=np.float32).reshape(-1)[0])
    return loc_min_np, loc_range_np, dt_range_f


def paper_output_scale_factor(model) -> float | None:
    """Return the paper-space output scaling used by notebook-faithful intensity plots.

    DeepSTPP / AutoSTPP internally parameterize intensity in their paper-space
    coordinates. Upstream-style visualization divides the queried values by the
    paper location/time scaling factor before treating them as comparable
    original-space intensities.
    """
    paper_stats = _paper_stats_from_model(model)
    if paper_stats is None:
        return None
    _loc_min, loc_range, dt_range = paper_stats
    return float(np.prod(np.maximum(loc_range, 1e-8)) * max(dt_range, 1e-8))


def _infer_notebook_lookback(model) -> int:
    event_model = getattr(model, "event_model", None)
    if hasattr(event_model, "seq_len"):
        return int(event_model.seq_len)
    if hasattr(event_model, "lookback"):
        return int(event_model.lookback)
    state_model = getattr(model, "state_model", None)
    if hasattr(state_model, "lookback"):
        return int(state_model.lookback)
    raise ValueError(
        "Could not infer notebook lookback from model. "
        "Expected event_model.seq_len or event_model.lookback."
    )


def _build_linspace_via_upstream_arange(lo: float, hi: float, n_step: int) -> np.ndarray:
    if n_step < 2:
        raise ValueError(f"n_step must be >= 2, got {n_step}.")
    step = (float(hi) - float(lo)) / float(n_step - 1)
    return np.arange(float(lo), float(hi) + 1e-5, step, dtype=np.float32)


def eval_intensity(
    *,
    model,
    t_query: float,
    s_grid: np.ndarray,
    history_times: np.ndarray,
    history_locs: np.ndarray,
    t_bias: float,
    t_scale: float,
    s_bias: np.ndarray,
    s_scale: np.ndarray,
    device=None,
    correct_for_normalization: bool = False,
    query_chunk_size: int | None = None,
) -> np.ndarray:
    """Evaluate intensity at one query time over a spatial grid.

    ``t_bias/t_scale/s_bias/s_scale`` define the caller-side normalization
    transform:
      ``x_norm = (x_orig - bias) / scale``
    """
    dev = _resolve_device(model, device)
    model = model.to(dev)
    model.eval()

    history_times = _as_float_array(history_times, ndim=1)
    history_locs = _as_float_array(history_locs)
    s_grid = _as_float_array(s_grid)
    if history_locs.ndim != 2 or s_grid.ndim != 2:
        raise ValueError("history_locs and s_grid must both have shape (N, d).")
    if history_locs.shape[-1] != s_grid.shape[-1]:
        raise ValueError(
            f"Spatial dim mismatch: history_locs dim={history_locs.shape[-1]} "
            f"vs s_grid dim={s_grid.shape[-1]}."
        )

    t_scale_safe = float(max(float(t_scale), 1e-8))
    s_scale_arr = np.maximum(np.asarray(s_scale, dtype=np.float32).reshape(-1), 1e-8)
    s_bias_arr = np.asarray(s_bias, dtype=np.float32).reshape(-1)
    if s_bias_arr.shape != s_scale_arr.shape:
        raise ValueError(f"s_bias shape {s_bias_arr.shape} != s_scale shape {s_scale_arr.shape}.")

    hist_t_norm = (history_times - float(t_bias)) / t_scale_safe
    hist_s_norm = (history_locs - s_bias_arr) / s_scale_arr
    query_t_norm = np.full((s_grid.shape[0],), (float(t_query) - float(t_bias)) / t_scale_safe, dtype=np.float32)
    query_s_norm = (s_grid - s_bias_arr) / s_scale_arr

    history_times_t = torch.as_tensor(hist_t_norm, dtype=torch.float32, device=dev).unsqueeze(0)
    history_locs_t = torch.as_tensor(hist_s_norm, dtype=torch.float32, device=dev).unsqueeze(0)
    history_lengths_t = torch.tensor([history_times.shape[0]], dtype=torch.long, device=dev)
    query_times_t = torch.as_tensor(query_t_norm, dtype=torch.float32, device=dev)
    query_locs_t = torch.as_tensor(query_s_norm, dtype=torch.float32, device=dev)

    with torch.no_grad():
        state_ctx = model.state_model.encode_history(
            times=history_times_t,
            locations=history_locs_t,
            lengths=history_lengths_t,
        )
        chunk_size = 0 if query_chunk_size is None else int(query_chunk_size)
        if chunk_size > 0 and query_locs_t.shape[0] > chunk_size:
            chunks = []
            for start in range(0, int(query_locs_t.shape[0]), chunk_size):
                stop = min(start + chunk_size, int(query_locs_t.shape[0]))
                chunks.append(
                    model.event_model.intensity(
                        state=state_ctx,
                        query_times=query_times_t[start:stop],
                        query_locations=query_locs_t[start:stop],
                        device=dev,
                    )
                )
            values = torch.cat(chunks, dim=0)
        else:
            values = model.event_model.intensity(
                state=state_ctx,
                query_times=query_times_t,
                query_locations=query_locs_t,
                device=dev,
            )

    out = values.detach().cpu().numpy().astype(np.float32).reshape(-1)
    if correct_for_normalization:
        factor = _intensity_scale_factor(
            model.event_model.surface_query_type,
            t_scale=t_scale_safe,
            s_scale=s_scale_arr,
        )
        out = out / float(max(factor, 1e-8))
    return out


def calc_lamb(
    *,
    model,
    history_times: np.ndarray,
    history_locs: np.ndarray,
    t_bias: float,
    t_scale: float,
    s_bias: np.ndarray,
    s_scale: np.ndarray,
    x_range: np.ndarray,
    y_range: np.ndarray,
    t_range: np.ndarray,
    device=None,
    correct_for_normalization: bool = False,
) -> np.ndarray:
    """Evaluate a full intensity cube from explicit x/y/t ranges."""
    x_range = _as_float_array(x_range, ndim=1)
    y_range = _as_float_array(y_range, ndim=1)
    t_range = _as_float_array(t_range, ndim=1)
    xx, yy = np.meshgrid(x_range, y_range, indexing="ij")
    s_grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

    frames: list[np.ndarray] = []
    for t_query in t_range:
        vals = eval_intensity(
            model=model,
            t_query=float(t_query),
            s_grid=s_grid,
            history_times=history_times,
            history_locs=history_locs,
            t_bias=t_bias,
            t_scale=t_scale,
            s_bias=s_bias,
            s_scale=s_scale,
            device=device,
            correct_for_normalization=correct_for_normalization,
        )
        frames.append(vals.reshape(len(x_range), len(y_range)))
    return np.stack(frames, axis=0).astype(np.float32)


def calc_lamb_sequence(
    *,
    model,
    sequence_times: np.ndarray,
    sequence_locs: np.ndarray,
    t_bias: float,
    t_scale: float,
    s_bias: np.ndarray,
    s_scale: np.ndarray,
    lookback: int,
    x_nstep: int = 101,
    y_nstep: int = 101,
    t_nstep: int = 201,
    round_time: bool = True,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    trunc: bool = False,
    max_history: int = 20,
    device=None,
    output_scale_factor: float = 1.0,
) -> IntensityCubeResult:
    """Notebook-faithful intensity cube for one chosen sequence."""
    times = _as_float_array(sequence_times, ndim=1)
    locs = _as_float_array(sequence_locs)
    if locs.ndim != 2 or locs.shape[-1] < 2:
        raise ValueError(f"Expected sequence_locs with shape (N, 2), got {locs.shape}.")
    if times.shape[0] != locs.shape[0]:
        raise ValueError(
            f"sequence_times length {times.shape[0]} != sequence_locs length {locs.shape[0]}."
        )
    if times.shape[0] <= int(lookback):
        raise ValueError(
            f"Sequence length {times.shape[0]} must exceed lookback {lookback}."
        )

    if xmin is None or xmax is None or ymin is None or ymax is None:
        paper_stats = _paper_stats_from_model(model)
        if paper_stats is None:
            raise ValueError(
                "Notebook-faithful calc_lamb requires paper_loc_min/range on the model "
                "when explicit bounds are not provided."
            )
        paper_loc_min, paper_loc_range, _paper_dt_range = paper_stats
        xmin = float(paper_loc_min[0] if xmin is None else xmin)
        xmax = float(paper_loc_min[0] + paper_loc_range[0] if xmax is None else xmax)
        ymin = float(paper_loc_min[1] if ymin is None else ymin)
        ymax = float(paper_loc_min[1] + paper_loc_range[1] if ymax is None else ymax)
    else:
        xmin = float(xmin)
        xmax = float(xmax)
        ymin = float(ymin)
        ymax = float(ymax)

    x_range = _build_linspace_via_upstream_arange(xmin, xmax, x_nstep)
    y_range = _build_linspace_via_upstream_arange(ymin, ymax, y_nstep)

    t_start = float(times[int(lookback)])
    t_end = float(times[-1])
    if t_nstep < 2:
        raise ValueError(f"t_nstep must be >= 2, got {t_nstep}.")
    t_step = (t_end - t_start) / float(t_nstep - 1)
    if round_time:
        t_range = np.arange(round(t_start), round(t_end) + 1e-5, t_step, dtype=np.float32)
    else:
        t_range = np.arange(t_start, t_end + 1e-5, t_step, dtype=np.float32)

    xx, yy = np.meshgrid(x_range, y_range, indexing="ij")
    s_grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

    frames: list[np.ndarray] = []
    max_history = int(max_history)
    for t_query in t_range:
        # Upstream notebook semantics: use <= on all but the final event.
        i = int(np.sum(times[:-1] <= float(t_query)) - 1)
        i = max(i, 0)
        hist_t = times[: i + 1]
        hist_s = locs[: i + 1]
        if trunc and max_history > 0 and hist_t.shape[0] > max_history:
            hist_t = hist_t[-max_history:]
            hist_s = hist_s[-max_history:]

        vals = eval_intensity(
            model=model,
            t_query=float(t_query),
            s_grid=s_grid,
            history_times=hist_t,
            history_locs=hist_s,
            t_bias=t_bias,
            t_scale=t_scale,
            s_bias=s_bias,
            s_scale=s_scale,
            device=device,
            correct_for_normalization=False,
        )
        vals = vals / float(max(output_scale_factor, 1e-8))
        frames.append(vals.reshape(len(x_range), len(y_range)))

    return IntensityCubeResult(
        lambs=np.stack(frames, axis=0).astype(np.float32),
        x_range=x_range.astype(np.float32),
        y_range=y_range.astype(np.float32),
        t_range=t_range.astype(np.float32),
        history_locs=locs.astype(np.float32),
        history_times=times.astype(np.float32),
    )


def calc_lamb_from_runner(
    *,
    runner: "STPPRunner",
    sequences: list[dict],
    seq_idx: int = 2,
    split: str = "test",
    x_nstep: int = 101,
    y_nstep: int = 101,
    t_nstep: int = 201,
    round_time: bool = True,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    trunc: Optional[bool] = None,
    max_history: Optional[int] = None,
    device=None,
) -> IntensityCubeResult:
    """Notebook-faithful ``calc_lamb`` wrapper for a loaded runner."""
    preset = runner.config.model.preset
    if preset not in {"deep_stpp", "auto_stpp"}:
        raise ValueError(
            "Notebook-faithful calc_lamb currently supports only "
            f"'deep_stpp' and 'auto_stpp', got {preset!r}."
        )
    if seq_idx < 0 or seq_idx >= len(sequences):
        raise IndexError(
            f"seq_idx={seq_idx} out of range for split {split!r} with {len(sequences)} sequences."
        )

    model = runner.model
    lookback = _infer_notebook_lookback(model)
    if trunc is None:
        trunc = bool(getattr(model.event_model, "trunc", False))
    if max_history is None:
        max_history = int(getattr(model.event_model, "max_history", lookback))

    seq = sequences[seq_idx]
    stats = runner.norm_stats
    s_bias = np.asarray(stats["loc_mean"], dtype=np.float32)
    s_scale = np.asarray(stats["loc_std"], dtype=np.float32)
    t_bias = float(stats["time_mean"])
    t_scale = float(stats["time_std"])

    paper_stats = _paper_stats_from_model(model)
    if paper_stats is None:
        raise ValueError(
            f"Preset {preset!r} does not expose paper-space stats required for "
            "notebook-faithful intensity plotting."
        )
    output_scale_factor = paper_output_scale_factor(model)
    if output_scale_factor is None:
        raise ValueError(
            f"Preset {preset!r} does not expose paper-space stats required for "
            "notebook-faithful intensity plotting."
        )

    return calc_lamb_sequence(
        model=model,
        sequence_times=np.asarray(seq["times"], dtype=np.float32),
        sequence_locs=np.asarray(seq["locations"], dtype=np.float32),
        t_bias=t_bias,
        t_scale=t_scale,
        s_bias=s_bias,
        s_scale=s_scale,
        lookback=lookback,
        x_nstep=x_nstep,
        y_nstep=y_nstep,
        t_nstep=t_nstep,
        round_time=round_time,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        trunc=bool(trunc),
        max_history=int(max_history),
        device=device,
        output_scale_factor=float(output_scale_factor),
    )


_DEFAULT_GRID_SPEC: dict[str, Any] = {
    "x_range": [0.0, 1.0],
    "y_range": [0.0, 1.0],
    "x_resolution": 50,
    "y_resolution": 50,
    "t_resolution": 100,
}


def compute_intensity_grid(
    runner: "STPPRunner",
    test_seqs: list[dict[str, np.ndarray]],
    *,
    grid_spec: dict[str, Any],
    ground_truth: GroundTruth | None,
    generative_rollouts: GenerativeRollouts | None,
    device: torch.device,
) -> IntensityGrid:
    """Compute a spatiotemporal intensity surface on the shared grid."""
    spec = dict(_DEFAULT_GRID_SPEC)
    spec.update(grid_spec)

    caps = runner.model.event_model.capabilities
    x_range = spec.get("x_range", [0.0, 1.0])
    y_range = spec.get("y_range", [0.0, 1.0])
    xs = np.linspace(
        float(x_range[0]),
        float(x_range[1]),
        int(spec["x_resolution"]),
        dtype=np.float32,
    )
    ys = np.linspace(
        float(y_range[0]),
        float(y_range[1]),
        int(spec["y_resolution"]),
        dtype=np.float32,
    )

    if test_seqs:
        t0 = float(test_seqs[0]["times"][0])
        t1 = float(test_seqs[0]["times"][-1])
    else:
        t0, t1 = 0.0, 1.0
    ts = np.linspace(t0, t1, int(spec["t_resolution"]), dtype=np.float32)

    if caps.has_intensity:
        lambda_hat = _direct_intensity_grid(runner, test_seqs, xs, ys, ts, device)
        method = "direct"
    elif generative_rollouts is not None:
        lambda_hat = _kde_intensity_grid(generative_rollouts, xs, ys, ts)
        method = "kde"
    else:
        lambda_hat = np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)
        method = "unavailable"

    lambda_true = None
    if ground_truth is not None and ground_truth.intensity_grid is not None:
        lambda_true = np.asarray(ground_truth.intensity_grid, dtype=np.float32)

    return IntensityGrid(
        lambda_hat=lambda_hat,
        lambda_true=lambda_true,
        xs=xs,
        ys=ys,
        ts=ts,
        method=method,
    )


def _direct_intensity_grid(
    runner: "STPPRunner",
    test_seqs: list[dict[str, np.ndarray]],
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    device: torch.device,
    chunk_size: int = 512,
) -> np.ndarray:
    """Evaluate intensity on the grid using the first test sequence as conditioning."""
    if not test_seqs:
        return np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)

    seq = test_seqs[0]
    norm_stats = runner.norm_stats
    normalize = bool(norm_stats.get("normalize", False))
    t_mean = float(norm_stats.get("time_mean", 0.0)) if normalize else 0.0
    t_std = max(float(norm_stats.get("time_std", 1.0)), 1e-8) if normalize else 1.0
    loc_mean = (
        np.asarray(norm_stats.get("loc_mean", [0.0, 0.0]), dtype=np.float32)
        if normalize
        else np.zeros(2, dtype=np.float32)
    )
    loc_std = (
        np.maximum(
            np.asarray(norm_stats.get("loc_std", [1.0, 1.0]), dtype=np.float32),
            1e-8,
        )
        if normalize
        else np.ones(2, dtype=np.float32)
    )

    result = np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    s_grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    runner.model.eval()
    with torch.no_grad():
        for ti, t_query in enumerate(ts):
            times = np.asarray(seq["times"], dtype=np.float32)
            hist_mask = times <= float(t_query)
            if not hist_mask.any():
                continue
            locs_full = np.asarray(seq["locations"], dtype=np.float32)
            history = {
                "times": times[hist_mask].copy(),
                "locations": locs_full[hist_mask].copy(),
            }
            intensity_vals = eval_intensity(
                model=runner.model,
                t_query=float(t_query),
                s_grid=s_grid,
                history_times=history["times"],
                history_locs=history["locations"],
                t_bias=t_mean,
                t_scale=t_std,
                s_bias=loc_mean,
                s_scale=loc_std,
                device=device,
                correct_for_normalization=normalize,
                query_chunk_size=int(chunk_size),
            )
            result[ti] = intensity_vals.reshape(len(xs), len(ys))

    return result


def _kde_intensity_grid(
    rollouts: GenerativeRollouts,
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:
    """Estimate intensity on the grid from generative rollout events via 3D KDE."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)

    points: list[np.ndarray] = []
    for seq_rollouts_t, seq_rollouts_s in zip(rollouts.rollout_times, rollouts.rollout_locs):
        for rollout_t, rollout_s in zip(seq_rollouts_t, seq_rollouts_s):
            if rollout_t.size > 0 and rollout_s.shape[0] > 0:
                points.append(np.column_stack([rollout_t, rollout_s[:, 0], rollout_s[:, 1]]))

    if not points:
        return np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)

    all_pts = np.concatenate(points, axis=0).astype(np.float64)
    n_pts = all_pts.shape[0]

    d = 3
    scott_bw = n_pts ** (-1.0 / (d + 4))
    t_range = float(ts[-1] - ts[0]) if len(ts) > 1 else 1.0
    x_range = float(xs[-1] - xs[0]) if len(xs) > 1 else 1.0
    y_range = float(ys[-1] - ys[0]) if len(ys) > 1 else 1.0
    bw = max(scott_bw, 0.02)

    pts_norm = all_pts.copy()
    pts_norm[:, 0] = (all_pts[:, 0] - float(ts[0])) / max(t_range, 1e-8)
    pts_norm[:, 1] = (all_pts[:, 1] - float(xs[0])) / max(x_range, 1e-8)
    pts_norm[:, 2] = (all_pts[:, 2] - float(ys[0])) / max(y_range, 1e-8)

    try:
        kde = gaussian_kde(pts_norm.T, bw_method=bw)
    except Exception:
        return np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)

    tt, xx, yy = np.meshgrid(
        (ts - float(ts[0])) / max(t_range, 1e-8),
        (xs - float(xs[0])) / max(x_range, 1e-8),
        (ys - float(ys[0])) / max(y_range, 1e-8),
        indexing="ij",
    )
    grid_pts = np.stack([tt.ravel(), xx.ravel(), yy.ravel()], axis=0)
    density = kde(grid_pts).reshape(len(ts), len(xs), len(ys)).astype(np.float32)

    total_rollouts = sum(len(r) for r in rollouts.rollout_times)
    scale = float(n_pts) / max(float(total_rollouts), 1.0)
    return density * scale
