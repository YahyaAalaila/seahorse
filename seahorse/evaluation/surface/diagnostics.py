"""Secondary exact and factorized surface diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from seahorse.evaluation.intensity import calc_lamb_from_runner
from seahorse.evaluation.runtime import (
    HistoryQuery,
    RunTarget,
    require_history_path,
    resolve_device,
)
from seahorse.runner.runner import STPPRunner
from seahorse.utils import load_jsonl


@dataclass(frozen=True)
class SurfaceDiagnosticSpec:
    profile: Literal["history_frame", "future_exact"] = "history_frame"
    x_nstep: int = 81
    y_nstep: int = 81
    t_nstep: int = 41
    future_horizon: float | None = None
    frame_index: int = -1
    round_time: bool = True
    trunc: bool | None = None
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    spatial_chunk_size: int | None = None
    device: str = "auto"


@dataclass
class SurfaceDiagnosticResult:
    run_dir: Path
    history_path: Path
    split: str
    seq_idx: int
    history_length: int
    preset: str
    profile: str
    device: str
    x_grid: np.ndarray
    y_grid: np.ndarray
    t_grid: np.ndarray
    history_times: np.ndarray
    history_locs: np.ndarray
    primary_cube: np.ndarray
    primary_value_name: str
    primary_value_label: str
    notes: list[str] = field(default_factory=list)
    provisional: bool = False
    extra_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class SurfaceDiagnosticEvaluator:
    """Evaluate one saved run under one exact/factorized surface profile."""

    def evaluate(
        self,
        run_target: RunTarget | str | Path,
        history_query: HistoryQuery,
        spec: SurfaceDiagnosticSpec,
    ) -> SurfaceDiagnosticResult:
        run_dir = Path(run_target.run if isinstance(run_target, RunTarget) else run_target).resolve()
        history_path = require_history_path(history_query.history_path)
        runner = STPPRunner.load(run_dir)
        preset = runner.config.model.preset
        prefer_cpu = spec.profile == "future_exact" and preset in {
            "njsde",
            "neural_jumpcnf",
            "neural_attncnf",
        }
        device = resolve_device(spec.device, prefer_cpu_for_neural_exact=prefer_cpu)
        runner.model.to(device)
        runner.model.eval()
        seq = self._load_surface_sequence(
            history_path,
            history_query.seq_idx,
            history_query.history_length,
        )

        if spec.profile == "history_frame":
            if preset not in {"auto_stpp", "deep_stpp"}:
                raise SystemExit(
                    "surface --profile history_frame currently supports only auto_stpp and deep_stpp."
                )
            payload = evaluate_notebook_profile(
                runner=runner,
                seq=seq,
                split=history_query.split,
                x_nstep=spec.x_nstep,
                y_nstep=spec.y_nstep,
                t_nstep=spec.t_nstep,
                round_time=bool(spec.round_time),
                trunc=spec.trunc,
                xmin=spec.xmin,
                xmax=spec.xmax,
                ymin=spec.ymin,
                ymax=spec.ymax,
                device=device,
            )
        else:
            if preset not in {"njsde", "neural_jumpcnf", "neural_attncnf"}:
                raise SystemExit(
                    "surface --profile future_exact currently supports only neural exact families."
                )
            payload = evaluate_neural_future_exact(
                runner=runner,
                seq=seq,
                split=history_query.split,
                x_nstep=spec.x_nstep,
                y_nstep=spec.y_nstep,
                t_nstep=spec.t_nstep,
                future_horizon=spec.future_horizon,
                spatial_chunk_size=spec.spatial_chunk_size,
                xmin=spec.xmin,
                xmax=spec.xmax,
                ymin=spec.ymin,
                ymax=spec.ymax,
                device=device,
            )

        extra_arrays: dict[str, np.ndarray] = {}
        if "lambda_t" in payload:
            extra_arrays["lambda_t"] = np.asarray(payload["lambda_t"], dtype=np.float32)
        if "spatial_density" in payload:
            extra_arrays["spatial_density"] = np.asarray(payload["spatial_density"], dtype=np.float32)

        extra_metadata = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "profile",
                "split",
                "preset",
                "device",
                "history_times",
                "history_locs",
                "t_grid",
                "x_grid",
                "y_grid",
                "primary_cube",
                "primary_value_name",
                "primary_value_label",
                "notes",
                "lambda_t",
                "spatial_density",
                "provisional",
            }
        }
        extra_metadata.setdefault("frame_index", int(spec.frame_index))
        return SurfaceDiagnosticResult(
            run_dir=run_dir,
            history_path=history_path,
            split=str(history_query.split),
            seq_idx=int(history_query.seq_idx),
            history_length=int(history_query.history_length),
            preset=str(payload["preset"]),
            profile=str(payload["profile"]),
            device=str(payload["device"]),
            x_grid=np.asarray(payload["x_grid"], dtype=np.float32),
            y_grid=np.asarray(payload["y_grid"], dtype=np.float32),
            t_grid=np.asarray(payload["t_grid"], dtype=np.float32),
            history_times=np.asarray(payload["history_times"], dtype=np.float32),
            history_locs=np.asarray(payload["history_locs"], dtype=np.float32),
            primary_cube=np.asarray(payload["primary_cube"], dtype=np.float32),
            primary_value_name=str(payload["primary_value_name"]),
            primary_value_label=str(payload["primary_value_label"]),
            notes=list(payload.get("notes", [])),
            provisional=bool(payload.get("provisional", False)),
            extra_arrays=extra_arrays,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def _load_surface_sequence(path: Path, seq_idx: int, history_length: int) -> dict[str, Any]:
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


DEFAULT_X_NSTEP = 81
DEFAULT_Y_NSTEP = 81
DEFAULT_T_NSTEP = 41


def resolve_neural_exact_profile(
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
    if preset == "njsde":
        if profile["spatial_chunk_size"] is None:
            profile["spatial_chunk_size"] = 4096
        profile["warnings"].append(
            "NJSDE exact-family surface support is intended for diagnostics."
        )
        return profile
    if preset == "neural_jumpcnf":
        if defaults_unchanged:
            profile.update({"x_nstep": 49, "y_nstep": 49, "t_nstep": 21, "auto_coarsened_grid": True})
        if profile["spatial_chunk_size"] is None:
            profile["spatial_chunk_size"] = 1024
        profile["warnings"].append(
            "Neural JumpCNF exact-family surface support uses a coarsened diagnostic grid."
        )
        return profile
    if preset == "neural_attncnf":
        if defaults_unchanged:
            profile.update({"x_nstep": 33, "y_nstep": 33, "t_nstep": 11, "auto_coarsened_grid": True})
        if profile["spatial_chunk_size"] is None:
            profile["spatial_chunk_size"] = 512
        profile["warnings"].append(
            "Neural AttnCNF exact-family surface support uses a coarsened diagnostic grid."
        )
        return profile
    raise ValueError(f"Unsupported neural exact profile for preset {preset!r}.")


def build_future_query_grid(*, last_t: float, horizon: float, n_steps: int) -> np.ndarray:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    t_end = float(last_t) + float(horizon)
    return np.linspace(float(last_t), t_end, int(n_steps) + 1, dtype=np.float32)[1:]


def history_overlay_z_level(frame_values: np.ndarray) -> float:
    values = np.asarray(frame_values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        return vmax
    return vmin + 0.88 * (vmax - vmin)


def representative_indices(n_frames: int) -> list[tuple[str, int]]:
    labels = [("start", 0), ("mid", n_frames // 2), ("end", n_frames - 1)]
    seen: set[int] = set()
    out: list[tuple[str, int]] = []
    for label, idx in labels:
        if idx not in seen:
            out.append((label, idx))
            seen.add(idx)
    return out


def normalize_history(
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


def infer_future_horizon(
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


def resolve_surface_bounds(
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
        return (float(lo) if lo is not None else v_lo - pad, float(hi) if hi is not None else v_hi + pad)

    x_lo, x_hi = _bounds(x_vals, lo=xmin, hi=xmax, pad_scale=0.5 * float(loc_std[0]))
    y_lo, y_hi = _bounds(y_vals, lo=ymin, hi=ymax, pad_scale=0.5 * float(loc_std[1]))
    return x_lo, x_hi, y_lo, y_hi


def evaluate_neural_future_exact(
    *,
    runner,
    seq: dict[str, Any],
    split: str,
    x_nstep: int,
    y_nstep: int,
    t_nstep: int,
    future_horizon: float | None,
    spatial_chunk_size: int | None,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
    device: torch.device,
) -> dict[str, Any]:
    model = runner.model.to(device)
    model.eval()
    preset = runner.config.model.preset
    profile = resolve_neural_exact_profile(
        preset=preset,
        x_nstep=x_nstep,
        y_nstep=y_nstep,
        t_nstep=t_nstep,
        spatial_chunk_size=spatial_chunk_size,
    )
    history_times = np.asarray(seq["times"], dtype=np.float32)
    history_locs = np.asarray(seq["locations"], dtype=np.float32)
    full_times = np.asarray(seq["full_times"], dtype=np.float32)
    full_locs = np.asarray(seq["full_locations"], dtype=np.float32)
    times_norm, locs_norm = normalize_history(
        history_times=history_times,
        history_locs=history_locs,
        norm_stats=runner.norm_stats,
    )
    lengths = torch.tensor([len(history_times)], dtype=torch.long, device=device)
    times_t = torch.as_tensor(times_norm, dtype=torch.float32, device=device).unsqueeze(0)
    locs_t = torch.as_tensor(locs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    state_ctx = model.state_model.encode_history(times=times_t, locations=locs_t, lengths=lengths)
    horizon = infer_future_horizon(
        history_times=history_times,
        full_times=full_times,
        future_horizon_arg=future_horizon,
    )
    t_grid = build_future_query_grid(last_t=float(history_times[-1]), horizon=float(horizon), n_steps=int(profile["t_nstep"]))
    x_lo, x_hi, y_lo, y_hi = resolve_surface_bounds(
        full_locs=full_locs,
        norm_stats=runner.norm_stats,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
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
    for i, t_query_raw in enumerate(t_grid):
        t_query_norm = (float(t_query_raw) - time_mean) / time_std
        terms = model.event_model.fixed_time_query_terms(state=state_ctx, query_time=t_query_norm, device=device)
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
    return {
        "profile": "future_exact",
        "split": split,
        "preset": preset,
        "device": str(device),
        "history_times": history_times,
        "history_locs": history_locs,
        "t_grid": t_grid,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "lambda_t": lambda_t,
        "spatial_density": spatial_density,
        "primary_cube": joint_intensity,
        "primary_value_name": "joint_intensity",
        "primary_value_label": "joint intensity",
        "future_horizon": float(horizon),
        "spatial_chunk_size": int(chunk_size),
        "auto_coarsened_grid": bool(profile["auto_coarsened_grid"]),
        "notes": list(profile["warnings"]),
        "query_complexity": {
            "spatial_points_per_frame": total_spatial_points,
            "chunk_calls_per_frame": chunk_calls_per_frame,
            "total_chunk_calls": chunk_calls_per_frame * len(t_grid),
        },
        "provisional": False,
    }


def evaluate_notebook_profile(
    *,
    runner,
    seq: dict[str, Any],
    split: str,
    x_nstep: int,
    y_nstep: int,
    t_nstep: int,
    round_time: bool,
    trunc: bool | None,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
    device: torch.device,
) -> dict[str, Any]:
    cube = calc_lamb_from_runner(
        runner=runner,
        sequences=[{"times": seq["times"], "locations": seq["locations"]}],
        seq_idx=0,
        split=split,
        x_nstep=x_nstep,
        y_nstep=y_nstep,
        t_nstep=t_nstep,
        round_time=round_time,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        trunc=trunc,
        device=device,
    )
    return {
        "profile": "history_frame",
        "split": split,
        "preset": runner.config.model.preset,
        "device": str(device),
        "history_times": cube.history_times.astype(np.float32),
        "history_locs": cube.history_locs.astype(np.float32),
        "t_grid": cube.t_range.astype(np.float32),
        "x_grid": cube.x_range.astype(np.float32),
        "y_grid": cube.y_range.astype(np.float32),
        "primary_cube": cube.lambs.astype(np.float32),
        "primary_value_name": "intensity",
        "primary_value_label": "intensity",
        "notes": [],
        "provisional": False,
    }
