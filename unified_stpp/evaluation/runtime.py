"""Shared helpers for post-fit evaluation commands."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from unified_stpp.models.configs import ConfigRegistry
from unified_stpp.runner.results import RunResult
from unified_stpp.runner.runner import STPPRunner
from unified_stpp.utils import load_jsonl


@dataclass(frozen=True)
class FrameWindow:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(float(self.end - self.start), 1e-8)


@dataclass(frozen=True)
class RunTarget:
    run: Path
    label: str | None = None


@dataclass(frozen=True)
class HistoryQuery:
    history_path: Path
    split: Literal["train", "val", "test"] = "test"
    seq_idx: int = 0
    start_event_idx: int = 0
    history_length: int = 0


@dataclass
class LoadedRun:
    label: str
    safe_label: str
    preset: str
    run_dir: Path
    runner: STPPRunner
    preset_status: str
    nll_kind: str
    nll_report_space: str


def resolve_device(spec: str, *, prefer_cpu_for_neural_exact: bool = False) -> torch.device:
    if spec == "auto":
        if prefer_cpu_for_neural_exact:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def sanitize_label(label: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    text = text.strip("._")
    return text or "model"


def parse_bandwidth(value: str | None) -> float | str | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(base_seed: int, *parts: object) -> int:
    payload = json.dumps(
        {"base_seed": int(base_seed), "parts": [str(part) for part in parts]},
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def require_history_path(path: str | Path) -> Path:
    history_path = Path(path).resolve()
    if not history_path.exists():
        raise SystemExit(f"History JSONL not found: {history_path}")
    return history_path


def load_sequence(path: Path, seq_idx: int) -> dict[str, np.ndarray]:
    seqs = load_jsonl(path)
    if not seqs:
        raise SystemExit(f"No sequences found in {path}")
    if seq_idx < 0 or seq_idx >= len(seqs):
        raise SystemExit(f"--seq-idx {seq_idx} out of range for {path} (n={len(seqs)})")

    seq = dict(seqs[seq_idx])
    times = np.asarray(seq["times"], dtype=np.float32)
    locs = np.asarray(seq["locations"], dtype=np.float32)
    if times.ndim != 1:
        raise SystemExit(f"Expected 1-D times, got {times.shape}")
    if locs.ndim != 2 or locs.shape[1] != 2:
        raise SystemExit(f"Expected 2-D locations, got {locs.shape}")
    if times.shape[0] != locs.shape[0]:
        raise SystemExit(
            f"Sequence length mismatch: {times.shape[0]} times vs {locs.shape[0]} locations"
        )
    if times.shape[0] == 0:
        raise SystemExit("Selected sequence is empty.")
    out = {"times": times, "locations": locs}
    if "marks" in seq and seq["marks"] is not None:
        out["marks"] = np.asarray(seq["marks"], dtype=np.int64)
    return out


def copy_history(history: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {
        "times": np.asarray(history["times"], dtype=np.float32).copy(),
        "locations": np.asarray(history["locations"], dtype=np.float32).copy(),
    }
    if "marks" in history and history["marks"] is not None:
        out["marks"] = np.asarray(history["marks"], dtype=np.int64).copy()
    return out


def cap_history(history: dict[str, np.ndarray], history_length: int) -> dict[str, np.ndarray]:
    out = copy_history(history)
    if history_length > 0 and out["times"].shape[0] > history_length:
        out["times"] = out["times"][-history_length:]
        out["locations"] = out["locations"][-history_length:]
        if "marks" in out and out["marks"] is not None:
            out["marks"] = out["marks"][-history_length:]
    return out


def append_events(
    history: dict[str, np.ndarray],
    new_times: np.ndarray,
    new_locs: np.ndarray,
) -> dict[str, np.ndarray]:
    out = copy_history(history)
    if new_times.size == 0:
        return out
    out["times"] = np.concatenate([out["times"], np.asarray(new_times, dtype=np.float32)], axis=0)
    out["locations"] = np.concatenate(
        [out["locations"], np.asarray(new_locs, dtype=np.float32).reshape(-1, 2)],
        axis=0,
    )
    return out


def slice_initial_history(seq: dict[str, np.ndarray], start_event_idx: int) -> dict[str, np.ndarray]:
    times = np.asarray(seq["times"], dtype=np.float32)
    locs = np.asarray(seq["locations"], dtype=np.float32)
    if start_event_idx < 0 or start_event_idx >= times.shape[0]:
        raise SystemExit(
            f"--start-event-idx {start_event_idx} out of range for selected sequence "
            f"(n={times.shape[0]})."
        )
    end = start_event_idx + 1
    return {
        "times": times[:end].copy(),
        "locations": locs[:end].copy(),
    }


def history_until_time(
    seq: dict[str, np.ndarray],
    *,
    t_cutoff: float,
    history_length: int,
) -> dict[str, np.ndarray]:
    times = np.asarray(seq["times"], dtype=np.float32)
    locs = np.asarray(seq["locations"], dtype=np.float32)
    mask = times <= float(t_cutoff)
    if not bool(mask.any()):
        raise SystemExit(
            "History construction produced an empty prefix. Choose a later start event "
            "or a smaller frame start."
        )
    return cap_history(
        {
            "times": times[mask].copy(),
            "locations": locs[mask].copy(),
        },
        history_length,
    )


def build_frame_schedule(
    start_time: float,
    n_frames: int,
    step_size: float,
    horizon: float,
) -> list[FrameWindow]:
    if n_frames <= 0:
        raise SystemExit("--n-frames must be positive.")
    if horizon <= 0.0:
        raise SystemExit("--horizon must be positive.")
    if step_size <= 0.0:
        raise SystemExit("--step-size must be positive.")
    out: list[FrameWindow] = []
    for idx in range(int(n_frames)):
        frame_start = float(start_time + idx * step_size)
        out.append(FrameWindow(index=idx, start=frame_start, end=frame_start + horizon))
    return out


def resolve_spatial_bounds(
    seq_locs: np.ndarray,
    *,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
) -> tuple[float, float, float, float]:
    locs = np.asarray(seq_locs, dtype=np.float32)
    lo = locs.min(axis=0)
    hi = locs.max(axis=0)
    span = np.maximum(hi - lo, 1e-4)
    pad = 0.08 * span
    auto_xmin = float(lo[0] - pad[0])
    auto_xmax = float(hi[0] + pad[0])
    auto_ymin = float(lo[1] - pad[1])
    auto_ymax = float(hi[1] + pad[1])
    return (
        float(auto_xmin if xmin is None else xmin),
        float(auto_xmax if xmax is None else xmax),
        float(auto_ymin if ymin is None else ymin),
        float(auto_ymax if ymax is None else ymax),
    )


def build_fixed_spatial_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if grid_size < 2:
        raise SystemExit("--grid-size must be >= 2.")
    xs = np.linspace(float(xmin), float(xmax), int(grid_size), dtype=np.float32)
    ys = np.linspace(float(ymin), float(ymax), int(grid_size), dtype=np.float32)
    return xs, ys


def load_run_result(run_dir: Path) -> RunResult | None:
    path = Path(run_dir) / "run_result.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return RunResult.from_dict(json.load(f))
    except Exception:
        return None


def load_runs(
    run_paths: list[str | Path],
    *,
    device: torch.device,
    supported_presets: set[str],
) -> list[LoadedRun]:
    resolved = [Path(p).resolve() for p in run_paths]
    raw: list[tuple[Path, STPPRunner, str, str, str, str]] = []
    for run_dir in resolved:
        runner = STPPRunner.load(run_dir)
        runner.model.to(device)
        runner.model.eval()
        preset = runner.config.model.preset
        if preset not in supported_presets:
            raise SystemExit(
                f"Unsupported preset for this evaluation path: {preset}. "
                f"Supported: {sorted(supported_presets)}"
            )
        result = load_run_result(run_dir)
        preset_status = (
            result.preset_status if result is not None else ConfigRegistry.canonical_status(preset)
        )
        nll_kind = result.nll_kind if result is not None else runner.model.event_model.capabilities.nll_kind
        nll_report_space = result.nll_report_space if result is not None else "native"
        raw.append((run_dir, runner, preset, preset_status, nll_kind, nll_report_space))

    preset_counts: dict[str, int] = {}
    for _, _, preset, _, _, _ in raw:
        preset_counts[preset] = preset_counts.get(preset, 0) + 1

    loaded: list[LoadedRun] = []
    for run_dir, runner, preset, preset_status, nll_kind, nll_report_space in raw:
        label = f"{preset}:{run_dir.name}" if preset_counts[preset] > 1 else preset
        loaded.append(
            LoadedRun(
                label=label,
                safe_label=sanitize_label(label),
                preset=preset,
                run_dir=run_dir,
                runner=runner,
                preset_status=preset_status,
                nll_kind=nll_kind,
                nll_report_space=nll_report_space,
            )
        )
    return loaded
