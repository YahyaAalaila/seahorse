"""Secondary exact and factorized surface diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from unified_stpp.evaluation.common import (
    HistoryQuery,
    RunTarget,
    require_history_path,
    resolve_device,
)
from unified_stpp.evaluation.surface_profiles import (
    evaluate_neural_future_exact,
    evaluate_notebook_profile,
)
from unified_stpp.runner.runner import STPPRunner
from unified_stpp.utils import load_jsonl


@dataclass(frozen=True)
class SurfaceDiagnosticSpec:
    profile: Literal["notebook_faithful", "future_exact"] = "notebook_faithful"
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
            "neural_cond_gmm",
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

        if spec.profile == "notebook_faithful":
            if preset not in {"auto_stpp", "deep_stpp"}:
                raise SystemExit(
                    "surface --profile notebook_faithful currently supports only auto_stpp and deep_stpp."
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
            if preset not in {"neural_cond_gmm", "neural_jumpcnf", "neural_attncnf"}:
                raise SystemExit(
                    "surface --profile future_exact currently supports only provisional neural exact families."
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
