"""Primary sample-based predictive comparison API."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from ..runtime import (
    FrameWindow,
    HistoryQuery,
    RunTarget,
    build_fixed_spatial_grid,
    build_frame_schedule,
    load_runs,
    load_sequence,
    parse_bandwidth,
    require_history_path,
    resolve_device,
    resolve_spatial_bounds,
    sanitize_label,
    slice_initial_history,
)
from .rollout import (
    ExactProposalConfig,
    SUPPORTED_PRESETS,
    evaluate_free_running_model,
    evaluate_teacher_forced_frame,
    is_exact_preset,
)


def kde_rate_surface(
    pooled_locs: np.ndarray,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    bandwidth: float | str | None,
    mean_events_per_rollout: float,
    window_duration: float,
) -> np.ndarray:
    """Derived KDE rate readout from sampled future-event payloads."""
    from scipy.stats import gaussian_kde

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    zero = np.zeros((xs.shape[0], ys.shape[0]), dtype=np.float32)
    if pooled_locs.size == 0 or mean_events_per_rollout <= 0.0:
        return zero

    pts = np.asarray(pooled_locs, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected pooled_locs with shape (N, 2), got {pts.shape}")

    if pts.shape[0] < 3:
        span = np.asarray([xs[-1] - xs[0], ys[-1] - ys[0]], dtype=np.float64)
        jitter_scale = max(float(np.max(np.abs(span))) * 1e-4, 1e-6)
        extra = np.random.default_rng(0).normal(scale=jitter_scale, size=(3 - pts.shape[0], 2))
        pts = np.concatenate([pts, pts[:1] + extra], axis=0)

    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=0)
    try:
        kde = gaussian_kde(pts.T, bw_method=bandwidth)
    except Exception:
        jitter = np.random.default_rng(0).normal(scale=1e-5, size=pts.shape)
        kde = gaussian_kde((pts + jitter).T, bw_method=bandwidth)

    values = kde(grid).reshape(xs.shape[0], ys.shape[0]).astype(np.float32)
    rate_scale = float(mean_events_per_rollout) / max(float(window_duration), 1e-8)
    return (values * rate_scale).astype(np.float32)


def compute_shared_colorscale(
    surfaces: np.ndarray,
    percentile: float,
) -> tuple[float, float]:
    arr = np.asarray(surfaces, dtype=np.float32)
    positive = arr[np.isfinite(arr) & (arr > 0.0)]
    if positive.size == 0:
        return 0.0, 1.0
    pct = float(np.clip(percentile, 50.0, 100.0))
    vmax = float(np.percentile(positive, pct))
    if not math.isfinite(vmax) or vmax <= 0.0:
        vmax = float(positive.max())
    return 0.0, max(vmax, 1e-8)


@dataclass(frozen=True)
class PredictiveCompareSpec:
    rollout_mode: Literal["teacher_forced", "free_running"] = "teacher_forced"
    n_frames: int = 6
    horizon: float = 1.0
    step_size: float | None = None
    n_rollouts: int = 128
    grid_size: int = 96
    bandwidth: float | str | None = None
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    lambda_bar: float = 10.0
    max_events_per_window: int = 64
    bridge_retries: int = 64
    adaptive_thinning: bool = True
    exact_proposal: ExactProposalConfig = field(default_factory=ExactProposalConfig)
    color_percentile: float = 99.0
    seed: int = 0
    device: str = "auto"


@dataclass
class PredictiveFrameResult:
    window: FrameWindow
    history_locs: np.ndarray | None
    pooled_event_times: np.ndarray
    pooled_event_locs: np.ndarray
    rollout_event_counts: np.ndarray
    true_event_times: np.ndarray
    true_event_locs: np.ndarray
    mean_events_per_rollout: float
    derived_kde_rate_surface: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class PredictiveModelResult:
    label: str
    safe_label: str
    preset: str
    preset_status: str
    nll_kind: str
    nll_report_space: str
    run_dir: Path
    sampling_backend: str
    frames: list[PredictiveFrameResult]


@dataclass
class PredictiveComparisonResult:
    history_path: Path
    split: str
    seq_idx: int
    start_event_idx: int
    initial_history_length: int
    sequence_length: int
    sequence_start_time: float
    sequence_end_time: float
    spec: PredictiveCompareSpec
    xs: np.ndarray
    ys: np.ndarray
    color_scale: dict[str, float]
    frame_schedule: list[FrameWindow]
    models: list[PredictiveModelResult]
    seed_policy: dict[str, Any]


class PredictiveComparator:
    """Compare multiple fitted runs under one shared predictive rollout spec."""

    def compare(
        self,
        run_targets: Iterable[RunTarget | str | Path],
        history_query: HistoryQuery,
        spec: PredictiveCompareSpec,
    ) -> PredictiveComparisonResult:
        step_size = float(spec.horizon if spec.step_size is None else spec.step_size)
        if not np.isclose(step_size, float(spec.horizon), rtol=0.0, atol=1e-8):
            raise SystemExit(
                "predictive-compare currently requires --step-size == --horizon so "
                "frames are non-overlapping fixed future windows."
            )
        if spec.exact_proposal.time_bins <= 0 or spec.exact_proposal.spatial_bins <= 0:
            raise SystemExit("exact proposal bins must be positive.")
        if spec.exact_proposal.safety <= 1.0:
            raise SystemExit("exact proposal safety must be > 1.0.")

        history_path = require_history_path(history_query.history_path)
        device = resolve_device(spec.device)
        runs = self._normalize_targets(run_targets)
        loaded_runs = load_runs(runs, device=device, supported_presets=SUPPORTED_PRESETS)
        loaded_runs = self._apply_labels(loaded_runs, run_targets)

        seq = load_sequence(history_path, history_query.seq_idx)
        initial_history = slice_initial_history(seq, history_query.start_event_idx)
        schedule = build_frame_schedule(
            start_time=float(initial_history["times"][-1]),
            n_frames=int(spec.n_frames),
            step_size=step_size,
            horizon=float(spec.horizon),
        )
        xmin, xmax, ymin, ymax = resolve_spatial_bounds(
            seq["locations"],
            xmin=spec.xmin,
            xmax=spec.xmax,
            ymin=spec.ymin,
            ymax=spec.ymax,
        )
        xs, ys = build_fixed_spatial_grid(xmin, xmax, ymin, ymax, int(spec.grid_size))
        bandwidth = parse_bandwidth(spec.bandwidth if isinstance(spec.bandwidth, str) else spec.bandwidth)

        model_results: list[PredictiveModelResult] = []
        all_surfaces: list[np.ndarray] = []

        for loaded in loaded_runs:
            if spec.rollout_mode == "teacher_forced":
                frame_meta = [
                    evaluate_teacher_forced_frame(
                        loaded,
                        seq=seq,
                        window=window,
                        history_length=int(history_query.history_length),
                        n_rollouts=int(spec.n_rollouts),
                        xmin=xmin,
                        xmax=xmax,
                        ymin=ymin,
                        ymax=ymax,
                        lambda_bar=float(spec.lambda_bar),
                        max_events_per_window=int(spec.max_events_per_window),
                        bridge_retries=int(spec.bridge_retries),
                        adaptive_thinning=bool(spec.adaptive_thinning),
                        exact_proposal=spec.exact_proposal,
                        device=device,
                        base_seed=int(spec.seed),
                    )
                    for window in schedule
                ]
            else:
                frame_meta = evaluate_free_running_model(
                    loaded,
                    initial_history=initial_history,
                    seq=seq,
                    schedule=schedule,
                    history_length=int(history_query.history_length),
                    n_rollouts=int(spec.n_rollouts),
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    lambda_bar=float(spec.lambda_bar),
                    max_events_per_window=int(spec.max_events_per_window),
                    bridge_retries=int(spec.bridge_retries),
                    adaptive_thinning=bool(spec.adaptive_thinning),
                    exact_proposal=spec.exact_proposal,
                    device=device,
                    base_seed=int(spec.seed),
                )

            frames: list[PredictiveFrameResult] = []
            for window, meta in zip(schedule, frame_meta):
                surface = kde_rate_surface(
                    meta["pooled_event_locs"],
                    xs=xs,
                    ys=ys,
                    bandwidth=bandwidth,
                    mean_events_per_rollout=float(meta["mean_events_per_rollout"]),
                    window_duration=window.duration,
                )
                all_surfaces.append(surface)
                frames.append(
                    PredictiveFrameResult(
                        window=window,
                        history_locs=meta["history_locs"],
                        pooled_event_times=np.asarray(meta["pooled_event_times"], dtype=np.float32),
                        pooled_event_locs=np.asarray(meta["pooled_event_locs"], dtype=np.float32),
                        rollout_event_counts=np.asarray(meta["rollout_event_counts"], dtype=np.int32),
                        true_event_times=np.asarray(meta["true_window_times"], dtype=np.float32),
                        true_event_locs=np.asarray(meta["true_window_locs"], dtype=np.float32),
                        mean_events_per_rollout=float(meta["mean_events_per_rollout"]),
                        derived_kde_rate_surface=np.asarray(surface, dtype=np.float32),
                        diagnostics={
                            key: value
                            for key, value in meta.items()
                            if key
                            not in {
                                "history_locs",
                                "pooled_event_times",
                                "pooled_event_locs",
                                "rollout_event_counts",
                                "true_window_times",
                                "true_window_locs",
                                "mean_events_per_rollout",
                            }
                        },
                    )
                )

            model_results.append(
                PredictiveModelResult(
                    label=loaded.label,
                    safe_label=loaded.safe_label,
                    preset=loaded.preset,
                    preset_status=loaded.preset_status,
                    nll_kind=loaded.nll_kind,
                    nll_report_space=loaded.nll_report_space,
                    run_dir=loaded.run_dir,
                    sampling_backend=(
                        "external_thinning_rollout" if is_exact_preset(loaded.preset) else "native_next_event_rollout"
                    ),
                    frames=frames,
                )
            )

        all_surfaces_np = np.asarray(all_surfaces, dtype=np.float32)
        vmin, vmax = compute_shared_colorscale(all_surfaces_np, float(spec.color_percentile))
        return PredictiveComparisonResult(
            history_path=history_path,
            split=str(history_query.split),
            seq_idx=int(history_query.seq_idx),
            start_event_idx=int(history_query.start_event_idx),
            initial_history_length=int(initial_history["times"].shape[0]),
            sequence_length=int(seq["times"].shape[0]),
            sequence_start_time=float(seq["times"][0]),
            sequence_end_time=float(seq["times"][-1]),
            spec=PredictiveCompareSpec(
                rollout_mode=spec.rollout_mode,
                n_frames=int(spec.n_frames),
                horizon=float(spec.horizon),
                step_size=step_size,
                n_rollouts=int(spec.n_rollouts),
                grid_size=int(spec.grid_size),
                bandwidth=bandwidth,
                xmin=float(xmin),
                xmax=float(xmax),
                ymin=float(ymin),
                ymax=float(ymax),
                lambda_bar=float(spec.lambda_bar),
                max_events_per_window=int(spec.max_events_per_window),
                bridge_retries=int(spec.bridge_retries),
                adaptive_thinning=bool(spec.adaptive_thinning),
                exact_proposal=spec.exact_proposal,
                color_percentile=float(spec.color_percentile),
                seed=int(spec.seed),
                device=str(device),
            ),
            xs=xs,
            ys=ys,
            color_scale={"vmin": float(vmin), "vmax": float(vmax)},
            frame_schedule=schedule,
            models=model_results,
            seed_policy={
                "base_seed": int(spec.seed),
                "derivation": (
                    "Deterministic sub-seeds are derived from the comparison base seed and "
                    "stable identifiers such as model label, frame index, rollout index, "
                    "and backend path."
                ),
                "caveat": (
                    "Reproducibility is expected for identical inputs and seeds, subject to "
                    "backend/device-level nondeterminism outside explicit seeding."
                ),
            },
        )

    @staticmethod
    def _normalize_targets(run_targets: Iterable[RunTarget | str | Path]) -> list[Path]:
        out: list[Path] = []
        for target in run_targets:
            if isinstance(target, RunTarget):
                out.append(Path(target.run).resolve())
            else:
                out.append(Path(target).resolve())
        if not out:
            raise SystemExit("predictive-compare requires at least one --run")
        return out

    @staticmethod
    def _apply_labels(
        loaded_runs,
        run_targets: Iterable[RunTarget | str | Path],
    ):
        targets = list(run_targets)
        if len(targets) != len(loaded_runs):
            return loaded_runs
        for loaded, target in zip(loaded_runs, targets):
            if isinstance(target, RunTarget) and target.label:
                loaded.label = str(target.label)
                loaded.safe_label = sanitize_label(str(target.label))
        safe_labels = [loaded.safe_label for loaded in loaded_runs]
        if len(set(safe_labels)) != len(safe_labels):
            raise SystemExit(
                "predictive-compare labels must be unique after sanitization so "
                "per-model bundle paths do not collide."
            )
        return loaded_runs
