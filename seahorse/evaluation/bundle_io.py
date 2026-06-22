"""Serialization helpers for the new evaluation bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from seahorse.evaluation.artifacts import (
    ARTIFACT_MANIFEST_FILENAME,
    PREDICTIVE_SAMPLES,
    PREDICTIVE_SAMPLES_SCHEMA_VERSION,
    ArtifactKey,
    ArtifactManifest,
    PredictiveSamples,
    artifact_dir_for_key,
    manifest_path_for_key,
    predictive_samples_manifest,
    predictive_samples_payload_path,
)
from seahorse.evaluation.predictive.compare import (
    PredictiveCompareSpec,
    PredictiveComparisonResult,
    PredictiveFrameResult,
    PredictiveModelResult,
)
from seahorse.evaluation.predictive.rollout import ExactProposalConfig
from seahorse.evaluation.runtime import FrameWindow
from seahorse.evaluation.surface import SurfaceDiagnosticResult


def write_predictive_samples_artifact(
    artifact_root: str | Path,
    key: ArtifactKey,
    samples: PredictiveSamples,
) -> dict[str, Path]:
    artifact_dir = artifact_dir_for_key(artifact_root, key)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload_path = predictive_samples_payload_path(artifact_root, key)
    np.savez_compressed(
        payload_path,
        next_times=np.asarray(samples.next_times, dtype=np.float32),
        next_locs=np.asarray(samples.next_locs, dtype=np.float32),
        true_next_times=np.asarray(samples.true_next_times, dtype=np.float32),
        true_next_locs=np.asarray(samples.true_next_locs, dtype=np.float32),
        history_end_times=np.asarray(samples.history_end_times, dtype=np.float32),
        sequence_index=np.asarray(samples.sequence_index, dtype=np.int64),
        target_event_index=np.asarray(samples.target_event_index, dtype=np.int64),
        history_length=np.asarray(samples.history_length, dtype=np.int64),
        is_last_context=np.asarray(samples.is_last_context, dtype=np.bool_),
        sampling_succeeded=np.asarray(samples.sampling_succeeded, dtype=np.bool_),
        sampling_backend=np.asarray(samples.sampling_backend, dtype="<U64"),
    )
    manifest = predictive_samples_manifest(key, samples)
    manifest_path = manifest_path_for_key(artifact_root, key)
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    return {"manifest": manifest_path, "payload": payload_path}


def load_predictive_samples_artifact(
    artifact_root: str | Path,
    key: ArtifactKey,
) -> PredictiveSamples | None:
    manifest_path = manifest_path_for_key(artifact_root, key)
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        manifest = ArtifactManifest.from_dict(json.load(f))
    if (
        manifest.family != PREDICTIVE_SAMPLES
        or manifest.schema_version != key.schema_version
        or manifest.key != key.digest
    ):
        raise ValueError(
            f"Artifact manifest at {manifest_path} does not match requested "
            f"{key.family}:{key.digest}."
        )
    payload_path = artifact_dir_for_key(artifact_root, key) / manifest.payload_file
    if not payload_path.exists():
        raise FileNotFoundError(
            f"Artifact manifest exists but payload is missing: {payload_path}"
        )
    payload = np.load(payload_path)
    return _predictive_samples_from_payload(payload)


def scan_and_load_predictive_samples(artifact_dir: str | Path) -> PredictiveSamples | None:
    """Load the first valid predictive_samples artifact found under ``artifact_dir``.

    Scans ``artifact_dir / "predictive_samples" / <digest> / manifest.json`` for
    any digest directory that contains a valid manifest and payload.  Returns the
    first one found (alphabetical digest order), or ``None`` if the directory does
    not exist or contains no valid artifacts.

    This is used by ``evaluate merge-artifacts`` to load shard artifacts without
    knowing the digest in advance.
    """
    root = Path(artifact_dir).resolve()
    family_dir = root / PREDICTIVE_SAMPLES
    if not family_dir.is_dir():
        return None
    for digest_dir in sorted(family_dir.iterdir()):
        if not digest_dir.is_dir():
            continue
        manifest_path = digest_dir / ARTIFACT_MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path) as f:
                manifest_data = json.load(f)
            manifest = ArtifactManifest.from_dict(manifest_data)
        except Exception:
            continue
        if manifest.family != PREDICTIVE_SAMPLES:
            continue
        payload_path = digest_dir / manifest.payload_file
        if not payload_path.exists():
            continue
        try:
            payload = np.load(payload_path)
            return _predictive_samples_from_payload(payload)
        except Exception:
            continue
    return None


def _predictive_samples_from_payload(payload) -> PredictiveSamples:
    next_times = np.asarray(payload["next_times"], dtype=np.float32)
    true_next_times = np.asarray(payload["true_next_times"], dtype=np.float32)
    sequence_index = np.asarray(
        payload["sequence_index"] if "sequence_index" in payload else payload["seq_indices"],
        dtype=np.int64,
    )
    target_event_index = np.asarray(
        payload["target_event_index"] if "target_event_index" in payload else np.zeros_like(sequence_index),
        dtype=np.int64,
    )
    history_length = np.asarray(
        payload["history_length"] if "history_length" in payload else np.maximum(target_event_index, 0),
        dtype=np.int64,
    )
    is_last_context = np.asarray(
        payload["is_last_context"] if "is_last_context" in payload else np.zeros_like(sequence_index, dtype=np.bool_),
        dtype=np.bool_,
    )
    sampling_succeeded = np.asarray(
        payload["sampling_succeeded"]
        if "sampling_succeeded" in payload
        else np.ones_like(sequence_index, dtype=np.bool_),
        dtype=np.bool_,
    )
    sampling_backend = str(
        np.asarray(
            payload["sampling_backend"] if "sampling_backend" in payload else payload["method"]
        ).item()
    )
    return PredictiveSamples(
        next_times=next_times,
        next_locs=np.asarray(payload["next_locs"], dtype=np.float32),
        true_next_times=true_next_times,
        true_next_locs=np.asarray(payload["true_next_locs"], dtype=np.float32),
        history_end_times=np.asarray(payload["history_end_times"], dtype=np.float32),
        sequence_index=sequence_index,
        target_event_index=target_event_index,
        history_length=history_length,
        is_last_context=is_last_context,
        sampling_succeeded=sampling_succeeded,
        sampling_backend=sampling_backend,
    )


def write_predictive_bundle(out_dir: Path, result: PredictiveComparisonResult) -> dict[str, Path]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_files: dict[str, list[str]] = {}
    surfaces = []
    model_labels = []
    for model in result.models:
        model_dir = out_dir / "samples" / model.safe_label
        model_dir.mkdir(parents=True, exist_ok=True)
        rel_files: list[str] = []
        model_surface_frames: list[np.ndarray] = []
        for frame in model.frames:
            frame_path = model_dir / f"frame_{frame.window.index:03d}.npz"
            np.savez_compressed(
                frame_path,
                pooled_event_times=frame.pooled_event_times.astype(np.float32),
                pooled_event_locs=frame.pooled_event_locs.astype(np.float32),
                rollout_event_counts=frame.rollout_event_counts.astype(np.int32),
                true_event_times=frame.true_event_times.astype(np.float32),
                true_event_locs=frame.true_event_locs.astype(np.float32),
                history_locs=(
                    np.zeros((0, 2), dtype=np.float32)
                    if frame.history_locs is None
                    else np.asarray(frame.history_locs, dtype=np.float32)
                ),
            )
            rel_files.append(str(frame_path.relative_to(out_dir)))
            model_surface_frames.append(frame.derived_kde_rate_surface.astype(np.float32))
        frame_files[model.label] = rel_files
        surfaces.append(np.stack(model_surface_frames, axis=0).astype(np.float32))
        model_labels.append(model.label)

    derived_surfaces_path = out_dir / "derived_surfaces.npz"
    np.savez_compressed(
        derived_surfaces_path,
        surfaces=np.stack(surfaces, axis=0).astype(np.float32),
        xs=result.xs.astype(np.float32),
        ys=result.ys.astype(np.float32),
        frame_starts=np.asarray([frame.start for frame in result.frame_schedule], dtype=np.float32),
        frame_ends=np.asarray([frame.end for frame in result.frame_schedule], dtype=np.float32),
        model_labels=np.asarray(model_labels, dtype="<U128"),
    )

    summary = {
        "history_path": str(result.history_path),
        "split": result.split,
        "seq_idx": result.seq_idx,
        "start_event_idx": result.start_event_idx,
        "initial_history_length": result.initial_history_length,
        "sequence_length": result.sequence_length,
        "sequence_start_time": result.sequence_start_time,
        "sequence_end_time": result.sequence_end_time,
        "spec": _predictive_spec_to_dict(result.spec),
        "seed_policy": dict(result.seed_policy),
        "frame_schedule": [
            {"index": frame.index, "start": frame.start, "end": frame.end}
            for frame in result.frame_schedule
        ],
        "color_scale": dict(result.color_scale),
        "models": [
            {
                "label": model.label,
                "safe_label": model.safe_label,
                "preset": model.preset,
                "preset_status": model.preset_status,
                "nll_kind": model.nll_kind,
                "nll_report_space": model.nll_report_space,
                "run_dir": str(model.run_dir),
                "sampling_backend": model.sampling_backend,
                "sample_frame_files": frame_files[model.label],
                "per_frame_mean_events_per_rollout": [
                    float(frame.mean_events_per_rollout) for frame in model.frames
                ],
                "per_frame_true_event_counts": [
                    int(frame.true_event_locs.shape[0]) for frame in model.frames
                ],
                "per_frame_pooled_event_counts": [
                    int(frame.pooled_event_locs.shape[0]) for frame in model.frames
                ],
                "per_frame_diagnostics": [dict(frame.diagnostics) for frame in model.frames],
            }
            for model in result.models
        ],
        "derived_surfaces_npz": str(derived_surfaces_path.relative_to(out_dir)),
        "notes": {
            "primary_artifact": "sampled future-event payloads stored per model and frame",
            "derived_artifact": "KDE rate surfaces stored separately as derived readouts",
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return {
        "summary": summary_path,
        "derived_surfaces": derived_surfaces_path,
    }


def load_predictive_bundle(out_dir: Path) -> PredictiveComparisonResult:
    out_dir = Path(out_dir).resolve()
    with open(out_dir / "summary.json") as f:
        summary = json.load(f)
    derived = np.load(out_dir / summary["derived_surfaces_npz"])
    xs = np.asarray(derived["xs"], dtype=np.float32)
    ys = np.asarray(derived["ys"], dtype=np.float32)
    surface_stack = np.asarray(derived["surfaces"], dtype=np.float32)
    schedule = [
        FrameWindow(index=int(item["index"]), start=float(item["start"]), end=float(item["end"]))
        for item in summary["frame_schedule"]
    ]
    models: list[PredictiveModelResult] = []
    for model_idx, model_summary in enumerate(summary["models"]):
        frames: list[PredictiveFrameResult] = []
        for frame_idx, frame_path in enumerate(model_summary["sample_frame_files"]):
            payload = np.load(out_dir / frame_path)
            history_locs = np.asarray(payload["history_locs"], dtype=np.float32)
            frames.append(
                PredictiveFrameResult(
                    window=schedule[frame_idx],
                    history_locs=history_locs if history_locs.size else None,
                    pooled_event_times=np.asarray(payload["pooled_event_times"], dtype=np.float32),
                    pooled_event_locs=np.asarray(payload["pooled_event_locs"], dtype=np.float32),
                    rollout_event_counts=np.asarray(payload["rollout_event_counts"], dtype=np.int32),
                    true_event_times=np.asarray(payload["true_event_times"], dtype=np.float32),
                    true_event_locs=np.asarray(payload["true_event_locs"], dtype=np.float32),
                    mean_events_per_rollout=float(model_summary["per_frame_mean_events_per_rollout"][frame_idx]),
                    derived_kde_rate_surface=surface_stack[model_idx, frame_idx].astype(np.float32),
                    diagnostics=dict(model_summary["per_frame_diagnostics"][frame_idx]),
                )
            )
        models.append(
            PredictiveModelResult(
                label=str(model_summary["label"]),
                safe_label=str(model_summary["safe_label"]),
                preset=str(model_summary["preset"]),
                preset_status=str(model_summary["preset_status"]),
                nll_kind=str(model_summary["nll_kind"]),
                nll_report_space=str(model_summary["nll_report_space"]),
                run_dir=Path(model_summary["run_dir"]),
                sampling_backend=str(model_summary["sampling_backend"]),
                frames=frames,
            )
        )
    return PredictiveComparisonResult(
        history_path=Path(summary["history_path"]),
        split=str(summary["split"]),
        seq_idx=int(summary["seq_idx"]),
        start_event_idx=int(summary["start_event_idx"]),
        initial_history_length=int(summary["initial_history_length"]),
        sequence_length=int(summary["sequence_length"]),
        sequence_start_time=float(summary["sequence_start_time"]),
        sequence_end_time=float(summary["sequence_end_time"]),
        spec=_predictive_spec_from_dict(summary["spec"]),
        xs=xs,
        ys=ys,
        color_scale={k: float(v) for k, v in summary["color_scale"].items()},
        frame_schedule=schedule,
        models=models,
        seed_policy=dict(summary["seed_policy"]),
    )


def write_surface_bundle(out_dir: Path, result: SurfaceDiagnosticResult) -> dict[str, Path]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.npz"
    payload = {
        "x_grid": result.x_grid.astype(np.float32),
        "y_grid": result.y_grid.astype(np.float32),
        "t_grid": result.t_grid.astype(np.float32),
        "history_times": result.history_times.astype(np.float32),
        "history_locs": result.history_locs.astype(np.float32),
        "primary_cube": result.primary_cube.astype(np.float32),
    }
    for name, value in result.extra_arrays.items():
        payload[name] = np.asarray(value, dtype=np.float32)
    np.savez_compressed(data_path, **payload)
    summary = {
        "run_dir": str(result.run_dir),
        "history_path": str(result.history_path),
        "split": result.split,
        "seq_idx": result.seq_idx,
        "history_length": result.history_length,
        "preset": result.preset,
        "profile": result.profile,
        "device": result.device,
        "primary_value_name": result.primary_value_name,
        "primary_value_label": result.primary_value_label,
        "notes": list(result.notes),
        "provisional": bool(result.provisional),
        "extra_metadata": dict(result.extra_metadata),
        "data_npz": str(data_path.relative_to(out_dir)),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return {"summary": summary_path, "data": data_path}


def load_surface_bundle(out_dir: Path) -> SurfaceDiagnosticResult:
    out_dir = Path(out_dir).resolve()
    with open(out_dir / "summary.json") as f:
        summary = json.load(f)
    payload = np.load(out_dir / summary["data_npz"])
    reserved = {"x_grid", "y_grid", "t_grid", "history_times", "history_locs", "primary_cube"}
    extra_arrays = {name: np.asarray(payload[name], dtype=np.float32) for name in payload.files if name not in reserved}
    return SurfaceDiagnosticResult(
        run_dir=Path(summary["run_dir"]),
        history_path=Path(summary["history_path"]),
        split=str(summary["split"]),
        seq_idx=int(summary["seq_idx"]),
        history_length=int(summary["history_length"]),
        preset=str(summary["preset"]),
        profile=str(summary["profile"]),
        device=str(summary["device"]),
        x_grid=np.asarray(payload["x_grid"], dtype=np.float32),
        y_grid=np.asarray(payload["y_grid"], dtype=np.float32),
        t_grid=np.asarray(payload["t_grid"], dtype=np.float32),
        history_times=np.asarray(payload["history_times"], dtype=np.float32),
        history_locs=np.asarray(payload["history_locs"], dtype=np.float32),
        primary_cube=np.asarray(payload["primary_cube"], dtype=np.float32),
        primary_value_name=str(summary["primary_value_name"]),
        primary_value_label=str(summary["primary_value_label"]),
        notes=list(summary.get("notes", [])),
        provisional=bool(summary.get("provisional", False)),
        extra_arrays=extra_arrays,
        extra_metadata=dict(summary.get("extra_metadata", {})),
    )


def _predictive_spec_to_dict(spec: PredictiveCompareSpec) -> dict:
    return {
        "rollout_mode": spec.rollout_mode,
        "n_frames": spec.n_frames,
        "horizon": spec.horizon,
        "step_size": spec.step_size,
        "n_rollouts": spec.n_rollouts,
        "grid_size": spec.grid_size,
        "bandwidth": spec.bandwidth,
        "xmin": spec.xmin,
        "xmax": spec.xmax,
        "ymin": spec.ymin,
        "ymax": spec.ymax,
        "lambda_bar": spec.lambda_bar,
        "max_events_per_window": spec.max_events_per_window,
        "bridge_retries": spec.bridge_retries,
        "adaptive_thinning": spec.adaptive_thinning,
        "exact_proposal": {
            "mode": spec.exact_proposal.mode,
            "time_bins": spec.exact_proposal.time_bins,
            "spatial_bins": spec.exact_proposal.spatial_bins,
            "safety": spec.exact_proposal.safety,
        },
        "color_percentile": spec.color_percentile,
        "seed": spec.seed,
        "device": spec.device,
    }


def _predictive_spec_from_dict(data: dict) -> PredictiveCompareSpec:
    exact = data.get("exact_proposal", {})
    def _maybe_float(value):
        return None if value is None else float(value)
    return PredictiveCompareSpec(
        rollout_mode=str(data["rollout_mode"]),
        n_frames=int(data["n_frames"]),
        horizon=float(data["horizon"]),
        step_size=float(data["step_size"]),
        n_rollouts=int(data["n_rollouts"]),
        grid_size=int(data["grid_size"]),
        bandwidth=data.get("bandwidth"),
        xmin=_maybe_float(data.get("xmin")),
        xmax=_maybe_float(data.get("xmax")),
        ymin=_maybe_float(data.get("ymin")),
        ymax=_maybe_float(data.get("ymax")),
        lambda_bar=float(data["lambda_bar"]),
        max_events_per_window=int(data["max_events_per_window"]),
        bridge_retries=int(data["bridge_retries"]),
        adaptive_thinning=bool(data["adaptive_thinning"]),
        exact_proposal=ExactProposalConfig(
            mode=str(exact["mode"]),
            time_bins=int(exact["time_bins"]),
            spatial_bins=int(exact["spatial_bins"]),
            safety=float(exact["safety"]),
        ),
        color_percentile=float(data["color_percentile"]),
        seed=int(data["seed"]),
        device=str(data["device"]),
    )
