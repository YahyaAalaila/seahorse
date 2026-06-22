"""Evaluation artifact schemas and cache keys.

This module owns persisted metric artifacts.  Phase 1 keeps the scope tight:
only predictive next-event samples are fully serializable here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seahorse.evaluation.profiles import PREDICTIVE_SAMPLES

PREDICTIVE_SAMPLES_SCHEMA_VERSION = 2
ARTIFACT_MANIFEST_FILENAME = "manifest.json"
PREDICTIVE_SAMPLES_PAYLOAD_FILENAME = "predictive_samples.npz"


@dataclass
class PredictiveSamples:
    """K next-event samples for each held-out next-event context.

    Arrays are in raw coordinates.

    next_times:          (N_contexts, K) float32 sampled next-event absolute times.
    next_locs:           (N_contexts, K, d) float32 sampled next-event locations.
    true_next_times:     (N_contexts,) float32 ground-truth next-event times.
    true_next_locs:      (N_contexts, d) float32 ground-truth next-event locations.
    history_end_times:   (N_contexts,) float32 conditioning-history end times.
    sequence_index:      (N_contexts,) int64 source sequence index per context.
    target_event_index:  (N_contexts,) int64 target-event index within the source sequence.
    history_length:      (N_contexts,) int64 number of events in the conditioning prefix.
    is_last_context:     (N_contexts,) bool marks H_{T-1} -> e_T for each sequence.
    sampling_succeeded:  (N_contexts,) bool marks whether all K samples were produced.
    sampling_backend:    Precise backend label for the predictive artifact.
    """

    next_times: np.ndarray
    next_locs: np.ndarray
    true_next_times: np.ndarray
    true_next_locs: np.ndarray
    history_end_times: np.ndarray
    sequence_index: np.ndarray
    target_event_index: np.ndarray
    history_length: np.ndarray
    is_last_context: np.ndarray
    sampling_succeeded: np.ndarray
    sampling_backend: str

    @property
    def seq_indices(self) -> np.ndarray:
        """Backward-compatible alias for older internal callers."""
        return self.sequence_index

    @property
    def method(self) -> str:
        """Backward-compatible alias for older metric/report code."""
        return self.sampling_backend


@dataclass(frozen=True)
class ArtifactKey:
    family: str
    schema_version: int
    digest: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ArtifactManifest:
    family: str
    schema_version: int
    key: str
    payload_file: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "schema_version": self.schema_version,
            "key": self.key,
            "payload_file": self.payload_file,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactManifest":
        return cls(
            family=str(data["family"]),
            schema_version=int(data["schema_version"]),
            key=str(data["key"]),
            payload_file=str(data["payload_file"]),
            metadata=dict(data.get("metadata", {})),
        )


def artifact_dir_for_key(root: str | Path, key: ArtifactKey) -> Path:
    return Path(root).resolve() / key.family / key.digest


def manifest_path_for_key(root: str | Path, key: ArtifactKey) -> Path:
    return artifact_dir_for_key(root, key) / ARTIFACT_MANIFEST_FILENAME


def predictive_samples_payload_path(root: str | Path, key: ArtifactKey) -> Path:
    return artifact_dir_for_key(root, key) / PREDICTIVE_SAMPLES_PAYLOAD_FILENAME


def predictive_samples_manifest(key: ArtifactKey, samples: PredictiveSamples) -> ArtifactManifest:
    sampling_succeeded = np.asarray(samples.sampling_succeeded, dtype=bool)
    sequence_index = np.asarray(samples.sequence_index, dtype=np.int64)
    is_last_context = np.asarray(samples.is_last_context, dtype=bool)
    return ArtifactManifest(
        family=PREDICTIVE_SAMPLES,
        schema_version=PREDICTIVE_SAMPLES_SCHEMA_VERSION,
        key=key.digest,
        payload_file=PREDICTIVE_SAMPLES_PAYLOAD_FILENAME,
        metadata={
            **key.metadata,
            "sampling_backend": samples.sampling_backend,
            "n_contexts": int(samples.next_times.shape[0]),
            "n_sequences": int(sequence_index.max()) + 1 if sequence_index.size > 0 else 0,
            "n_last_contexts": int(is_last_context.sum()),
            "n_sampling_failures": int((~sampling_succeeded).sum()),
            "k_pred": int(samples.next_times.shape[1]) if samples.next_times.ndim == 2 else 0,
        },
    )


def build_predictive_samples_key(
    runner: Any,
    test_seqs: list[dict[str, np.ndarray]],
    *,
    k: int,
    seed: int,
    device: str,
    exact_time_bins: int = 8,
    exact_spatial_bins: int = 8,
) -> ArtifactKey:
    run_dir = getattr(runner, "_run_dir", None)
    run_dir = None if run_dir is None else Path(run_dir).resolve()
    caps = runner.model.event_model.capabilities
    preset = getattr(getattr(runner, "config", None), "model", None)
    preset_id = None if preset is None else getattr(preset, "preset", None)
    metadata: dict[str, Any] = {
        "family": PREDICTIVE_SAMPLES,
        "schema_version": PREDICTIVE_SAMPLES_SCHEMA_VERSION,
        "evaluation_task": "held_out_next_event_prediction",
        "conditioning_protocol": "teacher_forced_test_prefixes",
        "target_protocol": "immediate_next_observed_event",
        "context_selection": "all_valid_nonempty_test_prefixes",
        "spatial_proposal_bounds": "test_sequence_bounds_pad_0.08",
        "run_dir": None if run_dir is None else str(run_dir),
        "run_id": None if run_dir is None else run_dir.name,
        "run_files": _run_file_fingerprints(run_dir),
        "preset": preset_id,
        "nll_kind": getattr(caps, "nll_kind", None),
        "has_intensity": bool(getattr(caps, "has_intensity", False)),
        "has_native_sampler": bool(getattr(caps, "has_native_sampler", False)),
        "sampling_backend": (
            "native_next_event_sampler"
            if getattr(caps, "has_native_sampler", False)
            else "exact_intensity_thinning"
        ),
        "exact_intensity_sampler": {
            "proposal_mode": "coarse",
            "proposal_time_bins": int(exact_time_bins),
            "proposal_spatial_bins": int(exact_spatial_bins),
            "initial_time_window_policy": "max(4*median_inter_event_time,1e-3)",
            "window_expansion_factor": 2.0,
            "max_window_expansions": 8,
        },
        "k_pred": int(k),
        "seed": int(seed),
        "device": str(device),
        "test_data": _sequence_fingerprint(test_seqs),
    }
    canonical = _canonical_json(metadata)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return ArtifactKey(
        family=PREDICTIVE_SAMPLES,
        schema_version=PREDICTIVE_SAMPLES_SCHEMA_VERSION,
        digest=digest,
        metadata=metadata,
    )


def _run_file_fingerprints(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    files = {
        "config": run_dir / "config.yaml",
        "resolved_config": run_dir / "resolved_config.yaml",
        "run_result": run_dir / "run_result.json",
        "checkpoint_best": run_dir / "checkpoints" / "best.ckpt",
        "checkpoint_last": run_dir / "checkpoints" / "last.ckpt",
        "model_ckpt": run_dir / "model.ckpt",
    }
    return {name: _file_stat(path) for name, path in files.items() if path.exists()}


def _file_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _sequence_fingerprint(seqs: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    h = hashlib.sha256()
    lengths: list[int] = []
    for seq in seqs:
        times = np.asarray(seq["times"], dtype=np.float32)
        locs = np.asarray(seq["locations"], dtype=np.float32)
        lengths.append(int(times.shape[0]))
        _hash_array(h, times)
        _hash_array(h, locs)
    return {
        "n_sequences": int(len(seqs)),
        "n_events": int(sum(lengths)),
        "sequence_lengths": lengths,
        "sha256": h.hexdigest(),
    }


def _hash_array(h: "hashlib._Hash", arr: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(arr)
    h.update(str(contiguous.shape).encode("utf-8"))
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(contiguous.tobytes())


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def merge_predictive_samples(parts: list[PredictiveSamples]) -> PredictiveSamples:
    """Merge shard PredictiveSamples into one artifact.

    Each shard's ``sequence_index`` array is assumed to be 0-based (relative to the
    shard's own sequence list). This function remaps them so that shard i's
    indices are offset by the total number of distinct sequences in shards 0..i-1.

    Parameters
    ----------
    parts:  Non-empty list of per-shard PredictiveSamples in shard order.

    Returns
    -------
    Merged PredictiveSamples with contiguous sequence indices across all shards.
    """
    if not parts:
        raise ValueError("merge_predictive_samples requires at least one part.")

    sampling_backends = {p.sampling_backend for p in parts}
    if len(sampling_backends) > 1:
        raise ValueError(
            "Cannot merge PredictiveSamples with different sampling backends: "
            f"{sorted(sampling_backends)}"
        )
    sampling_backend = parts[0].sampling_backend

    remapped_seq_indices: list[np.ndarray] = []
    offset = 0
    for part in parts:
        si = np.asarray(part.sequence_index, dtype=np.int64)
        remapped_seq_indices.append(si + offset)
        # next shard's offset = one past the highest seq index in this shard
        offset += int(si.max()) + 1 if si.size > 0 else 0

    return PredictiveSamples(
        next_times=np.concatenate([p.next_times for p in parts], axis=0),
        next_locs=np.concatenate([p.next_locs for p in parts], axis=0),
        true_next_times=np.concatenate([p.true_next_times for p in parts], axis=0),
        true_next_locs=np.concatenate([p.true_next_locs for p in parts], axis=0),
        history_end_times=np.concatenate([p.history_end_times for p in parts], axis=0),
        sequence_index=np.concatenate(remapped_seq_indices, axis=0),
        target_event_index=np.concatenate([p.target_event_index for p in parts], axis=0),
        history_length=np.concatenate([p.history_length for p in parts], axis=0),
        is_last_context=np.concatenate([p.is_last_context for p in parts], axis=0),
        sampling_succeeded=np.concatenate([p.sampling_succeeded for p in parts], axis=0),
        sampling_backend=sampling_backend,
    )
