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

from unified_stpp.evaluation.profiles import PREDICTIVE_SAMPLES

PREDICTIVE_SAMPLES_SCHEMA_VERSION = 1
ARTIFACT_MANIFEST_FILENAME = "manifest.json"
PREDICTIVE_SAMPLES_PAYLOAD_FILENAME = "predictive_samples.npz"


@dataclass
class PredictiveSamples:
    """K next-event samples for each test event.

    Arrays are in raw coordinates.

    next_times:        (N_events, K) float32 sampled next-event absolute times.
    next_locs:         (N_events, K, 2) float32 sampled next-event locations.
    true_next_times:   (N_events,) float32 ground-truth next-event times.
    true_next_locs:    (N_events, 2) float32 ground-truth next-event locations.
    history_end_times: (N_events,) float32 conditioning-history end times.
    seq_indices:       (N_events,) int64 source sequence index per event.
    method:            Sampling backend, currently "thinning" or "native".
    """

    next_times: np.ndarray
    next_locs: np.ndarray
    true_next_times: np.ndarray
    true_next_locs: np.ndarray
    history_end_times: np.ndarray
    seq_indices: np.ndarray
    method: str


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
    return ArtifactManifest(
        family=PREDICTIVE_SAMPLES,
        schema_version=PREDICTIVE_SAMPLES_SCHEMA_VERSION,
        key=key.digest,
        payload_file=PREDICTIVE_SAMPLES_PAYLOAD_FILENAME,
        metadata={
            **key.metadata,
            "method": samples.method,
            "n_contexts": int(samples.next_times.shape[0]),
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
) -> ArtifactKey:
    run_dir = getattr(runner, "_run_dir", None)
    run_dir = None if run_dir is None else Path(run_dir).resolve()
    caps = runner.model.event_model.capabilities
    preset = getattr(getattr(runner, "config", None), "model", None)
    preset_id = None if preset is None else getattr(preset, "preset", None)
    metadata: dict[str, Any] = {
        "family": PREDICTIVE_SAMPLES,
        "schema_version": PREDICTIVE_SAMPLES_SCHEMA_VERSION,
        "conditioning": "teacher_forced_next_event",
        "horizon_policy": "median_inter_event_time_x4",
        "spatial_bounds_policy": "test_sequence_bounds_pad_0.08",
        "run_dir": None if run_dir is None else str(run_dir),
        "run_id": None if run_dir is None else run_dir.name,
        "run_files": _run_file_fingerprints(run_dir),
        "preset": preset_id,
        "nll_kind": getattr(caps, "nll_kind", None),
        "has_intensity": bool(getattr(caps, "has_intensity", False)),
        "has_native_sampler": bool(getattr(caps, "has_native_sampler", False)),
        "sampling_backend": "native" if getattr(caps, "has_native_sampler", False) else "thinning",
        "exact_proposal": {"mode": "coarse"},
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
