"""Summary bundle writer for held-out next-event predictive evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from unified_stpp.evaluation.artifacts import PredictiveSamples
from unified_stpp.evaluation.result import Report


_SUMMARY_METRICS = (
    "temporal_crps",
    "spatial_energy_score",
    "temporal_mae",
    "spatial_mae",
    "joint_distance",
    "temporal_nll_sample_kde",
    "spatial_nll_sample_kde",
)


def write_next_event_benchmark_summary(
    out_dir: str | Path,
    report: Report,
    samples: PredictiveSamples,
) -> dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    sequence_index = np.asarray(samples.sequence_index, dtype=np.int64)
    target_event_index = np.asarray(samples.target_event_index, dtype=np.int64)
    history_length = np.asarray(samples.history_length, dtype=np.int64)
    is_last_context = np.asarray(samples.is_last_context, dtype=np.bool_)
    sampling_succeeded = np.asarray(samples.sampling_succeeded, dtype=np.bool_)

    context_index_path = out_dir / "next_event_context_index.npz"
    np.savez_compressed(
        context_index_path,
        sequence_index=sequence_index,
        target_event_index=target_event_index,
        history_length=history_length,
        is_last_context=is_last_context,
        sampling_succeeded=sampling_succeeded,
    )

    sequence_ids = np.unique(sequence_index)
    metric_entries: dict[str, Any] = {}
    score_files: dict[str, dict[str, str]] = {}

    for metric_name in _SUMMARY_METRICS:
        result = report.results.get(metric_name)
        if result is None or not result.available or result.per_event is None:
            continue
        per_context = np.asarray(result.per_event, dtype=np.float64)
        if per_context.shape != (sequence_index.shape[0],):
            continue
        per_context = per_context.copy()
        per_context[~sampling_succeeded] = np.nan

        per_sequence_mean = np.full(sequence_ids.shape, np.nan, dtype=np.float64)
        last_context_per_sequence = np.full(sequence_ids.shape, np.nan, dtype=np.float64)
        for idx, seq_id in enumerate(sequence_ids):
            seq_mask = sequence_index == seq_id
            seq_scores = per_context[seq_mask]
            if np.isfinite(seq_scores).any():
                per_sequence_mean[idx] = float(np.nanmean(seq_scores))
            last_mask = seq_mask & is_last_context
            if np.any(last_mask):
                last_values = per_context[last_mask]
                if np.isfinite(last_values).any():
                    last_context_per_sequence[idx] = float(last_values[np.flatnonzero(np.isfinite(last_values))[0]])

        per_context_path = scores_dir / f"{metric_name}_per_context.npy"
        per_sequence_mean_path = scores_dir / f"{metric_name}_per_sequence_mean.npy"
        last_context_path = scores_dir / f"{metric_name}_last_context_per_sequence.npy"
        np.save(per_context_path, per_context.astype(np.float64))
        np.save(per_sequence_mean_path, per_sequence_mean.astype(np.float64))
        np.save(last_context_path, last_context_per_sequence.astype(np.float64))

        metric_entries[metric_name] = {
            "all_context_mean": _nanmean(per_context),
            "all_context_std": _nanstd(per_context),
            "all_context_count": int(np.isfinite(per_context).sum()),
            "all_context_missing_count": int((~np.isfinite(per_context)).sum()),
            "per_sequence_mean_mean": _nanmean(per_sequence_mean),
            "per_sequence_mean_std": _nanstd(per_sequence_mean),
            "per_sequence_count": int(np.isfinite(per_sequence_mean).sum()),
            "last_context_mean": _nanmean(last_context_per_sequence),
            "last_context_std": _nanstd(last_context_per_sequence),
            "last_context_sequence_count": int(np.isfinite(last_context_per_sequence).sum()),
            "files": {
                "per_context": str(per_context_path.relative_to(out_dir)),
                "per_sequence_mean": str(per_sequence_mean_path.relative_to(out_dir)),
                "last_context_per_sequence": str(last_context_path.relative_to(out_dir)),
            },
        }
        score_files[metric_name] = {
            "per_context": str(per_context_path),
            "per_sequence_mean": str(per_sequence_mean_path),
            "last_context_per_sequence": str(last_context_path),
        }

    summary = {
        "schema_version": 1,
        "evaluation_task": {
            "name": "held_out_next_event_prediction",
            "conditioning_protocol": "teacher_forced_test_prefixes",
            "target_protocol": "immediate_next_observed_event",
            "context_selection": "all_valid_nonempty_test_prefixes",
            "n_sequences": int(sequence_ids.shape[0]),
            "n_contexts": int(sequence_index.shape[0]),
            "n_last_contexts": int(is_last_context.sum()),
            "n_sampling_failures": int((~sampling_succeeded).sum()),
        },
        "sampling_backend": samples.sampling_backend,
        "metrics": metric_entries,
    }
    summary_path = out_dir / "next_event_benchmark_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "summary_path": summary_path,
        "context_index_path": context_index_path,
        "score_files": score_files,
        "evaluation_task": dict(summary["evaluation_task"]),
    }


def _nanmean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).any():
        return None
    return float(np.nanmean(values))


def _nanstd(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).any():
        return None
    return float(np.nanstd(values, ddof=0))
