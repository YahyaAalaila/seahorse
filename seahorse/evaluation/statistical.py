"""Statistical models evaluated on Seahorse's observed test prefixes.

Fitting and posterior estimation belong to the runner/model. These helpers only
score frozen forecasts. They do not concatenate sequences or provide future
events, marks, or the test-window endpoint as prediction history.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from .artifacts import PredictiveSamples
from .backends import evaluation_capabilities


def _sequence_arrays(seq: dict) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(seq["times"], dtype=np.float64)
    locations = np.asarray(seq["locations"], dtype=np.float64)
    if not times.size and not locations.size:
        locations = locations.reshape(0, 2)
    if times.ndim != 1 or locations.shape != (len(times), 2):
        raise ValueError("Statistical evaluation requires times[N], locations[N,2].")
    if not np.isfinite(times).all() or not np.isfinite(locations).all():
        raise ValueError("Statistical evaluation requires finite times and locations.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Next-event evaluation requires strictly increasing event times; select an explicitly declared continuous-time view.")
    return times, locations


def _contexts(seqs: list[dict], min_history: int = 1) -> Iterator[tuple]:
    if min_history < 1:
        raise ValueError("min_history must be at least one observed event.")
    for seq_index, seq in enumerate(seqs):
        times, locations = _sequence_arrays(seq)
        marks = seq.get("marks")
        if marks is not None and len(marks) != len(times):
            raise ValueError("marks must have the same length as times.")
        for i in range(min_history, len(times)):
            history = {
                "times": times[:i].copy(),
                "locations": locations[:i].copy(),
                "t0": float(seq.get("t0", times[0])),
                "t1": float(times[i - 1]),
            }
            if marks is not None:
                history["marks"] = np.asarray(marks)[:i].copy()
            # The calendar origin is known before the forecast, unlike its
            # future endpoint, target marks, or total observed event count.
            for key in ("window_start", "calendar_window_start", "calendar_offset_from_window_start", "time_unit", "time_axis", "source_timezone"):
                if key in seq:
                    history[key] = seq[key]
            yield seq_index, i, i == len(times) - 1, history, float(times[i]), locations[i]


def compute_statistical_next_event_nll(
    runner: Any, seqs: list[dict], *, min_history: int = 1
) -> dict[str, Any]:
    values, sequence_indices, target_indices = [], [], []
    for seq_i, i, _last, history, t, location in _contexts(seqs, min_history):
        values.append(float(runner.model.next_event_nll(history, t, location)))
        sequence_indices.append(seq_i)
        target_indices.append(i)
    arr = np.asarray(values, dtype=np.float64)
    caps = evaluation_capabilities(runner)
    method = getattr(runner.model, "nll_method", "native_statistical_next_event")
    return {
        "mean_nll": float(np.mean(arr)) if arr.size else float("nan"),
        "method": method,
        "kind": caps.nll_kind,
        "report_space": "raw",
        "description": "held-out next-event NLL/event over teacher-forced test prefixes (raw/original data space)",
        "footnote": getattr(caps, "nll_footnote", ""),
        "per_context_nll": arr,
        "per_context_scored_mask": np.ones(arr.size, dtype=bool),
        "per_context_failure_mask": np.isnan(arr),
        "sequence_index": np.asarray(sequence_indices, dtype=np.int64),
        "target_event_index": np.asarray(target_indices, dtype=np.int64),
        "n_contexts": int(arr.size),
        "n_scored_contexts": int(arr.size),
        "n_missing_contexts": 0,
        "n_unscored_contexts": 0,
        "n_failed_contexts": int(np.isnan(arr).sum()),
        "n_nonfinite_scored_contexts": int((~np.isfinite(arr)).sum()),
        "sampling_backend": None,
    }


def compute_statistical_predictive_samples(
    runner: Any, seqs: list[dict], *, k: int, seed: int, min_history: int = 1
) -> PredictiveSamples:
    if k < 1:
        raise ValueError("k must be a positive number of forecast samples.")
    sampled_t, sampled_s, targets_t, targets_s = [], [], [], []
    history_ends, seq_indices, target_indices, last_contexts, successes = [], [], [], [], []
    for seq_i, i, last, history, target_t, target_s in _contexts(seqs, min_history):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), seq_i, i]))
        success = True
        try:
            t, s = runner.model.sample_next_event(history, n_samples=k, rng=rng)
            t, s = np.asarray(t, dtype=np.float64), np.asarray(s, dtype=np.float64)
            success = (
                t.shape == (k,) and s.shape == (k, 2)
                and np.isfinite(t).all() and np.isfinite(s).all()
                and np.all(t > history["times"][-1])
            )
            if success:
                from seahorse.models.statistical.core import contains
                success = bool(contains(runner.model.support, s).all())
        except (RuntimeError, ValueError, FloatingPointError):
            success = False
        if not success:
            t, s = np.full(k, np.nan), np.full((k, 2), np.nan)
        sampled_t.append(t)
        sampled_s.append(s)
        targets_t.append(target_t)
        targets_s.append(target_s)
        history_ends.append(history["times"][-1])
        seq_indices.append(seq_i)
        target_indices.append(i)
        last_contexts.append(last)
        successes.append(success)
    n = len(target_indices)
    return PredictiveSamples(
        next_times=np.asarray(sampled_t, dtype=np.float64).reshape(n, k),
        next_locs=np.asarray(sampled_s, dtype=np.float64).reshape(n, k, 2),
        true_next_times=np.asarray(targets_t, dtype=np.float64),
        true_next_locs=np.asarray(targets_s, dtype=np.float64).reshape(n, 2),
        history_end_times=np.asarray(history_ends, dtype=np.float64),
        sequence_index=np.asarray(seq_indices, dtype=np.int64),
        target_event_index=np.asarray(target_indices, dtype=np.int64),
        history_length=np.asarray(target_indices, dtype=np.int64),
        is_last_context=np.asarray(last_contexts, dtype=bool),
        sampling_succeeded=np.asarray(successes, dtype=bool),
        sampling_backend="statstpp_native_conditional",
    )
