"""Scalable data-registry protocol for immutable prepared real-data views."""
from __future__ import annotations

import bisect
import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from seahorse.data.dataset import (
    STPPDataset,
    _slice_event_trace,
    _strict_float32_times,
    observation_metadata_from_sequence,
)
from seahorse.data.prepared import (
    apply_train_mark_map,
    fit_train_mark_map,
    is_prepared_dataset_id,
    resolve_prepared_dataset,
)


class LazyWindowBatchSampler:
    """Re-iterable range batches that never allocate all window indices."""

    def __init__(self, n_windows: int, per_batch: int):
        self.n_windows, self.per_batch = int(n_windows), int(per_batch)

    def __iter__(self):
        for start in range(0, self.n_windows, self.per_batch):
            yield range(start, min(start + self.per_batch, self.n_windows))

    def __len__(self) -> int:
        return math.ceil(self.n_windows / self.per_batch) if self.n_windows else 0


def _copy_unmarked(records: list[dict]) -> list[dict]:
    """Hide model marks while retaining source indices in CPU trace metadata."""
    result = []
    for record in records:
        copied = dict(record)
        marks = copied.pop("marks", None)
        if marks is not None:
            event_metadata = dict(copied.get("event_metadata") or {})
            # Reuse the existing list: no duplicate event-scale allocation.
            event_metadata.setdefault("prepared_source_mark_index", marks)
            copied["event_metadata"] = event_metadata
        result.append(copied)
    return result


def _route_provenance(records: list[dict], *, route, split: str) -> list[dict]:
    result = []
    for record in records:
        copied = dict(record)
        metadata = dict(copied.get("observation_metadata") or {})
        metadata["prepared_route"] = {
            "dataset_id": route.dataset_id,
            "corpus": route.spec.corpus,
            "role": route.spec.role,
            "view": route.view,
            "split": split,
            "manifest_sha256": route.manifest_sha256,
            "split_sha256": route.split_sha256[split],
            "strictly_increasing_global_times": route.strictly_increasing_global_times,
            "exact_continuous_time_eligible": route.exact_continuous_time_eligible,
            "approximate_observation_model": route.approximate_observation_model,
        }
        copied["observation_metadata"] = metadata
        result.append(copied)
    return result


class LazyPreparedSlidingWindowDataset(Dataset):
    """Fixed-context view indexed without materialising one dict per window."""

    def __init__(self, sequences: list[dict], *, lookback: int, lookahead: int = 1):
        self.raw_sequences = list(sequences)
        self.lookback, self.lookahead = int(lookback), int(lookahead)
        if self.lookback < 1 or self.lookahead < 1:
            raise ValueError("lookback and lookahead must both be >= 1.")
        self.window_length = self.lookback + self.lookahead
        self.coordinate_space = "raw"
        self.normalize_time = self.normalize_space = self.normalize_covariates = False
        self.time_mean, self.time_std = 0.0, 1.0
        self.cov_mean = self.cov_std = None
        self.n_input_sequences = len(self.raw_sequences)
        self.n_empty_sequences = sum(len(s["times"]) == 0 for s in self.raw_sequences)
        # Source records, never eagerly generated H+1 windows. Consumers that
        # need train statistics can use this complete raw-source collection.
        self.stat_sequences = self.raw_sequences
        # Event-sized numerical arrays are converted exactly once per parent
        # source record. __getitem__ only takes cheap slices, rather than
        # repeatedly converting a full Citi/Uber/NYC record for every window.
        self._arrays: list[dict[str, np.ndarray]] = []
        for sequence in self.raw_sequences:
            arrays = {
                "times": np.asarray(sequence["times"], dtype=np.float64),
                "locations": np.asarray(sequence["locations"], dtype=np.float32),
            }
            for key in ("event_covariates", "field_covariates", "marks"):
                if sequence.get(key) is not None:
                    arrays[key] = np.asarray(sequence[key])
            self._arrays.append(arrays)
        self._eligible_indices: list[int] = []
        cumulative: list[int] = []
        total = 0
        self.removed_by_window_length = []
        for index, sequence in enumerate(self.raw_sequences):
            n = len(sequence["times"])
            if n < self.window_length:
                self.removed_by_window_length.append(sequence)
                continue
            total += n - self.window_length + 1
            self._eligible_indices.append(index)
            cumulative.append(total)
        self._cumulative_windows = cumulative
        self.n_retained_sequences = total
        self.n_removed_by_window_length = len(self.removed_by_window_length)
        self.sequence_filter_counts = {
            "input": self.n_input_sequences,
            "generated_windows": self.n_retained_sequences,
            "removed_by_window_length": self.n_removed_by_window_length,
            "empty_input": self.n_empty_sequences,
            "window_length": self.window_length,
            "materialization": "lazy_indexed",
        }
        first = next((arrays for arrays in self._arrays if len(arrays["times"]) > 0), None)
        dim = int(first["locations"].shape[-1]) if first is not None else 2
        self.loc_mean, self.loc_std = np.zeros(dim, np.float32), np.ones(dim, np.float32)

    def __len__(self) -> int:
        return self.n_retained_sequences

    def batch_by_size(self, max_events: int) -> LazyWindowBatchSampler:
        per_batch = max(1, int(max_events) // self.window_length) if max_events > 0 else 1
        return LazyWindowBatchSampler(len(self), per_batch)

    def _source_window(self, idx: int) -> tuple[int, int, int, dict]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        parent_pos = bisect.bisect_right(self._cumulative_windows, idx)
        before = self._cumulative_windows[parent_pos - 1] if parent_pos else 0
        source_idx = self._eligible_indices[parent_pos]
        start = idx - before
        return source_idx, start, start + self.window_length, self.raw_sequences[source_idx]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        source_idx, start, end, source = self._source_window(idx)
        n = len(source["times"])
        # Every H+1 window is conditioned independently. Rebase in float64 to
        # its first history event so that context index 0 is exactly t=0, then
        # run the canonical float32 strict-order repair only on that window.
        # This never crosses a source/calendar sequence boundary or drops an
        # event; absolute origin remains CPU-side provenance below.
        arrays = self._arrays[source_idx]
        raw_window_times = arrays["times"][start:end]
        local_times = _strict_float32_times(
            raw_window_times - raw_window_times[0],
            context=f"LazyPreparedSlidingWindowDataset[source_sequence_index={source_idx}]",
        )
        metadata = observation_metadata_from_sequence(source)
        for key in ("event_ids", "event_metadata"):
            if key in metadata:
                metadata[key] = _slice_event_trace(metadata[key], start=start, end=end, parent_n_events=n)
        metadata.update({
            "parent_source_sequence_index": source_idx,
            "parent_window_event_start_index": start,
            "parent_window_event_end_index_exclusive": end,
            "target_event_index": start + self.lookback,
            "target_parent_event_index": start + self.lookback,
            "history_length": self.lookback,
            "window_is_not_parent_observation_exposure": True,
            "window_time_coordinate": "float64_rebased_to_first_history_event",
            "raw_window_first_event_time": float(raw_window_times[0]),
            "conditioning_reset": "history is reset at this window; no event context crosses source/calendar sequence boundaries",
        })
        event_ids = source.get("event_ids")
        if isinstance(event_ids, (list, tuple)) and len(event_ids) == n:
            metadata["target_event_id"] = event_ids[start + self.lookback]
        item = {
            "times": torch.tensor(local_times, dtype=torch.float32),
            "locations": torch.tensor(arrays["locations"][start:end], dtype=torch.float32),
            "length": self.window_length,
            "coordinate_space": self.coordinate_space,
            "observation_metadata": metadata,
        }
        for key, dtype in (("event_covariates", torch.float32), ("field_covariates", torch.float32), ("marks", torch.long)):
            if key in arrays:
                item[key] = torch.tensor(arrays[key][start:end], dtype=dtype)
        return item


def _standard(train, val, test, *, normalize: bool):
    train_ds = STPPDataset(train, normalize_time=normalize, normalize_space=normalize)
    kwargs = {"normalize_time": normalize, "normalize_space": normalize, "cov_mean": train_ds.cov_mean, "cov_std": train_ds.cov_std}
    val_ds, test_ds = STPPDataset(val, **kwargs), STPPDataset(test, **kwargs) if test is not None else None
    for dataset in (val_ds, test_ds):
        if dataset is not None:
            dataset.time_mean, dataset.time_std = train_ds.time_mean, train_ds.time_std
            dataset.loc_mean, dataset.loc_std = train_ds.loc_mean, train_ds.loc_std
    return train_ds, val_ds, test_ds


def build_prepared_protocol_datasets(data_config, train_seqs, val_seqs, test_seqs):
    """Registry builder. Defaults to unmarked, full-sequence train-only stats."""
    kwargs = dict(getattr(data_config, "adapter_kwargs", None) or {})
    policy = str(kwargs.get("mark_policy", "unmarked"))
    if policy not in {"unmarked", "train_only"}:
        raise ValueError("prepared protocol requires data.adapter_kwargs.mark_policy 'unmarked' or 'train_only'.")
    dataset_id = getattr(data_config, "dataset", None)
    if not dataset_id or not is_prepared_dataset_id(dataset_id):
        raise ValueError("prepared protocol requires a data.dataset='prepared:<cohort>:<view>' ID.")
    route = resolve_prepared_dataset(dataset_id)
    if str(kwargs.get("continuous_time_mode", "none")) == "exact":
        route.assert_continuous_time_eligible(exact=True)
    elif str(kwargs.get("continuous_time_mode", "none")) == "approximate":
        route.assert_continuous_time_eligible(exact=False)
    train, val, test = (_route_provenance(train_seqs, route=route, split="train"), _route_provenance(val_seqs, route=route, split="val"), None if test_seqs is None else _route_provenance(test_seqs, route=route, split="test"))
    mark_map = None
    if policy == "unmarked":
        train, val = _copy_unmarked(train), _copy_unmarked(val)
        test = None if test is None else _copy_unmarked(test)
        n_marks = 0
    else:
        mark_map = fit_train_mark_map(train)
        train, val = apply_train_mark_map(train, mark_map), apply_train_mark_map(val, mark_map)
        test = None if test is None else apply_train_mark_map(test, mark_map)
        n_marks = 0 if mark_map is None else mark_map.n_marks
    view = str(kwargs.get("training_view", "full_sequence"))
    base_protocol = str(kwargs.get("base_protocol", "standard"))
    if view == "full_sequence":
        train_ds, val_ds, test_ds = _standard(
            train, val, test,
            normalize=bool(data_config.normalize) if base_protocol == "standard" else False,
        )
    elif view == "sliding_window":
        route.assert_continuous_time_eligible(exact=False)
        lookback, lookahead = int(kwargs.get("lookback", 20)), int(kwargs.get("lookahead", 1))
        train_ds = LazyPreparedSlidingWindowDataset(train, lookback=lookback, lookahead=lookahead)
        val_ds = LazyPreparedSlidingWindowDataset(val, lookback=lookback, lookahead=lookahead)
        test_ds = None if test is None else STPPDataset(test, normalize_time=False, normalize_space=False, normalize_covariates=False)
    else:
        raise ValueError("prepared protocol training_view must be 'full_sequence' or 'sliding_window'.")
    train_ds.prepared_mark_map = mark_map
    train_ds.prepared_n_marks = n_marks
    train_ds.prepared_route = route.machine_readable()
    return train_ds, val_ds, test_ds


__all__ = ["LazyPreparedSlidingWindowDataset", "LazyWindowBatchSampler", "build_prepared_protocol_datasets"]
