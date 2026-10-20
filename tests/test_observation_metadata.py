"""Regression tests for CPU-side observation-window metadata propagation."""

from __future__ import annotations

import json

import numpy as np
import torch

from seahorse.cli.evaluate import _load_metric_sequences
from seahorse.data.contract import validate_batch
from seahorse.data.dataset import SlidingWindowSTPPDataset, STPPDataset, collate_fn


def _sequence(sequence_id: str = "seq-1") -> dict:
    return {
        "times": np.array([10.0, 12.0, 15.0, 19.0], dtype=np.float64),
        "locations": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        "marks": np.array([0, 1, 1, 0]),
        "event_covariates": np.array([[1.0], [2.0], [3.0], [4.0]]),
        "field_covariates": np.array([[5.0], [6.0], [7.0], [8.0]]),
        "event_ids": ["a", "b", "c", "d"],
        "event_metadata": {"category": ["x", "y", "y", "x"]},
        "sequence_id": sequence_id,
        "stream_id": "city-A",
        "window_start": "2023-01-01 00:00:00",
        "window_end": "2023-01-02 00:00:00",
        "time_axis": "local_civil",
        "source_timezone": "America/Chicago",
        "time_unit": "day",
        "t0": 0.0,
        "t1": 1.0,
    }


def test_metadata_round_trip_is_raw_under_normalized_and_raw_inputs():
    seq = _sequence()
    raw = STPPDataset([seq], min_length=1, normalize_time=False, normalize_space=False)[0]
    normalized = STPPDataset([seq], min_length=1, normalize_time=True, normalize_space=True)[0]

    assert not torch.equal(raw["times"], normalized["times"])
    for item in (raw, normalized):
        metadata = item["observation_metadata"]
        assert metadata["sequence_id"] == "seq-1"
        assert metadata["stream_id"] == "city-A"
        assert metadata["t0"] == 0.0 and metadata["t1"] == 1.0
        assert metadata["time_axis"] == "local_civil"
        assert metadata["event_time_coordinate"] == "raw_sequence_coordinates"
        assert metadata["observation_bounds_coordinate"] == "raw_sequence_coordinates"
        assert metadata["model_input_bounds_available"] is False
        assert metadata["event_ids"] == ["a", "b", "c", "d"]


def test_collate_aligns_metadata_for_legacy_and_metadata_items():
    item = STPPDataset([_sequence()], min_length=1)[0]
    # Exercise metadata mixing independently of the established requirement
    # that optional event payloads be homogeneous within a collated batch.
    for key in ("marks", "event_covariates", "field_covariates"):
        item.pop(key)
    legacy = {
        "times": torch.tensor([1.0, 2.0, 3.0]),
        "locations": torch.zeros(3, 2),
        "length": 3,
        "coordinate_space": "raw",
    }
    batch = collate_fn([item, legacy])

    validate_batch(batch)
    assert len(batch["observation_metadata"]) == 2
    assert batch["observation_metadata"][0]["sequence_id"] == "seq-1"
    assert batch["observation_metadata"][1]["event_time_coordinate"] == "raw_sequence_coordinates"
    assert all(isinstance(metadata, dict) for metadata in batch["observation_metadata"])


def test_short_and_empty_records_remain_inspectable_after_min_length_filtering():
    empty = {"times": np.array([]), "locations": np.empty((0, 2)), "sequence_id": "empty"}
    short = {"times": np.array([1.0, 2.0]), "locations": np.zeros((2, 2)), "sequence_id": "short"}
    retained = _sequence("retained")
    dataset = STPPDataset([empty, short, retained], min_length=3)

    assert len(dataset) == 1
    assert dataset.n_input_sequences == 3
    assert dataset.n_retained_sequences == 1
    assert dataset.n_removed_by_min_length == 2
    assert dataset.n_empty_sequences == 1
    assert [seq["sequence_id"] for seq in dataset.raw_sequences] == ["empty", "short", "retained"]
    assert [seq["sequence_id"] for seq in dataset.removed_by_min_length] == ["empty", "short"]


def test_sliding_windows_keep_parent_identity_without_claiming_parent_exposure():
    parent = _sequence("parent-day")
    short = {"times": np.array([1.0]), "locations": np.zeros((1, 2)), "sequence_id": "too-short"}
    dataset = SlidingWindowSTPPDataset([parent, short], lookback=2, lookahead=1)

    assert dataset.n_input_sequences == 2
    assert dataset.n_removed_by_window_length == 1
    assert dataset.n_empty_sequences == 0
    item = dataset[1]
    metadata = item["observation_metadata"]
    assert metadata["sequence_id"] == "parent-day"
    assert metadata["stream_id"] == "city-A"
    assert metadata["parent_source_sequence_index"] == 0
    assert metadata["parent_window_event_start_index"] == 1
    assert metadata["parent_window_event_end_index_exclusive"] == 4
    assert metadata["window_is_not_parent_observation_exposure"] is True
    assert metadata["t0"] == 0.0 and metadata["t1"] == 1.0
    assert metadata["event_ids"] == ["b", "c", "d"]
    assert metadata["event_metadata"] == {"category": ["y", "y", "x"]}


def test_metric_reader_preserves_marks_covariates_and_labels_truncated_prefix(tmp_path):
    source = _sequence()
    path = tmp_path / "test.jsonl"
    with path.open("w") as handle:
        handle.write(json.dumps(_jsonable(source)) + "\n")

    records = _load_metric_sequences(
        path,
        max_seqs=None,
        max_events=3,
        load_jsonl=lambda source_path: [json.loads(line) for line in source_path.read_text().splitlines()],
    )
    record = records[0]

    assert record["times"].shape == (3,)
    assert record["marks"] == [0, 1, 1]
    assert record["event_covariates"] == [[1.0], [2.0], [3.0]]
    assert record["field_covariates"] == [[5.0], [6.0], [7.0]]
    assert record["event_ids"] == ["a", "b", "c"]
    assert record["event_metadata"] == {"category": ["x", "y", "y"]}
    metadata = record["observation_metadata"]
    assert metadata["sequence_id"] == "seq-1"
    assert metadata["t0"] == 0.0 and metadata["t1"] == 1.0
    assert metadata["event_truncation"] == {
        "applied": True,
        "original_event_count": 4,
        "retained_event_count": 3,
        "max_events": 3,
        "original_observation_window_is_not_complete_for_loaded_prefix": True,
    }


def test_metric_reader_keeps_float64_distinct_jittered_times_until_canonical_repair(tmp_path):
    source = _sequence()
    first = 100.0
    source["times"] = np.array([first, np.nextafter(first, np.inf), 101.0, 102.0])
    path = tmp_path / "close_times.jsonl"
    path.write_text(json.dumps(_jsonable(source)) + "\n")

    record = _load_metric_sequences(
        path,
        max_seqs=None,
        max_events=None,
        load_jsonl=lambda source_path: [json.loads(line) for line in source_path.read_text().splitlines()],
    )[0]

    assert record["times"].dtype == np.float64
    assert record["times"][1] > record["times"][0]
    assert record["times"][1].astype(np.float32) == record["times"][0].astype(np.float32)


def test_metric_reader_slices_trace_fields_already_nested_in_observation_metadata(tmp_path):
    source = _sequence()
    event_ids = source.pop("event_ids")
    event_metadata = source.pop("event_metadata")
    source["observation_metadata"] = {
        "event_ids": event_ids,
        "event_metadata": event_metadata,
        "sequence_id": "nested-only-trace",
    }
    path = tmp_path / "nested_trace.jsonl"
    path.write_text(json.dumps(_jsonable(source)) + "\n")

    record = _load_metric_sequences(
        path,
        max_seqs=None,
        max_events=2,
        load_jsonl=lambda source_path: [json.loads(line) for line in source_path.read_text().splitlines()],
    )[0]

    metadata = record["observation_metadata"]
    assert metadata["event_ids"] == ["a", "b"]
    assert metadata["event_metadata"] == {"category": ["x", "y"]}
    assert metadata["event_truncation"]["retained_event_count"] == 2


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
