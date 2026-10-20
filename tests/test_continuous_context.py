import numpy as np
import pytest

from seahorse.experiments.continuous_context import (
    ContinuousPreparedContextDataset,
    build_continuous_context_splits,
)
from seahorse.experiments.coordinates import CoordinateTransform


def _record(day: int, n: int, *, empty=False):
    times = [] if empty else np.linspace(.05, .95, n).tolist()
    return {
        "window_start": f"2020-01-{day:02d}T00:00:00",
        "window_end": f"2020-01-{day + 1:02d}T00:00:00",
        "t0": 0., "t1": 1., "time_unit": "day",
        "sequence_id": f"day-{day}", "stream_id": "catalogue",
        "times": times,
        "locations": [[float(i) / max(1, n), .25] for i in range(n)],
        "event_ids": [f"{day}:{i}" for i in range(n)],
    }


def _transform(scale=2.):
    return CoordinateTransform(
        version=1, history=20, input_time_unit="day", input_time_axis="utc",
        spatial_bounds_lon_lat=(0., 1., 0., 1.), time_mean=0., time_scale=scale,
        space_offset=(0., 0.), space_scale=(1., 1.), log_raw_volume_per_normalized_volume=0.,
        train_sequences=1, train_events=1, positive_train_intervals=1,
        nonpositive_train_intervals=0, median_positive_train_interval=.1,
    )


def test_contiguous_calendar_records_keep_targets_and_empty_waiting_time():
    records = [_record(1, 12), _record(2, 0, empty=True), _record(3, 10)]
    # Match campaign input: transformed local event times, raw observation
    # metadata unchanged.  The dataset must restore local native time before
    # anchoring it to calendar windows.
    transformed = [dict(row, times=(np.asarray(row["times"]) / 2.).tolist()) for row in records]
    dataset = ContinuousPreparedContextDataset(transformed, lookback=20, time_unit="day", transform=_transform(), split="train")
    assert len(dataset) == 2
    assert dataset.stat_sequences is dataset.sequences
    assert dataset.sequence_filter_counts["observation_gap_resets"] == 0
    assert dataset.sequence_filter_counts["empty_input_calendar_records"] == 1
    first = dataset[0]
    assert first["times"][0].item() == 0.
    # The context's twentieth history event comes after the observed empty day,
    # so its elapsed time is retained rather than resetting at day boundaries.
    assert first["times"][19].item() > .5
    meta = first["observation_metadata"]
    assert meta["target_source_sequence_index"] == 2
    assert meta["target_source_event_index"] == 8
    assert meta["target_source_event_id"] == "3:8"
    assert meta["target_conditioning"].startswith("events strictly before target")
    assert not hasattr(dataset, "_windows")


def test_missing_calendar_window_resets_history():
    # Day 2 is not observed at all: this is a collection gap, unlike an empty
    # explicit day-2 record, and neither side has enough events independently.
    dataset = ContinuousPreparedContextDataset([_record(1, 12), _record(3, 12)], lookback=20, time_unit="day")
    assert len(dataset) == 0
    assert dataset.sequence_filter_counts["observation_gap_resets"] == 1


def test_split_builds_are_independent_and_target_trace_is_source_calendar_index():
    train = [_record(1, 21)]
    val = [_record(10, 21)]
    datasets = build_continuous_context_splits({"train": train, "val": val}, lookback=20, time_unit="day")
    assert len(datasets["train"]) == len(datasets["val"]) == 1
    val_item = datasets["val"][0]
    meta = val_item["observation_metadata"]
    assert meta["split"] == "val"
    assert meta["target_source_sequence_index"] == 0
    assert meta["target_source_event_index"] == 20
    assert meta["target_global_event_index"] == 20
    assert meta["target_source_event_id"] == "10:20"


def test_overlapping_or_reordered_observation_windows_fail_explicitly():
    left, right = _record(2, 21), _record(1, 1)
    with pytest.raises(ValueError, match="out of order or overlap"):
        ContinuousPreparedContextDataset([left, right], lookback=20, time_unit="day")


def test_calendar_origin_is_subtracted_before_local_jitter_is_added():
    # Adding these values to a ~18,000-day Unix-epoch coordinate first can
    # collapse their float64 difference.  The continuous adapter must retain
    # native within-window ordering before applying its calendar origin.
    first = .1
    second = np.nextafter(first, np.inf)
    row = _record(1, 0, empty=True)
    row.update(times=[first, second], locations=[[.1, .2], [.2, .3]], event_ids=["a", "b"])
    dataset = ContinuousPreparedContextDataset([row], lookback=1, time_unit="day")
    assert len(dataset) == 1
    assert np.all(np.diff(dataset._segments[0].times) > 0)
    assert dataset[0]["times"][0].item() == 0.


def test_raw_context_is_source_unit_float64_and_has_same_target_identity():
    records = [_record(1, 12), _record(2, 0, empty=True), _record(3, 10)]
    transformed = [dict(row, times=(np.asarray(row["times"]) / 2.).tolist()) for row in records]
    dataset = ContinuousPreparedContextDataset(transformed, lookback=20, time_unit="day", transform=_transform(), split="test")
    raw = dataset.raw_context(0)
    neural = dataset[0]
    assert raw["times"].dtype == raw["locations"].dtype == np.float64
    assert len(raw["times"]) == len(raw["locations"]) == 21
    assert raw["times"][0] == 0.
    assert raw["times"][19] > 1.  # source days, including the empty day
    assert raw["history"]["t0"] == 0.
    assert raw["history"]["t1"] == raw["times"][-2]
    assert raw["next_time"] == raw["target"]["time"] == raw["times"][-1]
    assert raw["observation_metadata"]["target_source_event_id"] == neural["observation_metadata"]["target_source_event_id"] == "3:8"
    assert raw["observation_metadata"]["calendar_window_start"] == "2020-01-01T00:00:00"
    assert raw["target"]["time"] == raw["times"][-1]


def test_raw_context_inverts_spatial_transform_without_changing_neural_view():
    record = _record(1, 2)
    transform = CoordinateTransform(
        version=1, history=1, input_time_unit="day", input_time_axis="utc",
        spatial_bounds_lon_lat=(10., 30., 40., 70.), time_mean=0., time_scale=2.,
        space_offset=(10., 40.), space_scale=(20., 30.), log_raw_volume_per_normalized_volume=0.,
        train_sequences=1, train_events=2, positive_train_intervals=1,
        nonpositive_train_intervals=0, median_positive_train_interval=.1,
    )
    record["locations"] = [[12., 46.], [28., 67.]]
    transformed = dict(record, times=(np.asarray(record["times"]) / 2.).tolist(),
                       locations=((np.asarray(record["locations"]) - [10., 40.]) / [20., 30.]).tolist())
    dataset = ContinuousPreparedContextDataset([transformed], lookback=1, transform=transform)
    raw = dataset.raw_context(0)
    assert np.array_equal(raw["locations"], np.asarray(record["locations"], dtype=np.float64))
    assert np.allclose(dataset[0]["locations"].numpy(), np.asarray(transformed["locations"], dtype=np.float32))


def test_exposure_records_keep_zero_event_windows_and_do_not_cross_gaps():
    contiguous = ContinuousPreparedContextDataset([_record(1, 2), _record(2, 0, empty=True), _record(3, 1)], lookback=1)
    exposures = list(contiguous.exposure_records())
    assert len(exposures) == 1
    assert exposures[0]["t0"] == 0.
    assert exposures[0]["t1"] == 3.
    assert len(exposures[0]["times"]) == 3
    assert exposures[0]["observation_metadata"]["contains_empty_observation_windows"]
    gapped = ContinuousPreparedContextDataset([_record(1, 1), _record(3, 1)], lookback=1)
    assert len(list(gapped.exposure_records())) == 2


def test_raw_count_context_has_exact_history_and_requires_full_observation_horizon():
    record = _record(1, 4)
    record.update(times=[.1, .2, .4, .6], locations=[[.1, .2], [.2, .2], [.3, .2], [.4, .2]])
    dataset = ContinuousPreparedContextDataset([record], lookback=2, time_unit="day")
    count = dataset.raw_count_context(0, .4)
    next_event = dataset.raw_context(0)
    assert np.allclose(count["history"]["times"], [0., .1])
    assert np.allclose(count["true_future"]["times"], [.3])  # .5 is the exclusive endpoint
    assert count["origin"] == pytest.approx(.1)
    assert count["end"] == pytest.approx(.5)
    assert np.array_equal(count["history"]["times"], next_event["history"]["times"])
    assert count["history"]["calendar_offset_from_window_start"] == next_event["history"]["calendar_offset_from_window_start"]
    assert count["observation_metadata"]["target_source_event_id"] == "1:2"
    with pytest.raises(ValueError, match="continuous observed segment"):
        dataset.raw_count_context(0, .9)
