from __future__ import annotations

import math

import numpy as np
import pytest

from seahorse.experiments.coordinates import fit_coordinates, invert_samples, transform_sequences


def _train():
    return [
        {"times": [0.0, 1.0, 4.0], "locations": [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]], "event_ids": ["a", "b", "c"], "t0": 0.0, "t1": 5.0, "time_unit": "day", "time_axis": "utc"},
        {"times": [0.0, 2.0], "locations": [[2.0, 10.0], [1.0, 12.0]], "event_ids": ["d", "e"]},
    ]


def test_fit_is_train_only_affine_and_records_jacobian():
    artifact = fit_coordinates(_train(), {"longitude": [0.0, 2.0], "latitude": [10.0, 12.0]}, history=20)
    # Positive train gaps are [1, 3, 2], whose median is 2.
    assert artifact.median_positive_train_interval == 2.0
    assert artifact.time_scale == 40.0
    assert artifact.train_events == 5
    assert artifact.positive_train_intervals == 3
    assert artifact.log_raw_volume_per_normalized_volume == pytest.approx(math.log(40.0 * 2.0 * 2.0))


def test_transform_preserves_raw_bounds_ids_and_inverts():
    artifact = fit_coordinates(_train(), (0.0, 2.0, 10.0, 12.0))
    transformed = list(transform_sequences([_train()[0]], artifact))[0]
    assert transformed["times"] == pytest.approx([0.0, 0.025, 0.1])
    assert np.asarray(transformed["locations"]).tolist() == [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    assert transformed["event_ids"] == ["a", "b", "c"]
    assert transformed["t0"] == 0.0 and transformed["t1"] == 5.0
    assert transformed["observation_metadata"]["campaign_coordinate_transform"]["observation_bounds_remain_raw"]
    times, locations = invert_samples(transformed["times"], transformed["locations"], artifact)
    assert times == pytest.approx([0.0, 1.0, 4.0])
    assert locations == pytest.approx(np.asarray(_train()[0]["locations"]))


def test_transform_handles_empty_calendar_record_without_changing_provenance():
    artifact = fit_coordinates(_train(), (0.0, 2.0, 10.0, 12.0))
    empty = {"times": [], "locations": [], "sequence_id": "empty-day", "window_start": "2023-01-01", "window_end": "2023-01-02", "t0": 0.0, "t1": 1.0, "event_ids": []}
    transformed = next(transform_sequences([empty], artifact))
    assert transformed["times"] == [] and transformed["locations"] == []
    assert transformed["sequence_id"] == "empty-day"
    assert transformed["window_start"] == "2023-01-01" and transformed["t1"] == 1.0


def test_heldout_values_do_not_affect_fit_and_outside_rectangle_is_rejected():
    artifact = fit_coordinates(iter(_train()), (0.0, 2.0, 10.0, 12.0))
    assert artifact.train_events == 5
    with pytest.raises(ValueError, match="outside the declared spatial rectangle"):
        fit_coordinates([{"times": [0.0, 1.0], "locations": [[0.0, 10.0], [3.0, 11.0]]}], (0.0, 2.0, 10.0, 12.0))
