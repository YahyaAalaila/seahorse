"""Common affine coordinates for the frozen real-data campaign.

This module intentionally has no model dependency.  It fits only train-stream
statistics, maps declared longitude/latitude rectangles to ``[0, 1]^2``, and
leaves observation/exposure metadata in their original coordinates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Mapping

import numpy as np


def _bounds(bounds: Mapping[str, Any] | tuple[float, float, float, float] | list[float]) -> tuple[float, float, float, float]:
    if isinstance(bounds, Mapping):
        longitude, latitude = bounds.get("longitude"), bounds.get("latitude")
        if not (isinstance(longitude, (list, tuple)) and isinstance(latitude, (list, tuple)) and len(longitude) == len(latitude) == 2):
            raise ValueError("spatial_bounds mapping requires longitude=[min,max], latitude=[min,max].")
        lo, hi, bottom, top = map(float, (*longitude, *latitude))
    else:
        if len(bounds) != 4:
            raise ValueError("spatial_bounds must be (lon_min, lon_max, lat_min, lat_max).")
        lo, hi, bottom, top = map(float, bounds)
    if not np.isfinite([lo, hi, bottom, top]).all() or not (hi > lo and top > bottom):
        raise ValueError("spatial_bounds must be finite with positive longitude and latitude spans.")
    return lo, hi, bottom, top


@dataclass(frozen=True)
class CoordinateTransform:
    """Fitted common affine map from prepared source units to model inputs."""

    version: int
    history: int
    input_time_unit: str | None
    input_time_axis: str | None
    spatial_bounds_lon_lat: tuple[float, float, float, float]
    time_mean: float
    time_scale: float
    space_offset: tuple[float, float]
    space_scale: tuple[float, float]
    log_raw_volume_per_normalized_volume: float
    train_sequences: int
    train_events: int
    positive_train_intervals: int
    nonpositive_train_intervals: int
    median_positive_train_interval: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CoordinateTransform":
        copied = dict(value)
        for key in ("spatial_bounds_lon_lat", "space_offset", "space_scale"):
            if key in copied:
                copied[key] = tuple(copied[key])
        return cls(**copied)


def fit_coordinates(
    train_sequences: Iterable[Mapping[str, Any]],
    spatial_bounds: Mapping[str, Any] | tuple[float, float, float, float] | list[float],
    history: int = 20,
) -> CoordinateTransform:
    """Fit the campaign affine transform from a train-only source stream.

    Each event contributes once to the event count and each within-sequence
    positive interval contributes once to the global median.  Calendar/source
    boundaries are never crossed.  The median buffer is intentionally exact
    (rather than reservoir-sampled) for reproducibility.
    """
    if int(history) < 1:
        raise ValueError("history must be >= 1.")
    lo, hi, bottom, top = _bounds(spatial_bounds)
    positive_parts: list[np.ndarray] = []
    n_sequences = n_events = n_positive = n_nonpositive = 0
    time_unit: str | None = None
    time_axis: str | None = None
    for sequence in train_sequences:
        times = np.asarray(sequence["times"], dtype=np.float64).reshape(-1)
        locations = np.asarray(sequence["locations"], dtype=np.float64)
        if len(times) == 0 and locations.size == 0:
            locations = locations.reshape(0, 2)
        if locations.ndim != 2 or locations.shape != (len(times), 2):
            raise ValueError("Every coordinate-fit sequence must have locations shaped (n_events, 2).")
        if not np.isfinite(times).all() or not np.isfinite(locations).all():
            raise ValueError("Coordinate-fit inputs must be finite.")
        outside = (locations[:, 0] < lo) | (locations[:, 0] > hi) | (locations[:, 1] < bottom) | (locations[:, 1] > top)
        if outside.any():
            raise ValueError("Train event lies outside the declared spatial rectangle; do not clip or replace the domain with empirical extrema.")
        n_sequences += 1
        n_events += len(times)
        if time_unit is None and sequence.get("time_unit") is not None:
            time_unit = str(sequence["time_unit"])
        if time_axis is None and sequence.get("time_axis") is not None:
            time_axis = str(sequence["time_axis"])
        delta = np.diff(times)
        positive = delta[delta > 0]
        if positive.size:
            positive_parts.append(positive)
            n_positive += int(positive.size)
        n_nonpositive += int((delta <= 0).sum())
    if not n_events or not positive_parts:
        raise ValueError("Coordinate fit requires at least one positive train inter-event interval.")
    positive_intervals = np.concatenate(positive_parts)
    median = float(np.median(positive_intervals))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("Median positive train inter-event interval must be finite and positive.")
    time_scale = float(int(history) * median)
    dx, dy = hi - lo, top - bottom
    return CoordinateTransform(
        version=1,
        history=int(history),
        input_time_unit=time_unit,
        input_time_axis=time_axis,
        spatial_bounds_lon_lat=(lo, hi, bottom, top),
        time_mean=0.0,
        time_scale=time_scale,
        space_offset=(lo, bottom),
        space_scale=(dx, dy),
        log_raw_volume_per_normalized_volume=float(np.log(time_scale * dx * dy)),
        train_sequences=n_sequences,
        train_events=n_events,
        positive_train_intervals=n_positive,
        nonpositive_train_intervals=n_nonpositive,
        median_positive_train_interval=median,
    )


def transform_sequences(
    sequences: Iterable[Mapping[str, Any]], artifact: CoordinateTransform | Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    """Yield affine-transformed copies while retaining raw exposure and traceability."""
    transform = artifact if isinstance(artifact, CoordinateTransform) else CoordinateTransform.from_dict(artifact)
    lo, bottom = transform.space_offset
    dx, dy = transform.space_scale
    for sequence in sequences:
        copied = dict(sequence)
        times = np.asarray(sequence["times"], dtype=np.float64)
        locations = np.asarray(sequence["locations"], dtype=np.float64)
        if times.size == 0 and locations.size == 0:
            locations = locations.reshape(0, 2)
        if locations.ndim != 2 or locations.shape != (times.size, 2):
            raise ValueError("Every transformed sequence must have locations shaped (n_events, 2).")
        copied["times"] = ((times - transform.time_mean) / transform.time_scale).tolist()
        copied["locations"] = np.column_stack(((locations[:, 0] - lo) / dx, (locations[:, 1] - bottom) / dy)).tolist()
        metadata = dict(copied.get("observation_metadata") or {})
        metadata["campaign_coordinate_transform"] = {
            "version": transform.version,
            "time_mean": transform.time_mean,
            "time_scale": transform.time_scale,
            "space_offset": list(transform.space_offset),
            "space_scale": list(transform.space_scale),
            "log_raw_volume_per_normalized_volume": transform.log_raw_volume_per_normalized_volume,
            "event_coordinates": "affine_campaign_input",
            "observation_bounds_remain_raw": True,
        }
        copied["observation_metadata"] = metadata
        yield copied


def invert_samples(
    times: Any, locations: Any, artifact: CoordinateTransform | Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Invert transformed event times and [longitude, latitude] locations."""
    transform = artifact if isinstance(artifact, CoordinateTransform) else CoordinateTransform.from_dict(artifact)
    transformed_times = np.asarray(times, dtype=np.float64)
    transformed_locations = np.asarray(locations, dtype=np.float64)
    if transformed_locations.shape[-1] != 2:
        raise ValueError("locations must have final dimension 2.")
    raw_times = transformed_times * transform.time_scale + transform.time_mean
    raw_locations = transformed_locations * np.asarray(transform.space_scale) + np.asarray(transform.space_offset)
    return raw_times, raw_locations


__all__ = ["CoordinateTransform", "fit_coordinates", "invert_samples", "transform_sequences"]
