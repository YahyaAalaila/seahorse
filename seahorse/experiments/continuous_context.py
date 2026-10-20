"""Continuous, split-bounded fixed-history contexts for prepared observations.

Prepared calendar records are observation windows, not independent point-process
realizations.  This adapter joins adjacent observed windows within one supplied
split before extracting H+1 next-event contexts.  Empty windows therefore
contribute elapsed time, while a missing window or a split boundary resets the
history explicitly.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from seahorse.data.dataset import _strict_float32_times
from seahorse.experiments.count_baselines import prepared_windows


@dataclass(frozen=True)
class _SourceSpan:
    """One calendar source record's event range in a continuous segment."""

    event_start: int
    event_end: int
    source_sequence_index: int
    global_event_start: int
    window_start: str
    window_end: str
    t0: float
    t1: float
    calendar_start: float
    calendar_end: float


@dataclass(frozen=True)
class _Segment:
    """Numerical storage for one contiguous observed-calendar segment."""

    times: np.ndarray  # global, normalized float64 time
    locations: np.ndarray
    sources: tuple[_SourceSpan, ...]
    source_starts: tuple[int, ...]
    global_event_start: int


def _metadata_value(record: Mapping[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    metadata = record.get("observation_metadata")
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]
    raise ValueError(f"Prepared continuous contexts require frozen {key!r} metadata.")


def _optional_metadata_value(record: Mapping[str, Any], key: str) -> Any | None:
    if key in record:
        return record[key]
    metadata = record.get("observation_metadata")
    return metadata.get(key) if isinstance(metadata, Mapping) else None


def _raw_record_for_calendar(record: Mapping[str, Any], *, time_scale: float, time_mean: float) -> dict[str, Any]:
    """Restore record-local native time before applying the frozen calendar bounds.

    ``transform_sequences`` scales local source time but deliberately leaves
    observation metadata raw.  Calendar concatenation must occur on that raw
    axis, then apply the same scale exactly once to the resulting global axis.
    """
    times = np.asarray(record["times"], dtype=np.float64).reshape(-1) * time_scale + time_mean
    copied = dict(record)
    copied["times"] = times.tolist()
    for key in ("window_start", "window_end", "t0", "t1"):
        copied.setdefault(key, _metadata_value(record, key))
    return copied


def _transform_parameters(transform: Any | None) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return frozen raw-to-model affine coordinates without campaign imports."""
    if transform is None:
        return 1.0, 0.0, np.zeros(2, dtype=np.float64), np.ones(2, dtype=np.float64)
    if hasattr(transform, "time_scale"):
        scale, mean = transform.time_scale, transform.time_mean
        space_offset, space_scale = transform.space_offset, transform.space_scale
    elif isinstance(transform, Mapping):
        scale, mean = transform.get("time_scale"), transform.get("time_mean", 0.0)
        space_offset, space_scale = transform.get("space_offset"), transform.get("space_scale")
    else:
        raise TypeError("transform must be a CoordinateTransform, mapping, or None.")
    scale, mean = float(scale), float(mean)
    if not np.isfinite([scale, mean]).all() or scale <= 0:
        raise ValueError("Coordinate transform needs a finite positive time_scale and finite time_mean.")
    offset = np.asarray(space_offset, dtype=np.float64).reshape(-1)
    spatial_scale = np.asarray(space_scale, dtype=np.float64).reshape(-1)
    if offset.shape != (2,) or spatial_scale.shape != (2,) or not np.isfinite([*offset, *spatial_scale]).all() or np.any(spatial_scale <= 0):
        raise ValueError("Coordinate transform needs finite 2D space_offset and positive space_scale.")
    return scale, mean, offset, spatial_scale


class ContinuousPreparedContextDataset(Dataset):
    """Lazy H+1 contexts over calendar-contiguous prepared records.

    ``records`` must all come from exactly one frozen split.  They may already
    have campaign-transformed event coordinates.  Time is restored to native
    record-local units, anchored to ``window_start``/``t0``, concatenated only
    across adjacent declared windows, and finally divided by the train-fitted
    transform scale.  Each returned window is rebased to its first history
    event; no target or later event is used to build its condition.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        lookback: int = 20,
        time_unit: str = "day",
        transform: Any | None = None,
        split: str | None = None,
    ):
        if int(lookback) < 1:
            raise ValueError("lookback must be >= 1.")
        self.lookback = int(lookback)
        self.window_length = self.lookback + 1
        self.time_unit = str(time_unit)
        self.split = split
        # ``raw`` means no model-owned normalization remains to be fitted.  The
        # supplied campaign affine map is external, frozen preprocessing; its
        # provenance is carried separately so family fit hooks still run.
        self.coordinate_space = "raw"
        self.source_coordinate_space = "campaign_transformed" if transform is not None else "raw"
        self.normalize_time = self.normalize_space = self.normalize_covariates = False
        self.time_mean, self.time_std = 0.0, 1.0
        self.cov_mean = self.cov_std = None
        time_scale, time_mean, space_offset, space_scale = _transform_parameters(transform)
        self.time_scale, self.global_calendar_offset = time_scale, None
        self._source_time_mean = time_mean
        self._space_offset, self._space_scale = space_offset, space_scale

        # Preserve transformed source records for model-owned train statistics;
        # context arrays below are the only event-scale additional allocation.
        self.raw_sequences = list(records)
        self.stat_sequences = self.raw_sequences
        self.sequences = self.stat_sequences
        self.n_input_sequences = len(self.raw_sequences)
        self.n_empty_sequences = sum(len(np.asarray(row["times"]).reshape(-1)) == 0 for row in self.raw_sequences)

        self._segments: list[_Segment] = []
        self._target_cumulative: list[int] = []
        self.n_observation_gap_resets = 0
        self.n_stream_resets = 0
        global_event = 0
        running_targets = 0
        current_sources: list[_SourceSpan] = []
        current_event_count = 0
        previous_end: float | None = None
        previous_stream: str | None = None

        def finish_segment() -> None:
            nonlocal current_sources, current_event_count, running_targets
            if not current_sources:
                return
            # Allocate exactly one time and one location array per retained
            # segment.  Reconstruct one source record at a time while filling;
            # retaining daily/hourly fragment arrays would double Citi/Uber's
            # event-scale memory at concatenation time.
            times = np.empty(current_event_count, dtype=np.float64)
            # Preserve source coordinates for statistical scoring.  The neural
            # view below reapplies the frozen affine transform and casts to
            # float32 exactly at its existing tensor boundary.
            locs = np.empty((current_event_count, 2), dtype=np.float64)
            for source in current_sources:
                row = self.raw_sequences[source.source_sequence_index]
                raw_row = _raw_record_for_calendar(row, time_scale=time_scale, time_mean=time_mean)
                window = prepared_windows((raw_row,), time_unit=self.time_unit)[0]
                # ``prepared_windows`` validates the frozen bounds but exposes
                # an epoch-absolute convenience time.  Do not subtract that
                # large value after it has rounded sub-second jitter away.
                # Anchor the calendar offset before adding record-local time.
                local_times = np.asarray(raw_row["times"], dtype=np.float64)
                event_times = ((float(window["start"]) - self.global_calendar_offset)
                               + (local_times - float(raw_row["t0"]))) / time_scale
                locations = np.asarray(row["locations"], dtype=np.float64)
                if locations.size == 0:
                    locations = locations.reshape(0, 2)
                if locations.shape != (len(event_times), 2):
                    raise ValueError("Prepared continuous contexts require locations shaped (n_events, 2).")
                times[source.event_start:source.event_end] = event_times
                locs[source.event_start:source.event_end] = locations * self._space_scale + self._space_offset
            if times.size > 1 and np.any(np.diff(times) <= 0):
                raise ValueError("Calendar-contiguous prepared records contain non-increasing global event times.")
            sources = tuple(current_sources)
            segment = _Segment(times=times, locations=locs, sources=sources,
                               source_starts=tuple(source.event_start for source in sources),
                               global_event_start=current_sources[0].global_event_start)
            self._segments.append(segment)
            n_targets = max(0, len(times) - self.lookback)
            running_targets += n_targets
            self._target_cumulative.append(running_targets)
            current_sources, current_event_count = [], 0

        for source_index, row in enumerate(self.raw_sequences):
            raw_row = _raw_record_for_calendar(row, time_scale=time_scale, time_mean=time_mean)
            window = prepared_windows((raw_row,), time_unit=self.time_unit)[0]
            start, end = float(window["start"]), float(window["end"])
            stream = row.get("stream_id")
            stream_key = None if stream is None else str(stream)
            if previous_end is not None:
                if start < previous_end and not np.isclose(start, previous_end, rtol=0., atol=1e-8):
                    raise ValueError("Prepared calendar records are out of order or overlap; do not reorder them implicitly.")
                stream_changed = stream_key != previous_stream
                has_gap = start > previous_end and not np.isclose(start, previous_end, rtol=0., atol=1e-8)
                if stream_changed or has_gap:
                    finish_segment()
                    self.n_stream_resets += int(stream_changed)
                    self.n_observation_gap_resets += int(has_gap)
            if self.global_calendar_offset is None:
                self.global_calendar_offset = start
            locations = np.asarray(row["locations"], dtype=np.float64)
            if locations.size == 0:
                locations = locations.reshape(0, 2)
            n_events = len(window["times"])
            if locations.shape != (n_events, 2):
                raise ValueError("Prepared continuous contexts require locations shaped (n_events, 2).")
            local_start = current_event_count
            current_sources.append(_SourceSpan(
                event_start=local_start, event_end=local_start + n_events,
                source_sequence_index=source_index, global_event_start=global_event,
                window_start=str(_metadata_value(raw_row, "window_start")),
                window_end=str(_metadata_value(raw_row, "window_end")),
                t0=float(_metadata_value(raw_row, "t0")), t1=float(_metadata_value(raw_row, "t1")),
                calendar_start=start - self.global_calendar_offset,
                calendar_end=end - self.global_calendar_offset,
            ))
            current_event_count += n_events
            global_event += n_events
            previous_end, previous_stream = end, stream_key
        finish_segment()

        self.n_retained_sequences = running_targets
        self.n_removed_by_window_length = sum(len(segment.times) <= self.lookback for segment in self._segments)
        self.sequence_filter_counts = {
            "input_calendar_records": self.n_input_sequences,
            "empty_input_calendar_records": self.n_empty_sequences,
            "contiguous_observation_segments": len(self._segments),
            "observation_gap_resets": self.n_observation_gap_resets,
            "stream_resets": self.n_stream_resets,
            "generated_contexts": self.n_retained_sequences,
            "history_length": self.lookback,
            "materialization": "lazy_segment_indexed",
        }
        first = next((segment.locations for segment in self._segments if len(segment.locations)), None)
        dim = int(first.shape[-1]) if first is not None else 2
        self.loc_mean, self.loc_std = np.zeros(dim, np.float32), np.ones(dim, np.float32)

    def __len__(self) -> int:
        return self.n_retained_sequences

    def batch_by_size(self, max_events: int):
        """Yield bounded index ranges without materialising a window list."""
        per_batch = max(1, int(max_events) // self.window_length) if max_events > 0 else 1
        for start in range(0, len(self), per_batch):
            yield range(start, min(start + per_batch, len(self)))

    def _target(self, index: int) -> tuple[_Segment, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment_index = bisect.bisect_right(self._target_cumulative, index)
        before = self._target_cumulative[segment_index - 1] if segment_index else 0
        return self._segments[segment_index], self.lookback + index - before

    @staticmethod
    def _source_for_event(segment: _Segment, event_index: int) -> _SourceSpan:
        # Source starts are cached once per calendar record, not rebuilt for
        # each context.  ``bisect_right`` selects the following nonempty source
        # when one or more empty records share the same event offset.
        source_index = bisect.bisect_right(segment.source_starts, event_index) - 1
        source = segment.sources[source_index]
        if not source.event_start <= event_index < source.event_end:
            raise RuntimeError("Continuous-context source span lookup failed.")
        return source

    def _target_metadata(self, segment: _Segment, target: int) -> dict[str, Any]:
        """Trace the target through the exact source-calendar identity."""
        source = self._source_for_event(segment, target)
        source_event_index = target - source.event_start
        source_record = self.raw_sequences[source.source_sequence_index]
        event_ids = source_record.get("event_ids")
        target_event_id = (
            event_ids[source_event_index]
            if isinstance(event_ids, (list, tuple, np.ndarray)) and len(event_ids) == source.event_end - source.event_start
            else None
        )
        if isinstance(target_event_id, np.generic):
            target_event_id = target_event_id.item()
        return {
            "split": self.split,
            "history_length": self.lookback,
            "target_global_event_index": source.global_event_start + source_event_index,
            "target_source_sequence_index": source.source_sequence_index,
            "target_source_event_index": source_event_index,
            "target_source_event_id": target_event_id,
            "target_observation_window": {"window_start": source.window_start, "window_end": source.window_end,
                                          "t0": source.t0, "t1": source.t1},
        }

    def raw_context(self, index: int) -> dict[str, Any]:
        """Return one H+1 next-event context in source units and float64.

        The first ``lookback`` entries are the history and the final entry is
        the target.  Times are rebased only to the first history *event*, so
        elapsed time through explicit empty windows remains present.  The
        accompanying calendar fields retain the offset from that event back to
        its source observation window, which lets a seasonal model recover a
        fixed phase without deriving anything from the target or test end.
        """
        segment, target = self._target(index)
        begin, end = target - self.lookback, target + 1
        source = self._source_for_event(segment, begin)
        normalized_times = segment.times[begin:end]
        times = (normalized_times - normalized_times[0]) * self.time_scale
        locations = np.asarray(segment.locations[begin:end], dtype=np.float64)
        identity = self._target_metadata(segment, target)
        calendar_offset = float(segment.times[begin] * self.time_scale - source.calendar_start)
        first_record = self.raw_sequences[source.source_sequence_index]
        history = {
            "times": np.asarray(times[:-1], dtype=np.float64),
            "locations": np.asarray(locations[:-1], dtype=np.float64),
            "t0": 0.0,
            # This is the final observed history event, not an exposure end.
            "t1": float(times[-2]),
            "window_start": source.window_start,
            "calendar_window_start": source.window_start,
            "time_unit": self.time_unit,
            "calendar_offset_from_window_start": calendar_offset,
            "time_axis": _optional_metadata_value(first_record, "time_axis"),
            "timezone": _optional_metadata_value(first_record, "timezone"),
        }
        metadata = {
            "continuous_calendar_context": True,
            **identity,
            "target_conditioning": "events strictly before target only; context may cross contiguous observed calendar windows",
            "time_unit": self.time_unit,
            "window_start": source.window_start,
            "calendar_window_start": source.window_start,
            "calendar_window_end": source.window_end,
            "calendar_offset_from_window_start": calendar_offset,
            "calendar_origin_note": "absolute calendar phase = calendar_window_start + calendar_offset_from_window_start + rebased source-unit time",
            "source_coordinate_space": "raw",
        }
        return {
            "times": np.asarray(times, dtype=np.float64),
            "locations": locations,
            "history": history,
            "next_time": float(times[-1]),
            "next_location": np.asarray(locations[-1], dtype=np.float64),
            "target": {"time": float(times[-1]), "location": np.asarray(locations[-1], dtype=np.float64)},
            "observation_metadata": metadata,
        }

    def exposure_records(self):
        """Yield complete observed-calendar segments for count diagnostics.

        Segments retain empty observation windows in their exposure bounds.
        They are intentionally emitted separately: callers must choose whether
        an aggregate is appropriate for a model instead of silently joining
        histories across windows or gaps.
        """
        for segment_index, segment in enumerate(self._segments):
            first, last = segment.sources[0], segment.sources[-1]
            origin = first.calendar_start
            times = segment.times * self.time_scale - origin
            yield {
                "t0": 0.0,
                "t1": float(last.calendar_end - origin),
                "times": np.asarray(times, dtype=np.float64),
                "locations": np.asarray(segment.locations, dtype=np.float64),
                "window_start": first.window_start,
                "window_end": last.window_end,
                "calendar_window_start": first.window_start,
                "calendar_window_end": last.window_end,
                "time_unit": self.time_unit,
                "observation_metadata": {
                    "continuous_calendar_exposure": True,
                    "split": self.split,
                    "segment_index": segment_index,
                    "source_sequence_indices": [source.source_sequence_index for source in segment.sources],
                    "n_observation_windows": len(segment.sources),
                    "n_events": int(len(segment.times)),
                    "contains_empty_observation_windows": any(source.event_start == source.event_end for source in segment.sources),
                    "source_coordinate_space": "raw",
                    "aggregation_policy": "separate observed segment; do not merge point-process histories implicitly",
                },
            }

    def raw_count_context(self, index: int, horizon: float) -> dict[str, Any]:
        """Return an event-triggered, fully observed raw-unit count target.

        The conditioning history is exactly the H events used by
        :meth:`raw_context`, ending at its last history event.  The held-out
        events are then every subsequent event in the same observed segment up
        to ``horizon`` source-time units later.  A request whose endpoint is
        outside the declared observation segment is rejected: lack of an
        observation is never converted into a zero-count outcome.
        """
        horizon = float(horizon)
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("horizon must be a finite positive source-time duration.")
        segment, target = self._target(index)
        begin = target - self.lookback
        raw_segment_times = segment.times * self.time_scale
        rebase = float(raw_segment_times[begin])
        raw_segment_times = raw_segment_times - rebase
        origin = float(raw_segment_times[target - 1])
        end = origin + horizon
        exposure_end = float(segment.sources[-1].calendar_end - rebase)
        if end > exposure_end and not np.isclose(end, exposure_end, rtol=0.0, atol=1e-10):
            raise ValueError(
                "Requested count horizon extends beyond this continuous observed segment; "
                "the unobserved region cannot be scored as zero events."
            )
        # [origin, end): an event at exactly the forecast endpoint belongs to
        # neither this fixed-horizon outcome nor an earlier endpoint bin.
        future_end = int(np.searchsorted(raw_segment_times, end, side="left"))
        history_times = np.asarray(raw_segment_times[begin:target], dtype=np.float64)
        history_locations = np.asarray(segment.locations[begin:target], dtype=np.float64)
        future_times = np.asarray(raw_segment_times[target:future_end], dtype=np.float64)
        future_locations = np.asarray(segment.locations[target:future_end], dtype=np.float64)
        first_source = self._source_for_event(segment, begin)
        first_record = self.raw_sequences[first_source.source_sequence_index]
        calendar_offset = float(rebase - first_source.calendar_start)
        return {
            "origin": origin,
            "end": end,
            "horizon": horizon,
            "history": {
                "times": history_times,
                "locations": history_locations,
                "t0": 0.0,
                "t1": origin,
                "window_start": first_source.window_start,
                "calendar_window_start": first_source.window_start,
                "time_unit": self.time_unit,
                "calendar_offset_from_window_start": calendar_offset,
                "time_axis": _optional_metadata_value(first_record, "time_axis"),
                "timezone": _optional_metadata_value(first_record, "timezone"),
            },
            "true_future": {"times": future_times, "locations": future_locations},
            "observation_metadata": {
                "continuous_calendar_count_context": True,
                **self._target_metadata(segment, target),
                "time_unit": self.time_unit,
                "calendar_window_start": first_source.window_start,
                "calendar_offset_from_window_start": calendar_offset,
                "forecast_endpoint_observed": True,
                "target_conditioning": "events through origin only; future events are truth only",
                "source_coordinate_space": "raw",
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        segment, target = self._target(index)
        begin, end = target - self.lookback, target + 1
        raw_times = segment.times[begin:end]
        times = _strict_float32_times(raw_times - raw_times[0], context=f"ContinuousPreparedContextDataset[target={index}]")
        metadata = {
            "continuous_calendar_context": True,
            **self._target_metadata(segment, target),
            "target_conditioning": "events strictly before target only; context may cross contiguous observed calendar windows",
            "window_time_coordinate": "global_calendar_time_rebased_to_first_history_event",
            "global_calendar_offset": self.global_calendar_offset,
            "global_calendar_offset_coordinate": f"native_{self.time_unit}_from_1970-01-01",
            "time_scale": self.time_scale,
            "source_coordinate_space": self.source_coordinate_space,
            "window_is_not_parent_observation_exposure": True,
        }
        return {"times": torch.tensor(times, dtype=torch.float32),
                "locations": torch.tensor((segment.locations[begin:end] - self._space_offset) / self._space_scale,
                                          dtype=torch.float32),
                "length": self.window_length, "coordinate_space": self.coordinate_space,
                "observation_metadata": metadata}


def build_continuous_context_splits(
    split_records: Mapping[str, Sequence[Mapping[str, Any]]], **kwargs: Any
) -> dict[str, ContinuousPreparedContextDataset]:
    """Build separate datasets, making a split-crossing context impossible by construction."""
    return {split: ContinuousPreparedContextDataset(records, split=split, **kwargs)
            for split, records in split_records.items()}


__all__ = ["ContinuousPreparedContextDataset", "build_continuous_context_splits"]
