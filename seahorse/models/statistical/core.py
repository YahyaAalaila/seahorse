"""Shared validation and geometry for native statistical STPPs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np


Support = tuple[float, float, float, float]


def as_support(value: Sequence[float]) -> Support:
    """Validate a declared rectangular spatial support; never estimate one."""
    if len(value) != 4:
        raise ValueError("support must be (xmin, xmax, ymin, ymax).")
    xmin, xmax, ymin, ymax = (float(x) for x in value)
    if (not np.isfinite([xmin, xmax, ymin, ymax]).all()
            or not xmax > xmin or not ymax > ymin
            or not np.isfinite((xmax - xmin) * (ymax - ymin))):
        raise ValueError("support must have positive area.")
    return xmin, xmax, ymin, ymax


def area(support: Support) -> float:
    xmin, xmax, ymin, ymax = support
    return (xmax - xmin) * (ymax - ymin)


def contains(support: Support, locations: np.ndarray) -> np.ndarray:
    """Closed declared rectangle membership for raw observation coordinates.

    Spatial bounds are preprocessing support, not array-bin edges.  Their
    upstream Rectangle implementation admits both boundary faces, and a
    continuous density has the same one-sided value at a measure-zero edge.
    """
    xmin, xmax, ymin, ymax = support
    x = locations[:, 0]
    y = locations[:, 1]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


@dataclass(frozen=True)
class SequenceData:
    """One independent observation window.

    Events are a simple process: strictly increasing timestamps and no
    cross-window excitation.  Bounds must be supplied by the prepared data,
    rather than inferred from the first/last event.
    """

    t0: float
    t1: float
    times: np.ndarray
    locations: np.ndarray
    calendar_offset: float = 0.0

    @classmethod
    def from_mapping(cls, seq: Mapping, support: Support, *, require_window: bool = True) -> "SequenceData":
        needed = ("t0", "t1", "times", "locations") if require_window else ("times", "locations")
        missing = [name for name in needed if name not in seq]
        if missing:
            raise ValueError(f"Sequence is missing declared fields: {missing}.")
        t0 = float(seq.get("t0", -np.inf))
        t1 = float(seq.get("t1", np.inf))
        times = np.asarray(seq["times"], dtype=float)
        locations = np.asarray(seq["locations"], dtype=float)
        if locations.size == 0:
            locations = locations.reshape(0, 2)
        if times.ndim != 1 or locations.ndim != 2 or locations.shape[1] != 2 or len(times) != len(locations):
            raise ValueError("times and locations must have matching shapes (n,) and (n, 2).")
        if require_window and (not np.isfinite([t0, t1]).all() or not t1 > t0):
            raise ValueError("Observation endpoint t1 must be strictly after t0.")
        if not np.isfinite(times).all() or not np.isfinite(locations).all():
            raise ValueError("Events must be finite.")
        if len(times) > 1 and not np.all(np.diff(times) > 0):
            raise ValueError("Event times must be strictly chronological; ties require an explicit policy.")
        if require_window and (np.any(times < t0) or np.any(times >= t1)):
            raise ValueError("Events must lie in their declared half-open observation window [t0, t1).")
        if not bool(contains(support, locations).all()):
            raise ValueError("Events outside the declared spatial support have zero intensity.")
        return cls(t0=t0, t1=t1, times=times, locations=locations, calendar_offset=calendar_offset(seq))


def calendar_offset(seq: Mapping) -> float:
    """Day offset for prepared records whose event times are window-relative.

    A naive ISO origin denotes the source's declared local civil clock. An
    offset-aware ISO origin is converted to UTC. Neither case guesses a zone.
    Records without a calendar origin already use the caller's declared axis.
    """
    origin = seq.get("window_start", seq.get("calendar_window_start"))
    offset = float(seq.get("calendar_offset_from_window_start", 0.0))
    if not np.isfinite(offset):
        raise ValueError("calendar_offset_from_window_start must be finite.")
    if origin is None:
        return offset
    unit = str(seq.get("time_unit", "days")).lower()
    # Synthetic controls carry descriptive record IDs such as ``synthetic:0``
    # rather than civil timestamps.  They are independent horizon-one
    # realizations, so the label must not be parsed as a date or create a
    # calendar phase; its only valid offset is the explicitly supplied one.
    if unit == "synthetic_horizon" and str(origin).startswith("synthetic:"):
        return offset
    if unit not in {"day", "days", "d"}:
        raise ValueError("Calendar-origin statistical records currently require time units in days.")
    date = datetime.fromisoformat(str(origin).replace("Z", "+00:00"))
    epoch = datetime(1970, 1, 1)
    if date.tzinfo is not None:
        date = date.astimezone(timezone.utc)
        epoch = epoch.replace(tzinfo=timezone.utc)
    return (date - epoch).total_seconds() / 86400.0 + offset


def forecast_origin(history: Mapping, times: np.ndarray) -> float:
    """End of observed history, including any observed event-free interval."""
    origin = history.get("t1", times[-1] if len(times) else None)
    if origin is None or not np.isfinite(float(origin)):
        raise ValueError("Forecast history needs a finite observation end t1 or an event.")
    if len(times) and float(origin) < times[-1]:
        raise ValueError("Forecast origin cannot precede the last observed event.")
    return float(origin)


def sequences(values: Iterable[Mapping], support: Support) -> list[SequenceData]:
    data = [SequenceData.from_mapping(value, support) for value in values]
    if not data:
        raise ValueError("At least one independently observed training sequence is required.")
    return data


def history_arrays(history: Mapping, support: Support) -> tuple[np.ndarray, np.ndarray]:
    """Validate a walk-forward prefix.  ``last_time`` is its last event."""
    if "times" not in history or "locations" not in history:
        raise ValueError("History must contain times and locations.")
    times = np.asarray(history["times"], dtype=float)
    locations = np.asarray(history["locations"], dtype=float)
    if locations.size == 0:
        locations = locations.reshape(0, 2)
    if times.ndim != 1 or locations.ndim != 2 or locations.shape[1] != 2 or len(times) != len(locations):
        raise ValueError("History times and locations must have matching shapes (n,) and (n, 2).")
    if len(times) and (not np.isfinite(times).all() or not np.isfinite(locations).all()):
        raise ValueError("History events must be finite.")
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("History must be strictly chronological.")
    if not bool(contains(support, locations).all()):
        raise ValueError("History events outside the declared support are invalid.")
    return times, locations


def json_array(value: np.ndarray) -> list:
    return np.asarray(value, dtype=float).tolist()
