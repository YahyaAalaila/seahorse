"""Native coarse-calendar count baselines.

These models score binned counts, including zero-count cells/bins. They are not
continuous-time point-process likelihoods and must be reported in that lane.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
from datetime import datetime, timedelta

import numpy as np


Window = dict
_CALENDAR_EPOCH = datetime(1970, 1, 1)


def _parse_civil(value: str) -> datetime:
    """Parse prepared calendar bounds without asserting a timezone conversion."""
    cleaned = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    return dt.replace(tzinfo=None)


def prepared_windows(records: Iterable[dict], *, time_unit: str) -> list[Window]:
    """Adapt immutable prepared records using recorded bounds, never event extrema.

    The returned coordinate is in the source's declared native units from a
    fixed civil epoch. This is calendar indexing, not a claim of UTC physical
    time. Empty records are retained.
    """
    unit_seconds = {"day": 86400.0, "hour": 3600.0, "second": 1.0}.get(time_unit)
    if unit_seconds is None:
        raise ValueError(f"Unsupported prepared time_unit {time_unit!r}.")
    output: list[Window] = []
    for row in records:
        required = ("window_start", "window_end", "t0", "t1", "times", "locations")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(
                f"Prepared count route needs frozen observation metadata; missing {missing}."
            )
        begin, end = _parse_civil(str(row["window_start"])), _parse_civil(str(row["window_end"]))
        start = (begin - _CALENDAR_EPOCH).total_seconds() / unit_seconds
        finish = (end - _CALENDAR_EPOCH).total_seconds() / unit_seconds
        t0, t1 = float(row["t0"]), float(row["t1"])
        if not finish > start or not np.isclose(finish - start, t1 - t0, rtol=0.0, atol=1e-8):
            raise ValueError(
                "Prepared calendar bounds disagree with declared native observation duration."
            )
        relative = np.asarray(row["times"], dtype=float)
        if np.any(relative < t0) or np.any(relative >= t1):
            raise ValueError(
                "Prepared event falls outside its declared half-open observation window."
            )
        loc = np.asarray(row["locations"], dtype=float).reshape(-1, 2)
        if len(relative) != len(loc):
            raise ValueError("Prepared times and locations length mismatch.")
        output.append(
            {
                "start": start,
                "end": finish,
                "times": (start + relative - t0).tolist(),
                "locations": loc.tolist(),
                "sequence_id": row.get("sequence_id"),
                "stream_id": row.get("stream_id"),
                "calendar_start": begin.isoformat(sep=" "),
                "calendar_end": end.isoformat(sep=" "),
                "time_unit": time_unit,
                "observation_bounds_source": "prepared.window_start/window_end,t0/t1",
            }
        )
    return output


def load_prepared_windows(
    dataset_id: str, *, root: str | None = None
) -> tuple[dict[str, list[Window]], dict]:
    """Load all immutable prepared JSONL records for count evaluation only."""
    from seahorse.data.prepared import resolve_prepared_dataset

    route = resolve_prepared_dataset(dataset_id, root=root)
    metadata_path = route.view_root / "dataset_meta.json"
    metadata = json.loads(metadata_path.read_text())
    time_unit = metadata.get("time_unit")
    if not isinstance(time_unit, str):
        raise ValueError("Prepared dataset_meta.json has no declared time_unit.")

    def read(path: Path) -> list[dict]:
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]

    return (
        {
            split: prepared_windows(read(path), time_unit=time_unit)
            for split, path in route.split_paths.items()
        },
        {
            "dataset_id": dataset_id,
            "view": route.view,
            "time_unit": time_unit,
            "manifest_sha256": route.manifest_sha256,
            "calendar_time_axis": route.observation.get("time_axis"),
            "timestamp_semantics": route.observation.get("timestamp_semantics"),
        },
    )


def _whole_bins(window: Window, width: float) -> np.ndarray:
    """Calendar bins fully observed by a window; boundary fragments are excluded."""
    start, end = float(window["start"]), float(window["end"])
    first, last = math.ceil(start / width), math.floor(end / width)
    return np.arange(first, last, dtype=int)


def _counts_in_bins(window: Window, bins: np.ndarray, width: float) -> np.ndarray:
    t = np.asarray(window.get("times", []), dtype=float)
    index = np.floor(t / width).astype(int)
    return np.array([(index == b).sum() for b in bins], dtype=float)


def _whole_event_locations(window: Window, width: float) -> np.ndarray:
    """Locations of events whose timestamp belongs to a scored whole calendar bin."""
    times = np.asarray(window.get("times", []), dtype=float)
    loc = np.asarray(window.get("locations", []), dtype=float).reshape(-1, 2)
    if len(times) != len(loc):
        raise ValueError("times and locations must have equal length.")
    bins = _whole_bins(window, width)
    keep = np.isin(np.floor(times / width).astype(int), bins)
    return loc[keep]


def _poisson_nll(counts: np.ndarray, mean: np.ndarray) -> float:
    mean = np.asarray(mean, dtype=float)
    if np.asarray(counts).size == 0:
        return 0.0
    if np.any(mean < 0):
        raise ValueError("Poisson means must be non-negative.")
    if np.any((mean == 0) & (counts > 0)):
        return float("inf")
    safe = np.maximum(mean, 1e-300)
    return float(np.sum(mean - counts * np.log(safe) + np.vectorize(math.lgamma)(counts + 1)))


@dataclass(frozen=True)
class CalendarPoisson:
    bin_width: float
    rate: float
    train_exposure: float

    @classmethod
    def fit(cls, windows: Iterable[Window], *, bin_width: float) -> "CalendarPoisson":
        counts = exposure = 0.0
        for window in windows:
            bins = _whole_bins(window, bin_width)
            counts += _counts_in_bins(window, bins, bin_width).sum()
            exposure += len(bins) * bin_width
        if exposure <= 0:
            raise ValueError("No whole calendar-bin exposure in training data.")
        return cls(bin_width, counts / exposure, exposure)

    def score(self, windows: Iterable[Window]) -> dict:
        nll = exposure = count = 0.0
        for window in windows:
            bins = _whole_bins(window, self.bin_width)
            y = _counts_in_bins(window, bins, self.bin_width)
            nll += _poisson_nll(y, np.full(y.shape, self.rate * self.bin_width))
            exposure += len(bins) * self.bin_width
            count += y.sum()
        n_bins = int(round(exposure / self.bin_width))
        if n_bins == 0:
            return {
                "coarse_count_nll": None,
                "n_events": 0.0,
                "whole_bin_exposure": 0.0,
                "n_scored_bins": 0,
                "status": "unavailable: no complete calendar bins in declared observation windows",
            }
        return {
            "coarse_count_nll": nll,
            "n_events": count,
            "whole_bin_exposure": exposure,
            "n_scored_bins": n_bins,
            "nll_per_bin": nll / n_bins,
        }


@dataclass(frozen=True)
class SeasonalCalendarPoisson:
    bin_width: float
    period_bins: int
    rates: tuple[float, ...]
    train_exposure_by_phase: tuple[float, ...]

    @classmethod
    def fit(
        cls, windows: Iterable[Window], *, bin_width: float, period_bins: int
    ) -> "SeasonalCalendarPoisson":
        if period_bins < 1:
            raise ValueError("period_bins must be positive.")
        counts, exposure = np.zeros(period_bins), np.zeros(period_bins)
        for window in windows:
            bins = _whole_bins(window, bin_width)
            y = _counts_in_bins(window, bins, bin_width)
            phase = bins % period_bins
            np.add.at(counts, phase, y)
            np.add.at(exposure, phase, bin_width)
        if not np.all(exposure > 0):
            raise ValueError("Every seasonal phase needs train exposure.")
        return cls(bin_width, period_bins, tuple(counts / exposure), tuple(exposure))

    def score(self, windows: Iterable[Window]) -> dict:
        nll = exposure = count = 0.0
        rate = np.asarray(self.rates)
        for window in windows:
            bins = _whole_bins(window, self.bin_width)
            y = _counts_in_bins(window, bins, self.bin_width)
            nll += _poisson_nll(y, rate[bins % self.period_bins] * self.bin_width)
            exposure += len(bins) * self.bin_width
            count += y.sum()
        n_bins = int(round(exposure / self.bin_width))
        if n_bins == 0:
            return {
                "coarse_count_nll": None,
                "n_events": 0.0,
                "whole_bin_exposure": 0.0,
                "n_scored_bins": 0,
                "status": "unavailable: no complete calendar bins in declared observation windows",
            }
        return {
            "coarse_count_nll": nll,
            "n_events": count,
            "whole_bin_exposure": exposure,
            "n_scored_bins": n_bins,
            "nll_per_bin": nll / n_bins,
        }


@dataclass(frozen=True)
class CalendarMonthPoisson:
    """Month-of-year rates for day-indexed civil-calendar windows.

    This avoids treating leap-day years as a fixed 365-bin periodic process.
    """

    bin_width: float
    rates: tuple[float, ...]
    train_exposure_by_month: tuple[float, ...]

    @staticmethod
    def _month(bin_index: int) -> int:
        return (_CALENDAR_EPOCH + timedelta(days=int(bin_index))).month - 1

    @classmethod
    def fit(cls, windows: Iterable[Window], *, bin_width: float = 1.0) -> "CalendarMonthPoisson":
        if bin_width != 1.0:
            raise ValueError("CalendarMonthPoisson requires native day bins (bin_width=1).")
        counts, exposure = np.zeros(12), np.zeros(12)
        for window in windows:
            bins = _whole_bins(window, bin_width)
            y = _counts_in_bins(window, bins, bin_width)
            month = np.asarray([cls._month(int(b)) for b in bins])
            np.add.at(counts, month, y)
            np.add.at(exposure, month, 1.0)
        if not np.all(exposure > 0):
            raise ValueError("Every calendar month needs train exposure for month-of-year rates.")
        return cls(bin_width, tuple(counts / exposure), tuple(exposure))

    def score(self, windows: Iterable[Window]) -> dict:
        nll = exposure = count = 0.0
        rates = np.asarray(self.rates)
        for window in windows:
            bins = _whole_bins(window, self.bin_width)
            y = _counts_in_bins(window, bins, self.bin_width)
            means = rates[[self._month(int(b)) for b in bins]]
            nll += _poisson_nll(y, means)
            exposure += len(bins)
            count += y.sum()
        n_bins = int(exposure)
        if n_bins == 0:
            return {
                "coarse_count_nll": None,
                "n_events": 0.0,
                "whole_bin_exposure": 0.0,
                "n_scored_bins": 0,
                "status": "unavailable: no complete calendar bins in declared observation windows",
            }
        return {
            "coarse_count_nll": nll,
            "n_events": count,
            "whole_bin_exposure": exposure,
            "n_scored_bins": n_bins,
            "nll_per_bin": nll / n_bins,
            "seasonality": "calendar_month_of_year",
        }


@dataclass(frozen=True)
class SpatialCellPoisson:
    """Train-fit coarse time×space Poisson cells on a preregistered rectangle."""

    rectangle: tuple[float, float, float, float]
    grid: tuple[int, int]
    bin_width: float
    total_rate: float
    probabilities: tuple[float, ...]
    alpha: float

    @classmethod
    def fit(
        cls,
        windows: Iterable[Window],
        *,
        rectangle: Sequence[float],
        grid: tuple[int, int],
        bin_width: float,
        alpha: float = 0.5,
    ) -> "SpatialCellPoisson":
        xmin, xmax, ymin, ymax = map(float, rectangle)
        if not xmax > xmin or not ymax > ymin or min(grid) < 1:
            raise ValueError("Invalid preregistered rectangle or grid.")
        nx, ny = grid
        counts = np.zeros((nx, ny))
        exposure = 0.0
        total = 0.0
        for window in windows:
            bins = _whole_bins(window, bin_width)
            exposure += len(bins) * bin_width
            loc = _whole_event_locations(window, bin_width)
            if not len(loc):
                continue
            inside = (
                (loc[:, 0] >= xmin) & (loc[:, 0] < xmax) & (loc[:, 1] >= ymin) & (loc[:, 1] < ymax)
            )
            if not bool(inside.all()):
                raise ValueError(
                    "Training events fall outside the preregistered rectangle; choose a declared support before fitting."
                )
            ix = np.minimum(((loc[inside, 0] - xmin) / (xmax - xmin) * nx).astype(int), nx - 1)
            iy = np.minimum(((loc[inside, 1] - ymin) / (ymax - ymin) * ny).astype(int), ny - 1)
            np.add.at(counts, (ix, iy), 1)
            total += inside.sum()
        if exposure <= 0:
            raise ValueError("No whole-bin training exposure.")
        prob = (counts + alpha) / (total + alpha * nx * ny)
        return cls(
            (xmin, xmax, ymin, ymax),
            grid,
            bin_width,
            total / exposure,
            tuple(prob.reshape(-1)),
            alpha,
        )

    def score(self, windows: Iterable[Window]) -> dict:
        xmin, xmax, ymin, ymax = self.rectangle
        nx, ny = self.grid
        prob = np.asarray(self.probabilities).reshape(nx, ny)
        nll = exposure = 0.0
        outside = 0
        for window in windows:
            bins = _whole_bins(window, self.bin_width)
            m = len(bins)
            exposure += m * self.bin_width
            cell_counts = np.zeros((m, nx, ny))
            times = np.asarray(window.get("times", []), dtype=float)
            loc = _whole_event_locations(window, self.bin_width)
            whole = np.isin(np.floor(times / self.bin_width).astype(int), bins)
            event_bins = np.floor(times[whole] / self.bin_width).astype(int)
            if len(loc):
                inside = (
                    (loc[:, 0] >= xmin)
                    & (loc[:, 0] < xmax)
                    & (loc[:, 1] >= ymin)
                    & (loc[:, 1] < ymax)
                )
                outside += int((~inside).sum())
                ix = np.minimum(((loc[inside, 0] - xmin) / (xmax - xmin) * nx).astype(int), nx - 1)
                iy = np.minimum(((loc[inside, 1] - ymin) / (ymax - ymin) * ny).astype(int), ny - 1)
                ib = np.searchsorted(bins, event_bins[inside])
                np.add.at(cell_counts, (ib, ix, iy), 1)
            # Repeated full calendar bins contribute all spatial zero cells.
            nll += _poisson_nll(
                cell_counts,
                np.broadcast_to(self.total_rate * prob * self.bin_width, cell_counts.shape),
            )
        if outside:
            nll = float("inf")  # zero probability under the declared spatial support
        n_bins = int(round(exposure / self.bin_width))
        if n_bins == 0:
            return {
                "coarse_spatiotemporal_count_nll": None,
                "whole_bin_exposure": 0.0,
                "n_scored_bins": 0,
                "outside_preregistered_rectangle": 0,
                "status": "unavailable: no complete calendar bins in declared observation windows",
            }
        return {
            "coarse_spatiotemporal_count_nll": nll,
            "whole_bin_exposure": exposure,
            "n_scored_bins": n_bins,
            "outside_preregistered_rectangle": outside,
            "nll_per_time_space_cell": nll / (n_bins * nx * ny),
        }


def select_spatial_grid(
    train: Iterable[Window],
    val: Iterable[Window],
    *,
    rectangle: Sequence[float],
    grids: Sequence[tuple[int, int]],
    bin_width: float,
    alpha: float = 0.5,
) -> SpatialCellPoisson:
    """Validation chooses only among predeclared grids; test is never consulted."""
    train, val = list(train), list(val)
    candidates = [
        SpatialCellPoisson.fit(
            train, rectangle=rectangle, grid=tuple(g), bin_width=bin_width, alpha=alpha
        )
        for g in grids
    ]
    return min(candidates, key=lambda model: model.score(val)["coarse_spatiotemporal_count_nll"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit/score native calendar-count controls from JSON windows."
    )
    parser.add_argument("--train", type=Path)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--dataset", help="prepared:<cohort>:<view> immutable JSONL route")
    parser.add_argument(
        "--prepared-root", help="prepared corpus root; defaults to SEAHORSE_PREPARED_DATA_ROOT"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, required=True)
    parser.add_argument("--period-bins", type=int, default=7)
    parser.add_argument("--rectangle", nargs=4, type=float)
    parser.add_argument("--grids", default="8x8,16x16")
    args = parser.parse_args(argv)
    prepared_info = None
    if args.dataset:
        if any(value is not None for value in (args.train, args.val, args.test)):
            parser.error("--dataset is exclusive with --train/--val/--test")
        prepared, prepared_info = load_prepared_windows(args.dataset, root=args.prepared_root)
        train, val, test = prepared["train"], prepared["val"], prepared["test"]
    else:
        if any(value is None for value in (args.train, args.val, args.test)):
            parser.error("provide all --train/--val/--test, or --dataset")

        def read(path: Path, split: str) -> list[Window]:
            data = json.loads(path.read_text())
            return data.get("windows", data.get("splits", {}).get(split, []))

        train, val, test = read(args.train, "train"), read(args.val, "val"), read(args.test, "test")
    constant = CalendarPoisson.fit(train, bin_width=args.bin_width)
    seasonal = SeasonalCalendarPoisson.fit(
        train, bin_width=args.bin_width, period_bins=args.period_bins
    )
    result = {
        "provenance": {
            "metric": "whole-calendar-bin count likelihood; not continuous-time likelihood",
            "bin_width": args.bin_width,
            "prepared": prepared_info,
        },
        "poisson": asdict(constant),
        "poisson_test": constant.score(test),
        "seasonal": asdict(seasonal),
        "seasonal_test": seasonal.score(test),
    }
    if prepared_info and prepared_info["time_unit"] == "day" and args.bin_width == 1.0:
        monthly = CalendarMonthPoisson.fit(train)
        result["calendar_month"] = asdict(monthly)
        result["calendar_month_test"] = monthly.score(test)
    if args.rectangle:
        grids = [tuple(map(int, g.split("x"))) for g in args.grids.split(",")]
        spatial = select_spatial_grid(
            train, val, rectangle=args.rectangle, grids=grids, bin_width=args.bin_width
        )
        result["spatial"] = asdict(spatial)
        result["spatial_test"] = spatial.score(test)
    else:
        result["spatial"] = {
            "status": "not fitted: no preregistered rectangle; use total-count track (e.g. FPA California)."
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
