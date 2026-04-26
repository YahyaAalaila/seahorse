#!/usr/bin/env python3
"""Rebuild HawkesNest easy-hard v2 JSONL into short equal-time windows."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_SUITE_NAMES = ("combined", "sweep_E", "sweep_H")


@dataclass(frozen=True)
class RepairStats:
    seed: int
    source_npz: str
    n_time_ties_repaired: int
    min_original_dt: float
    min_repaired_dt: float
    time_repair_policy: str

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "source_npz": self.source_npz,
            "n_time_ties_repaired": self.n_time_ties_repaired,
            "min_original_dt": self.min_original_dt,
            "min_repaired_dt": self.min_repaired_dt,
            "time_repair_policy": self.time_repair_policy,
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default="data/hawkesnest_easy_hard_v2",
        help="Root containing combined/sweep_E/sweep_H suite directories.",
    )
    p.add_argument(
        "--n-windows",
        type=int,
        default=20,
        help="Number of equal-time windows to carve from each 8000-event source sequence.",
    )
    p.add_argument(
        "--train-windows",
        type=int,
        default=16,
        help="Number of early windows assigned to train.",
    )
    p.add_argument(
        "--val-windows",
        type=int,
        default=2,
        help="Number of middle windows assigned to val.",
    )
    p.add_argument(
        "--test-windows",
        type=int,
        default=2,
        help="Number of late windows assigned to test.",
    )
    p.add_argument(
        "--suite",
        action="append",
        dest="suites",
        help="Optional subset of suite names to rebuild. Defaults to all easy-hard v2 suites.",
    )
    return p.parse_args(argv)


def _seed_from_source_name(name: str) -> int:
    match = re.search(r"_r(\d+)\.npz$", name)
    if match is None:
        raise ValueError(f"Could not parse seed from source file name {name!r}")
    return int(match.group(1))


def _json_default(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def _summarize_source_times(times: np.ndarray, *, source_name: str, seed: int) -> RepairStats:
    times64 = np.asarray(times, dtype=np.float64).reshape(-1)
    if times64.size <= 1:
        return RepairStats(
            seed=seed,
            source_npz=source_name,
            n_time_ties_repaired=0,
            min_original_dt=float("inf"),
            min_repaired_dt=float("inf"),
            time_repair_policy="none",
        )

    raw_dt = np.diff(times64)
    if (raw_dt <= 0).any():
        bad = raw_dt <= 0
        first_bad = int(np.flatnonzero(bad)[0] + 1)
        raise ValueError(
            f"Non-increasing source times in {source_name}; "
            f"seed={seed}, event_index={first_bad}, bad_count={int(bad.sum())}, "
            f"min_raw_dt={float(raw_dt.min()):.6e}"
        )

    repaired = times64.copy()
    repair_count = 0
    for idx in range(1, repaired.size):
        if not (np.float32(repaired[idx]) > np.float32(repaired[idx - 1])):
            repaired[idx] = max(
                repaired[idx],
                np.nextafter(repaired[idx - 1], np.float64(np.inf)),
            )
            repair_count += 1

    repaired_dt = np.diff(repaired)
    if (repaired_dt <= 0).any():
        bad = repaired_dt <= 0
        first_bad = int(np.flatnonzero(bad)[0] + 1)
        raise ValueError(
            f"Time repair failed in {source_name}; "
            f"seed={seed}, event_index={first_bad}, bad_count={int(bad.sum())}, "
            f"min_repaired_dt={float(repaired_dt.min()):.6e}"
        )

    return RepairStats(
        seed=seed,
        source_npz=source_name,
        n_time_ties_repaired=repair_count,
        min_original_dt=float(raw_dt.min()),
        min_repaired_dt=float(repaired_dt.min()),
        time_repair_policy="nextafter_forward_pass" if repair_count else "none",
    )


def _split_for_chunk(chunk_idx: int, *, train_windows: int, val_windows: int, test_windows: int) -> str:
    if chunk_idx < train_windows:
        return "train"
    if chunk_idx < train_windows + val_windows:
        return "val"
    if chunk_idx < train_windows + val_windows + test_windows:
        return "test"
    raise ValueError(f"Chunk index {chunk_idx} exceeds configured window split")


def _chunk_rows(
    *,
    times: np.ndarray,
    locations: np.ndarray,
    t_window: float,
    n_windows: int,
) -> list[tuple[int, float, float, np.ndarray, np.ndarray]]:
    edges = np.linspace(0.0, float(t_window), n_windows + 1, dtype=np.float64)
    rows: list[tuple[int, float, float, np.ndarray, np.ndarray]] = []
    for chunk_idx, (t_lo, t_hi) in enumerate(zip(edges[:-1], edges[1:])):
        if chunk_idx == n_windows - 1:
            mask = (times >= t_lo) & (times <= t_hi)
        else:
            mask = (times >= t_lo) & (times < t_hi)
        rows.append((chunk_idx, float(t_lo), float(t_hi), times[mask], locations[mask]))
    return rows


def _rebuild_config(
    *,
    suite_name: str,
    suite_root: Path,
    config_label: str,
    expected_seeds: int,
    n_windows: int,
    train_windows: int,
    val_windows: int,
    test_windows: int,
) -> list[dict[str, object]]:
    config_dir = suite_root / "jsonl" / config_label
    source_paths = sorted((suite_root / "sequences").glob(f"{config_label}_r*.npz"))
    if len(source_paths) != expected_seeds:
        raise FileNotFoundError(
            f"Expected {expected_seeds} sources for {suite_name}/{config_label}, found {len(source_paths)}"
        )

    rows_by_split: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    manifest_rows: list[dict[str, object]] = []
    repair_log: list[dict[str, object]] = []

    for source_path in source_paths:
        with np.load(source_path) as npz:
            times = np.asarray(npz["times"], dtype=np.float64).reshape(-1)
            locations = np.asarray(npz["locations"], dtype=np.float32)
            t_window = float(np.asarray(npz["T_window"]).reshape(()))
        if times.shape[0] != locations.shape[0]:
            raise ValueError(
                f"Time/location length mismatch in {source_path.name}: "
                f"{times.shape[0]} vs {locations.shape[0]}"
            )
        if locations.ndim != 2:
            raise ValueError(f"Expected 2D locations in {source_path.name}, got shape {locations.shape}")

        seed = _seed_from_source_name(source_path.name)
        repair_stats = _summarize_source_times(times, source_name=source_path.name, seed=seed)
        repair_log.append(repair_stats.to_dict())

        for chunk_idx, t_lo, t_hi, chunk_times, chunk_locs in _chunk_rows(
            times=times,
            locations=locations,
            t_window=t_window,
            n_windows=n_windows,
        ):
            split = _split_for_chunk(
                chunk_idx,
                train_windows=train_windows,
                val_windows=val_windows,
                test_windows=test_windows,
            )
            rows_by_split[split].append(
                {
                    "times": chunk_times.tolist(),
                    "locations": chunk_locs.tolist(),
                }
            )
            manifest_rows.append(
                {
                    "suite": suite_name,
                    "config": config_label,
                    "seed": seed,
                    "source_npz": source_path.name,
                    "n_time_ties_repaired": repair_stats.n_time_ties_repaired,
                    "min_original_dt": repair_stats.min_original_dt,
                    "min_repaired_dt": repair_stats.min_repaired_dt,
                    "time_repair_policy": repair_stats.time_repair_policy,
                    "chunk_idx": chunk_idx,
                    "t_lo": t_lo,
                    "t_hi": t_hi,
                    "n_events": int(chunk_times.shape[0]),
                    "split": split,
                }
            )

    for split_name in ("train", "val", "test"):
        _write_jsonl(config_dir / f"{split_name}.jsonl", rows_by_split[split_name])
    _write_jsonl(config_dir / "manifest.jsonl", manifest_rows)
    return repair_log


def _rebuild_suite(
    suite_root: Path,
    *,
    n_windows: int,
    train_windows: int,
    val_windows: int,
    test_windows: int,
) -> dict[str, list[dict[str, object]]]:
    metadata_path = suite_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {suite_root}")
    metadata = json.loads(metadata_path.read_text())
    suite_name = str(metadata["suite"])
    expected_seeds = int(metadata.get("n_seeds", 5))

    repair_log: dict[str, list[dict[str, object]]] = {}
    for level in metadata.get("levels", []):
        config_label = str(level["label"])
        repair_log[config_label] = _rebuild_config(
            suite_name=suite_name,
            suite_root=suite_root,
            config_label=config_label,
            expected_seeds=expected_seeds,
            n_windows=n_windows,
            train_windows=train_windows,
            val_windows=val_windows,
            test_windows=test_windows,
        )

    metadata["serialization"] = {
        "strategy": "equal_time_windows",
        "n_windows": n_windows,
        "split_windows": {
            "train": train_windows,
            "val": val_windows,
            "test": test_windows,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    (suite_root / "repair_log.json").write_text(json.dumps(repair_log, indent=2) + "\n")
    return repair_log


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    total_windows = args.train_windows + args.val_windows + args.test_windows
    if total_windows != args.n_windows:
        raise ValueError(
            f"Split windows must sum to n_windows; got {args.train_windows}+{args.val_windows}+"
            f"{args.test_windows}!={args.n_windows}"
        )

    root = Path(args.root).expanduser().resolve()
    suites = args.suites if args.suites else list(_SUITE_NAMES)
    for suite_name in suites:
        suite_root = root / suite_name
        if not suite_root.is_dir():
            raise FileNotFoundError(f"Missing suite directory {suite_root}")
        _rebuild_suite(
            suite_root,
            n_windows=args.n_windows,
            train_windows=args.train_windows,
            val_windows=args.val_windows,
            test_windows=args.test_windows,
        )
        print(f"[reprocess] rebuilt {suite_root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
