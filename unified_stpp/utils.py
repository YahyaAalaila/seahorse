"""Shared utility functions for the unified_stpp package."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _canonicalize_sequence_record(record: dict) -> dict:
    """Return a shallow canonicalized copy of one STPP JSONL record.

    Accepted minimal aliases are intentionally small and benchmark-oriented:
    ``t``/``time`` for ``times`` and ``x``/``y``[/``z``] for ``locations``.
    Existing canonical keys win when already present.
    """
    if not isinstance(record, dict):
        return record

    out = dict(record)

    if "times" not in out and "locations" not in out and "events" in out:
        events = out.get("events")
        if isinstance(events, list):
            times: list = []
            locations: list[list] = []
            marks: list = []
            have_marks = True

            for event in events:
                if not isinstance(event, dict):
                    have_marks = False
                    break

                time_value = None
                for key in ("t", "time", "times"):
                    if key in event:
                        time_value = event[key]
                        break
                if time_value is None:
                    have_marks = False
                    break

                location_value = None
                if "locations" in event:
                    location_value = event["locations"]
                elif "location" in event:
                    location_value = event["location"]
                elif "loc" in event:
                    location_value = event["loc"]
                else:
                    coord_keys = [key for key in ("x", "y", "z") if key in event]
                    if len(coord_keys) >= 2:
                        location_value = [event[key] for key in coord_keys]

                if location_value is None:
                    have_marks = False
                    break

                times.append(time_value)
                locations.append(list(location_value))

                mark_value = None
                for key in ("mark", "marks", "m"):
                    if key in event:
                        mark_value = event[key]
                        break
                if mark_value is None:
                    have_marks = False
                else:
                    marks.append(mark_value)

            if len(times) == len(events) and len(locations) == len(events):
                out["times"] = times
                out["locations"] = locations
                if have_marks and len(marks) == len(events):
                    out["marks"] = marks

    if "times" not in out:
        for key in ("t", "time"):
            if key in out:
                out["times"] = out[key]
                break

    if "locations" not in out:
        if "locs" in out:
            out["locations"] = out["locs"]
        elif "location" in out:
            out["locations"] = out["location"]
        else:
            coord_keys = [key for key in ("x", "y", "z") if key in out]
            if len(coord_keys) >= 2:
                coord_arrays = [out[key] for key in coord_keys]
                out["locations"] = [
                    list(coords) for coords in zip(*coord_arrays, strict=True)
                ]

    return out


def deep_update(base: dict, override: dict) -> None:
    """Recursively update *base* in-place with values from *override*.

    Nested dicts are merged rather than replaced; all other types are
    overwritten directly.
    """
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            deep_update(base[key], val)
        else:
            base[key] = val


def load_jsonl(path) -> list[dict]:
    """Load a newline-delimited JSON file into a list of dicts.

    Dataset records are canonicalized on read so common compact aliases like
    ``t`` and ``x``/``y`` work throughout the benchmark path.
    """
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seqs.append(_canonicalize_sequence_record(json.loads(line)))
    return seqs


def load_splits_dir(
    splits_dir: str,
    datasets: "list[str] | None" = None,
) -> dict[str, tuple]:
    """Load (train/val/test).jsonl from a directory.

    Expects one sub-directory per dataset:
      splits_dir/
        dataset_a/train.jsonl  val.jsonl  test.jsonl
        dataset_b/...
    If the directory itself contains train.jsonl, treats it as a single dataset.

    Parameters
    ----------
    splits_dir : root splits directory.
    datasets   : if given, load only these dataset sub-directories and raise
                 ``ValueError`` if any are missing.
    """
    p = Path(splits_dir)

    if datasets is not None:
        splits = {}
        missing = []
        for d in datasets:
            ds_dir = p / d
            if not ds_dir.is_dir() or not (ds_dir / "train.jsonl").exists():
                missing.append(d)
            else:
                splits[d] = (
                    load_jsonl(ds_dir / "train.jsonl"),
                    load_jsonl(ds_dir / "val.jsonl"),
                    load_jsonl(ds_dir / "test.jsonl") if (ds_dir / "test.jsonl").exists() else None,
                )
        if missing:
            raise ValueError(
                f"Dataset directories not found under {splits_dir!r}: {missing}"
            )
        return splits

    if (p / "train.jsonl").exists():
        return {
            p.name: (
                load_jsonl(p / "train.jsonl"),
                load_jsonl(p / "val.jsonl"),
                load_jsonl(p / "test.jsonl") if (p / "test.jsonl").exists() else None,
            )
        }
    splits = {}
    for ds_dir in sorted(p.iterdir()):
        if ds_dir.is_dir() and (ds_dir / "train.jsonl").exists():
            splits[ds_dir.name] = (
                load_jsonl(ds_dir / "train.jsonl"),
                load_jsonl(ds_dir / "val.jsonl"),
                load_jsonl(ds_dir / "test.jsonl") if (ds_dir / "test.jsonl").exists() else None,
            )
    if not splits:
        raise ValueError(f"No dataset directories with train.jsonl found in {splits_dir}")
    return splits


def parse_overrides(override_list: list[str]) -> dict:
    """Convert ['training.n_epochs=50', 'model.hidden_dim=128'] to nested dict."""
    result = {}
    for item in (override_list or []):
        if "=" not in item:
            print(f"WARNING: ignoring malformed override {item!r} (expected key=value)", file=sys.stderr)
            continue
        key, _, raw_val = item.partition("=")
        try:
            val = int(raw_val)
        except ValueError:
            try:
                val = float(raw_val)
            except ValueError:
                if raw_val.lower() in ("true", "false"):
                    val = raw_val.lower() == "true"
                else:
                    val = raw_val
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return result
