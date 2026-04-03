"""Shared utility functions for the unified_stpp package."""

from __future__ import annotations

import json
import sys
from pathlib import Path


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
    """Load a newline-delimited JSON file into a list of dicts."""
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seqs.append(json.loads(line))
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
