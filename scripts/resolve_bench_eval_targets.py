#!/usr/bin/env python
"""Resolve per-run evaluation targets from one benchmark output directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _resolve_split_paths(
    *,
    splits_dir: Path,
    dataset_id: str,
    split: str,
) -> tuple[Path, Path | None]:
    if (splits_dir / "train.jsonl").exists():
        data_path = splits_dir / f"{split}.jsonl"
        train_path = splits_dir / "train.jsonl"
    else:
        ds_dir = splits_dir / dataset_id
        data_path = ds_dir / f"{split}.jsonl"
        train_path = ds_dir / "train.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Expected evaluation split '{split}.jsonl' for dataset {dataset_id!r} at {data_path}"
        )
    if not train_path.exists():
        train_path = None
    return data_path.resolve(), None if train_path is None else train_path.resolve()


def build_targets(
    *,
    bench_root: Path,
    split: str,
) -> list[dict[str, Any]]:
    bench_root = bench_root.expanduser().resolve()
    meta_path = bench_root / "bench_meta.json"
    cell_index_path = bench_root / "cell_index.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"bench_meta.json not found under {bench_root}")
    if not cell_index_path.exists():
        raise FileNotFoundError(f"cell_index.json not found under {bench_root}")

    meta = _load_json(meta_path)
    cell_index = _load_json(cell_index_path)
    data_manifest_path = bench_root / "data_manifest.json"
    data_manifest = (
        _load_json(data_manifest_path)
        if data_manifest_path.exists()
        else meta.get("data_manifest", {})
    )
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path} must contain a JSON object.")
    if not isinstance(cell_index, list):
        raise ValueError(f"{cell_index_path} must contain a JSON list.")
    if not isinstance(data_manifest, dict):
        data_manifest = {}

    splits_dir_raw = meta.get("splits_dir")
    bench_id = meta.get("bench_id") or bench_root.name
    if not splits_dir_raw:
        raise ValueError(f"{meta_path} does not contain 'splits_dir'.")
    splits_dir = Path(str(splits_dir_raw)).expanduser().resolve()
    requested = data_manifest.get("requested", {}) if isinstance(data_manifest, dict) else {}
    dataset_ref = requested.get("dataset")
    dataset_revision = requested.get("dataset_revision")

    targets: list[dict[str, Any]] = []
    for row in cell_index:
        if not isinstance(row, dict):
            continue
        run_dir_raw = row.get("run_dir")
        dataset_id = row.get("dataset_id")
        preset = row.get("preset")
        seed = row.get("seed")
        if not run_dir_raw or not dataset_id or not preset or seed is None:
            continue
        run_dir = Path(str(run_dir_raw)).expanduser().resolve()
        if dataset_ref:
            data_path = None
            train_path = None
        else:
            data_path, train_path = _resolve_split_paths(
                splits_dir=splits_dir,
                dataset_id=str(dataset_id),
                split=split,
            )
        targets.append(
            {
                "bench_id": str(bench_id),
                "bench_root": str(bench_root),
                "dataset_id": str(dataset_id),
                "preset": str(preset),
                "seed": int(seed),
                "run_dir": str(run_dir),
                "data_path": None if data_path is None else str(data_path),
                "train_data": None if train_path is None else str(train_path),
                "split": str(split),
                "dataset_ref": None if dataset_ref is None else str(dataset_ref),
                "dataset_revision": (
                    None if dataset_revision is None else str(dataset_revision)
                ),
            }
        )
    return targets


def _emit_tsv(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        fields = (
            row["bench_id"],
            row["dataset_id"],
            row["preset"],
            str(row["seed"]),
            row["run_dir"],
            row["data_path"] or "",
            row["train_data"] or "",
            row["split"],
            row["dataset_ref"] or "",
            row["dataset_revision"] or "",
        )
        print("\t".join(fields))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", required=True, help="Benchmark output directory.")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Evaluation split to resolve.",
    )
    parser.add_argument(
        "--format",
        default="tsv",
        choices=("tsv", "json"),
        help="Output format.",
    )
    args = parser.parse_args()

    targets = build_targets(
        bench_root=Path(args.bench_root),
        split=str(args.split),
    )
    if args.format == "json":
        json.dump(targets, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _emit_tsv(targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
