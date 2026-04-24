#!/usr/bin/env python
"""Resolve dataset-backed inputs, then invoke `unified_stpp evaluate metrics`."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _resolve_data_paths(
    *,
    data: str | None,
    dataset: str | None,
    dataset_revision: str | None,
    split: str,
    train_data: str | None,
) -> tuple[Path, Path | None]:
    if data is not None:
        data_path = Path(data).expanduser().resolve()
        train_path = None if train_data is None else Path(train_data).expanduser().resolve()
        return data_path, train_path

    if dataset is None:
        raise ValueError("run_eval_metrics requires either --data or --dataset.")

    from unified_stpp.data.hub import download_dataset

    root = Path(
        download_dataset(
            dataset,
            revision=dataset_revision,
        )
    ).expanduser().resolve()
    data_path = root / f"{split}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Resolved dataset root {root} does not contain {split}.jsonl"
        )
    if train_data is not None:
        train_path = Path(train_data).expanduser().resolve()
    else:
        candidate = root / "train.jsonl"
        train_path = candidate.resolve() if candidate.exists() else None
    return data_path, train_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data")
    source.add_argument("--dataset")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--metric-profile", default="core")
    parser.add_argument("--artifact-mode", default="load_or_compute")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-pred", type=int, default=32)
    parser.add_argument("--k-gen", type=int, default=20)
    parser.add_argument("--exact-time-bins", type=int, default=8)
    parser.add_argument("--exact-spatial-bins", type=int, default=8)
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data_path, train_path = _resolve_data_paths(
        data=args.data,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        split=args.split,
        train_data=args.train_data,
    )

    cmd = [
        sys.executable,
        "-m",
        "unified_stpp",
        "evaluate",
        "metrics",
        "--run",
        str(Path(args.run).expanduser().resolve()),
        "--data",
        str(data_path),
        "--metric-profile",
        str(args.metric_profile),
        "--artifact-mode",
        str(args.artifact_mode),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--k-pred",
        str(int(args.k_pred)),
        "--k-gen",
        str(int(args.k_gen)),
        "--exact-time-bins",
        str(int(args.exact_time_bins)),
        "--exact-spatial-bins",
        str(int(args.exact_spatial_bins)),
        "--out",
        str(Path(args.out).expanduser().resolve()),
    ]
    if train_path is not None:
        cmd.extend(["--train-data", str(train_path)])
    if args.split:
        cmd.extend(["--split", str(args.split)])
    if args.benchmark_id:
        cmd.extend(["--benchmark-id", str(args.benchmark_id)])

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
