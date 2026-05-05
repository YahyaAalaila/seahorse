#!/usr/bin/env python3
"""Run one real-data benchmark cell with periodic test-NLL curve logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from unified_stpp.config import STPPConfig
from unified_stpp.config.schema import DataConfig
from unified_stpp.runner import STPPRunner
from unified_stpp.training.callbacks import PeriodicTestNLLCallback
from unified_stpp.utils import deep_update, parse_overrides


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a single real-data benchmark cell with test_nll_curve.csv."
    )
    p.add_argument("--preset", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True, help="Benchmark output root.")
    p.add_argument("--curve-step", type=float, default=0.05)
    p.add_argument("--suite", default=None)
    p.add_argument("--config-id", default=None)
    p.add_argument("--override", nargs="*", default=[])
    norm = p.add_mutually_exclusive_group()
    norm.add_argument("--normalize", action="store_true", default=False)
    norm.add_argument("--no-normalize", dest="normalize", action="store_false")
    return p


def _single_dataset_id(resolved) -> str:
    ids = sorted(resolved.splits)
    if len(ids) != 1:
        raise ValueError(f"Expected exactly one resolved dataset, got {ids!r}")
    return ids[0]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    data_config = DataConfig(
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
    )
    resolved = data_config.resolve_data(mode="benchmark")
    dataset_id = _single_dataset_id(resolved)
    train_seqs, val_seqs, test_seqs = resolved.splits[dataset_id]

    raw: dict[str, Any] = STPPConfig.raw_source_dict(preset=args.preset)
    deep_update(
        raw,
        {
            "data": {
                "protocol": "raw",
                "normalize": bool(args.normalize),
                "seed": int(args.seed),
                "dataset": args.dataset,
                "dataset_revision": args.dataset_revision,
            },
            "logging": {
                "out_dir": str(Path(args.out)),
            },
            "training": {
                "checkpoint_select": "best",
                "test_nll_space": "raw",
            },
        },
    )
    overrides = parse_overrides(args.override)
    if overrides:
        deep_update(raw, overrides)

    cfg = STPPConfig(**raw)
    runner = STPPRunner(cfg)
    suite = args.suite or dataset_id
    config_id = args.config_id or dataset_id
    callback = PeriodicTestNLLCallback(
        suite=suite,
        config_id=config_id,
        preset=args.preset,
        seed=args.seed,
        curve_step=args.curve_step,
        config=runner.config,
    )

    manifest_path = Path(args.out) / "curve_bench_invocation.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "preset": args.preset,
                "dataset": args.dataset,
                "dataset_revision": args.dataset_revision,
                "dataset_id": dataset_id,
                "seed": args.seed,
                "out": str(Path(args.out)),
                "curve_step": args.curve_step,
                "overrides": args.override,
            },
            f,
            indent=2,
        )

    result = runner.fit(
        train_seqs,
        val_seqs,
        test_seqs,
        dataset_id=dataset_id,
        extra_callbacks=[callback],
    )
    print(f"[curve-bench] run_dir={result.run_dir}")
    print(f"[curve-bench] test_nll_curve={Path(result.run_dir) / 'test_nll_curve.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
