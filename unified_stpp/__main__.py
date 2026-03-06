"""
CLI entrypoint for unified_stpp.

Usage
-----
python -m unified_stpp fit   --preset auto_stpp --train data/train.jsonl --val data/val.jsonl
python -m unified_stpp fit   --config my.yaml   --train data/train.jsonl --val data/val.jsonl [--test data/test.jsonl] [--out runs/]
python -m unified_stpp tune  --preset auto_stpp --train data/train.jsonl --val data/val.jsonl [--n_trials 50]
python -m unified_stpp bench --presets auto_stpp deep_stpp --splits_dir data/ --seeds 42 43 [--out bench_out/]

Data format: newline-delimited JSON (.jsonl), one sequence per line:
  {"times": [0.1, 0.4, ...], "locations": [[x1, y1], [x2, y2], ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seqs.append(json.loads(line))
    return seqs


def _load_splits_dir(splits_dir: str) -> dict[str, tuple]:
    """Load (train/val/test).jsonl from a directory.

    Expects one sub-directory per dataset:
      splits_dir/
        dataset_a/train.jsonl  val.jsonl  test.jsonl
        dataset_b/...
    If the directory itself contains train.jsonl, treats it as a single dataset.
    """
    p = Path(splits_dir)
    # Single dataset flat layout
    if (p / "train.jsonl").exists():
        return {
            p.name: (
                _load_jsonl(p / "train.jsonl"),
                _load_jsonl(p / "val.jsonl"),
                _load_jsonl(p / "test.jsonl") if (p / "test.jsonl").exists() else None,
            )
        }
    # Multi-dataset layout
    splits = {}
    for ds_dir in sorted(p.iterdir()):
        if ds_dir.is_dir() and (ds_dir / "train.jsonl").exists():
            splits[ds_dir.name] = (
                _load_jsonl(ds_dir / "train.jsonl"),
                _load_jsonl(ds_dir / "val.jsonl"),
                _load_jsonl(ds_dir / "test.jsonl") if (ds_dir / "test.jsonl").exists() else None,
            )
    if not splits:
        raise ValueError(f"No dataset directories with train.jsonl found in {splits_dir}")
    return splits


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_fit(args):
    from unified_stpp.runner import STPPRunner

    if args.config:
        runner = STPPRunner.from_yaml(args.config)
    elif args.preset:
        runner = STPPRunner.from_preset(args.preset)
        if args.out:
            runner.config.logging.out_dir = args.out
    else:
        print("ERROR: provide --preset or --config", file=sys.stderr)
        sys.exit(1)

    train_seqs = _load_jsonl(args.train)
    val_seqs = _load_jsonl(args.val)
    test_seqs = _load_jsonl(args.test) if args.test else None

    result = runner.fit(
        train_seqs,
        val_seqs,
        test_seqs,
        dataset_id=Path(args.train).stem,
    )
    print(result)

    if args.save:
        saved = runner.save(args.save)
        print(f"Saved to {saved}")


def cmd_tune(args):
    import yaml
    from unified_stpp.benchmark.hpo import run_hpo
    from pathlib import Path as _Path

    # Load raw YAML dict — do NOT go through STPPConfig so that search-space
    # values ({min, max} dicts, lists) pass through without type validation.
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
    elif args.preset:
        yaml_path = _Path(__file__).parent / "configs" / f"{args.preset}.yaml"
        if not yaml_path.exists():
            print(f"ERROR: no YAML found for preset '{args.preset}' at {yaml_path}", file=sys.stderr)
            sys.exit(1)
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
    else:
        print("ERROR: provide --preset or --config", file=sys.stderr)
        sys.exit(1)

    train_seqs = _load_jsonl(args.train)
    val_seqs = _load_jsonl(args.val)

    best_config = run_hpo(
        config_dict=config_dict,
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        n_trials=args.n_trials,
        algorithm=args.algorithm,
    )
    out_path = Path(args.out) if args.out else Path("best_config.yaml")
    best_config.to_yaml(out_path)
    print(f"Best config written to {out_path}")


def cmd_bench(args):
    from unified_stpp.benchmark import Benchmark

    splits = _load_splits_dir(args.splits_dir)
    seeds = [int(s) for s in args.seeds]
    bench = Benchmark(presets=args.presets, splits=splits, seeds=seeds)

    if args.tune:
        bench.tune_all(n_trials=args.n_trials, algorithm=args.algorithm)

    table = bench.run(n_workers=args.n_workers)
    out = args.out or "bench_out"
    table.report(out)
    print(f"Report written to {out}/report.html")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="python -m unified_stpp",
        description="Unified Neural STPP CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- fit ------------------------------------------------------------------
    fit_p = sub.add_parser("fit", help="Train a single model")
    fit_group = fit_p.add_mutually_exclusive_group(required=True)
    fit_group.add_argument("--preset", help="Model preset name (e.g. auto_stpp)")
    fit_group.add_argument("--config", help="Path to YAML config file")
    fit_p.add_argument("--train", required=True, help="Path to train .jsonl")
    fit_p.add_argument("--val",   required=True, help="Path to val .jsonl")
    fit_p.add_argument("--test",  default=None,  help="Path to test .jsonl (optional)")
    fit_p.add_argument("--out",   default=None,  help="Output directory for logs")
    fit_p.add_argument("--save",  default=None,  help="Directory to save the runner")

    # -- tune -----------------------------------------------------------------
    tune_p = sub.add_parser("tune", help="HPO for a preset (requires Ray Tune)")
    tune_group = tune_p.add_mutually_exclusive_group(required=True)
    tune_group.add_argument("--preset")
    tune_group.add_argument("--config")
    tune_p.add_argument("--train",     required=True)
    tune_p.add_argument("--val",       required=True)
    tune_p.add_argument("--n_trials",  type=int, default=50)
    tune_p.add_argument("--algorithm", default="asha", choices=["asha", "bayesian", "grid"])
    tune_p.add_argument("--out",       default=None, help="Output path for best config YAML")

    # -- bench ----------------------------------------------------------------
    bench_p = sub.add_parser("bench", help="Multi-preset × multi-dataset benchmark")
    bench_p.add_argument("--presets",    nargs="+", required=True)
    bench_p.add_argument("--splits_dir", required=True, help="Directory with train/val/test.jsonl splits")
    bench_p.add_argument("--seeds",      nargs="+", default=["42"])
    bench_p.add_argument("--out",        default="bench_out")
    bench_p.add_argument("--n_workers",  type=int, default=1)
    bench_p.add_argument("--tune",       action="store_true", help="Run HPO before evaluation")
    bench_p.add_argument("--n_trials",   type=int, default=50)
    bench_p.add_argument("--algorithm",  default="asha", choices=["asha", "bayesian", "grid"])

    args = parser.parse_args()
    {
        "fit":   cmd_fit,
        "tune":  cmd_tune,
        "bench": cmd_bench,
    }[args.command](args)


if __name__ == "__main__":
    main()
