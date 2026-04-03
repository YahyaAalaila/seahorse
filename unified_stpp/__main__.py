"""
CLI entrypoint for unified_stpp.

Usage
-----
python -m unified_stpp fit   --preset auto_stpp --train data/train.jsonl --val data/val.jsonl
python -m unified_stpp fit   --config my.yaml   --train data/train.jsonl --val data/val.jsonl [--test data/test.jsonl] [--out runs/]
python -m unified_stpp tune  --preset auto_stpp --train data/train.jsonl --val data/val.jsonl [--n_trials 50] [--search-alg random] [--scheduler asha]
python -m unified_stpp bench --presets auto_stpp deep_stpp --splits_dir data/ --seeds 42 43 [--out bench_out/]

Data format: newline-delimited JSON (.jsonl), one sequence per line:
  {"times": [0.1, 0.4, ...], "locations": [[x1, y1], [x2, y2], ...]}
"""

from __future__ import annotations

import argparse

from unified_stpp.cli import fit, tune, evaluate, bench


def main():
    parser = argparse.ArgumentParser(
        prog="python -m unified_stpp",
        description="Unified Neural STPP CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fit.add_subparser(sub)
    tune.add_subparser(sub)
    evaluate.add_subparser(sub)
    bench.add_subparser(sub)
    args = parser.parse_args()
    {
        "fit":      fit.execute,
        "tune":     tune.execute,
        "evaluate": evaluate.execute,
        "bench":    bench.execute,
    }[args.command](args)


if __name__ == "__main__":
    main()
