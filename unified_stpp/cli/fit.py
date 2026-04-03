"""``fit`` subcommand — train a single model."""

from __future__ import annotations

import math
from pathlib import Path

from unified_stpp.cli._args import add_config_source_args, add_data_args
from unified_stpp.cli._bridge import extract_explicit_cli_values


FIT_CLI_FIELD_MAP = {
    "out": "logging.out_dir",
}


def add_subparser(sub) -> None:
    p = sub.add_parser("fit", help="Train a single model")
    add_config_source_args(p)
    add_data_args(p, include_test=True)
    p.add_argument("--out",      default=None, help="Output directory for logs")
    p.add_argument("--save",     default=None, help="Directory to save the runner")
    p.add_argument("--override", nargs="*",   default=[], metavar="KEY=VALUE",
                   help="Dotted-key config overrides, e.g. training.lr=1e-4")


def execute(args) -> None:
    from unified_stpp.runner import STPPRunner
    from unified_stpp.utils import load_jsonl

    cli_values = extract_explicit_cli_values(args, FIT_CLI_FIELD_MAP)
    runner = STPPRunner.from_config_source(
        args.preset,
        args.config,
        cli_values=cli_values,
        override_list=args.override,
    )

    train_seqs = load_jsonl(args.train)
    val_seqs = load_jsonl(args.val)
    test_seqs = load_jsonl(args.test) if args.test else None

    result = runner.fit(
        train_seqs,
        val_seqs,
        test_seqs,
        dataset_id=Path(args.train).stem,
    )

    print(f"\n  val_{result.val_metric_key}:  {result.val_objective:.4f}")
    if not math.isnan(result.test_nll):
        print(f"  test_nll: {result.test_nll:.4f}")
    if result.run_dir is not None:
        print(f"  saved to: {result.run_dir}\n")

    if args.save:
        saved = runner.save(args.save)
        print(f"Saved to {saved}")
