"""``fit`` subcommand — train a single model."""

from __future__ import annotations

import math
from pathlib import Path
import sys

from seahorse.cli._args import add_config_source_args, add_data_args
from seahorse.cli._bridge import extract_explicit_cli_values


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
    from seahorse.runner import STPPRunner
    from seahorse.utils import load_jsonl

    cli_values = extract_explicit_cli_values(args, FIT_CLI_FIELD_MAP)
    cli_values.setdefault("data", {})
    for key in ("dataset", "dataset_revision", "train", "val", "test"):
        value = getattr(args, key, None)
        if value is None:
            continue
        target = {
            "dataset": "dataset",
            "dataset_revision": "dataset_revision",
            "train": "train_path",
            "val": "val_path",
            "test": "test_path",
        }[key]
        cli_values["data"][target] = value
    runner = STPPRunner.from_config_source(
        args.preset,
        args.config,
        cli_values=cli_values,
        override_list=args.override,
    )

    try:
        resolved = runner.config.data.resolve_data(mode="single", include_test=True)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        sys.exit(f"error: {exc}")

    train_seqs = load_jsonl(resolved.train_path)
    val_seqs = load_jsonl(resolved.val_path)
    test_seqs = load_jsonl(resolved.test_path) if resolved.test_path else None

    result = runner.fit(
        train_seqs,
        val_seqs,
        test_seqs,
        dataset_id=resolved.dataset_id,
    )

    print(f"\n  val_{result.val_metric_key}:  {result.val_objective:.4f}")
    if not math.isnan(result.test_nll):
        print(f"  test_nll: {result.test_nll:.4f}")
    if result.run_dir is not None:
        print(f"  saved to: {result.run_dir}\n")

    if args.save:
        saved = runner.save(args.save)
        print(f"Saved to {saved}")
