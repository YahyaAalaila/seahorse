"""``tune`` subcommand — HPO for a preset (requires Ray Tune)."""

from __future__ import annotations

from pathlib import Path

from unified_stpp.cli._args import add_config_source_args, add_data_args, add_hpo_args
from unified_stpp.cli._bridge import extract_explicit_cli_values


TUNE_CLI_FIELD_MAP = {
    "n_trials": "n_trials",
    "search_alg": "search_alg",
    "scheduler": "scheduler",
    "seed": "seed",
    "fail_fast": "fail_fast",
    "max_concurrent_trials": "max_concurrent_trials",
}


def add_subparser(sub) -> None:
    p = sub.add_parser("tune", help="HPO for a preset (requires Ray Tune)")
    add_config_source_args(p)
    add_data_args(p, include_test=False)
    # All HPO args default to None (sentinel) so cmd_tune can distinguish
    # "explicitly provided" from "not provided", enabling YAML tuning: as base.
    add_hpo_args(p, sentinel_defaults=True)
    p.add_argument("--seed",      type=int, default=None,
                   help="RNG seed for reproducibility (overrides YAML tuning.seed if set)")
    p.add_argument("--fail-fast", dest="fail_fast", action="store_true", default=None,
                   help="Stop all trials on first failure")
    p.add_argument("--max-concurrent-trials", dest="max_concurrent_trials",
                   type=int, default=None,
                   help="Maximum number of concurrent trials (overrides YAML if set)")
    p.add_argument("--out", default=None, help="Output path for best config YAML")


def execute(args) -> None:
    from unified_stpp.benchmark.hpo import run_hpo
    from unified_stpp.config.schema import STPPConfig
    from unified_stpp.config.tuning import TuningConfig

    # Preserve raw search-space syntax for HPO while centralizing precedence.
    config_dict = STPPConfig.raw_source_dict(args.preset, args.config)

    # Split tuning: section from model config (HPO search-space parser must not
    # see tuning fields as model params).
    config_dict, raw_tuning = STPPConfig.split_tuning_dict(config_dict)
    cli_values = extract_explicit_cli_values(args, TUNE_CLI_FIELD_MAP)
    tuning = TuningConfig.from_sources(yaml_tuning=raw_tuning, cli_values=cli_values)

    best_config = run_hpo(
        config_dict=config_dict,
        tuning=tuning,
        train_path=args.train,
        val_path=args.val,
    )
    out_path = Path(args.out) if args.out else Path("best_config.yaml")
    best_config.to_yaml(out_path)
    print(f"Best config written to {out_path}")
