"""``tune`` subcommand — HPO for a preset (requires Ray Tune)."""

from __future__ import annotations

import sys
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
    import json

    from unified_stpp.benchmark.hpo import build_hpo_manifest, run_hpo, write_trial_history
    from unified_stpp.config.schema import DataConfig, STPPConfig
    from unified_stpp.config.tuning import TuningConfig
    from unified_stpp.data.resolution import build_single_data_provenance

    # Preserve raw search-space syntax for HPO while centralizing precedence.
    config_dict = STPPConfig.raw_source_dict(args.preset, args.config)

    # Split tuning: section from model config (HPO search-space parser must not
    # see tuning fields as model params).
    config_dict, raw_tuning = STPPConfig.split_tuning_dict(config_dict)
    cli_values = extract_explicit_cli_values(args, TUNE_CLI_FIELD_MAP)
    tuning = TuningConfig.from_sources(yaml_tuning=raw_tuning, cli_values=cli_values)
    data_config = DataConfig(
        dataset=getattr(args, "dataset", None),
        dataset_revision=getattr(args, "dataset_revision", None),
        train_path=getattr(args, "train", None),
        val_path=getattr(args, "val", None),
    )
    resolved = data_config.resolve_data(mode="single", include_test=False)

    hpo_result = run_hpo(
        config_dict=config_dict,
        tuning=tuning,
        train_path=str(resolved.train_path),
        val_path=str(resolved.val_path),
        dataset_id=resolved.dataset_id,
        return_analysis=True,
    )
    if isinstance(hpo_result, tuple):
        best_config, analysis = hpo_result
    else:
        best_config, analysis = hpo_result, None
    out_path = Path(args.out) if args.out else Path("best_config.yaml")
    best_raw = best_config.model_dump(mode="json")
    best_raw.setdefault("data", {})
    if data_config.dataset is not None:
        best_raw["data"]["dataset"] = data_config.dataset
    if data_config.dataset_revision is not None:
        best_raw["data"]["dataset_revision"] = data_config.dataset_revision
    best_config = STPPConfig(**best_raw)
    best_config.to_yaml(out_path)
    data_manifest = build_single_data_provenance(data_config, resolved)
    with open(out_path.with_suffix(".data_manifest.json"), "w") as f:
        json.dump(data_manifest, f, indent=2, default=str)
    trials_json_path = out_path.with_suffix(".trials.json")
    trials_csv_path = out_path.with_suffix(".trials.csv")
    if analysis is not None:
        write_trial_history(
            analysis,
            json_path=trials_json_path,
            csv_path=trials_csv_path,
        )
    manifest = build_hpo_manifest(
        source="fresh_hpo",
        preset=best_config.model.preset,
        dataset_id=resolved.dataset_id,
        data_source_fingerprint=data_manifest.get("source_fingerprint"),
        tuning=tuning,
        best_config_path=out_path,
        trials_json_path=trials_json_path,
        trials_csv_path=trials_csv_path,
        analysis=analysis,
        argv=sys.argv[1:],
        extra={
            "data_manifest_path": str(out_path.with_suffix(".data_manifest.json").resolve()),
            "requested_dataset": data_config.dataset,
            "requested_dataset_revision": data_config.dataset_revision,
        },
    )
    with open(out_path.with_suffix(".hpo_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Best config written to {out_path}")
