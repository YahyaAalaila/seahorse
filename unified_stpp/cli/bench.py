"""``bench`` subcommand — multi-preset × multi-dataset benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

from unified_stpp.cli._args import add_hpo_args


def add_subparser(sub) -> None:
    p = sub.add_parser("bench", help="Multi-preset × multi-dataset benchmark")
    preset_group = p.add_mutually_exclusive_group(required=True)
    preset_group.add_argument("--preset", default=None,
                              help="Single model preset name.")
    preset_group.add_argument("--presets", nargs="+", default=None,
                              help="One or more model presets.")
    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--splits_dir", default=None,
                            help="Directory with train/val/test.jsonl benchmark splits.")
    data_group.add_argument("--dataset", default=None,
                            help="Single named dataset or local dataset directory for one-dataset benchmarking.")
    p.add_argument("--dataset-revision", default=None,
                   help="Optional dataset revision when --dataset resolves through the dataset hub.")
    p.add_argument("--datasets",   nargs="+", default=None, metavar="DATASET",
                   help="Restrict to these dataset names from splits_dir (default: all found)")
    p.add_argument("--seeds",      nargs="+", default=["42"])
    p.add_argument("--out",        default=None, metavar="DIR",
                   help="Output directory (default: bench_YYYYMMDD_HHMMSS_<sha8>)")
    p.add_argument("--n_workers",  type=int, default=1)
    p.add_argument("--tune",       action="store_true", help="Run HPO before evaluation")
    p.add_argument("--tune-dataset", default=None, metavar="DATASET",
                   help="Dataset used for HPO. Required when --tune is enabled.")
    p.add_argument("--hpo-seed", type=int, default=0,
                   help="Explicit RNG seed for HPO search.")
    add_hpo_args(p, sentinel_defaults=False)
    p.add_argument("--hpo_configs_dir", default=None, metavar="DIR",
                   help="Directory with {preset}_best.yaml files from a prior --tune run "
                        "(presets found here skip re-tuning)")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override config values for all presets, e.g. training.n_epochs=50")
    p.add_argument("--normalize", action="store_true", default=False,
                   help="Z-score normalize time and space for all presets (default: off)")
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="Disable normalization (all presets see raw coordinates)")


def execute(args) -> None:
    from unified_stpp.benchmark import Benchmark
    from unified_stpp.config import BenchmarkConfig, TuningConfig
    from unified_stpp.config.schema import DataConfig
    from unified_stpp.data.resolution import build_benchmark_data_provenance
    from unified_stpp.runner.artifacts import make_bench_run_id
    from unified_stpp.utils import parse_overrides

    data_config = DataConfig(
        splits_dir=getattr(args, "splits_dir", None),
        dataset=getattr(args, "dataset", None),
        dataset_revision=getattr(args, "dataset_revision", None),
        datasets=list(getattr(args, "datasets", None) or []),
    )
    resolved = data_config.resolve_data(mode="benchmark")
    splits = resolved.splits
    data_manifest = build_benchmark_data_provenance(data_config, resolved)

    base_overrides = parse_overrides(args.override)
    out = args.out or make_bench_run_id()
    out_path = Path(out)
    presets = list(getattr(args, "presets", None) or [])
    if not presets and getattr(args, "preset", None):
        presets = [args.preset]
    if not presets:
        raise ValueError("bench requires --preset or --presets.")

    benchmark_config = BenchmarkConfig(
        run_hpo=args.tune,
        tuning=TuningConfig(
            n_trials=args.n_trials,
            search_alg=args.search_alg,
            scheduler=args.scheduler,
            seed=getattr(args, "hpo_seed", 0),
        ) if args.tune else None,
        tune_dataset=getattr(args, "tune_dataset", None),
        seeds=[int(s) for s in args.seeds],
        n_workers=args.n_workers,
        normalize=args.normalize,
    )

    bench = Benchmark(
        presets=presets,
        splits=splits,
        config=benchmark_config,
        base_overrides=base_overrides,
        hpo_configs_dir=args.hpo_configs_dir,
        out_dir=out_path,
        data_manifest=data_manifest,
        argv=sys.argv[1:],
        splits_dir_str=str(resolved.splits_dir),
        raw_overrides=args.override or [],
    )
    table = bench.run()
    table.report(out, metric=benchmark_config.primary_metric)
    print(f"Report written to {out}/report.html")
