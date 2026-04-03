"""``bench`` subcommand — multi-preset × multi-dataset benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

from unified_stpp.cli._args import add_hpo_args


def add_subparser(sub) -> None:
    p = sub.add_parser("bench", help="Multi-preset × multi-dataset benchmark")
    p.add_argument("--presets",    nargs="+", required=True)
    p.add_argument("--splits_dir", required=True,
                   help="Directory with train/val/test.jsonl splits")
    p.add_argument("--datasets",   nargs="+", default=None, metavar="DATASET",
                   help="Restrict to these dataset names from splits_dir (default: all found)")
    p.add_argument("--seeds",      nargs="+", default=["42"])
    p.add_argument("--out",        default=None, metavar="DIR",
                   help="Output directory (default: bench_YYYYMMDD_HHMMSS_<sha8>)")
    p.add_argument("--n_workers",  type=int, default=1)
    p.add_argument("--tune",       action="store_true", help="Run HPO before evaluation")
    add_hpo_args(p, sentinel_defaults=False)
    p.add_argument("--hpo_configs_dir", default=None, metavar="DIR",
                   help="Directory with {preset}_best.yaml files from a prior --tune run "
                        "(presets found here skip re-tuning)")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override config values for all presets, e.g. training.n_epochs=50")
    p.add_argument("--normalize", action="store_true", default=True,
                   help="Z-score normalize time and space for all presets (default: on)")
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="Disable normalization (all presets see raw coordinates)")


def execute(args) -> None:
    from unified_stpp.benchmark import Benchmark
    from unified_stpp.config import BenchmarkConfig, TuningConfig
    from unified_stpp.runner.artifacts import make_bench_run_id
    from unified_stpp.utils import load_splits_dir, parse_overrides

    splits = load_splits_dir(args.splits_dir, datasets=args.datasets)

    base_overrides = parse_overrides(args.override)
    out = args.out or make_bench_run_id()
    out_path = Path(out)

    benchmark_config = BenchmarkConfig(
        run_hpo=args.tune,
        tuning=TuningConfig(
            n_trials=args.n_trials,
            search_alg=args.search_alg,
            scheduler=args.scheduler,
        ) if args.tune else None,
        seeds=[int(s) for s in args.seeds],
        n_workers=args.n_workers,
        normalize=args.normalize,
    )

    bench = Benchmark(
        presets=args.presets,
        splits=splits,
        config=benchmark_config,
        base_overrides=base_overrides,
        hpo_configs_dir=args.hpo_configs_dir,
        out_dir=out_path,
        argv=sys.argv[1:],
        splits_dir_str=args.splits_dir,
        raw_overrides=args.override or [],
    )
    table = bench.run()
    table.report(out, metric=benchmark_config.primary_metric)
    print(f"Report written to {out}/report.html")
