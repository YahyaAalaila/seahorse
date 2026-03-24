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
import math
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

    print(f"\n  val_nll:  {result.val_nll:.4f}")
    if not math.isnan(result.test_nll):
        print(f"  test_nll: {result.test_nll:.4f}")
    if result.run_dir is not None:
        print(f"  saved to: {result.run_dir}\n")

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


def _parse_overrides(override_list: list[str]) -> dict:
    """Convert ['training.n_epochs=50', 'model.hidden_dim=128'] to nested dict."""
    result = {}
    for item in (override_list or []):
        if "=" not in item:
            print(f"WARNING: ignoring malformed override {item!r} (expected key=value)", file=sys.stderr)
            continue
        key, _, raw_val = item.partition("=")
        # Auto-cast: int → float → bool → str
        try:
            val = int(raw_val)
        except ValueError:
            try:
                val = float(raw_val)
            except ValueError:
                if raw_val.lower() in ("true", "false"):
                    val = raw_val.lower() == "true"
                else:
                    val = raw_val
        # Expand dotted key into nested dict
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return result


def cmd_evaluate(args):
    """Post-fit analysis on a saved run (primary evaluation path)."""
    from unified_stpp.runner import STPPRunner
    from unified_stpp.evaluation.surface import SurfaceEvalSpec
    from unified_stpp.viz.workflow import SurfaceVizConfig

    runner = STPPRunner.load(args.run)
    val_seqs = _load_jsonl(args.val)

    surface_viz = None
    if args.surface_viz:
        # When --surface_animate is set and the user did not explicitly pass
        # --surface_history_mode, default to rolling (each frame conditions on
        # events before its query time, giving genuinely distinct surfaces).
        # For static panels the conservative 'fixed' default applies.
        history_mode = args.surface_history_mode
        if history_mode is None:
            history_mode = "rolling" if args.surface_animate else "fixed"
        eval_spec = SurfaceEvalSpec(
            split=args.surface_history_split,
            history_mode=history_mode,
            history_length=args.surface_history_length,
            t_query_mode=args.surface_t_query_mode,
            n_time_steps=args.surface_n_time_steps,
            horizon=args.surface_horizon,
            n_grid=args.surface_n_grid,
        )
        reference_provider = None
        if args.surface_reference_mode == "sthp_gt":
            if not args.surface_sthp_meta:
                raise ValueError(
                    "--surface_sthp_meta PATH is required with "
                    "--surface_reference_mode sthp_gt"
                )
            from unified_stpp.viz.reference import STHPGroundTruthProvider
            reference_provider = STHPGroundTruthProvider.from_meta_file(
                args.surface_sthp_meta
            )
        surface_viz = SurfaceVizConfig(
            eval_spec=eval_spec,
            enabled=True,
            render_mode=args.surface_render_mode,
            animate=args.surface_animate,
            save_panel=True,
            save_individual=True,
            reference_mode=args.surface_reference_mode,
            reference_provider=reference_provider,
            reference_first=args.surface_reference_first,
            animate_share_colorscale=not args.surface_no_share_colorscale,
        )

    run_dir = Path(args.out) if args.out else None
    artifacts = runner.evaluate(val_seqs=val_seqs, surface_viz=surface_viz, run_dir=run_dir)

    if artifacts:
        for name, path in artifacts.items():
            print(f"  {name}: {path}")
    else:
        print("No evaluation workflows were enabled.  Pass --surface_viz to generate surface plots.")


def cmd_bench(args):
    from pathlib import Path

    from unified_stpp.benchmark import Benchmark
    from unified_stpp.runner.artifacts import make_bench_run_id, write_bench_meta
    from unified_stpp.utils import deep_update

    if args.datasets:
        # Load each named dataset directly by path — avoids the root-train.jsonl
        # shortcut in _load_splits_dir when splits_dir also contains flat splits.
        p = Path(args.splits_dir)
        splits = {}
        missing = []
        for d in args.datasets:
            ds_dir = p / d
            if not ds_dir.is_dir() or not (ds_dir / "train.jsonl").exists():
                missing.append(d)
            else:
                splits[d] = (
                    _load_jsonl(ds_dir / "train.jsonl"),
                    _load_jsonl(ds_dir / "val.jsonl"),
                    _load_jsonl(ds_dir / "test.jsonl") if (ds_dir / "test.jsonl").exists() else None,
                )
        if missing:
            raise ValueError(
                f"Dataset directories not found under {args.splits_dir!r}: {missing}"
            )
    else:
        splits = _load_splits_dir(args.splits_dir)

    seeds = [int(s) for s in args.seeds]
    base_overrides = _parse_overrides(args.override)
    out = args.out or make_bench_run_id()
    out_path = Path(out)

    # Route all per-run dirs inside the bench output dir so everything is
    # co-located and moveable as a unit.  User's --override logging.out_dir=...
    # takes precedence: deep_update puts base_overrides values on top of bench_local.
    bench_local = {"logging": {"out_dir": str(out_path)}}
    deep_update(bench_local, base_overrides)
    base_overrides = bench_local

    write_bench_meta(
        out_dir=out_path,
        bench_id=out_path.name,
        argv=sys.argv[1:],
        splits_dir=args.splits_dir,
        datasets=sorted(splits.keys()),
        presets=args.presets,
        seeds=seeds,
        normalize=args.normalize,
        n_workers=args.n_workers,
        overrides=args.override or [],
        hpo_configs_dir=args.hpo_configs_dir,
    )

    # Write a one-click rerun script alongside bench_meta.json
    rerun_path = out_path / "rerun.sh"
    out_path.mkdir(parents=True, exist_ok=True)
    with open(rerun_path, "w") as f:
        f.write("#!/bin/bash\n" + " ".join(sys.argv) + "\n")
    rerun_path.chmod(0o755)

    bench = Benchmark(presets=args.presets, splits=splits, seeds=seeds,
                      base_overrides=base_overrides,
                      hpo_configs_dir=args.hpo_configs_dir,
                      normalize=args.normalize)

    if args.tune:
        bench.tune_all(n_trials=args.n_trials, algorithm=args.algorithm, out_dir=out)

    table = bench.run(n_workers=args.n_workers)
    table.report(out)
    print(f"Report written to {out}/report.html")

    if args.intensity_video != "none":
        print(f"\nRendering intensity animations (fmt={args.intensity_video})...")
        produced = table.plot_intensities(
            splits=splits,
            out_dir=out,
            fmt=args.intensity_video,
            n_frames=args.intensity_frames,
            n_grid=args.intensity_grid,
            fps=args.intensity_fps,
        )
        for p in produced:
            print(f"  {p}")


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

    # -- evaluate -------------------------------------------------------------
    eval_p = sub.add_parser(
        "evaluate",
        help="Post-fit analysis on a saved run (primary evaluation path)",
    )
    eval_p.add_argument("--run", required=True,
                        help="Path to a saved run directory (produced by fit)")
    eval_p.add_argument("--val", required=True,
                        help="Path to val .jsonl — sequences used as evaluation history")
    eval_p.add_argument("--out", default=None,
                        help="Output directory for artifacts (default: the run directory)")
    eval_p.add_argument("--surface_viz", action="store_true",
                        help="Enable surface visualization workflow")
    eval_p.add_argument("--surface_n_grid", type=int, default=50,
                        metavar="N", help="Grid resolution per axis (default: 50)")
    eval_p.add_argument("--surface_n_time_steps", type=int, default=3,
                        metavar="N", help="Number of time steps to query (default: 3)")
    eval_p.add_argument("--surface_render_mode", default="2d", choices=["2d", "3d"],
                        help="Render mode: '2d' heatmap (default) or '3d' surface")
    eval_p.add_argument("--surface_animate", action="store_true",
                        help="Also save a GIF animation of the surface sequence")
    eval_p.add_argument("--surface_history_length", type=int, default=10,
                        metavar="N", help="Number of history events to condition on (default: 10)")
    eval_p.add_argument("--surface_history_split", default="val",
                        choices=["train", "val", "test"],
                        help="Which data split to draw history from (default: val)")
    eval_p.add_argument(
        "--surface_history_mode",
        choices=["fixed", "rolling"], default=None,
        help=(
            "History strategy per animation frame. "
            "'rolling' (recommended for animation) uses all events strictly before "
            "each t_query, giving genuinely different surfaces per frame. "
            "Default when --surface_animate is set: 'rolling'. "
            "Default otherwise: 'fixed'."
        ),
    )
    eval_p.add_argument(
        "--surface_t_query_mode",
        choices=["after_history", "uniform"], default="after_history",
        help=(
            "How to place query times across the sequence. "
            "'uniform' spans the entire sequence time range and typically gives "
            "more visibly distinct frames. "
            "'after_history' with small --surface_horizon can yield subtle changes."
        ),
    )
    eval_p.add_argument(
        "--surface_horizon",
        type=float, default=1.0,
        help="Time horizon past the last history event for 'after_history' mode (default: 1.0).",
    )
    eval_p.add_argument(
        "--surface_reference_mode",
        choices=["none", "empirical_kde", "sthp_gt"], default="none",
        help=(
            "Reference surface to show alongside the model. "
            "'empirical_kde': marginal spatial KDE from the sequence events (time-independent proxy). "
            "'sthp_gt': exact conditional intensity λ*(t,s|H) from an STHP model — "
            "requires --surface_sthp_meta pointing to the dataset_meta.json produced by "
            "gen_sthp_splits.py. "
            "'none' (default): no reference."
        ),
    )
    eval_p.add_argument(
        "--surface_sthp_meta",
        default=None,
        metavar="PATH",
        help=(
            "Path to dataset_meta.json for STHP ground-truth reference. "
            "Required when --surface_reference_mode sthp_gt is set."
        ),
    )
    eval_p.add_argument(
        "--surface_reference_first",
        action="store_true", default=False,
        help=(
            "Put the reference surface on the LEFT and the model on the RIGHT in "
            "all outputs (animation frames, multi-panel figures, individual files). "
            "Default: model left, reference right."
        ),
    )
    eval_p.add_argument(
        "--surface_no_share_colorscale",
        action="store_true", default=False,
        help=(
            "Disable the fixed global colorscale in GIF animations. "
            "By default every frame uses the same vmin/vmax per surface type so "
            "colors are comparable across time steps. "
            "Pass this flag to let each frame auto-scale independently."
        ),
    )

    # -- bench ----------------------------------------------------------------
    bench_p = sub.add_parser("bench", help="Multi-preset × multi-dataset benchmark")
    bench_p.add_argument("--presets",    nargs="+", required=True)
    bench_p.add_argument("--splits_dir", required=True, help="Directory with train/val/test.jsonl splits")
    bench_p.add_argument("--datasets",   nargs="+", default=None, metavar="DATASET",
                         help="Restrict to these dataset names from splits_dir (default: all found)")
    bench_p.add_argument("--seeds",      nargs="+", default=["42"])
    bench_p.add_argument("--out",        default=None, metavar="DIR",
                         help="Output directory (default: bench_YYYYMMDD_HHMMSS_<sha8>)")
    bench_p.add_argument("--n_workers",  type=int, default=1)
    bench_p.add_argument("--tune",       action="store_true", help="Run HPO before evaluation")
    bench_p.add_argument("--n_trials",         type=int, default=50)
    bench_p.add_argument("--algorithm",        default="asha", choices=["asha", "bayesian", "grid"])
    bench_p.add_argument("--intensity_video",  default="none",
                         choices=["none", "gif", "mp4", "png"],
                         help="Render per-preset intensity animation after benchmarking")
    bench_p.add_argument("--intensity_frames", type=int, default=24,
                         help="Number of animation frames (gif/mp4 only)")
    bench_p.add_argument("--intensity_grid",   type=int, default=40,
                         help="Spatial grid resolution (points per axis)")
    bench_p.add_argument("--intensity_fps",    type=int, default=8,
                         help="Animation frames per second")
    bench_p.add_argument("--hpo_configs_dir", default=None,
                         metavar="DIR",
                         help="Directory with {preset}_best.yaml files from a prior --tune run "
                              "(presets found here skip re-tuning)")
    bench_p.add_argument("--override", nargs="*", default=[],
                         metavar="KEY=VALUE",
                         help="Override config values for all presets, e.g. training.n_epochs=50")
    bench_p.add_argument("--normalize", action="store_true", default=True,
                         help="Z-score normalize time and space for all presets (default: on)")
    bench_p.add_argument("--no-normalize", dest="normalize", action="store_false",
                         help="Disable normalization (all presets see raw coordinates)")

    args = parser.parse_args()
    {
        "fit":      cmd_fit,
        "tune":     cmd_tune,
        "evaluate": cmd_evaluate,
        "bench":    cmd_bench,
    }[args.command](args)


if __name__ == "__main__":
    main()
