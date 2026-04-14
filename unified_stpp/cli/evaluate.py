"""``evaluate`` subcommand — post-fit-only analysis on saved runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

_HISTORY_SPLITS = frozenset(("train", "val", "test"))


def add_subparser(sub) -> None:
    p = sub.add_parser(
        "evaluate",
        help="Post-fit evaluation on saved runs",
    )
    modes = p.add_subparsers(dest="evaluate_mode", required=True)
    _add_metrics_subparser(modes)
    _add_predictive_compare_subparser(modes)
    _add_surface_subparser(modes)


def _add_metrics_subparser(sub) -> None:
    from unified_stpp.evaluation.profiles import profile_names

    profile_help = "|".join(profile_names())
    p = sub.add_parser(
        "metrics",
        help="Post-fit metric reports with explicit artifact-backed profiles",
    )
    p.add_argument("--run", required=True, help="Path to a saved run directory.")
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument(
        "--data",
        dest="data",
        help="JSONL file containing evaluation sequences.",
    )
    data.add_argument(
        "--history",
        dest="data",
        help="Deprecated alias for --data.",
    )
    p.add_argument(
        "--train-data",
        default=None,
        help="Optional JSONL training split for train/test NLL gap metrics.",
    )
    p.add_argument(
        "--split",
        default=None,
        choices=("train", "val", "test"),
        help=(
            "Metadata label for --data. Defaults to the filename when it is "
            "train.jsonl, val.jsonl, or test.jsonl."
        ),
    )
    p.add_argument(
        "--metric-profile",
        default="core",
        help=f"Metric profile to run when --metric is not provided: {profile_help}.",
    )
    p.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Explicit metric name to run. Repeat to select multiple metrics.",
    )
    p.add_argument("--max-seqs", type=int, default=None, help="Limit evaluation sequences.")
    p.add_argument("--max-events", type=int, default=None, help="Limit events per sequence.")
    p.add_argument("--k-pred", type=int, default=200, help="Next-event samples per context.")
    p.add_argument("--k-gen", type=int, default=20, help="Full-rollout samples per sequence.")
    p.add_argument(
        "--exact-time-bins",
        type=int,
        default=12,
        help="Number of proposal time bins for thinning-based next-event sampling.",
    )
    p.add_argument(
        "--exact-spatial-bins",
        type=int,
        default=12,
        help="Number of proposal bins per spatial axis for thinning-based next-event sampling.",
    )
    p.add_argument("--seed", type=int, default=0, help="Evaluation sampling seed.")
    p.add_argument(
        "--device",
        default="auto",
        help="Device for loading and evaluation: auto, cpu, cuda, cuda:0, mps, ...",
    )
    p.add_argument(
        "--artifact-dir",
        default=None,
        help="Root directory for persisted evaluation artifacts. Defaults under --out.",
    )
    p.add_argument(
        "--artifact-mode",
        default="load_or_compute",
        choices=("load_or_compute", "load_only"),
        help="Artifact policy for planned heavy artifact families.",
    )
    p.add_argument(
        "--benchmark-id",
        default=None,
        help="Optional upstream benchmark identifier to persist in the evaluation manifest.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to <run>/evaluate/metrics/<profile>_<split>/.",
    )


def _add_predictive_compare_subparser(sub) -> None:
    p = sub.add_parser(
        "predictive-compare",
        help="Primary benchmark path: sample-based predictive rollout comparison",
    )
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help="Path to a saved run directory. Repeat --run to compare multiple models.",
    )
    p.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional display label. Repeat to align with --run order.",
    )
    p.add_argument(
        "--history",
        required=True,
        help="JSONL file containing candidate conditioning histories.",
    )
    p.add_argument(
        "--split",
        default=None,
        choices=("train", "val", "test"),
        help=(
            "Metadata label for the chosen history source. Defaults to the "
            "--history filename when it is train.jsonl, val.jsonl, or test.jsonl."
        ),
    )
    p.add_argument("--seq-idx", type=int, default=0, help="Sequence index inside --history.")
    p.add_argument(
        "--start-event-idx",
        type=int,
        default=0,
        help="Frame-0 history ends at this observed event index (inclusive).",
    )
    p.add_argument(
        "--history-length",
        type=int,
        default=0,
        help="If > 0, keep only the last N events in the working history.",
    )
    p.add_argument(
        "--rollout-mode",
        default="teacher_forced",
        choices=("teacher_forced", "free_running"),
        help="Primary default is teacher_forced. Free-running remains diagnostic.",
    )
    p.add_argument("--n-frames", type=int, default=6, help="Number of future windows to compare.")
    p.add_argument(
        "--horizon",
        type=float,
        required=True,
        help="Duration of each future prediction window.",
    )
    p.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Window stride. Defaults to --horizon. Current freeze requires equality.",
    )
    p.add_argument(
        "--n-rollouts",
        type=int,
        default=128,
        help="Number of predictive trajectories per frame and model.",
    )
    p.add_argument(
        "--grid-size",
        type=int,
        default=96,
        help="Shared spatial grid size per axis for derived KDE rate surfaces.",
    )
    p.add_argument(
        "--bandwidth",
        default=None,
        help="Optional scipy gaussian_kde bw_method for derived KDE surfaces.",
    )
    p.add_argument("--xmin", type=float, default=None, help="Optional shared xmin override.")
    p.add_argument("--xmax", type=float, default=None, help="Optional shared xmax override.")
    p.add_argument("--ymin", type=float, default=None, help="Optional shared ymin override.")
    p.add_argument("--ymax", type=float, default=None, help="Optional shared ymax override.")
    p.add_argument(
        "--lambda-bar",
        type=float,
        default=10.0,
        help="Initial thinning upper bound for exact-intensity presets.",
    )
    p.add_argument(
        "--max-events-per-window",
        type=int,
        default=64,
        help="Safety cap on sampled events per rollout and window.",
    )
    p.add_argument(
        "--bridge-retries",
        type=int,
        default=64,
        help="Maximum redraw attempts when native samplers must bridge into a window.",
    )
    p.add_argument(
        "--no-adaptive-thinning",
        dest="adaptive_thinning",
        action="store_false",
        help="Disable adaptive thinning bound updates for exact-intensity presets.",
    )
    p.add_argument(
        "--exact-proposal",
        default="coarse",
        choices=("coarse",),
        help="Proposal-envelope mode for exact-model thinning (default: coarse).",
    )
    p.add_argument(
        "--exact-time-bins",
        type=int,
        default=12,
        help="Number of proposal time bins for exact-model thinning.",
    )
    p.add_argument(
        "--exact-spatial-bins",
        type=int,
        default=12,
        help="Number of proposal bins per spatial axis for exact-model thinning.",
    )
    p.add_argument(
        "--exact-safety",
        type=float,
        default=2.0,
        help="Envelope safety factor for exact-model thinning proposals.",
    )
    p.add_argument(
        "--color-percentile",
        type=float,
        default=99.0,
        help="Percentile used for a shared derived-surface colorscale.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Comparison-level base seed. Persisted in bundle metadata and used to "
            "derive deterministic sub-seeds from stable model/frame/rollout identifiers."
        ),
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device for loading and evaluation: auto, cpu, cuda, cuda:0, mps, ...",
    )
    p.add_argument(
        "--plot-style",
        default="2d",
        choices=("2d", "3d", "both"),
        help="Rendered visualization style for derived predictive panels.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Animation frames per second when --gif is set.",
    )
    p.add_argument(
        "--gif",
        action="store_true",
        help="Also write an animation over the derived predictive panel frames.",
    )
    p.add_argument(
        "--out",
        required=True,
        help=(
            "Output directory for the comparison bundle. Required because "
            "predictive comparison is a multi-run artifact, not a child of one run."
        ),
    )
    p.set_defaults(adaptive_thinning=True)


def _add_surface_subparser(sub) -> None:
    p = sub.add_parser(
        "surface",
        help="Secondary diagnostic path: single-run exact/factorized surfaces",
    )
    p.add_argument("--run", required=True, help="Path to a saved run directory.")
    p.add_argument(
        "--history",
        required=True,
        help="JSONL file containing candidate conditioning histories.",
    )
    p.add_argument(
        "--split",
        default=None,
        choices=("train", "val", "test"),
        help=(
            "Metadata label for the chosen history source. Defaults to the "
            "--history filename when it is train.jsonl, val.jsonl, or test.jsonl."
        ),
    )
    p.add_argument("--seq-idx", type=int, default=0, help="Sequence index inside --history.")
    p.add_argument(
        "--history-length",
        type=int,
        default=0,
        help="If > 0, keep only the last N events from the chosen history.",
    )
    p.add_argument(
        "--profile",
        default="notebook_faithful",
        choices=("notebook_faithful", "future_exact"),
        help=(
            "Surface profile. Neural future_exact support remains provisional until "
            "packaged parity is proven."
        ),
    )
    p.add_argument("--x-nstep", type=int, default=81, help="Number of x-grid points.")
    p.add_argument("--y-nstep", type=int, default=81, help="Number of y-grid points.")
    p.add_argument("--t-nstep", type=int, default=41, help="Number of time frames.")
    p.add_argument(
        "--future-horizon",
        type=float,
        default=None,
        help="Future horizon for future_exact queries. Omit to infer a local horizon.",
    )
    p.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Static-frame index to render for notebook_faithful mode. Negative means middle.",
    )
    p.add_argument(
        "--no-round-time",
        action="store_true",
        help="Disable notebook-style rounded time-grid start/end behavior.",
    )
    p.add_argument(
        "--trunc",
        dest="trunc",
        action="store_true",
        default=None,
        help="Force truncation to the model's max_history in notebook_faithful mode.",
    )
    p.add_argument(
        "--no-trunc",
        dest="trunc",
        action="store_false",
        help="Disable truncation override in notebook_faithful mode.",
    )
    p.add_argument("--xmin", type=float, default=None, help="Optional original-space xmin override.")
    p.add_argument("--xmax", type=float, default=None, help="Optional original-space xmax override.")
    p.add_argument("--ymin", type=float, default=None, help="Optional original-space ymin override.")
    p.add_argument("--ymax", type=float, default=None, help="Optional original-space ymax override.")
    p.add_argument(
        "--spatial-chunk-size",
        type=int,
        default=None,
        help="Chunk size for provisional neural exact-family spatial queries.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device for loading and evaluation: auto, cpu, cuda, cuda:0, mps, ...",
    )
    p.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Disable interactive HTML output.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to <run>/evaluate/surface/<profile>_<split>_seq<idx>/.",
    )
    p.set_defaults(interactive=True)


def execute(args) -> None:
    mode = getattr(args, "evaluate_mode", None)
    if mode == "metrics":
        _execute_metrics(args)
        return
    if mode == "predictive-compare":
        _execute_predictive_compare(args)
        return
    if mode == "surface":
        _execute_surface(args)
        return
    raise SystemExit("evaluate requires one of: metrics, predictive-compare, surface")


def _execute_metrics(args) -> None:
    from unified_stpp.evaluation.evaluator import evaluate
    from unified_stpp.evaluation.profiles import (
        MetricPlanError,
        metric_profile as resolve_metric_profile,
        validate_metric_plan,
    )
    from unified_stpp.evaluation.registry import metric_by_name
    from unified_stpp.runner.runner import STPPRunner
    from unified_stpp.utils import load_jsonl

    from unified_stpp.evaluation.common import load_run_result, resolve_device

    run_dir = Path(args.run).resolve()
    data_path = Path(args.data).resolve()
    split = _resolve_history_split(data_path, args.split)
    requested_profile = str(args.metric_profile)
    profile = resolve_metric_profile(requested_profile)
    canonical_profile = profile.name
    out_dir = _default_metrics_out_dir(
        run_dir=run_dir,
        profile=canonical_profile,
        split=split,
        out_override=args.out,
    )
    artifact_dir = (
        Path(args.artifact_dir).resolve()
        if args.artifact_dir is not None
        else out_dir / "artifacts"
    )
    test_seqs = _load_metric_sequences(
        data_path,
        max_seqs=args.max_seqs,
        max_events=args.max_events,
        load_jsonl=load_jsonl,
    )
    train_seqs = (
        None
        if args.train_data is None
        else _load_metric_sequences(
            Path(args.train_data).resolve(),
            max_seqs=args.max_seqs,
            max_events=args.max_events,
            load_jsonl=load_jsonl,
        )
    )
    selected_metrics = [metric_by_name(name) for name in list(args.metric or [])] or None
    if selected_metrics is not None:
        try:
            validate_metric_plan(
                selected_metrics,
                allowed_artifact_families=profile.allowed_artifact_families,
            )
        except MetricPlanError as exc:
            raise SystemExit(str(exc)) from None
    device = resolve_device(str(args.device))
    runner = STPPRunner.load(run_dir)
    runner.model.to(device)
    runner.model.eval()

    try:
        report = evaluate(
            runner,
            test_seqs,
            train_data=train_seqs,
            k_pred=int(args.k_pred),
            k_gen=int(args.k_gen),
            exact_time_bins=int(args.exact_time_bins),
            exact_spatial_bins=int(args.exact_spatial_bins),
            seed=int(args.seed),
            device=device,
            metric_profile=canonical_profile,
            artifact_dir=artifact_dir,
            artifact_mode=str(args.artifact_mode),
            allowed_artifact_families=(
                profile.allowed_artifact_families
                if selected_metrics is not None
                else None
            ),
            metrics=selected_metrics,
        )
    except MetricPlanError as exc:
        raise SystemExit(str(exc)) from None

    out_dir.mkdir(parents=True, exist_ok=True)
    report.save(out_dir)
    metrics_path = out_dir / "metrics.json"
    per_event_files = sorted(out_dir.glob("*_per_event.npy"))
    result = load_run_result(run_dir)
    manifest_path = out_dir / "evaluation_manifest.json"
    manifest = _metrics_evaluation_manifest(
        run_dir=run_dir,
        data_path=data_path,
        split=split,
        out_dir=out_dir,
        artifact_dir=artifact_dir,
        metrics_path=metrics_path,
        per_event_files=per_event_files,
        args=args,
        report=report,
        result=result,
        test_seqs=test_seqs,
        selected_metrics=selected_metrics,
        device=device,
        requested_profile=requested_profile,
        canonical_profile=canonical_profile,
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    artifacts = {
        "metrics": metrics_path,
        "evaluation_manifest": manifest_path,
    }
    for path in per_event_files:
        artifacts[f"per_event_{path.stem.removesuffix('_per_event')}"] = path
    artifacts["manifest"] = _write_manifest(out_dir, artifacts)
    print(report.summary())
    _print_artifacts(artifacts)


def _load_metric_sequences(
    path: Path,
    *,
    max_seqs: int | None,
    max_events: int | None,
    load_jsonl,
) -> list[dict[str, np.ndarray]]:
    if not path.exists():
        raise SystemExit(f"Evaluation data JSONL not found: {path}")
    seqs = load_jsonl(path)
    if max_seqs is not None:
        seqs = seqs[: int(max_seqs)]
    out: list[dict[str, np.ndarray]] = []
    for seq in seqs:
        times = np.asarray(seq["times"], dtype=np.float32)
        locs = np.asarray(seq["locations"], dtype=np.float32)
        if max_events is not None and times.shape[0] > int(max_events):
            times = times[: int(max_events)]
            locs = locs[: int(max_events)]
        out.append({"times": times, "locations": locs})
    return out


def _default_metrics_out_dir(
    *,
    run_dir: Path,
    profile: str,
    split: str,
    out_override: str | None,
) -> Path:
    if out_override:
        return Path(out_override).resolve()
    return (run_dir / "evaluate" / "metrics" / f"{profile}_{split}").resolve()


def _metrics_evaluation_manifest(
    *,
    run_dir: Path,
    data_path: Path,
    split: str,
    out_dir: Path,
    artifact_dir: Path,
    metrics_path: Path,
    per_event_files: list[Path],
    args,
    report,
    result,
    test_seqs: list[dict[str, np.ndarray]],
    selected_metrics,
    device,
    requested_profile: str,
    canonical_profile: str,
) -> dict[str, Any]:
    run_id = run_dir.name
    preset = getattr(result, "preset", None)
    preset_status = getattr(result, "preset_status", None)
    nll_kind = getattr(result, "nll_kind", None)
    nll_report_space = getattr(result, "nll_report_space", None)
    if result is None:
        preset = None
        preset_status = None
        nll_kind = None
        nll_report_space = None
    metric_names = (
        [metric.name for metric in selected_metrics]
        if selected_metrics is not None
        else list(report.results.keys())
    )
    return {
        "schema_version": 1,
        "evaluation_id": _metrics_evaluation_id(
            run_dir=run_dir,
            data_path=data_path,
            split=split,
            profile=canonical_profile,
            seed=int(args.seed),
            k_pred=int(args.k_pred),
            k_gen=int(args.k_gen),
            metric_names=metric_names,
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": args.benchmark_id,
        "source_run": {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "preset": preset,
            "preset_status": preset_status,
            "nll_kind": nll_kind,
            "nll_report_space": nll_report_space,
        },
        "data": {
            "path": str(data_path),
            "split": split,
            "max_seqs": args.max_seqs,
            "max_events": args.max_events,
            "n_sequences": len(test_seqs),
            "n_events": int(sum(np.asarray(seq["times"]).shape[0] for seq in test_seqs)),
        },
        "metrics": {
            "profile": canonical_profile,
            "requested_profile": (
                requested_profile if requested_profile != canonical_profile else None
            ),
            "explicit_metric_names": (
                [metric.name for metric in selected_metrics]
                if selected_metrics is not None
                else None
            ),
            "result_names": list(report.results.keys()),
            "report": str(metrics_path),
            "per_event_files": [str(path) for path in per_event_files],
        },
        "artifacts": {
            "artifact_dir": str(artifact_dir),
            "artifact_mode": str(args.artifact_mode),
            "events": dict(report.artifact_events),
            "links": _artifact_links(artifact_dir, report.artifact_events),
        },
        "execution": {
            "device": str(device),
            "seed": int(args.seed),
            "k_pred": int(args.k_pred),
            "k_gen": int(args.k_gen),
            "exact_time_bins": int(args.exact_time_bins),
            "exact_spatial_bins": int(args.exact_spatial_bins),
            "out_dir": str(out_dir),
        },
    }


def _metrics_evaluation_id(
    *,
    run_dir: Path,
    data_path: Path,
    split: str,
    profile: str,
    seed: int,
    k_pred: int,
    k_gen: int,
    metric_names: list[str],
) -> str:
    import hashlib

    payload = {
        "run_dir": str(run_dir),
        "data_path": str(data_path),
        "split": split,
        "profile": profile,
        "seed": int(seed),
        "k_pred": int(k_pred),
        "k_gen": int(k_gen),
        "metric_names": list(metric_names),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def _artifact_links(artifact_dir: Path, events: dict[str, str]) -> dict[str, dict[str, str | None]]:
    links: dict[str, dict[str, str | None]] = {}
    for family, event in events.items():
        action, sep, key = str(event).partition(":")
        entry: dict[str, str | None] = {
            "event": action,
            "key": key if sep else None,
            "manifest": None,
            "payload": None,
        }
        if sep and key:
            base = artifact_dir / family / key
            entry["manifest"] = str(base / "manifest.json")
            entry["payload"] = str(base / f"{family}.npz")
        links[family] = entry
    return links


def _execute_predictive_compare(args) -> None:
    from unified_stpp.evaluation.common import HistoryQuery, RunTarget
    from unified_stpp.evaluation.io import load_predictive_bundle, write_predictive_bundle
    from unified_stpp.evaluation.predictive_compare import (
        PredictiveComparator,
        PredictiveCompareSpec,
    )
    from unified_stpp.evaluation.predictive_sampling import ExactProposalConfig
    from unified_stpp.viz import PredictiveRenderConfig, render_predictive_bundle

    labels = list(args.label or [])
    runs = list(args.run or [])
    if labels and len(labels) != len(runs):
        raise SystemExit("When --label is used, it must be repeated exactly once per --run.")
    run_targets = [
        RunTarget(run=Path(run), label=(labels[idx] if labels else None))
        for idx, run in enumerate(runs)
    ]
    split = _resolve_history_split(args.history, args.split)
    query = HistoryQuery(
        history_path=Path(args.history),
        split=split,
        seq_idx=int(args.seq_idx),
        start_event_idx=int(args.start_event_idx),
        history_length=int(args.history_length),
    )
    spec = PredictiveCompareSpec(
        rollout_mode=str(args.rollout_mode),
        n_frames=int(args.n_frames),
        horizon=float(args.horizon),
        step_size=None if args.step_size is None else float(args.step_size),
        n_rollouts=int(args.n_rollouts),
        grid_size=int(args.grid_size),
        bandwidth=args.bandwidth,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        lambda_bar=float(args.lambda_bar),
        max_events_per_window=int(args.max_events_per_window),
        bridge_retries=int(args.bridge_retries),
        adaptive_thinning=bool(args.adaptive_thinning),
        exact_proposal=ExactProposalConfig(
            mode=str(args.exact_proposal),
            time_bins=int(args.exact_time_bins),
            spatial_bins=int(args.exact_spatial_bins),
            safety=float(args.exact_safety),
        ),
        color_percentile=float(args.color_percentile),
        seed=int(args.seed),
        device=str(args.device),
    )

    result = PredictiveComparator().compare(run_targets, query, spec)
    out_dir = Path(args.out).resolve()
    artifacts = {}
    artifacts.update(write_predictive_bundle(out_dir, result))
    loaded = load_predictive_bundle(out_dir)
    artifacts.update(
        render_predictive_bundle(
            loaded,
            out_dir,
            PredictiveRenderConfig(
                plot_style=str(args.plot_style),
                fps=float(args.fps),
                write_gif=bool(args.gif),
            ),
        )
    )
    artifacts["manifest"] = _write_manifest(out_dir, artifacts)
    _print_artifacts(artifacts)


def _execute_surface(args) -> None:
    from unified_stpp.evaluation.common import HistoryQuery, RunTarget
    from unified_stpp.evaluation.io import load_surface_bundle, write_surface_bundle
    from unified_stpp.evaluation.surface import (
        SurfaceDiagnosticEvaluator,
        SurfaceDiagnosticSpec,
    )
    from unified_stpp.viz import SurfaceRenderConfig, render_surface_bundle

    split = _resolve_history_split(args.history, args.split)
    query = HistoryQuery(
        history_path=Path(args.history),
        split=split,
        seq_idx=int(args.seq_idx),
        history_length=int(args.history_length),
    )
    spec = SurfaceDiagnosticSpec(
        profile=str(args.profile),
        x_nstep=int(args.x_nstep),
        y_nstep=int(args.y_nstep),
        t_nstep=int(args.t_nstep),
        future_horizon=None if args.future_horizon is None else float(args.future_horizon),
        frame_index=int(args.frame_index),
        round_time=not bool(args.no_round_time),
        trunc=args.trunc,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        spatial_chunk_size=(
            None if args.spatial_chunk_size is None else int(args.spatial_chunk_size)
        ),
        device=str(args.device),
    )
    run_dir = Path(args.run).resolve()
    out_dir = _default_surface_out_dir(
        run_dir=run_dir,
        profile=str(args.profile),
        split=split,
        seq_idx=int(args.seq_idx),
        out_override=args.out,
    )
    result = SurfaceDiagnosticEvaluator().evaluate(RunTarget(run=run_dir), query, spec)
    artifacts = {}
    artifacts.update(write_surface_bundle(out_dir, result))
    loaded = load_surface_bundle(out_dir)
    artifacts.update(
        render_surface_bundle(
            loaded,
            out_dir,
            SurfaceRenderConfig(interactive=bool(args.interactive)),
        )
    )
    artifacts["manifest"] = _write_manifest(out_dir, artifacts)
    _print_artifacts(artifacts)


def _default_surface_out_dir(
    *,
    run_dir: Path,
    profile: str,
    split: str,
    seq_idx: int,
    out_override: str | None,
) -> Path:
    if out_override:
        return Path(out_override).resolve()
    return (
        run_dir
        / "evaluate"
        / "surface"
        / f"{profile}_{split}_seq{int(seq_idx):03d}"
    ).resolve()


def _resolve_history_split(history: str | Path, split: str | None) -> str:
    inferred = Path(history).stem
    inferred = inferred if inferred in _HISTORY_SPLITS else None
    if split is None:
        if inferred is None:
            raise SystemExit(
                "Could not infer --split from the input JSONL. Use a filename "
                "train.jsonl, val.jsonl, or test.jsonl, or pass --split explicitly."
            )
        return inferred
    if inferred is not None and split != inferred:
        raise SystemExit(
            f"--split {split!r} does not match input filename "
            f"{Path(history).name!r}; use --split {inferred!r} or choose a matching "
            "input file."
        )
    return split


def _write_manifest(out_dir: Path, artifacts: dict[str, Path]) -> Path:
    manifest_path = Path(out_dir) / "artifacts.json"
    payload = {name: str(path) for name, path in sorted(artifacts.items())}
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2)
    return manifest_path


def _print_artifacts(artifacts: dict[str, Path]) -> None:
    for name, path in sorted(artifacts.items()):
        print(f"  {name}: {path}")
