#!/usr/bin/env python3
"""Quick predictive-surface RMSE/MAE curves for HawkesNest benchmark runs.

This utility is intentionally a temporary analysis script. It discovers fitted
runs from a benchmark output directory, evaluates each run on the matching
dataset split, and compares the model-derived predictive KDE surface against a
KDE surface built from the true future events in the same frame.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_PRESETS = {"smash", "diffusion_stpp", "deep_stpp", "auto_stpp"}


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    preset: str
    dataset_id: str
    family: str
    level: int
    seed: int
    test_nll: float | None
    val_objective: float | None


def _record_to_payload(record: RunRecord) -> dict[str, Any]:
    return {
        "run_dir": str(record.run_dir),
        "preset": record.preset,
        "dataset_id": record.dataset_id,
        "family": record.family,
        "level": record.level,
        "seed": record.seed,
        "test_nll": record.test_nll,
        "val_objective": record.val_objective,
    }


def _record_from_payload(payload: dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_dir=Path(payload["run_dir"]),
        preset=str(payload["preset"]),
        dataset_id=str(payload["dataset_id"]),
        family=str(payload["family"]),
        level=int(payload["level"]),
        seed=int(payload["seed"]),
        test_nll=_as_float(payload.get("test_nll")),
        val_objective=_as_float(payload.get("val_objective")),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bench-run-root",
        required=True,
        help="Benchmark output root, e.g. runs/bench_hawkesnest_v20260409_deep_stpp.",
    )
    p.add_argument("--preset", required=True, help="Preset name to select, e.g. deep_stpp.")
    p.add_argument(
        "--splits-dir",
        required=True,
        help="Root containing <dataset_id>/{train,val,test}.jsonl.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for metric CSV/JSON and cached predictive bundles.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional explicit dataset ids, e.g. bg_BG0 bg_BG1.",
    )
    p.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Optional family filters, e.g. bg echo pulse. Defaults to all discovered.",
    )
    p.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=None,
        help="Optional complexity levels to keep, e.g. 0 1 2 3 4 5.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional seed filters. Defaults to all discovered seeds.",
    )
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--seq-idx", type=int, default=0)
    p.add_argument("--start-event-idx", type=int, default=20)
    p.add_argument("--history-length", type=int, default=0)
    p.add_argument(
        "--rollout-mode",
        default="teacher_forced",
        choices=("teacher_forced", "free_running"),
    )
    p.add_argument("--n-frames", type=int, default=2)
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--n-rollouts", type=int, default=16)
    p.add_argument("--grid-size", type=int, default=32)
    p.add_argument("--bandwidth", default=None)
    p.add_argument("--lambda-bar", type=float, default=10.0)
    p.add_argument("--max-events-per-window", type=int, default=64)
    p.add_argument("--bridge-retries", type=int, default=64)
    p.add_argument(
        "--no-adaptive-thinning",
        dest="adaptive_thinning",
        action="store_false",
        help="Disable adaptive thinning for exact-family rollouts.",
    )
    p.set_defaults(adaptive_thinning=True)
    p.add_argument("--exact-proposal", default="coarse", choices=("coarse", "uniform"))
    p.add_argument("--exact-time-bins", type=int, default=12)
    p.add_argument("--exact-spatial-bins", type=int, default=12)
    p.add_argument("--exact-safety", type=float, default=2.0)
    p.add_argument("--color-percentile", type=float, default=99.0)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute predictive bundles even if summary.json already exists.",
    )
    p.add_argument(
        "--no-isolate-process",
        dest="isolate_process",
        action="store_false",
        help=(
            "Evaluate all datasets in the parent process. By default, each dataset "
            "runs in a child process so PyTorch/MPS memory is released between levels."
        ),
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="Record failures and continue instead of stopping at the first failed dataset.",
    )
    p.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip writing rmse_mae_by_level.png.",
    )
    p.add_argument(
        "--with-renders",
        action="store_true",
        help="Write optional predictive 2D/3D render artifacts for each cached bundle.",
    )
    p.add_argument(
        "--plot-style",
        default="both",
        choices=("2d", "3d", "both"),
        help="Render style used when --with-renders is enabled.",
    )
    p.add_argument(
        "--gif",
        action="store_true",
        help="Also render predictive panel GIFs when --with-renders is enabled.",
    )
    p.add_argument("--fps", type=float, default=2.0)
    p.set_defaults(plot=True, isolate_process=True)
    return p.parse_args()


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("._") or "item"


def _parse_dataset_id(dataset_id: str) -> tuple[str, int]:
    family = dataset_id.split("_", 1)[0]
    suffix = dataset_id.rsplit("_", 1)[-1]
    match = re.search(r"(\d+)$", suffix)
    if match is None:
        raise ValueError(f"Could not parse numeric complexity level from dataset_id={dataset_id!r}")
    return family, int(match.group(1))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _discover_runs(bench_run_root: Path, preset: str) -> list[RunRecord]:
    records: list[RunRecord] = []
    seen: set[Path] = set()
    for result_path in sorted(bench_run_root.rglob("run_result.json")):
        run_dir = result_path.parent
        if run_dir.name == "latest":
            continue
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        with open(result_path) as f:
            payload = json.load(f)
        if str(payload.get("preset")) != preset:
            continue
        dataset_id = str(payload.get("dataset_id", ""))
        if not dataset_id:
            continue
        family, level = _parse_dataset_id(dataset_id)
        records.append(
            RunRecord(
                run_dir=run_dir,
                preset=preset,
                dataset_id=dataset_id,
                family=family,
                level=level,
                seed=int(payload.get("seed", 0)),
                test_nll=_as_float(payload.get("test_nll")),
                val_objective=_as_float(payload.get("val_objective")),
            )
        )
    return sorted(records, key=lambda r: (r.family, r.level, r.seed, str(r.run_dir)))


def _filter_runs(records: list[RunRecord], args: argparse.Namespace) -> list[RunRecord]:
    datasets = set(args.datasets or [])
    families = set(args.families or [])
    levels = set(args.levels or [])
    seeds = set(args.seeds or [])
    out = []
    for record in records:
        if datasets and record.dataset_id not in datasets:
            continue
        if families and record.family not in families:
            continue
        if levels and record.level not in levels:
            continue
        if seeds and record.seed not in seeds:
            continue
        out.append(record)
    return out


def _bundle_dir(out_dir: Path, record: RunRecord, args: argparse.Namespace) -> Path:
    name = (
        f"{record.preset}_{record.dataset_id}_seed{record.seed}_"
        f"{args.split}_seq{args.seq_idx:03d}_start{args.start_event_idx}_"
        f"{args.rollout_mode}_f{args.n_frames}_r{args.n_rollouts}_g{args.grid_size}"
    )
    return out_dir / "bundles" / _safe_name(name)


def _render_dir(bundle_dir: Path) -> Path:
    return bundle_dir / "renders"


def _surface_rmse(reference_surfaces: np.ndarray, predicted_surfaces: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_surfaces, dtype=np.float32)
    predicted = np.asarray(predicted_surfaces, dtype=np.float32)
    if reference.shape != predicted.shape:
        raise ValueError(
            f"reference/predicted surface shape mismatch: {reference.shape} vs {predicted.shape}"
        )
    diff = predicted - reference
    return np.sqrt(np.mean(np.square(diff), axis=(1, 2))).astype(np.float32)


def _surface_mae(reference_surfaces: np.ndarray, predicted_surfaces: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_surfaces, dtype=np.float32)
    predicted = np.asarray(predicted_surfaces, dtype=np.float32)
    if reference.shape != predicted.shape:
        raise ValueError(
            f"reference/predicted surface shape mismatch: {reference.shape} vs {predicted.shape}"
        )
    return np.mean(np.abs(predicted - reference), axis=(1, 2)).astype(np.float32)


def _count_error_metrics(true_counts: list[int], pred_mean_counts: list[float]) -> tuple[np.ndarray, np.ndarray]:
    true_arr = np.asarray(true_counts, dtype=np.float32)
    pred_arr = np.asarray(pred_mean_counts, dtype=np.float32)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"true/predicted count shape mismatch: {true_arr.shape} vs {pred_arr.shape}"
        )
    abs_err = np.abs(pred_arr - true_arr).astype(np.float32)
    signed_err = (pred_arr - true_arr).astype(np.float32)
    return abs_err, signed_err


def _run_or_load_bundle(
    record: RunRecord,
    args: argparse.Namespace,
    *,
    splits_dir: Path,
    out_dir: Path,
):
    from unified_stpp.evaluation.bundle_io import load_predictive_bundle, write_predictive_bundle
    from unified_stpp.evaluation.predictive import (
        ExactProposalConfig,
        PredictiveComparator,
        PredictiveCompareSpec,
    )
    from unified_stpp.evaluation.runtime import HistoryQuery, RunTarget
    from unified_stpp.viz import PredictiveRenderConfig, render_predictive_bundle

    bundle_dir = _bundle_dir(out_dir, record, args)
    reused = (bundle_dir / "summary.json").exists() and not args.force
    if reused:
        bundle = load_predictive_bundle(bundle_dir)
    else:
        history_path = splits_dir / record.dataset_id / f"{args.split}.jsonl"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history split for {record.dataset_id}: {history_path}")

        label = f"{record.preset}_{record.dataset_id}_seed{record.seed}"
        query = HistoryQuery(
            history_path=history_path,
            split=args.split,
            seq_idx=int(args.seq_idx),
            start_event_idx=int(args.start_event_idx),
            history_length=int(args.history_length),
        )
        spec = PredictiveCompareSpec(
            rollout_mode=args.rollout_mode,
            n_frames=int(args.n_frames),
            horizon=float(args.horizon),
            step_size=float(args.step_size),
            n_rollouts=int(args.n_rollouts),
            grid_size=int(args.grid_size),
            bandwidth=args.bandwidth,
            lambda_bar=float(args.lambda_bar),
            max_events_per_window=int(args.max_events_per_window),
            bridge_retries=int(args.bridge_retries),
            adaptive_thinning=bool(args.adaptive_thinning),
            exact_proposal=ExactProposalConfig(
                mode=args.exact_proposal,
                time_bins=int(args.exact_time_bins),
                spatial_bins=int(args.exact_spatial_bins),
                safety=float(args.exact_safety),
            ),
            color_percentile=float(args.color_percentile),
            seed=int(args.eval_seed),
            device=str(args.device),
        )
        bundle = PredictiveComparator().compare(
            [RunTarget(run=record.run_dir, label=label)],
            query,
            spec,
        )
        write_predictive_bundle(bundle_dir, bundle)

    render_dir = None
    if args.with_renders:
        render_dir = _render_dir(bundle_dir)
        render_predictive_bundle(
            bundle,
            render_dir,
            PredictiveRenderConfig(
                plot_style=str(args.plot_style),
                fps=float(args.fps),
                write_gif=bool(args.gif),
            ),
        )
    return bundle, bundle_dir, reused, render_dir


def _reference_surfaces(bundle, model) -> np.ndarray:
    from unified_stpp.evaluation.predictive.compare import kde_rate_surface

    surfaces = []
    for frame in model.frames:
        true_count = int(frame.true_event_locs.shape[0])
        surfaces.append(
            kde_rate_surface(
                frame.true_event_locs,
                xs=bundle.xs,
                ys=bundle.ys,
                bandwidth=bundle.spec.bandwidth,
                mean_events_per_rollout=float(true_count),
                window_duration=frame.window.duration,
            )
        )
    return np.stack(surfaces, axis=0).astype(np.float32)


def _metric_row(
    record: RunRecord,
    bundle,
    bundle_dir: Path,
    reused: bool,
    render_dir: Path | None,
) -> dict[str, Any]:
    if not bundle.models:
        raise ValueError(f"No models stored in bundle: {bundle_dir}")
    model = bundle.models[0]
    reference = _reference_surfaces(bundle, model)
    predicted = np.stack(
        [frame.derived_kde_rate_surface for frame in model.frames],
        axis=0,
    ).astype(np.float32)
    rmse = _surface_rmse(reference, predicted)
    mae = _surface_mae(reference, predicted)
    true_counts = [int(frame.true_event_locs.shape[0]) for frame in model.frames]
    pred_mean_counts = [float(frame.mean_events_per_rollout) for frame in model.frames]
    count_abs_err, count_signed_err = _count_error_metrics(true_counts, pred_mean_counts)
    elapsed_sec = [
        _as_float(frame.diagnostics.get("elapsed_sec"))
        for frame in model.frames
        if isinstance(frame.diagnostics, dict)
    ]
    elapsed_total = float(np.nansum([v for v in elapsed_sec if v is not None])) if elapsed_sec else 0.0
    render_path_2d = None
    render_path_3d = None
    if render_dir is not None:
        panel_2d = render_dir / "panels" / "frame_000.png"
        panel_3d = render_dir / "panels" / "frame_000_3d.png"
        if panel_2d.exists():
            render_path_2d = str(panel_2d)
        if panel_3d.exists():
            render_path_3d = str(panel_3d)
    return {
        "family": record.family,
        "level": record.level,
        "dataset_id": record.dataset_id,
        "preset": record.preset,
        "seed": record.seed,
        "run_dir": str(record.run_dir),
        "bundle_dir": str(bundle_dir),
        "bundle_reused": bool(reused),
        "render_dir": None if render_dir is None else str(render_dir),
        "panel_frame0_2d": render_path_2d,
        "panel_frame0_3d": render_path_3d,
        "model_label": model.label,
        "split": bundle.split,
        "seq_idx": bundle.seq_idx,
        "start_event_idx": bundle.start_event_idx,
        "n_frames": len(model.frames),
        "n_rollouts": int(bundle.spec.n_rollouts),
        "grid_size": int(bundle.spec.grid_size),
        "horizon": float(bundle.spec.horizon),
        "step_size": float(bundle.spec.step_size),
        "bandwidth": bundle.spec.bandwidth,
        "test_nll": record.test_nll,
        "val_objective": record.val_objective,
        "rmse_mean": float(np.mean(rmse)),
        "mae_mean": float(np.mean(mae)),
        "count_mae_mean": float(np.mean(count_abs_err)),
        "count_bias_mean": float(np.mean(count_signed_err)),
        "rmse_per_frame": [float(v) for v in rmse],
        "mae_per_frame": [float(v) for v in mae],
        "count_mae_per_frame": [float(v) for v in count_abs_err],
        "count_bias_per_frame": [float(v) for v in count_signed_err],
        "true_event_counts": true_counts,
        "pred_mean_event_counts": pred_mean_counts,
        "eval_elapsed_sec": elapsed_total,
    }


def _clear_backend_caches() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except Exception:
        pass


def _compute_metric_for_record(
    record: RunRecord,
    args: argparse.Namespace,
    *,
    splits_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    bundle = None
    try:
        bundle, bundle_dir, reused, render_dir = _run_or_load_bundle(
            record,
            args,
            splits_dir=splits_dir,
            out_dir=out_dir,
        )
        return _metric_row(record, bundle, bundle_dir, reused, render_dir)
    finally:
        del bundle
        _clear_backend_caches()


def _worker_entry(
    record_payload: dict[str, Any],
    args_payload: dict[str, Any],
    splits_dir_text: str,
    out_dir_text: str,
    row_path_text: str,
) -> None:
    row_path = Path(row_path_text)
    row_path.parent.mkdir(parents=True, exist_ok=True)
    record = _record_from_payload(record_payload)
    args = argparse.Namespace(**args_payload)
    try:
        row = _compute_metric_for_record(
            record,
            args,
            splits_dir=Path(splits_dir_text),
            out_dir=Path(out_dir_text),
        )
        payload = {"ok": True, "row": row}
    except BaseException as exc:
        payload = {
            "ok": False,
            "error": repr(exc),
            "record": record_payload,
        }
        with open(row_path, "w") as f:
            json.dump(payload, f, indent=2)
        raise
    with open(row_path, "w") as f:
        json.dump(payload, f, indent=2)


def _isolated_row_path(out_dir: Path, record: RunRecord, args: argparse.Namespace) -> Path:
    name = (
        f"{record.preset}_{record.dataset_id}_seed{record.seed}_"
        f"{args.split}_seq{args.seq_idx:03d}_start{args.start_event_idx}_"
        f"{args.rollout_mode}_f{args.n_frames}_r{args.n_rollouts}_g{args.grid_size}.json"
    )
    return out_dir / "_worker_rows" / _safe_name(name)


def _run_record_isolated(
    record: RunRecord,
    args: argparse.Namespace,
    *,
    splits_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    row_path = _isolated_row_path(out_dir, record, args)
    if row_path.exists():
        row_path.unlink()
    args_payload = vars(args).copy()
    args_payload["isolate_process"] = False
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_worker_entry,
        args=(
            _record_to_payload(record),
            args_payload,
            str(splits_dir),
            str(out_dir),
            str(row_path),
        ),
    )
    proc.start()
    proc.join()

    payload: dict[str, Any] | None = None
    if row_path.exists():
        with open(row_path) as f:
            payload = json.load(f)
    if proc.exitcode != 0:
        if payload and not payload.get("ok", False):
            raise RuntimeError(payload.get("error", f"worker exited with {proc.exitcode}"))
        raise RuntimeError(f"worker exited with {proc.exitcode}")
    if not payload:
        raise RuntimeError(f"worker did not write a row file: {row_path}")
    if not payload.get("ok", False):
        raise RuntimeError(payload.get("error", "worker failed"))
    return dict(payload["row"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    if value is None:
        return ""
    return value


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["family"]),
            int(row["level"]),
            str(row["dataset_id"]),
            str(row["preset"]),
        )
        groups[key].append(row)

    out = []
    for (family, level, dataset_id, preset), group in sorted(groups.items()):
        rmse = np.asarray([row["rmse_mean"] for row in group], dtype=np.float64)
        mae = np.asarray([row["mae_mean"] for row in group], dtype=np.float64)
        count_mae = np.asarray([row["count_mae_mean"] for row in group], dtype=np.float64)
        count_bias = np.asarray([row["count_bias_mean"] for row in group], dtype=np.float64)
        test_nll_values = [row["test_nll"] for row in group if row["test_nll"] is not None]
        test_nll = np.asarray(test_nll_values, dtype=np.float64) if test_nll_values else np.asarray([])
        out.append(
            {
                "family": family,
                "level": level,
                "dataset_id": dataset_id,
                "preset": preset,
                "n_runs": len(group),
                "seeds": sorted(int(row["seed"]) for row in group),
                "rmse_mean": float(np.mean(rmse)),
                "rmse_std": float(np.std(rmse, ddof=1)) if len(rmse) > 1 else 0.0,
                "mae_mean": float(np.mean(mae)),
                "mae_std": float(np.std(mae, ddof=1)) if len(mae) > 1 else 0.0,
                "count_mae_mean": float(np.mean(count_mae)),
                "count_mae_std": float(np.std(count_mae, ddof=1)) if len(count_mae) > 1 else 0.0,
                "count_bias_mean": float(np.mean(count_bias)),
                "count_bias_std": float(np.std(count_bias, ddof=1)) if len(count_bias) > 1 else 0.0,
                "test_nll_mean": float(np.mean(test_nll)) if test_nll.size else None,
                "test_nll_std": (
                    float(np.std(test_nll, ddof=1))
                    if test_nll.size > 1
                    else (0.0 if test_nll.size else None)
                ),
            }
        )
    return out


def _write_plot(path: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] Could not import matplotlib; skipping plot: {exc}")
        return

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_family[str(row["family"])].append(row)
    if not by_family:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for family, rows in sorted(by_family.items()):
        rows = sorted(rows, key=lambda r: int(r["level"]))
        levels = [int(row["level"]) for row in rows]
        axes[0].plot(levels, [float(row["rmse_mean"]) for row in rows], marker="o", label=family)
        axes[1].plot(levels, [float(row["mae_mean"]) for row in rows], marker="o", label=family)
        axes[2].plot(levels, [float(row["count_mae_mean"]) for row in rows], marker="o", label=family)
    axes[0].set_title("Predictive Surface RMSE")
    axes[1].set_title("Predictive Surface MAE")
    axes[2].set_title("Predictive Count MAE")
    for ax in axes:
        ax.set_xlabel("complexity level")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("MAE")
    axes[2].set_ylabel("count MAE")
    axes[2].legend(title="family", loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No metric rows produced.")
        return
    header = (
        f"{'family':<10} {'lvl':>3} {'n':>2} "
        f"{'rmse':>10} {'mae':>10} {'cnt_mae':>10} {'test_nll':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        test_nll = row["test_nll_mean"]
        test_text = "" if test_nll is None else f"{float(test_nll):10.4f}"
        print(
            f"{row['family']:<10} {int(row['level']):>3} {int(row['n_runs']):>2} "
            f"{float(row['rmse_mean']):10.4f} {float(row['mae_mean']):10.4f} "
            f"{float(row['count_mae_mean']):10.4f} {test_text:>10}"
        )


def main() -> None:
    args = _parse_args()
    if args.preset not in SUPPORTED_PRESETS:
        raise SystemExit(
            f"{args.preset!r} is not supported by the predictive-sampling evaluator yet. "
            f"Supported presets: {sorted(SUPPORTED_PRESETS)}"
        )
    bench_run_root = Path(args.bench_run_root)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _filter_runs(_discover_runs(bench_run_root, args.preset), args)
    if not records:
        raise SystemExit("No matching runs found. Check --bench-run-root, --preset, and filters.")

    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for idx, record in enumerate(records, start=1):
        print(
            f"[{idx}/{len(records)}] {record.preset} {record.dataset_id} "
            f"seed={record.seed} level={record.level}"
        )
        try:
            if bool(args.isolate_process):
                row = _run_record_isolated(
                    record,
                    args,
                    splits_dir=splits_dir,
                    out_dir=out_dir,
                )
            else:
                row = _compute_metric_for_record(
                    record,
                    args,
                    splits_dir=splits_dir,
                    out_dir=out_dir,
                )
            metric_rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    "dataset_id": record.dataset_id,
                    "seed": str(record.seed),
                    "run_dir": str(record.run_dir),
                    "error": repr(exc),
                }
            )
            if not args.keep_going:
                raise
            print(f"[warn] failed {record.dataset_id} seed={record.seed}: {exc!r}")

    aggregate_rows = _aggregate(metric_rows)
    _write_csv(out_dir / "metrics_by_run.csv", metric_rows)
    _write_csv(out_dir / "metrics_by_family_level.csv", aggregate_rows)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "metrics_by_run": metric_rows,
                "metrics_by_family_level": aggregate_rows,
                "failures": failures,
            },
            f,
            indent=2,
        )
    if args.plot:
        _write_plot(out_dir / "rmse_mae_by_level.png", aggregate_rows)

    _print_table(aggregate_rows)
    print(f"\nwrote: {out_dir / 'metrics_by_family_level.csv'}")
    print(f"wrote: {out_dir / 'metrics_by_run.csv'}")
    if failures:
        print(f"failures: {len(failures)} (see {out_dir / 'metrics.json'})")


if __name__ == "__main__":
    main()
