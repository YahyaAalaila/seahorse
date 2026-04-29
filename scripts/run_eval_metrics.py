#!/usr/bin/env python
"""Resolve dataset-backed inputs, then invoke the metrics evaluator directly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from unified_stpp.cli.evaluate import (
    _default_metrics_out_dir,
    _load_predictive_samples_for_report,
    _metrics_evaluation_manifest,
    _print_artifacts,
    _resolve_history_split,
    _write_manifest,
)
from unified_stpp.data.hub import download_dataset
from unified_stpp.evaluation.context import GroundTruth
from unified_stpp.evaluation.evaluator import evaluate
from unified_stpp.evaluation.predictive.benchmark import (
    write_next_event_benchmark_summary,
)
from unified_stpp.evaluation.profiles import (
    MetricPlanError,
    metric_profile as resolve_metric_profile,
)
from unified_stpp.evaluation.runtime import load_run_result, resolve_device
from unified_stpp.runner.runner import STPPRunner
from unified_stpp.utils import load_jsonl


def _resolve_data_paths(
    *,
    data: str | None,
    dataset: str | None,
    dataset_revision: str | None,
    split: str,
    train_data: str | None,
) -> tuple[Path, Path | None]:
    if data is not None:
        data_path = Path(data).expanduser().resolve()
        train_path = None if train_data is None else Path(train_data).expanduser().resolve()
        return data_path, train_path

    if dataset is None:
        raise ValueError("run_eval_metrics requires either --data or --dataset.")

    root = Path(
        download_dataset(
            dataset,
            revision=dataset_revision,
        )
    ).expanduser().resolve()
    data_path = root / f"{split}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Resolved dataset root {root} does not contain {split}.jsonl"
        )
    if train_data is not None:
        train_path = Path(train_data).expanduser().resolve()
    else:
        candidate = root / "train.jsonl"
        train_path = candidate.resolve() if candidate.exists() else None
    return data_path, train_path


def _load_metric_sequences(path: Path) -> list[dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation data JSONL not found: {path}")
    seqs = load_jsonl(path)
    out: list[dict[str, np.ndarray]] = []
    for seq in seqs:
        out.append(
            {
                "times": np.asarray(seq["times"], dtype=np.float32),
                "locations": np.asarray(seq["locations"], dtype=np.float32),
            }
        )
    return out


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path) as f:
        payload = json.load(f)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_optional_mask(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        payload = np.load(path)
        if "mask" in payload.files:
            data = payload["mask"]
        elif len(payload.files) == 1:
            data = payload[payload.files[0]]
        else:
            raise ValueError(
                f"{path} must contain exactly one array or a 'mask' array entry."
            )
    else:
        raise ValueError(f"Unsupported domain-mask format: {path}")
    return np.asarray(data, dtype=bool)


def _load_ground_truth_bundle(
    *,
    intensity_path: Path | None,
    params_path: Path | None,
    domain_mask_path: Path | None,
) -> tuple[GroundTruth | None, np.ndarray | None, dict[str, Any] | None]:
    params = _load_optional_json(params_path)
    domain_mask = _load_optional_mask(domain_mask_path)
    if intensity_path is None and params is None:
        return None, domain_mask, None

    intensity_grid = None
    grid_spec = None
    if intensity_path is not None:
        payload = np.load(intensity_path)
        if "lambda_" in payload.files:
            intensity_grid = np.asarray(payload["lambda_"], dtype=np.float32)
        elif "lambda_true" in payload.files:
            intensity_grid = np.asarray(payload["lambda_true"], dtype=np.float32)
        else:
            raise ValueError(
                f"{intensity_path} must contain 'lambda_' or 'lambda_true'."
            )

        x_grid = np.asarray(payload["x_grid"], dtype=np.float32) if "x_grid" in payload.files else None
        y_grid = np.asarray(payload["y_grid"], dtype=np.float32) if "y_grid" in payload.files else None
        t_grid = np.asarray(payload["t_grid"], dtype=np.float32) if "t_grid" in payload.files else None
        if x_grid is not None and y_grid is not None and t_grid is not None:
            grid_spec = {
                "x_range": [float(x_grid[0]), float(x_grid[-1])],
                "y_range": [float(y_grid[0]), float(y_grid[-1])],
                "x_resolution": int(x_grid.shape[0]),
                "y_resolution": int(y_grid.shape[0]),
                "t_resolution": int(t_grid.shape[0]),
            }

    return GroundTruth(intensity_grid=intensity_grid, params=params), domain_mask, grid_spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data")
    source.add_argument("--dataset")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--metric-profile", default="core")
    parser.add_argument("--artifact-mode", default="load_or_compute")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-pred", type=int, default=32)
    parser.add_argument("--k-gen", type=int, default=20)
    parser.add_argument("--exact-time-bins", type=int, default=8)
    parser.add_argument("--exact-spatial-bins", type=int, default=8)
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ground-truth-intensity", default=None)
    parser.add_argument("--ground-truth-params", default=None)
    parser.add_argument("--domain-mask", default=None)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    data_path, train_path = _resolve_data_paths(
        data=args.data,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        split=args.split,
        train_data=args.train_data,
    )
    split = _resolve_history_split(data_path, args.split)
    requested_profile = str(args.metric_profile)
    profile = resolve_metric_profile(requested_profile)
    canonical_profile = profile.name

    run_dir = Path(args.run).expanduser().resolve()
    out_dir = _default_metrics_out_dir(
        run_dir=run_dir,
        profile=canonical_profile,
        split=split,
        out_override=args.out,
    )
    artifact_dir = out_dir / "artifacts"
    device = resolve_device(str(args.device))
    test_seqs = _load_metric_sequences(data_path)
    train_seqs = None if train_path is None else _load_metric_sequences(train_path)

    gt_intensity_path = (
        None
        if args.ground_truth_intensity in (None, "")
        else Path(str(args.ground_truth_intensity)).expanduser().resolve()
    )
    gt_params_path = (
        None
        if args.ground_truth_params in (None, "")
        else Path(str(args.ground_truth_params)).expanduser().resolve()
    )
    domain_mask_path = (
        None
        if args.domain_mask in (None, "")
        else Path(str(args.domain_mask)).expanduser().resolve()
    )
    ground_truth, domain_mask, grid_spec = _load_ground_truth_bundle(
        intensity_path=gt_intensity_path,
        params_path=gt_params_path,
        domain_mask_path=domain_mask_path,
    )

    runner = STPPRunner.load(run_dir)
    runner.model.to(device)
    runner.model.eval()

    try:
        report = evaluate(
            runner,
            test_seqs,
            ground_truth=ground_truth,
            domain_mask=domain_mask,
            train_data=train_seqs,
            k_pred=int(args.k_pred),
            k_gen=int(args.k_gen),
            exact_time_bins=int(args.exact_time_bins),
            exact_spatial_bins=int(args.exact_spatial_bins),
            grid_spec=grid_spec,
            seed=int(args.seed),
            device=device,
            metric_profile=canonical_profile,
            artifact_dir=artifact_dir,
            artifact_mode=str(args.artifact_mode),
        )
    except MetricPlanError as exc:
        raise SystemExit(str(exc)) from None

    out_dir.mkdir(parents=True, exist_ok=True)
    report.save(out_dir)
    metrics_path = out_dir / "metrics.json"
    per_event_files = sorted(out_dir.glob("*_per_event.npy"))
    result = load_run_result(run_dir)

    predictive_outputs = None
    predictive_samples = _load_predictive_samples_for_report(
        artifact_dir=artifact_dir,
        report=report,
    )
    if predictive_samples is not None:
        predictive_outputs = write_next_event_benchmark_summary(
            out_dir=out_dir,
            report=report,
            samples=predictive_samples,
        )

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
        selected_metrics=None,
        device=device,
        requested_profile=requested_profile,
        canonical_profile=canonical_profile,
        predictive_samples=predictive_samples,
        predictive_outputs=predictive_outputs,
    )
    manifest["synthetic_ground_truth"] = {
        "intensity_path": None if gt_intensity_path is None else str(gt_intensity_path),
        "params_path": None if gt_params_path is None else str(gt_params_path),
        "domain_mask_path": None if domain_mask_path is None else str(domain_mask_path),
        "grid_spec": grid_spec,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    artifacts = {
        "metrics": metrics_path,
        "evaluation_manifest": manifest_path,
    }
    if predictive_outputs is not None:
        artifacts["next_event_benchmark_summary"] = predictive_outputs["summary_path"]
        artifacts["next_event_context_index"] = predictive_outputs["context_index_path"]
    for path in per_event_files:
        artifacts[f"per_event_{path.stem.removesuffix('_per_event')}"] = path
    artifacts["manifest"] = _write_manifest(out_dir, artifacts)
    print(report.summary())
    _print_artifacts(artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
