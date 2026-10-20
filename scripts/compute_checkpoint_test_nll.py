#!/usr/bin/env python3
"""Compute benchmark test NLL directly from an existing checkpoint.

This is intended for interrupted runs that have `config.yaml` and a Lightning
checkpoint but no final `run_result.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from seahorse.config import STPPConfig
from seahorse.runner.artifacts import load_norm_stats, load_state_dict
from seahorse.runner.runner import STPPRunner
from seahorse.training.lightning_module import STPPLightningModule
from seahorse.evaluation.likelihood import compute_next_event_test_nll


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Directory containing config.yaml and checkpoints/")
    p.add_argument("--checkpoint-select", choices=["best", "last"], default="last")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.add_argument("--max-seqs", type=int, default=None, help="Optional emergency subset; omit for full test split.")
    p.add_argument("--predictive-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def _raw_norm_stats() -> dict[str, Any]:
    return {
        "normalize": False,
        "coordinate_space": "raw",
        "time_mean": 0.0,
        "time_std": 1.0,
        "loc_mean": [0.0, 0.0],
        "loc_std": [1.0, 1.0],
    }


def _fallback_norm_stats(config: STPPConfig) -> dict[str, Any]:
    if not bool(config.data.normalize):
        return _raw_norm_stats()
    model = config.model
    transform = getattr(model, "input_transform", None)
    if transform is not None:
        return {
            "normalize": True,
            "coordinate_space": getattr(transform, "type", "zscore"),
            "time_mean": float(getattr(transform, "time_mean", 0.0)),
            "time_std": float(getattr(transform, "time_std", 1.0)),
            "loc_mean": list(getattr(transform, "loc_mean", [0.0, 0.0])),
            "loc_std": list(getattr(transform, "loc_std", [1.0, 1.0])),
        }
    return _raw_norm_stats()


def _load_runner(
    run_dir: Path,
    checkpoint_select: str,
    train_seqs: list[dict[str, Any]],
    val_seqs: list[dict[str, Any]],
    test_seqs: list[dict[str, Any]],
) -> STPPRunner:
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml or resolved_config.yaml under {run_dir}")

    config = STPPConfig.from_yaml(cfg_path, sanitize=False)
    config.training.checkpoint_select = checkpoint_select

    runner = STPPRunner(config)
    dm = runner._prepare_data_module(train_seqs, val_seqs, test_seqs, None)
    runner._sync_model_spatial_dim_from_data(dm)
    model = runner._build_model(dm)
    model.load_state_dict(load_state_dict(run_dir, selection=checkpoint_select))

    runner._lightning_module = STPPLightningModule(model=model, tc=config.training)
    runner._data_module = dm
    runner._run_dir = run_dir
    runner._norm_stats = load_norm_stats(run_dir) or dm.get_norm_stats(config.data.normalize) or _fallback_norm_stats(config)
    return runner


def _resolve_benchmark_split(
    config: STPPConfig,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    resolved = config.data.resolve_data(mode="benchmark")
    ids = sorted(resolved.splits)
    if len(ids) != 1:
        raise ValueError(f"Expected one dataset split, got {ids!r}")
    dataset_id = ids[0]
    train, val, test = resolved.splits[dataset_id]
    return dataset_id, train, val, test


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml or resolved_config.yaml under {run_dir}")
    split_config = STPPConfig.from_yaml(cfg_path, sanitize=False)
    dataset_id, train_seqs, val_seqs, test_seqs = _resolve_benchmark_split(split_config)
    runner = _load_runner(run_dir, args.checkpoint_select, train_seqs, val_seqs, test_seqs)
    n_test_total = len(test_seqs)
    if args.max_seqs is not None:
        test_seqs = test_seqs[: int(args.max_seqs)]

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    runner.model.to(device)
    result = compute_next_event_test_nll(
        runner,
        test_seqs,
        device=device,
        predictive_samples=args.predictive_samples,
        seed=args.seed,
    )

    elapsed = float(time.perf_counter() - t0)
    payload = {
        "run_dir": str(run_dir),
        "checkpoint_select": args.checkpoint_select,
        "checkpoint_path": str(run_dir / "checkpoints" / f"{args.checkpoint_select}.ckpt"),
        "dataset_id": dataset_id,
        "preset": runner.config.model.preset,
        "seed": runner.config.data.seed,
        "device": str(device),
        "test_nll": _finite(float(result["mean_nll"])),
        "nll_kind": result.get("kind"),
        "nll_report_space": result.get("report_space"),
        "test_nll_method": result.get("method"),
        "test_nll_contexts": int(result.get("n_contexts", 0)),
        "test_nll_scored_contexts": int(result.get("n_scored_contexts", 0)),
        "test_nll_missing_contexts": int(result.get("n_missing_contexts", 0)),
        "test_nll_unscored_contexts": int(result.get("n_unscored_contexts", 0)),
        "test_nll_failed_contexts": int(result.get("n_failed_contexts", 0)),
        "test_nll_nonfinite_scored_contexts": int(
            result.get("n_nonfinite_scored_contexts", 0)
        ),
        "test_sequences_used": len(test_seqs),
        "test_sequences_total": n_test_total,
        "max_seqs": args.max_seqs,
        "elapsed_sec": elapsed,
        "description": result.get("description"),
        "footnote": result.get("footnote"),
    }

    with open(out_dir / "checkpoint_test_nll.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_dir / "checkpoint_test_nll.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload))
        writer.writeheader()
        writer.writerow(payload)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
