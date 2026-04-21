#!/usr/bin/env python3
"""Tune-once, run-many synthetic-suite campaign pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unified_stpp.config.schema import STPPConfig
from unified_stpp.runner import STPPRunner
from unified_stpp.training.callbacks import PeriodicTestNLLCallback
from unified_stpp.utils import load_jsonl
from unified_stpp.utils import deep_update


@dataclass(frozen=True)
class SuiteConfig:
    config_id: str
    train_path: Path
    val_path: Path
    test_path: Path


@dataclass(frozen=True)
class RunIndexRecord:
    suite: str
    config_id: str
    preset: str
    seed: int
    run_dir: Path
    run_result_path: Path
    curve_jsonl_path: Path
    curve_csv_path: Path
    train_path: Path
    val_path: Path
    test_path: Path

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.suite, self.config_id, self.preset, self.seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "config_id": self.config_id,
            "preset": self.preset,
            "seed": self.seed,
            "run_dir": str(self.run_dir),
            "run_result_path": str(self.run_result_path),
            "curve_jsonl_path": str(self.curve_jsonl_path),
            "curve_csv_path": str(self.curve_csv_path),
            "train_path": str(self.train_path),
            "val_path": str(self.val_path),
            "test_path": str(self.test_path),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunIndexRecord":
        return cls(
            suite=str(payload["suite"]),
            config_id=str(payload["config_id"]),
            preset=str(payload["preset"]),
            seed=int(payload["seed"]),
            run_dir=Path(payload["run_dir"]),
            run_result_path=Path(payload["run_result_path"]),
            curve_jsonl_path=Path(payload["curve_jsonl_path"]),
            curve_csv_path=Path(payload["curve_csv_path"]),
            train_path=Path(payload["train_path"]),
            val_path=Path(payload["val_path"]),
            test_path=Path(payload["test_path"]),
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--suite-path", help="Path to one suite directory, e.g. hawkesnest_suitesv2/suite4_heterogeneity")
    group.add_argument("--suite-root", help="Root containing multiple suites, used together with --suite")
    p.add_argument("--suite", help="Suite name under --suite-root, e.g. suite4_heterogeneity")
    p.add_argument("--presets", nargs="+", required=True, help="Preset names to tune and run.")
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="Training seeds for the main suite runs.")
    p.add_argument("--out", required=True, help="Campaign output root.")
    p.add_argument("--stage", default="all", choices=("tune", "run", "plot", "all"))
    p.add_argument("--hpo-config-dir", default="unified_stpp/configs")
    p.add_argument("--curve-step", type=float, default=0.1)
    p.add_argument("--hpo-seed", type=int, default=42)
    p.add_argument(
        "--device",
        default=None,
        help="Optional training.device override for suite runs. When omitted, use the tuned YAML value.",
    )
    p.add_argument(
        "--run-batch-size",
        type=int,
        default=None,
        help="Optional training.batch_size override for suite runs only. Tune stage still uses the HPO YAML.",
    )
    resume_group = p.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", help="Skip completed tune/run outputs.")
    resume_group.add_argument("--force", dest="resume", action="store_false", help="Re-run tune/run stages and overwrite indexes.")
    p.set_defaults(resume=True)
    return p.parse_args(argv)


def _resolve_suite_path(args: argparse.Namespace) -> Path:
    if args.suite_path is not None:
        return Path(args.suite_path).expanduser().resolve()
    if not args.suite:
        raise ValueError("--suite is required when using --suite-root")
    return (Path(args.suite_root).expanduser() / args.suite).resolve()


def _discover_suite_configs(suite_path: Path) -> list[SuiteConfig]:
    jsonl_root = suite_path / "jsonl"
    if not jsonl_root.is_dir():
        raise FileNotFoundError(f"Expected suite jsonl directory at {jsonl_root}")
    configs: list[SuiteConfig] = []
    for ds_dir in sorted(p for p in jsonl_root.iterdir() if p.is_dir()):
        train_path = ds_dir / "train.jsonl"
        val_path = ds_dir / "val.jsonl"
        test_path = ds_dir / "test.jsonl"
        if not train_path.exists() or not val_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Suite config {ds_dir.name!r} must provide train.jsonl, val.jsonl, and test.jsonl"
            )
        configs.append(
            SuiteConfig(
                config_id=ds_dir.name,
                train_path=train_path.resolve(),
                val_path=val_path.resolve(),
                test_path=test_path.resolve(),
            )
        )
    if not configs:
        raise FileNotFoundError(f"No config directories found under {jsonl_root}")
    return configs


def _anchor_config(configs: list[SuiteConfig]) -> SuiteConfig:
    return sorted(configs, key=lambda cfg: cfg.config_id)[0]


def _safe_name(value: str) -> str:
    chars = []
    for ch in value:
        chars.append(ch if ch.isalnum() or ch in "._-" else "_")
    text = "".join(chars).strip("._")
    return text or "item"


def _ensure_hpo_configs(presets: list[str], hpo_config_dir: Path) -> None:
    missing = [preset for preset in presets if not (hpo_config_dir / f"{preset}_hpo.yaml").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing preset HPO YAML(s) in {hpo_config_dir}: {', '.join(sorted(missing))}"
        )


def _load_run_index(path: Path) -> dict[tuple[str, str, str, int], RunIndexRecord]:
    if not path.exists():
        return {}
    records: dict[tuple[str, str, str, int], RunIndexRecord] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = RunIndexRecord.from_dict(json.loads(line))
            records[record.key] = record
    return records


def _write_run_index(path: Path, records: dict[tuple[str, str, str, int], RunIndexRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for key in sorted(records):
            f.write(json.dumps(records[key].to_dict(), default=str) + "\n")


def _record_complete(record: RunIndexRecord) -> bool:
    return (
        record.run_dir.exists()
        and record.run_result_path.exists()
        and record.curve_jsonl_path.exists()
        and record.curve_csv_path.exists()
    )


def _campaign_paths(out_root: Path) -> dict[str, Path]:
    return {
        "root": out_root,
        "tune": out_root / "tune",
        "manifests": out_root / "manifests",
        "plots": out_root / "plots",
        "tables": out_root / "tables",
        "run_index": out_root / "manifests" / "run_index.jsonl",
        "campaign_manifest": out_root / "manifests" / "campaign_manifest.json",
    }


def _write_campaign_manifest(
    *,
    out_root: Path,
    suite_name: str,
    suite_path: Path,
    configs: list[SuiteConfig],
    anchor: SuiteConfig,
    presets: list[str],
    seeds: list[int],
    stage: str,
    curve_step: float,
    hpo_seed: int,
    device: str | None,
    run_batch_size: int | None,
) -> None:
    paths = _campaign_paths(out_root)
    paths["manifests"].mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": suite_name,
        "suite_path": str(suite_path),
        "generated_at": datetime.now().isoformat(),
        "stage": stage,
        "curve_step": curve_step,
        "hpo_seed": hpo_seed,
        "device_override": device,
        "run_batch_size": run_batch_size,
        "anchor_config_id": anchor.config_id,
        "presets": list(presets),
        "seeds": list(seeds),
        "configs": [
            {
                "config_id": cfg.config_id,
                "train_path": str(cfg.train_path),
                "val_path": str(cfg.val_path),
                "test_path": str(cfg.test_path),
            }
            for cfg in configs
        ],
    }
    with open(paths["campaign_manifest"], "w") as f:
        json.dump(payload, f, indent=2)


def _run_tune_subprocess(
    *,
    preset: str,
    hpo_yaml: Path,
    train_path: Path,
    val_path: Path,
    hpo_seed: int,
    out_path: Path,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "unified_stpp",
        "tune",
        "--config",
        str(hpo_yaml),
        "--train",
        str(train_path),
        "--val",
        str(val_path),
        "--seed",
        str(hpo_seed),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def _run_tune_stage(
    *,
    suite_name: str,
    anchor: SuiteConfig,
    presets: list[str],
    hpo_config_dir: Path,
    out_root: Path,
    hpo_seed: int,
    resume: bool,
) -> None:
    tune_dir = _campaign_paths(out_root)["tune"]
    tune_dir.mkdir(parents=True, exist_ok=True)
    for preset in presets:
        best_yaml = tune_dir / f"{preset}_best.yaml"
        if resume and best_yaml.exists():
            continue
        _run_tune_subprocess(
            preset=preset,
            hpo_yaml=(hpo_config_dir / f"{preset}_hpo.yaml").resolve(),
            train_path=anchor.train_path,
            val_path=anchor.val_path,
            hpo_seed=hpo_seed,
            out_path=best_yaml,
        )


def _fit_campaign_run(
    *,
    suite_name: str,
    config: SuiteConfig,
    preset: str,
    seed: int,
    best_yaml: Path,
    out_root: Path,
    curve_step: float,
    device: str | None,
    run_batch_size: int | None,
) -> RunIndexRecord:
    cli_values: dict[str, Any] = {
        "data": {
            "train_path": str(config.train_path),
            "val_path": str(config.val_path),
            "test_path": str(config.test_path),
            "seed": int(seed),
        },
        "logging": {
            "out_dir": str(out_root),
            "experiment_name": f"{suite_name}/{config.config_id}/{preset}/seed_{seed}",
        },
        "training": {
            "patience": None,
            "checkpoint_select": "best",
            "test_nll_space": "raw",
        },
    }
    if device is not None:
        cli_values["training"]["device"] = device
    if run_batch_size is not None:
        cli_values["training"]["batch_size"] = int(run_batch_size)

    tuned_cfg = STPPConfig.from_yaml(best_yaml, sanitize=False)
    merged_cfg = tuned_cfg.model_dump(mode="json")
    deep_update(merged_cfg, cli_values)
    runner = STPPRunner(STPPConfig(**merged_cfg))
    callback = PeriodicTestNLLCallback(
        suite=suite_name,
        config_id=config.config_id,
        preset=preset,
        seed=seed,
        curve_step=curve_step,
        config=runner.config,
    )
    train_seqs = load_jsonl(config.train_path)
    val_seqs = load_jsonl(config.val_path)
    test_seqs = load_jsonl(config.test_path)
    result = runner.fit(
        train_seqs,
        val_seqs,
        test_seqs,
        dataset_id=config.config_id,
        extra_callbacks=[callback],
    )
    run_dir = Path(result.run_dir).resolve()
    return RunIndexRecord(
        suite=suite_name,
        config_id=config.config_id,
        preset=preset,
        seed=seed,
        run_dir=run_dir,
        run_result_path=run_dir / "run_result.json",
        curve_jsonl_path=run_dir / "test_nll_curve.jsonl",
        curve_csv_path=run_dir / "test_nll_curve.csv",
        train_path=config.train_path,
        val_path=config.val_path,
        test_path=config.test_path,
    )


def _run_suite_stage(
    *,
    suite_name: str,
    configs: list[SuiteConfig],
    presets: list[str],
    seeds: list[int],
    out_root: Path,
    curve_step: float,
    device: str | None,
    run_batch_size: int | None,
    resume: bool,
) -> dict[tuple[str, str, str, int], RunIndexRecord]:
    paths = _campaign_paths(out_root)
    existing = _load_run_index(paths["run_index"])
    for preset in presets:
        best_yaml = paths["tune"] / f"{preset}_best.yaml"
        if not best_yaml.exists():
            raise FileNotFoundError(
                f"Missing tuned YAML for preset {preset!r}: expected {best_yaml}"
            )
        for config in configs:
            for seed in seeds:
                key = (suite_name, config.config_id, preset, int(seed))
                existing_record = existing.get(key)
                if resume and existing_record is not None and _record_complete(existing_record):
                    continue
                record = _fit_campaign_run(
                    suite_name=suite_name,
                    config=config,
                    preset=preset,
                    seed=seed,
                    best_yaml=best_yaml,
                    out_root=out_root,
                    curve_step=curve_step,
                    device=device,
                    run_batch_size=run_batch_size,
                )
                existing[key] = record
                _write_run_index(paths["run_index"], existing)
    return existing


def _read_curve_rows(records: dict[tuple[str, str, str, int], RunIndexRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(records):
        record = records[key]
        if not record.curve_csv_path.exists():
            continue
        with open(record.curve_csv_path, newline="") as f:
            for row in csv.DictReader(f):
                enriched = dict(row)
                enriched["suite"] = record.suite
                enriched["config_id"] = record.config_id
                enriched["preset"] = record.preset
                enriched["seed"] = int(record.seed)
                enriched["run_dir"] = str(record.run_dir)
                enriched["test_path"] = str(record.test_path)
                rows.append(enriched)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_curve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[float]] = {}
    meta: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    for row in rows:
        suite = str(row["suite"])
        preset = str(row["preset"])
        config_id = str(row["config_id"])
        progress_percent = float(row["train_progress_percent"])
        key = (suite, preset, config_id, progress_percent)
        grouped.setdefault(key, []).append(float(row["test_nll"]))
        meta[key] = {
            "suite": suite,
            "preset": preset,
            "config_id": config_id,
            "train_progress_fraction": float(row["train_progress_fraction"]),
            "train_progress_percent": progress_percent,
        }

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = np.asarray(grouped[key], dtype=np.float64)
        summary_rows.append(
            {
                **meta[key],
                "test_nll_mean": float(np.mean(values)) if values.size else float("nan"),
                "test_nll_std": float(np.std(values, ddof=0)) if values.size else float("nan"),
                "n": int(values.size),
            }
        )
    return summary_rows


def _plot_curve_family(*, suite_name: str, preset: str, rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    configs = sorted({str(row["config_id"]) for row in rows})
    for config_id in configs:
        subset = [row for row in rows if str(row["config_id"]) == config_id]
        subset.sort(key=lambda row: float(row["train_progress_percent"]))
        x = np.asarray([float(row["train_progress_percent"]) for row in subset], dtype=np.float64)
        y = np.asarray([float(row["test_nll_mean"]) for row in subset], dtype=np.float64)
        y_std = np.asarray([float(row["test_nll_std"]) for row in subset], dtype=np.float64)
        n = np.asarray([int(row["n"]) for row in subset], dtype=np.int64)

        ax.plot(x, y, marker="o", linewidth=1.8, label=config_id)
        if np.any(n > 1):
            band = np.where(n > 1, y_std, 0.0)
            ax.fill_between(x, y - band, y + band, alpha=0.18)

    ax.set_title(f"{suite_name} · {preset}")
    ax.set_xlabel("Training progress (%)")
    ax.set_ylabel("Benchmark test_nll")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Config", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _run_plot_stage(*, suite_name: str, out_root: Path) -> None:
    paths = _campaign_paths(out_root)
    records = _load_run_index(paths["run_index"])
    if not records:
        raise FileNotFoundError(f"No run index found at {paths['run_index']}")

    long_rows = _read_curve_rows(records)
    if not long_rows:
        raise FileNotFoundError("No test_nll_curve.csv files found for indexed runs.")

    long_fields = [
        "suite",
        "config_id",
        "preset",
        "seed",
        "epoch",
        "train_progress_fraction",
        "train_progress_percent",
        "test_nll",
        "nll_kind",
        "nll_report_space",
        "test_nll_method",
        "test_nll_contexts",
        "test_nll_scored_contexts",
        "test_nll_missing_contexts",
        "wall_time_sec",
        "run_dir",
        "test_path",
    ]
    _write_csv(paths["tables"] / "test_nll_curve_long.csv", long_rows, long_fields)

    summary_rows = _summarize_curve_rows(long_rows)
    summary_fields = [
        "suite",
        "preset",
        "config_id",
        "train_progress_fraction",
        "train_progress_percent",
        "test_nll_mean",
        "test_nll_std",
        "n",
    ]
    _write_csv(paths["tables"] / "test_nll_curve_summary.csv", summary_rows, summary_fields)

    target_rows = [
        {
            "suite": record.suite,
            "config_id": record.config_id,
            "preset": record.preset,
            "seed": record.seed,
            "run_dir": str(record.run_dir),
            "test_path": str(record.test_path),
        }
        for _, record in sorted(records.items())
        if _record_complete(record)
    ]
    _write_csv(
        paths["tables"] / "evaluate_targets.csv",
        target_rows,
        ["suite", "config_id", "preset", "seed", "run_dir", "test_path"],
    )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["suite"]), str(row["preset"])), []).append(row)
    for (suite, preset), rows in sorted(grouped.items()):
        out_path = paths["plots"] / f"{_safe_name(suite)}__{_safe_name(preset)}__test_nll_curve.png"
        _plot_curve_family(suite_name=suite, preset=preset, rows=rows, out_path=out_path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    suite_path = _resolve_suite_path(args)
    suite_name = suite_path.name
    out_root = Path(args.out).expanduser().resolve()
    paths = _campaign_paths(out_root)
    for key in ("tune", "manifests", "plots", "tables"):
        paths[key].mkdir(parents=True, exist_ok=True)

    configs = _discover_suite_configs(suite_path)
    anchor = _anchor_config(configs)
    hpo_config_dir = Path(args.hpo_config_dir).expanduser().resolve()
    if args.stage in {"tune", "all"}:
        _ensure_hpo_configs(args.presets, hpo_config_dir)

    _write_campaign_manifest(
        out_root=out_root,
        suite_name=suite_name,
        suite_path=suite_path,
        configs=configs,
        anchor=anchor,
        presets=args.presets,
        seeds=[int(seed) for seed in args.seeds],
        stage=args.stage,
        curve_step=float(args.curve_step),
        hpo_seed=int(args.hpo_seed),
        device=args.device,
        run_batch_size=args.run_batch_size,
    )

    if args.stage in {"tune", "all"}:
        _run_tune_stage(
            suite_name=suite_name,
            anchor=anchor,
            presets=args.presets,
            hpo_config_dir=hpo_config_dir,
            out_root=out_root,
            hpo_seed=int(args.hpo_seed),
            resume=bool(args.resume),
        )
    if args.stage in {"run", "all"}:
        _run_suite_stage(
            suite_name=suite_name,
            configs=configs,
            presets=args.presets,
            seeds=[int(seed) for seed in args.seeds],
            out_root=out_root,
            curve_step=float(args.curve_step),
            device=args.device,
            run_batch_size=args.run_batch_size,
            resume=bool(args.resume),
        )
    if args.stage in {"plot", "all"}:
        _run_plot_stage(
            suite_name=suite_name,
            out_root=out_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
