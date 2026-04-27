#!/usr/bin/env python3
"""Collect and render paper-readiness status for training and evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SYNTHETIC_METHODS = (
    "auto_stpp",
    "deep_stpp",
    "smash",
    "diffusion_stpp",
    "nsmpp",
    "rmtpp_gmm",
    "thp_gmm",
    "neural_attncnf",
    "neural_jumpcnf",
    "neural_cond_gmm",
)

REALDATA_METHODS = (
    "auto_stpp",
    "deep_stpp",
    "smash",
    "diffusion_stpp",
    "nsmpp",
    "rmtpp_gmm",
    "thp_gmm",
    "neural_attncnf",
    "neural_jumpcnf",
    "neural_cond_gmm",
    "poisson_gmm",
    "hawkes_gmm",
    "selfcorrecting_gmm",
    "poisson_cnf",
    "hawkes_cnf",
    "selfcorrecting_cnf",
    "poisson_tvcnf",
    "hawkes_tvcnf",
    "selfcorrecting_tvcnf",
)

SUITES = (
    "suite1_branching",
    "suite2_training_size",
    "suite3_entanglement",
    "suite4_heterogeneity",
    "suite5_topology",
    "suite6_corner",
)

REALDATA_DATASETS = (
    "covid-stpp",
    "earthquakes-stpp",
    "citibike-stpp",
    "bold5000-stpp",
)


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    job_name: str
    state: str
    exit_code: str
    start: str
    end: str


ERROR_PATTERNS = (
    "Traceback",
    "RuntimeError",
    "ValueError",
    "FileNotFoundError",
    "AssertionError",
    "torch.OutOfMemoryError",
    "CUDA out of memory",
    "Out Of Memory",
    "oom_kill",
    "Killed",
    "TIMEOUT",
    "CANCELLED",
    "srun: error",
)


def _safe_run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_repo_path(repo_root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _parse_space_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item for item in raw.split() if item]


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _job_record_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["job_id"]: row for row in rows if row.get("job_id")}


def _state_bucket(state: str | None) -> str:
    text = (state or "").upper()
    if "RUNNING" in text:
        return "running"
    if "PENDING" in text:
        return "pending"
    if "COMPLETED" in text:
        return "done"
    if "FAILED" in text:
        return "failed"
    if "TIMEOUT" in text:
        return "timeout"
    if "CANCELLED" in text:
        return "cancelled"
    return "unknown"


def _job_failure_summary(repo_root: Path, job_id: str, cache: dict[str, str]) -> str:
    if not job_id:
        return ""
    cached = cache.get(job_id)
    if cached is not None:
        return cached
    logs_root = repo_root / "logs"
    matches = sorted(list(logs_root.glob(f"*_{job_id}.out")) + list(logs_root.glob(f"*_{job_id}.err")))
    summary = ""
    for path in matches:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for line in reversed(lines):
            if any(pattern in line for pattern in ERROR_PATTERNS):
                summary = re.sub(r"^\s+", "", line.strip())
                break
        if summary:
            break
    cache[job_id] = summary
    return summary


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _format_metric(mean: float | None, std: float | None, n: int) -> str:
    if mean is None:
        return ""
    if n <= 1:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {0.0 if std is None else std:.3f}"


def _display_status(
    *,
    n_done: int,
    n_expected: int,
    active_jobs: list[str],
    latest_job_state: str | None,
    mean: float | None,
    std: float | None,
) -> str:
    if n_expected > 0 and n_done == n_expected:
        return _format_metric(mean, std, n_done)
    if active_jobs:
        return "SR"
    bucket = _state_bucket(latest_job_state)
    if n_done > 0:
        return f"P{n_done}/{n_expected}"
    if bucket in {"failed", "timeout", "cancelled"}:
        return "F"
    return "M"


def _suite_configs(repo_root: Path) -> dict[str, list[str]]:
    suites: dict[str, list[str]] = {}
    suite_root = repo_root / "data" / "hawkesnest_suitesv2"
    for suite in SUITES:
        jsonl_root = suite_root / suite / "jsonl"
        if not jsonl_root.is_dir():
            suites[suite] = []
            continue
        suites[suite] = sorted(path.name for path in jsonl_root.iterdir() if path.is_dir())
    return suites


def _collect_live_jobs() -> list[dict[str, str]]:
    out = _safe_run(["squeue", "-u", "aalaila", "-h", "-o", "%i|%j|%T|%M|%R"])
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        rows.append(
            {
                "job_id": parts[0],
                "job_name": parts[1],
                "state": parts[2],
                "elapsed": parts[3],
                "reason_or_node": parts[4],
            }
        )
    return rows


def _collect_recent_jobs(since: str) -> list[dict[str, str]]:
    out = _safe_run(
        [
            "sacct",
            "-u",
            "aalaila",
            "-X",
            "-S",
            since,
            "--format=JobIDRaw,JobName%45,State,ExitCode,Start,End",
            "--noheader",
            "-P",
        ]
    )
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 6:
            continue
        rows.append(
            {
                "job_id": parts[0],
                "job_name": parts[1],
                "state": parts[2],
                "exit_code": parts[3],
                "start": parts[4],
                "end": parts[5],
            }
        )
    return rows


def _recent_job_index(rows: list[dict[str, str]]) -> dict[str, list[JobRecord]]:
    index: dict[str, list[JobRecord]] = defaultdict(list)
    for row in rows:
        index[row["job_name"]].append(
            JobRecord(
                job_id=row["job_id"],
                job_name=row["job_name"],
                state=row["state"],
                exit_code=row["exit_code"],
                start=row["start"],
                end=row["end"],
            )
        )
    return index


def _active_job_names_for_pattern(live_jobs: list[dict[str, str]], pattern: str) -> list[str]:
    return sorted({row["job_name"] for row in live_jobs if pattern in row["job_name"]})


def _active_job_names_exact(live_jobs: list[dict[str, str]], names: list[str]) -> list[str]:
    target = set(names)
    return sorted({row["job_name"] for row in live_jobs if row["job_name"] in target})


def _latest_job_state(job_names: list[str], recent_jobs: dict[str, list[JobRecord]]) -> str | None:
    latest: JobRecord | None = None
    for name in job_names:
        for record in recent_jobs.get(name, []):
            if latest is None or (record.start or "") > (latest.start or ""):
                latest = record
    return None if latest is None else latest.state


def _latest_job_state_by_pattern(pattern: str, recent_jobs: dict[str, list[JobRecord]]) -> str | None:
    latest: JobRecord | None = None
    for name, records in recent_jobs.items():
        if pattern not in name:
            continue
        for record in records:
            if latest is None or (record.start or "") > (latest.start or ""):
                latest = record
    return None if latest is None else latest.state


def _collect_synthetic_training(
    repo_root: Path,
    expected_seeds: list[int],
    live_jobs: list[dict[str, str]],
    recent_jobs: dict[str, list[JobRecord]],
) -> list[dict[str, Any]]:
    suite_configs = _suite_configs(repo_root)
    campaign_root = repo_root / "runs" / "hawkesnest_campaigns"
    manifests_by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    completed_cells: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    completed_results: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for manifest_path in campaign_root.rglob("manifests/campaign_manifest.json"):
        campaign_dir = manifest_path.parent.parent
        manifest = json.loads(manifest_path.read_text())
        suite = str(manifest.get("suite") or "")
        manifests_by_suite[suite].append({"campaign_name": campaign_dir.name, "manifest": manifest})
        run_index_path = manifest_path.parent / "run_index.jsonl"
        if not run_index_path.exists():
            continue
        for line in run_index_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            run_result_path = Path(row["run_result_path"])
            if run_result_path.exists():
                completed_cells[(suite, str(row["preset"]))].add((str(row["config_id"]), int(row["seed"])))
                payload = json.loads(run_result_path.read_text())
                completed_results[(suite, str(row["preset"]))].append(
                    {
                        "config_id": str(row["config_id"]),
                        "seed": int(row["seed"]),
                        "test_nll": payload.get("test_nll"),
                        "val_objective": payload.get("val_objective"),
                        "run_dir": str(payload.get("run_dir") or row.get("run_dir")),
                    }
                )

    rows: list[dict[str, Any]] = []
    for suite in SUITES:
        configs = suite_configs.get(suite, [])
        expected_cells = len(configs) * len(expected_seeds)
        suite_manifests = manifests_by_suite.get(suite, [])
        suite_campaign_names = [row["campaign_name"] for row in suite_manifests]
        for preset in SYNTHETIC_METHODS:
            done = completed_cells.get((suite, preset), set())
            missing = [
                {"config_id": cfg, "seed": seed}
                for cfg in configs
                for seed in expected_seeds
                if (cfg, seed) not in done
            ]
            active_jobs: list[str] = []
            candidate_job_names: list[str] = []
            for campaign_name in suite_campaign_names:
                if f"__{preset}__" in campaign_name or campaign_name.endswith(f"__{preset}"):
                    candidate_job_names.append(campaign_name)
                if preset in {"smash", "diffusion_stpp"} and "__gen__" in campaign_name:
                    candidate_job_names.append(campaign_name)
                if preset in {"auto_stpp", "deep_stpp", "nsmpp", "rmtpp_gmm", "thp_gmm"} and "__rest__" in campaign_name:
                    candidate_job_names.append(campaign_name)
                if preset in {"neural_attncnf", "neural_jumpcnf", "neural_cond_gmm"} and "__neural__" in campaign_name:
                    candidate_job_names.append(campaign_name)
            candidate_job_names = sorted(set(candidate_job_names))
            active_jobs = _active_job_names_exact(live_jobs, candidate_job_names)
            latest_state = _latest_job_state(candidate_job_names, recent_jobs)
            if expected_cells == 0:
                status = "not_configured"
            elif len(done) == expected_cells:
                status = "done"
            elif active_jobs:
                job_states = {row["state"] for row in live_jobs if row["job_name"] in active_jobs}
                status = "running" if "RUNNING" in job_states else "pending"
            elif len(done) > 0:
                status = "partial"
            else:
                status = "missing"
            rows.append(
                {
                    "kind": "synthetic_training",
                    "suite": suite,
                    "preset": preset,
                    "expected_cells": expected_cells,
                    "done_cells": len(done),
                    "status": status,
                    "configs": configs,
                    "expected_seeds": expected_seeds,
                    "missing_cells": missing,
                    "active_jobs": active_jobs,
                    "latest_job_state": latest_state,
                    "campaign_names": suite_campaign_names,
                    "done_results": sorted(
                        completed_results.get((suite, preset), []),
                        key=lambda item: (item["config_id"], item["seed"]),
                    ),
                }
            )
    return rows


def _collect_realdata_training(
    repo_root: Path,
    expected_seeds: list[int],
    live_jobs: list[dict[str, str]],
    recent_jobs: dict[str, list[JobRecord]],
) -> list[dict[str, Any]]:
    root = repo_root / "runs" / "exp1"
    done_seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
    bench_campaigns: dict[tuple[str, str], set[str]] = defaultdict(set)
    done_results: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run_result_path in root.rglob("run_result.json"):
        parts = run_result_path.parts
        try:
            bench_idx = parts.index("bench")
        except ValueError:
            continue
        dataset = parts[bench_idx - 1]
        campaign = parts[bench_idx + 1]
        preset = parts[bench_idx + 3]
        payload = json.loads(run_result_path.read_text())
        seed = payload.get("seed")
        if seed is None:
            continue
        done_seeds[(dataset, preset)].add(int(seed))
        bench_campaigns[(dataset, preset)].add(campaign)
        done_results[(dataset, preset)].append(
            {
                "seed": int(seed),
                "test_nll": payload.get("test_nll"),
                "val_objective": payload.get("val_objective"),
                "run_dir": str(payload.get("run_dir") or run_result_path.parent),
                "campaign": campaign,
            }
        )

    rows: list[dict[str, Any]] = []
    for dataset in REALDATA_DATASETS:
        for preset in REALDATA_METHODS:
            done = sorted(done_seeds.get((dataset, preset), set()))
            missing = [seed for seed in expected_seeds if seed not in done]
            pattern = f"{dataset}__{preset}__"
            active_jobs = _active_job_names_for_pattern(live_jobs, pattern)
            latest_state = _latest_job_state_by_pattern(pattern, recent_jobs)
            if len(done) == len(expected_seeds):
                status = "done"
            elif active_jobs:
                job_states = {row["state"] for row in live_jobs if row["job_name"] in active_jobs}
                status = "running" if "RUNNING" in job_states else "pending"
            elif done:
                status = "partial"
            else:
                status = "missing"
            rows.append(
                {
                    "kind": "realdata_training",
                    "dataset": dataset,
                    "preset": preset,
                    "expected_seeds": expected_seeds,
                    "done_seeds": done,
                    "missing_seeds": missing,
                    "status": status,
                    "active_jobs": active_jobs,
                    "latest_job_state": latest_state,
                    "campaigns": sorted(bench_campaigns.get((dataset, preset), set())),
                    "done_results": sorted(done_results.get((dataset, preset), []), key=lambda item: item["seed"]),
                }
            )
    return rows


def _collect_protocol_evals(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((repo_root / "runs" / "eval_protocol").rglob("evaluation_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        metrics_path = manifest_path.parent / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        available = sorted(name for name, payload in metrics.items() if isinstance(payload, dict) and payload.get("available"))
        unavailable = sorted(name for name, payload in metrics.items() if isinstance(payload, dict) and not payload.get("available"))
        src = manifest.get("source_run", {})
        rows.append(
            {
                "kind": "protocol_eval",
                "subject": str(manifest_path.parent.relative_to(repo_root / "runs" / "eval_protocol")),
                "profile": manifest.get("metrics", {}).get("profile"),
                "preset": src.get("preset"),
                "run_id": src.get("run_id"),
                "status": "done" if metrics_path.exists() else "missing",
                "available_metrics": available,
                "unavailable_metrics": unavailable,
                "root": str(manifest_path.parent),
            }
        )
    return rows


def _collect_predictive_kde_evals(
    repo_root: Path,
    live_jobs: list[dict[str, str]],
    recent_jobs: dict[str, list[JobRecord]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    campaign_root = repo_root / "runs" / "hawkesnest_campaigns"
    for manifest_path in sorted(campaign_root.rglob("manifests/campaign_manifest.json")):
        campaign_dir = manifest_path.parent.parent
        manifest = json.loads(manifest_path.read_text())
        suite = manifest.get("suite")
        if not str(suite).startswith("suite"):
            continue
        kde_root = campaign_dir / "evaluate" / "predictive_kde"
        if not kde_root.exists():
            continue
        for preset_dir in sorted(path for path in kde_root.iterdir() if path.is_dir()):
            metrics_json = preset_dir / "metrics.json"
            by_run_csv = preset_dir / "metrics_by_run.csv"
            by_level_csv = preset_dir / "metrics_by_family_level.csv"
            worker_rows = list((preset_dir / "_worker_rows").glob("*.json")) if (preset_dir / "_worker_rows").exists() else []
            complete = metrics_json.exists() and by_run_csv.exists() and by_level_csv.exists()
            job_name = f"{campaign_dir.name}__{preset_dir.name}__pkde"
            active_jobs = _active_job_names_for_pattern(live_jobs, job_name)
            latest_state = _latest_job_state([job_name], recent_jobs)
            if complete:
                status = "done"
            elif active_jobs:
                job_states = {row["state"] for row in live_jobs if row["job_name"] in active_jobs}
                status = "running" if "RUNNING" in job_states else "pending"
            elif latest_state and "FAILED" in latest_state:
                status = "failed"
            elif latest_state and "CANCELLED" in latest_state:
                status = "cancelled"
            elif worker_rows:
                status = "partial"
            else:
                status = "missing"
            rows.append(
                {
                    "kind": "predictive_kde_eval",
                    "suite": suite,
                    "campaign": campaign_dir.name,
                    "preset": preset_dir.name,
                    "status": status,
                    "worker_rows": len(worker_rows),
                    "metrics_json": metrics_json.exists(),
                    "metrics_by_run_csv": by_run_csv.exists(),
                    "metrics_by_family_level_csv": by_level_csv.exists(),
                    "active_jobs": active_jobs,
                    "latest_job_state": latest_state,
                    "root": str(preset_dir),
                }
            )
    return rows


def _collect_submission_reconciliation(
    repo_root: Path,
    live_jobs: list[dict[str, str]],
    recent_job_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    live_by_id = _job_record_by_id(live_jobs)
    recent_by_id = _job_record_by_id(recent_job_rows)
    failure_cache: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    def add_row(base: dict[str, Any], out_path: Path | None, expected_outputs: int | None, done_outputs: int | None) -> None:
        job_id = str(base.get("job_id") or "")
        live = live_by_id.get(job_id, {})
        recent = recent_by_id.get(job_id, {})
        state = live.get("state") or recent.get("state") or "UNKNOWN"
        base["current_state"] = state
        base["current_bucket"] = _state_bucket(state)
        base["job_started_at"] = recent.get("start", "")
        base["job_finished_at"] = recent.get("end", "")
        base["out_path"] = "" if out_path is None else str(out_path)
        base["expected_outputs"] = expected_outputs
        base["done_outputs"] = done_outputs
        base["failure_summary"] = _job_failure_summary(repo_root, job_id, failure_cache)
        rows.append(base)

    synthetic_ledger = repo_root / "runs" / "hawkesnest_campaigns" / "submissions.csv"
    for row in _load_csv_rows(synthetic_ledger):
        suite = row.get("suite", "")
        if not str(suite).startswith("suite"):
            continue
        out_root = _resolve_repo_path(repo_root, row.get("out_root"))
        expected = None
        done = None
        if out_root and out_root.exists():
            manifest_path = out_root / "manifests" / "campaign_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                configs = manifest.get("configs", [])
                seeds = manifest.get("seeds", [])
                presets = manifest.get("presets", [])
                expected = len(configs) * len(seeds) * len(presets)
            done = sum(1 for _ in out_root.rglob("run_result.json"))
        add_row(
            {
                "kind": "synthetic_training_submission",
                "submitted_at": row.get("submitted_at", ""),
                "job_id": row.get("job_id", ""),
                "job_name": row.get("job_name", ""),
                "scope": suite,
                "family_or_profile": row.get("family", ""),
                "presets": row.get("presets", ""),
            },
            out_root,
            expected,
            done,
        )

    for path in sorted((repo_root / "runs" / "exp1").glob("*/bench/submissions.csv")):
        for row in _load_csv_rows(path):
            out_root = _resolve_repo_path(repo_root, row.get("bench_out"))
            presets = _parse_space_list(row.get("presets"))
            seeds = _parse_space_list(row.get("seeds"))
            expected = len(presets) * len(seeds) if presets and seeds else None
            done = None
            if out_root and out_root.exists():
                done = sum(1 for _ in out_root.rglob("run_result.json"))
            add_row(
                {
                    "kind": "realdata_training_submission",
                    "submitted_at": row.get("submitted_at", ""),
                    "job_id": row.get("job_id", ""),
                    "job_name": row.get("job_name", ""),
                    "scope": row.get("dataset", ""),
                    "family_or_profile": row.get("family", ""),
                    "presets": row.get("presets", ""),
                },
                out_root,
                expected,
                done,
            )

    protocol_ledgers = sorted((repo_root / "runs" / "exp1").rglob("evaluate/submissions.csv"))
    for path in protocol_ledgers:
        for row in _load_csv_rows(path):
            out_dir = _resolve_repo_path(repo_root, row.get("out_dir"))
            done = 1 if out_dir and (out_dir / "metrics.json").exists() else 0
            add_row(
                {
                    "kind": "protocol_eval_submission",
                    "submitted_at": row.get("submitted_at", ""),
                    "job_id": row.get("job_id", ""),
                    "job_name": row.get("job_name", ""),
                    "scope": row.get("dataset_id", ""),
                    "family_or_profile": row.get("metric_profile", ""),
                    "presets": row.get("preset", ""),
                },
                out_dir,
                1,
                done,
            )

    pkde_ledger = repo_root / "runs" / "hawkesnest_campaigns" / "predictive_kde_submissions.csv"
    for row in _load_csv_rows(pkde_ledger):
        out_dir = _resolve_repo_path(repo_root, row.get("out_dir"))
        done = 0
        if out_dir and out_dir.exists():
            done = int(
                all((out_dir / name).exists() for name in ("metrics.json", "metrics_by_run.csv", "metrics_by_family_level.csv"))
            )
        add_row(
            {
                "kind": "predictive_kde_submission",
                "submitted_at": row.get("submitted_at", ""),
                "job_id": row.get("job_id", ""),
                "job_name": row.get("job_name", ""),
                "scope": row.get("suite", ""),
                "family_or_profile": "predictive_kde",
                "presets": row.get("preset", ""),
            },
            out_dir,
            1,
            done,
        )

    return rows


def build_snapshot(repo_root: Path, expected_seeds: list[int], sacct_since: str) -> dict[str, Any]:
    live_jobs = _collect_live_jobs()
    recent_job_rows = _collect_recent_jobs(sacct_since)
    recent_jobs = _recent_job_index(recent_job_rows)
    return {
        "created_at": _now_utc(),
        "repo_root": str(repo_root),
        "expected_seeds": expected_seeds,
        "sacct_since": sacct_since,
        "live_jobs": live_jobs,
        "recent_jobs": recent_job_rows,
        "synthetic_training": _collect_synthetic_training(repo_root, expected_seeds, live_jobs, recent_jobs),
        "realdata_training": _collect_realdata_training(repo_root, expected_seeds, live_jobs, recent_jobs),
        "protocol_evals": _collect_protocol_evals(repo_root),
        "predictive_kde_evals": _collect_predictive_kde_evals(repo_root, live_jobs, recent_jobs),
        "submission_reconciliation": _collect_submission_reconciliation(repo_root, live_jobs, recent_job_rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _render_synthetic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for row in rows:
        missing_preview = ", ".join(
            f"{item['config_id']}:seed_{item['seed']}" for item in row["missing_cells"][:6]
        )
        rendered.append(
            {
                "suite": row["suite"],
                "preset": row["preset"],
                "status": row["status"],
                "done_cells": row["done_cells"],
                "expected_cells": row["expected_cells"],
                "missing_cells_preview": missing_preview,
                "active_jobs": "; ".join(row["active_jobs"]),
                "latest_job_state": row["latest_job_state"] or "",
            }
        )
    return rendered


def _render_realdata_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for row in rows:
        rendered.append(
            {
                "dataset": row["dataset"],
                "preset": row["preset"],
                "status": row["status"],
                "done_seeds": ",".join(str(seed) for seed in row["done_seeds"]),
                "missing_seeds": ",".join(str(seed) for seed in row["missing_seeds"]),
                "active_jobs": "; ".join(row["active_jobs"]),
                "latest_job_state": row["latest_job_state"] or "",
            }
        )
    return rendered


def _render_protocol_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for row in rows:
        rendered.append(
            {
                "subject": row["subject"],
                "profile": row["profile"],
                "preset": row["preset"],
                "status": row["status"],
                "available_metrics": ",".join(row["available_metrics"]),
                "unavailable_metrics": ",".join(row["unavailable_metrics"]),
                "root": row["root"],
            }
        )
    return rendered


def _render_pkde_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for row in rows:
        rendered.append(
            {
                "suite": row["suite"],
                "campaign": row["campaign"],
                "preset": row["preset"],
                "status": row["status"],
                "worker_rows": row["worker_rows"],
                "metrics_json": row["metrics_json"],
                "metrics_by_run_csv": row["metrics_by_run_csv"],
                "metrics_by_family_level_csv": row["metrics_by_family_level_csv"],
                "active_jobs": "; ".join(row["active_jobs"]),
                "latest_job_state": row["latest_job_state"] or "",
                "root": row["root"],
            }
        )
    return rendered


def _render_submission_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for row in rows:
        rendered.append(
            {
                "kind": row["kind"],
                "submitted_at": row["submitted_at"],
                "job_id": row["job_id"],
                "job_name": row["job_name"],
                "scope": row["scope"],
                "family_or_profile": row["family_or_profile"],
                "presets": row["presets"],
                "current_state": row["current_state"],
                "current_bucket": row["current_bucket"],
                "expected_outputs": row["expected_outputs"],
                "done_outputs": row["done_outputs"],
                "failure_summary": row["failure_summary"],
                "out_path": row["out_path"],
            }
        )
    return rendered


def _synthetic_result_cells(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in snapshot["synthetic_training"]:
        values_by_config: dict[str, list[float]] = defaultdict(list)
        seeds_by_config: dict[str, list[int]] = defaultdict(list)
        for item in row["done_results"]:
            if item["test_nll"] is not None:
                values_by_config[item["config_id"]].append(float(item["test_nll"]))
            seeds_by_config[item["config_id"]].append(int(item["seed"]))
        missing_by_config: dict[str, list[int]] = defaultdict(list)
        for item in row["missing_cells"]:
            missing_by_config[item["config_id"]].append(int(item["seed"]))
        for config_id in row["configs"]:
            values = sorted(values_by_config.get(config_id, []))
            seeds_done = sorted(seeds_by_config.get(config_id, []))
            n_expected = len(row["expected_seeds"])
            mean, std = _mean_std(values)
            active_jobs = row["active_jobs"] if row["status"] in {"running", "pending"} else []
            display = _display_status(
                n_done=len(seeds_done),
                n_expected=n_expected,
                active_jobs=active_jobs,
                latest_job_state=row["latest_job_state"],
                mean=mean,
                std=std,
            )
            rows.append(
                {
                    "suite": row["suite"],
                    "preset": row["preset"],
                    "config_id": config_id,
                    "display": display,
                    "status": row["status"] if len(seeds_done) != n_expected else "done",
                    "test_nll_mean": mean,
                    "test_nll_std": std,
                    "n_done": len(seeds_done),
                    "n_expected": n_expected,
                    "done_seeds": ",".join(str(seed) for seed in seeds_done),
                    "missing_seeds": ",".join(str(seed) for seed in sorted(missing_by_config.get(config_id, []))),
                    "latest_job_state": row["latest_job_state"] or "",
                }
            )
    return rows


def _realdata_result_cells(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in snapshot["realdata_training"]:
        values = [float(item["test_nll"]) for item in row["done_results"] if item["test_nll"] is not None]
        mean, std = _mean_std(values)
        display = _display_status(
            n_done=len(row["done_seeds"]),
            n_expected=len(row["expected_seeds"]),
            active_jobs=row["active_jobs"],
            latest_job_state=row["latest_job_state"],
            mean=mean,
            std=std,
        )
        rows.append(
            {
                "dataset": row["dataset"],
                "preset": row["preset"],
                "display": display,
                "status": row["status"],
                "test_nll_mean": mean,
                "test_nll_std": std,
                "n_done": len(row["done_seeds"]),
                "n_expected": len(row["expected_seeds"]),
                "done_seeds": ",".join(str(seed) for seed in row["done_seeds"]),
                "missing_seeds": ",".join(str(seed) for seed in row["missing_seeds"]),
                "latest_job_state": row["latest_job_state"] or "",
            }
        )
    return rows


def _missing_targets(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in snapshot["synthetic_training"]:
        if row["status"] == "done":
            continue
        rows.append(
            {
                "kind": "synthetic_training",
                "scope": row["suite"],
                "preset": row["preset"],
                "status": row["status"],
                "missing_units": len(row["missing_cells"]),
                "details": ", ".join(f"{item['config_id']}:seed_{item['seed']}" for item in row["missing_cells"][:10]),
                "active_jobs": "; ".join(row["active_jobs"]),
            }
        )
    for row in snapshot["realdata_training"]:
        if row["status"] == "done":
            continue
        rows.append(
            {
                "kind": "realdata_training",
                "scope": row["dataset"],
                "preset": row["preset"],
                "status": row["status"],
                "missing_units": len(row["missing_seeds"]),
                "details": ",".join(str(seed) for seed in row["missing_seeds"]),
                "active_jobs": "; ".join(row["active_jobs"]),
            }
        )
    for row in snapshot["predictive_kde_evals"]:
        if row["status"] == "done":
            continue
        rows.append(
            {
                "kind": "predictive_kde_eval",
                "scope": row["suite"],
                "preset": row["preset"],
                "status": row["status"],
                "missing_units": 1,
                "details": row["campaign"],
                "active_jobs": "; ".join(row["active_jobs"]),
            }
        )
    return rows


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def render_report(snapshot: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / "cluster_status_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2))

    synthetic_rows = _render_synthetic_rows(snapshot["synthetic_training"])
    realdata_rows = _render_realdata_rows(snapshot["realdata_training"])
    protocol_rows = _render_protocol_rows(snapshot["protocol_evals"])
    pkde_rows = _render_pkde_rows(snapshot["predictive_kde_evals"])
    submission_rows = _render_submission_rows(snapshot["submission_reconciliation"])
    synthetic_result_rows = _synthetic_result_cells(snapshot)
    realdata_result_rows = _realdata_result_cells(snapshot)
    missing_rows = _missing_targets(snapshot)

    _write_csv(
        out_dir / "synthetic_training_status.csv",
        synthetic_rows,
        [
            "suite",
            "preset",
            "status",
            "done_cells",
            "expected_cells",
            "missing_cells_preview",
            "active_jobs",
            "latest_job_state",
        ],
    )
    _write_csv(
        out_dir / "realdata_training_status.csv",
        realdata_rows,
        [
            "dataset",
            "preset",
            "status",
            "done_seeds",
            "missing_seeds",
            "active_jobs",
            "latest_job_state",
        ],
    )
    _write_csv(
        out_dir / "protocol_eval_status.csv",
        protocol_rows,
        ["subject", "profile", "preset", "status", "available_metrics", "unavailable_metrics", "root"],
    )
    _write_csv(
        out_dir / "predictive_kde_eval_status.csv",
        pkde_rows,
        [
            "suite",
            "campaign",
            "preset",
            "status",
            "worker_rows",
            "metrics_json",
            "metrics_by_run_csv",
            "metrics_by_family_level_csv",
            "active_jobs",
            "latest_job_state",
            "root",
        ],
    )
    _write_csv(
        out_dir / "missing_targets.csv",
        missing_rows,
        ["kind", "scope", "preset", "status", "missing_units", "details", "active_jobs"],
    )
    _write_csv(
        out_dir / "submission_reconciliation.csv",
        submission_rows,
        [
            "kind",
            "submitted_at",
            "job_id",
            "job_name",
            "scope",
            "family_or_profile",
            "presets",
            "current_state",
            "current_bucket",
            "expected_outputs",
            "done_outputs",
            "failure_summary",
            "out_path",
        ],
    )
    _write_csv(
        out_dir / "live_jobs.csv",
        snapshot["live_jobs"],
        ["job_id", "job_name", "state", "elapsed", "reason_or_node"],
    )
    _write_csv(
        out_dir / "synthetic_test_nll_cells.csv",
        synthetic_result_rows,
        [
            "suite",
            "preset",
            "config_id",
            "display",
            "status",
            "test_nll_mean",
            "test_nll_std",
            "n_done",
            "n_expected",
            "done_seeds",
            "missing_seeds",
            "latest_job_state",
        ],
    )
    _write_csv(
        out_dir / "realdata_test_nll_cells.csv",
        realdata_result_rows,
        [
            "dataset",
            "preset",
            "display",
            "status",
            "test_nll_mean",
            "test_nll_std",
            "n_done",
            "n_expected",
            "done_seeds",
            "missing_seeds",
            "latest_job_state",
        ],
    )

    synthetic_done = sum(1 for row in snapshot["synthetic_training"] if row["status"] == "done")
    realdata_done = sum(1 for row in snapshot["realdata_training"] if row["status"] == "done")
    protocol_done = sum(1 for row in snapshot["protocol_evals"] if row["status"] == "done")
    pkde_done = sum(1 for row in snapshot["predictive_kde_evals"] if row["status"] == "done")

    md: list[str] = []
    md.append("# Paper Readiness Report")
    md.append("")
    md.append(f"- Snapshot created: `{snapshot['created_at']}`")
    md.append(f"- Repo root: `{snapshot['repo_root']}`")
    md.append(f"- Expected seeds: `{', '.join(map(str, snapshot['expected_seeds']))}`")
    md.append(f"- `sacct` window start: `{snapshot['sacct_since']}`")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- Synthetic suite method cells done: `{synthetic_done}/{len(snapshot['synthetic_training'])}`")
    md.append(f"- Real-data dataset/method rows done: `{realdata_done}/{len(snapshot['realdata_training'])}`")
    md.append(f"- Protocol eval bundles done: `{protocol_done}/{len(snapshot['protocol_evals'])}`")
    md.append(f"- Predictive-KDE eval bundles done: `{pkde_done}/{len(snapshot['predictive_kde_evals'])}`")
    md.append("")

    md.append("## Test NLL Tables")
    md.append("")
    md.append("Legend: `SR` still running, `F` failed/timed out/cancelled, `M` missing, `Pk/n` partial seeds complete.")
    md.append("")

    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in synthetic_result_rows:
        by_suite[row["suite"]].append(row)
    for suite in SUITES:
        suite_rows = by_suite.get(suite, [])
        if not suite_rows:
            continue
        config_ids = sorted({row["config_id"] for row in suite_rows})
        md.append(f"### {suite}")
        md.append("")
        table_rows: list[list[str]] = []
        for preset in SYNTHETIC_METHODS:
            preset_rows = {row["config_id"]: row for row in suite_rows if row["preset"] == preset}
            table_rows.append([preset] + [preset_rows.get(config_id, {}).get("display", "M") for config_id in config_ids])
        md.append(_table(["preset"] + config_ids, table_rows))
        md.append("")

    by_dataset: dict[str, dict[str, Any]] = {
        (row["dataset"], row["preset"]): row for row in realdata_result_rows
    }
    md.append("### Real Data")
    md.append("")
    real_results_table_rows: list[list[str]] = []
    for preset in REALDATA_METHODS:
        real_results_table_rows.append(
            [preset] + [by_dataset.get((dataset, preset), {}).get("display", "M") for dataset in REALDATA_DATASETS]
        )
    md.append(_table(["preset"] + list(REALDATA_DATASETS), real_results_table_rows))
    md.append("")

    live_rows = [
        [row["job_id"], row["job_name"], row["state"], row["elapsed"], row["reason_or_node"]]
        for row in snapshot["live_jobs"]
    ]
    md.append("## Live Jobs")
    md.append("")
    if live_rows:
        md.append(_table(["job_id", "job_name", "state", "elapsed", "where"], live_rows))
    else:
        md.append("_No live jobs found._")
    md.append("")

    md.append("## Synthetic Training")
    md.append("")
    synth_table_rows = [
        [
            row["suite"],
            row["preset"],
            row["status"],
            f"{row['done_cells']}/{row['expected_cells']}",
            row["missing_cells_preview"] or "-",
            row["active_jobs"] or "-",
        ]
        for row in synthetic_rows
    ]
    md.append(_table(["suite", "preset", "status", "done/expected", "missing preview", "active jobs"], synth_table_rows))
    md.append("")

    md.append("## Real-Data Training")
    md.append("")
    real_training_table_rows = [
        [
            row["dataset"],
            row["preset"],
            row["status"],
            row["done_seeds"] or "-",
            row["missing_seeds"] or "-",
            row["active_jobs"] or "-",
        ]
        for row in realdata_rows
    ]
    md.append(_table(["dataset", "preset", "status", "done seeds", "missing seeds", "active jobs"], real_training_table_rows))
    md.append("")

    md.append("## Protocol Evals")
    md.append("")
    proto_table_rows = [
        [
            row["subject"],
            row["profile"],
            row["preset"],
            row["status"],
            row["available_metrics"] or "-",
            row["unavailable_metrics"] or "-",
        ]
        for row in protocol_rows
    ]
    md.append(_table(["subject", "profile", "preset", "status", "available metrics", "unavailable metrics"], proto_table_rows))
    md.append("")

    md.append("## Predictive-KDE Evals")
    md.append("")
    pkde_table_rows = [
        [
            row["suite"],
            row["campaign"],
            row["preset"],
            row["status"],
            str(row["worker_rows"]),
            row["active_jobs"] or "-",
        ]
        for row in pkde_rows
    ]
    md.append(_table(["suite", "campaign", "preset", "status", "worker rows", "active jobs"], pkde_table_rows))
    md.append("")

    md.append("## Submission Reconciliation")
    md.append("")
    submission_table_rows = [
        [
            row["kind"],
            row["job_id"],
            row["job_name"],
            row["scope"],
            row["current_state"],
            (
                "-"
                if row["expected_outputs"] in ("", None) or row["done_outputs"] in ("", None)
                else f"{row['done_outputs']}/{row['expected_outputs']}"
            ),
            row["failure_summary"] or "-",
        ]
        for row in submission_rows
    ]
    md.append(_table(["kind", "job_id", "job_name", "scope", "state", "outputs", "failure summary"], submission_table_rows))
    md.append("")

    md.append("## Missing Targets")
    md.append("")
    missing_table_rows = [
        [
            row["kind"],
            row["scope"],
            row["preset"],
            row["status"],
            str(row["missing_units"]),
            row["details"] or "-",
            row["active_jobs"] or "-",
        ]
        for row in missing_rows
    ]
    md.append(_table(["kind", "scope", "preset", "status", "missing units", "details", "active jobs"], missing_table_rows))
    md.append("")

    (out_dir / "paper_readiness_report.md").write_text("\n".join(md) + "\n")

    synthetic_md: list[str] = ["# Synthetic Test NLL Tables", "", "Legend: `SR`, `F`, `M`, `Pk/n`.", ""]
    for suite in SUITES:
        suite_rows = by_suite.get(suite, [])
        if not suite_rows:
            continue
        config_ids = sorted({row["config_id"] for row in suite_rows})
        synthetic_md.append(f"## {suite}")
        synthetic_md.append("")
        table_rows = []
        for preset in SYNTHETIC_METHODS:
            preset_rows = {row["config_id"]: row for row in suite_rows if row["preset"] == preset}
            table_rows.append([preset] + [preset_rows.get(config_id, {}).get("display", "M") for config_id in config_ids])
        synthetic_md.append(_table(["preset"] + config_ids, table_rows))
        synthetic_md.append("")
    (out_dir / "synthetic_test_nll_tables.md").write_text("\n".join(synthetic_md) + "\n")

    real_md = [
        "# Real-Data Test NLL Table",
        "",
        "Legend: `SR`, `F`, `M`, `Pk/n`.",
        "",
        _table(["preset"] + list(REALDATA_DATASETS), real_results_table_rows),
        "",
    ]
    (out_dir / "realdata_test_nll_table.md").write_text("\n".join(real_md) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    snapshot_parser = subparsers.add_parser("snapshot", help="Collect a status snapshot from the current filesystem/Slurm host.")
    snapshot_parser.add_argument("--repo-root", default=".", help="Path to repo root on the machine running the snapshot.")
    snapshot_parser.add_argument("--expected-seeds", nargs="+", type=int, default=[42, 3, 555])
    snapshot_parser.add_argument(
        "--sacct-since",
        default=(datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%S"),
        help="Start time for recent accounting data, in sacct-compatible format.",
    )
    snapshot_parser.add_argument("--out", default="-", help="Output JSON path, or - for stdout.")

    render_parser = subparsers.add_parser("render", help="Render CSV/Markdown report from a snapshot JSON.")
    render_parser.add_argument("--snapshot", required=True, help="Snapshot JSON path.")
    render_parser.add_argument("--out-dir", required=True, help="Directory to write report artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.cmd == "snapshot":
        snapshot = build_snapshot(Path(args.repo_root).expanduser().resolve(), list(args.expected_seeds), args.sacct_since)
        if args.out == "-":
            json.dump(snapshot, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            out_path = Path(args.out).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(snapshot, indent=2))
        return 0

    snapshot_path = Path(args.snapshot).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    snapshot = json.loads(snapshot_path.read_text())
    render_report(snapshot, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
