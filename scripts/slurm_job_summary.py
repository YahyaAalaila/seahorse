#!/usr/bin/env python3
"""Write a compact Slurm job completion summary.

This script is intentionally stdlib-only so it can run on the host after a
containerized job exits. It is called from Slurm batch-script EXIT traps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ERROR_PATTERNS = (
    "Traceback",
    "RuntimeError",
    "ValueError",
    "AssertionError",
    "OutOfMemoryError",
    "CUDA out of memory",
    "Out Of Memory",
    "oom",
    "Killed",
    "TIMEOUT",
    "CANCELLED",
    "srun: error",
    "Exited with exit code",
)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _repo_root() -> Path:
    raw = _env("REPO_ROOT") or _env("SLURM_SUBMIT_DIR") or os.getcwd()
    return Path(raw).expanduser().resolve()


def _resolve_path(raw: str | None, repo_root: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _safe_job_id() -> str:
    return _env("SLURM_JOB_ID") or _env("JOB_ID") or "unknown-job"


def _safe_job_name() -> str:
    return _env("SLURM_JOB_NAME") or _env("JOB_NAME") or "unknown-job-name"


def _json_load(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_run_result(run_dir: Path | None) -> dict[str, Any] | None:
    return _json_load(None if run_dir is None else run_dir / "run_result.json")


def _read_metrics(out_dir: Path | None) -> dict[str, Any] | None:
    return _json_load(None if out_dir is None else out_dir / "metrics.json")


def _numeric_metrics(metrics: dict[str, Any] | None, limit: int = 12) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value == value:
            out[key] = float(value)
        if len(out) >= limit:
            break
    return out


def _count_files(root: Path | None, pattern: str) -> int:
    if root is None or not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def _latest_files(root: Path | None, patterns: tuple[str, ...], limit: int = 8) -> list[Path]:
    if root is None or not root.exists():
        return []
    hits: list[Path] = []
    for pattern in patterns:
        hits.extend(path for path in root.rglob(pattern) if path.is_file())
    hits = sorted(set(hits), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[:limit]


def _find_logs(repo_root: Path, job_id: str, job_name: str) -> list[Path]:
    logs = repo_root / "logs"
    if not logs.exists():
        return []
    candidates = list(logs.glob(f"*{job_id}*"))
    if not candidates and job_name:
        candidates = list(logs.glob(f"*{job_name}*"))
    return sorted(path for path in candidates if path.is_file())


def _failure_hint(logs: list[Path]) -> str:
    for path in reversed(logs):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for line in reversed(lines[-300:]):
            if any(pattern.lower() in line.lower() for pattern in ERROR_PATTERNS):
                return line.strip()
    return ""


def _sacct(job_id: str) -> dict[str, str]:
    if not job_id or job_id == "unknown-job" or shutil.which("sacct") is None:
        return {}
    cmd = [
        "sacct",
        "-j",
        job_id,
        "-X",
        "--format=JobIDRaw,JobName%80,State,ExitCode,Elapsed,Start,End,NodeList%30",
        "--noheader",
        "-P",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=8, check=False)
    except Exception:
        return {}
    rows = [line for line in proc.stdout.splitlines() if line.strip()]
    if not rows:
        return {}
    parts = rows[0].split("|")
    keys = ["job_id", "job_name", "state", "exit_code", "elapsed", "start", "end", "node_list"]
    return {key: parts[i] if i < len(parts) else "" for i, key in enumerate(keys)}


def _infer_seed(env: dict[str, str], run_result: dict[str, Any] | None) -> str:
    if run_result and run_result.get("seed") is not None:
        return str(run_result.get("seed"))
    for name in ("SEED", "EVAL_SEED"):
        if env.get(name):
            return env[name]
    text = " ".join(env.get(name, "") for name in ("BENCHMARK_ID", "SLURM_JOB_NAME", "JOB_NAME"))
    match = re.search(r"(?:^|__)s(?:eed_)?(\d+)(?:__|$)", text)
    if match:
        return match.group(1)
    return env.get("SEEDS", "")


def _infer_model(env: dict[str, str], run_result: dict[str, Any] | None) -> str:
    if run_result and run_result.get("preset"):
        return str(run_result.get("preset"))
    return env.get("PRESET") or env.get("PRESETS") or ""


def _infer_data(env: dict[str, str], run_result: dict[str, Any] | None) -> str:
    if run_result and run_result.get("dataset_id"):
        return str(run_result.get("dataset_id"))
    for name in ("DATASET", "DATASET_ID", "DATA_PATH", "HISTORY_PATH", "SPLITS_DIR", "SUITE_PATH", "CAMPAIGN_ROOT"):
        if env.get(name):
            return env[name]
    return ""


def _result_root(env: dict[str, str], repo_root: Path) -> Path | None:
    for name in (
        "UNIFIED_STPP_JOB_RESULT_ROOT",
        "OUT_DIR",
        "BENCH_OUT",
        "OUT_ROOT",
        "RUN_DIR",
        "CAMPAIGN_ROOT",
    ):
        path = _resolve_path(env.get(name), repo_root)
        if path is not None:
            return path
    return None


def _artifact_status(result_root: Path | None) -> dict[str, Any]:
    if result_root is None:
        return {}
    files = {
        "metrics_json": result_root / "metrics.json",
        "evaluation_manifest": result_root / "evaluation_manifest.json",
        "artifacts_json": result_root / "artifacts.json",
        "summary_json": result_root / "summary.json",
        "run_result_json": result_root / "run_result.json",
        "bench_meta_json": result_root / "bench_meta.json",
        "results_json": result_root / "results.json",
        "report_html": result_root / "report.html",
        "run_index_jsonl": result_root / "manifests" / "run_index.jsonl",
    }
    existing = {name: str(path) for name, path in files.items() if path.exists()}
    return {
        "result_root": str(result_root),
        "exists": result_root.exists(),
        "key_artifacts": existing,
        "run_result_count": _count_files(result_root, "run_result.json"),
        "metrics_count": _count_files(result_root, "metrics.json"),
        "latest_artifacts": [
            str(path)
            for path in _latest_files(
                result_root,
                (
                    "run_result.json",
                    "metrics.json",
                    "evaluation_manifest.json",
                    "summary.json",
                    "artifacts.json",
                    "results.json",
                    "report.html",
                    "test_nll_curve.csv",
                ),
            )
        ],
    }


def _write_outputs(summary: dict[str, Any], markdown: str, notify_dir: Path) -> tuple[Path, Path]:
    notify_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(summary["job"]["id"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{stamp}_{job_id}"
    json_path = notify_dir / f"{stem}.json"
    md_path = notify_dir / f"{stem}.md"
    ledger = notify_dir / "notifications.jsonl"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md_path.write_text(markdown)
    with ledger.open("a") as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")
    return json_path, md_path


def _send_email(to_addr: str, subject: str, body: str) -> bool:
    if not to_addr:
        return False
    mail_bin = shutil.which("mail") or shutil.which("mailx")
    if mail_bin:
        proc = subprocess.run(
            [mail_bin, "-s", subject, to_addr],
            input=body,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0
    sendmail = shutil.which("sendmail")
    if sendmail:
        payload = f"To: {to_addr}\nSubject: {subject}\n\n{body}"
        proc = subprocess.run(
            [sendmail, "-t"],
            input=payload,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0
    return False


def _markdown(summary: dict[str, Any]) -> str:
    job = summary["job"]
    run = summary["run"]
    artifacts = summary.get("artifacts", {})
    metrics = summary.get("metrics", {})
    lines = [
        f"# Unified STPP job {job['id']} {job['status_label']}",
        "",
        f"- job: `{job['name']}`",
        f"- state: `{job.get('slurm_state') or job['status_label']}` exit_code=`{job['exit_code']}` elapsed=`{job.get('elapsed') or ''}`",
        f"- pipeline: `{run.get('pipeline') or ''}`",
        f"- data: `{run.get('data') or ''}`",
        f"- model: `{run.get('model') or ''}`",
        f"- seed: `{run.get('seed') or ''}`",
        f"- results: `{artifacts.get('result_root') or ''}`",
    ]
    if artifacts.get("key_artifacts"):
        lines.append("- key artifacts:")
        for name, path in sorted(artifacts["key_artifacts"].items()):
            lines.append(f"  - `{name}`: `{path}`")
    if metrics:
        lines.append("- metric preview:")
        for key, value in metrics.items():
            lines.append(f"  - `{key}`: `{value:.6g}`")
    if summary.get("failure_hint"):
        lines.append(f"- failure hint: `{summary['failure_hint']}`")
    if summary.get("logs"):
        lines.append("- logs:")
        for path in summary["logs"]:
            lines.append(f"  - `{path}`")
    if artifacts.get("latest_artifacts"):
        lines.append("- latest artifacts:")
        for path in artifacts["latest_artifacts"]:
            lines.append(f"  - `{path}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args(argv)

    env = dict(os.environ)
    repo_root = _repo_root()
    job_id = _safe_job_id()
    job_name = _safe_job_name()
    run_dir = _resolve_path(env.get("RUN_DIR"), repo_root)
    run_result = _read_run_result(run_dir)
    result_root = _result_root(env, repo_root)
    metrics = _numeric_metrics(_read_metrics(result_root))
    sacct = _sacct(job_id)
    logs = _find_logs(repo_root, job_id, job_name)
    failure_hint = "" if args.exit_code == 0 else _failure_hint(logs)
    status_label = "OK" if args.exit_code == 0 else "FAILED"

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "job": {
            "id": job_id,
            "name": job_name,
            "status_label": status_label,
            "exit_code": int(args.exit_code),
            "slurm_state": sacct.get("state", ""),
            "slurm_exit_code": sacct.get("exit_code", ""),
            "elapsed": sacct.get("elapsed", ""),
            "start": sacct.get("start", ""),
            "end": sacct.get("end", ""),
            "node_list": sacct.get("node_list", _env("SLURM_JOB_NODELIST")),
        },
        "run": {
            "pipeline": env.get("UNIFIED_STPP_JOB_PIPELINE", ""),
            "data": _infer_data(env, run_result),
            "model": _infer_model(env, run_result),
            "seed": _infer_seed(env, run_result),
            "run_dir": "" if run_dir is None else str(run_dir),
            "benchmark_id": env.get("BENCHMARK_ID", ""),
            "metric_profile": env.get("METRIC_PROFILE", ""),
            "split": env.get("SPLIT", ""),
        },
        "artifacts": _artifact_status(result_root),
        "metrics": metrics,
        "failure_hint": failure_hint,
        "logs": [str(path) for path in logs],
    }

    notify_dir = _resolve_path(env.get("UNIFIED_STPP_NOTIFY_DIR"), repo_root)
    if notify_dir is None:
        notify_dir = repo_root / "runs" / "job_notifications"
    markdown = _markdown(summary)
    json_path, md_path = _write_outputs(summary, markdown, notify_dir)
    summary["notification_json"] = str(json_path)
    summary["notification_md"] = str(md_path)

    email = env.get("UNIFIED_STPP_NOTIFY_EMAIL", "")
    if email:
        prefix = env.get("UNIFIED_STPP_NOTIFY_SUBJECT_PREFIX", "[uni-stpp]")
        subject = f"{prefix} {status_label} {job_id} {job_name}"
        sent = _send_email(email, subject, markdown)
        print(f"[job-summary] email_sent={sent} to={email}")

    print(f"[job-summary] wrote {md_path}")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
