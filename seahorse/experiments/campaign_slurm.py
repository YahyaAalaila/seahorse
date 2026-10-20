"""Slurm plan, bounded dispatcher, and reconciliation for frozen campaigns.

The scheduler owns physical GPU placement.  This module records immutable task
intent and scheduler evidence; campaign workers retain ownership of model work
and atomic per-cell attempt claims.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from seahorse.experiments.attempts import status_path
from seahorse.experiments.campaign import canonical_hash, verify, write_json

TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL"}
FAILURES = TERMINAL - {"COMPLETED"}
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "UNKNOWN", "MISSING_FROM_SACCT", "SUBMISSION_INTENT"}
DEFAULT_RESOURCES = {
    "heavy": {"partitions": ["H100", "H200", "H100-PCI", "H200-PCI", "A100-80GB"], "max_concurrent": 2,
              "time": "24:00:00", "cpus": 16, "mem": "96G"},
    "general": {"partitions": ["L40S", "RTXA6000", "A100-40GB", "A100-80GB"], "max_concurrent": 4,
                "time": "24:00:00", "cpus": 12, "mem": "64G"},
    "cpu": {"partitions": [], "max_concurrent": 4, "time": "72:00:00", "cpus": 16, "mem": "64G"},
}
HEAVY_MODELS = frozenset({"njsde", "neural_attncnf", "neural_jumpcnf"})


def _utc() -> str: return datetime.now(timezone.utc).isoformat()
def _root(plan: Mapping[str, Any]) -> Path: return Path(plan["manifest"]).parent
def _receipt(plan: Mapping[str, Any], task: Mapping[str, Any]) -> Path: return _root(plan) / "slurm" / "receipts" / f"{task['task_id']}.json"
def _phase(task: Mapping[str, Any]) -> str: return "evaluate" if task["action"] == "evaluate" else "fit"


def _load_resources(value: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    source: Mapping[str, Any] = DEFAULT_RESOURCES if value is None else (json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value)
    result = {key: dict(item) for key, item in source.items()}
    for name in ("heavy", "general", "cpu"):
        if name not in result: raise ValueError(f"resource config missing {name}")
        item = result[name]
        if not isinstance(item.get("partitions"), list) or not isinstance(item.get("max_concurrent"), int) or item["max_concurrent"] < 1:
            raise ValueError(f"invalid resource class {name}")
        if not item.get("time"): raise ValueError(f"resource class {name} needs Slurm time")
    return result


def _task_command(manifest: Path, job: Mapping[str, Any], action: str, python: str, prefix: Iterable[str]) -> list[str]:
    device = "cpu" if job.get("runner", "nnstpp") == "statstpp" else "cuda"
    return [*prefix, python, "-m", "seahorse", "campaign", action, "--manifest", str(manifest), "--job", job["id"], "--device", device]


def build_plan(manifest_path: str | Path, *, out: str | Path | None = None, phase: str = "all", runner: str = "all",
               resources: str | Path | Mapping[str, Any] | None = None, python: str = "python",
               working_directory: str | Path | None = None, container_prefix: Iterable[str] = (), environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Create a source-verified plan.  It never calls Slurm."""
    if phase not in {"fit", "evaluate", "all"} or runner not in {"all", "nnstpp", "statstpp"}: raise ValueError("invalid phase or runner")
    manifest_path = Path(manifest_path).resolve(); manifest = verify(manifest_path)
    raw_resources = (json.loads(Path(resources).read_text()) if isinstance(resources, (str, Path)) else resources)
    meta = dict(raw_resources) if isinstance(raw_resources, Mapping) and "classes" in raw_resources else {}
    config = _load_resources(meta.pop("classes") if meta else raw_resources)
    if not container_prefix and meta.get("container_prefix"): container_prefix = meta["container_prefix"]
    if working_directory is None and meta.get("working_directory"): working_directory = meta["working_directory"]
    merged_environment = dict(meta.get("environment", {})); merged_environment.update(environment or {})
    tasks: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        if runner != "all" and job.get("runner", "nnstpp") != runner: continue
        group = "cpu" if job.get("runner", "nnstpp") == "statstpp" else ("heavy" if job["model"] in HEAVY_MODELS else "general")
        for action in (("run",) if phase in {"fit", "all"} else ()) + (("evaluate",) if job["action"] == "refit" and phase in {"evaluate", "all"} else ()):
            dependencies = list(job.get("dependencies", ())) if action == "run" else [job["id"]]
            task_id = f"{job['id']}__{action}"
            tasks.append({"task_id": task_id, "job_id": job["id"], "action": action, "runner": job.get("runner", "nnstpp"),
                          "resource_class": group, "dependency_job_ids": dependencies,
                          "command": _task_command(manifest_path, job, action, python, container_prefix),
                          "log": str(manifest_path.parent / "slurm" / "logs" / f"{task_id}-%j.out")})
    # Heavy jobs first, but dependencies—not this ordering—determine admissibility.
    rank = {"heavy": 0, "general": 1, "cpu": 2}
    tasks.sort(key=lambda x: (rank[x["resource_class"]], x["job_id"].rsplit("__", 1)[0], x["job_id"], x["action"] != "run"))
    plan = {"schema_version": 2, "created_utc": _utc(), "manifest": str(manifest_path), "manifest_sha256": manifest["sha256"],
            "working_directory": str(Path(working_directory or Path(__file__).resolve().parents[2]).resolve()),
            "environment": {"cuda_binding": "Slurm allocation; do not override CUDA_VISIBLE_DEVICES", "exports": merged_environment,
                            "path_binding": "mount frozen manifest and prepared-data absolute paths unchanged"},
            "phase": phase, "runner": runner, "resource_config": config, "stop_before_utc": manifest["spec"].get("compute", {}).get("stop_before_utc"), "tasks": tasks,
            "dependency_policy": "all HPO trials terminal before refit; refit selects finite completed trial; evaluate requires completed refit"}
    plan["sha256"] = canonical_hash(plan)
    if out is not None:
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True); (manifest_path.parent / "slurm" / "logs").mkdir(parents=True, exist_ok=True); write_json(out, plan)
    return plan


def _load_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text()); expected = plan.pop("sha256", None)
    if expected != canonical_hash(plan): raise ValueError("Slurm plan changed after freezing")
    plan["sha256"] = expected
    current = verify(plan["manifest"])
    if current["sha256"] != plan["manifest_sha256"]: raise ValueError("Manifest/source changed after Slurm plan freeze")
    return plan


def _worker_status(plan: Mapping[str, Any], job_id: str, phase: str) -> dict[str, Any] | None:
    path = status_path(_root(plan) / "jobs" / job_id, phase)
    return json.loads(path.read_text()) if path.exists() else None


def _dependency_ready(plan: Mapping[str, Any], task: Mapping[str, Any]) -> tuple[bool, str | None]:
    if task["action"] == "evaluate":
        prior = _worker_status(plan, task["job_id"], "fit")
        if prior is not None and prior.get("state") == "completed": return True, None
        if prior is not None and prior.get("state") in {"failed", "timeout", "cancelled", "blocked", "stale"}: return False, "refit_failed"
        return False, "refit_pending"
    for dep in task["dependency_job_ids"]:
        state = _worker_status(plan, dep, "fit")
        receipt = next((_receipt(plan, t) for t in plan["tasks"] if t["job_id"] == dep and t["action"] == "run"), None)
        scheduled = json.loads(receipt.read_text()).get("scheduler_state") if receipt and receipt.exists() else None
        if (state or {}).get("state") in {"completed", "failed", "timeout", "cancelled", "blocked", "stale"}: continue
        if scheduled in TERMINAL: continue
        return False, "dependencies_not_terminal"
    return True, None


def _exit_fields(value: str | None) -> dict[str, int | None]:
    """Slurm ExitCode is ``returncode:signal``; retain both without guessing cause."""
    try:
        code, signal = str(value).split(":", 1)
        return {"returncode": int(code), "signal": int(signal)}
    except (TypeError, ValueError):
        return {"returncode": None, "signal": None}


def _parse_sacct(text: str) -> dict[str, dict[str, str]]:
    rows = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 7 and parts[0] and "." not in parts[0]:
            rows[parts[0]] = dict(zip(("job_id", "state", "reason", "exit_code", "elapsed", "node", "alloc_tres"), parts))
            rows[parts[0]]["state"] = rows[parts[0]]["state"].split()[0]
    return rows


def reconcile(plan_path: str | Path, *, execute: bool = False, runner: Callable = subprocess.run) -> dict[str, Any]:
    plan = _load_plan(plan_path); pairs = [(_receipt(plan, t), t) for t in plan["tasks"] if _receipt(plan, t).exists()]
    ids = [json.loads(path.read_text()).get("slurm_job_id") for path, _ in pairs if json.loads(path.read_text()).get("slurm_job_id")]
    if not execute: return {"state": "dry_run", "scheduled_jobs": ids}
    if not ids: return {"state": "no_submissions", "scheduled_jobs": []}
    answer = runner(["sacct", "-n", "-P", "-j", ",".join(ids), "--format", "JobIDRaw,State,Reason,ExitCode,Elapsed,NodeList,AllocTRES"], check=True, text=True, capture_output=True)
    rows = _parse_sacct(answer.stdout); evidence = []
    for path, task in pairs:
        receipt = json.loads(path.read_text()); row = rows.get(receipt.get("slurm_job_id")); old = receipt.get("scheduler_state", "PENDING")
        state = row["state"] if row else old
        receipt.update(scheduler_state=state, scheduler_reason=row["reason"] if row else "sacct not yet visible", scheduler_failure=state in FAILURES,
                       sacct=row, reconciled_utc=_utc()); write_json(path, receipt)
        output = _root(plan) / "jobs" / task["job_id"]
        evidence_path = output / f"scheduler_{_phase(task)}_status.json"
        scheduler_evidence = {**receipt, "task_id": task["task_id"], "action": task["action"], "log": task["log"]}
        write_json(evidence_path, scheduler_evidence)
        worker = status_path(output, _phase(task)); prior = json.loads(worker.read_text()) if worker.exists() else None
        if prior is not None:
            # Append scheduler facts without replacing the worker's Python exception.
            prior["scheduler"] = {"slurm_job_id": receipt.get("slurm_job_id"), "state": state, "reason": receipt["scheduler_reason"], "elapsed": (row or {}).get("elapsed"), "exit_code": (row or {}).get("exit_code"), **_exit_fields((row or {}).get("exit_code")), "node": (row or {}).get("node"), "log": task["log"]}
            write_json(worker, prior)
        missing_worker = state == "COMPLETED" and (prior is None or prior.get("state") == "running")
        if (state in FAILURES or missing_worker) and (prior is None or prior.get("state") == "running"):
            label = "scheduler_completed_missing_worker_receipt" if missing_worker else f"scheduler_{state.lower()}"
            synthetic = {"job_id": task["job_id"], "state": "timeout" if state == "TIMEOUT" else "failed", "failure_class": label,
                         "failure_origin": "slurm_reconcile", "error_type": f"Slurm{state.title().replace('_', '')}",
                         "error": "Slurm completed but the worker did not write a terminal receipt" if missing_worker else receipt["scheduler_reason"],
                         "scheduler_evidence": str(evidence_path), "slurm_job_id": receipt.get("slurm_job_id"), "scheduler_state": state, "reconciled_utc": _utc()}
            write_json(worker, synthetic)
        evidence.append(receipt)
    return {"state": "reconciled", "evidence": evidence}


def _recover_submission_intents(plan: Mapping[str, Any], runner: Callable) -> None:
    """Recover an accepted sbatch whose reply was lost, by its immutable comment/name."""
    for task in plan["tasks"]:
        path = _receipt(plan, task)
        if not path.exists(): continue
        receipt = json.loads(path.read_text())
        if receipt.get("scheduler_state") != "SUBMISSION_INTENT": continue
        answer = runner(["squeue", "-h", "-n", task["task_id"], "-o", "%A"], check=True, text=True, capture_output=True)
        ids = [line.strip() for line in answer.stdout.splitlines() if line.strip().isdigit()]
        if len(ids) == 1:
            receipt.update(slurm_job_id=ids[0], scheduler_state="PENDING", recovered_utc=_utc())
            write_json(path, receipt)


def _active(plan: Mapping[str, Any]) -> dict[str, int]:
    used = {key: 0 for key in plan["resource_config"]}
    for task in plan["tasks"]:
        receipt = _receipt(plan, task)
        if receipt.exists() and json.loads(receipt.read_text()).get("scheduler_state") in ACTIVE: used[task["resource_class"]] += 1
    return used


def _sbatch(plan: Mapping[str, Any], task: Mapping[str, Any]) -> list[str]:
    resource = plan["resource_config"][task["resource_class"]]
    cmd = ["sbatch", "--parsable", "--chdir", plan["working_directory"], "--job-name", task["task_id"], "--comment", task["task_id"],
           "--output", task["log"], "--time", str(resource["time"]), "--cpus-per-task", str(resource["cpus"]), "--mem", str(resource["mem"])]
    if resource["partitions"]: cmd += ["--partition", ",".join(resource["partitions"])]
    if resource.get("account"): cmd += ["--account", str(resource["account"])]
    if task["runner"] == "nnstpp": cmd += ["--gpus", "1"]
    exports = plan["environment"].get("exports", {})
    if exports: cmd += ["--export", "ALL," + ",".join(f"{key}={value}" for key, value in sorted(exports.items()))]
    return cmd + ["--wrap", shlex.join(task["command"])]


def _intent(plan: Mapping[str, Any], task: Mapping[str, Any], command: list[str]) -> None:
    path = _receipt(plan, task)
    write_json(path, {"schema_version": 2, "intent_utc": _utc(), "plan_sha256": plan["sha256"], "manifest_sha256": plan["manifest_sha256"],
                      "task_id": task["task_id"], "job_id": task["job_id"], "action": task["action"], "scheduler_state": "SUBMISSION_INTENT",
                      "resource": plan["resource_config"][task["resource_class"]], "command": command, "log": task["log"]})


def dispatch(plan_path: str | Path, *, execute: bool = False, watch: bool = False, poll_seconds: int = 30, runner: Callable = subprocess.run) -> dict[str, Any]:
    """Submit only globally available, dependency-ready slots; ``watch`` is a controller loop."""
    if poll_seconds < 1: raise ValueError("poll_seconds must be positive")
    plan = _load_plan(plan_path); root = _root(plan)
    import fcntl
    launched = []
    lock_context = contextlib.nullcontext()
    if execute:
        (root / "slurm" / "logs").mkdir(parents=True, exist_ok=True)
        lock_context = (root / "slurm" / "dispatch.lock").open("a")
    with lock_context as lock:
        if execute: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if execute:
                _recover_submission_intents(plan, runner)
                reconcile(plan_path, execute=True, runner=runner)
            active = _active(plan)
            for task in plan["tasks"]:
                receipt = _receipt(plan, task)
                if receipt.exists(): continue
                # Import a worker receipt made before this controller started;
                # it is a completed campaign fact, not a candidate to resubmit.
                own = _worker_status(plan, task["job_id"], _phase(task))
                if own is not None:
                    if execute:
                        state = "COMPLETED" if own.get("state") == "completed" else ("FAILED" if own.get("state") in {"failed", "timeout", "cancelled", "blocked", "stale"} else "RUNNING")
                        write_json(receipt, {"schema_version": 2, "imported_utc": _utc(), "task_id": task["task_id"], "job_id": task["job_id"], "action": task["action"], "scheduler_state": state, "worker_status": str(status_path(root / "jobs" / task["job_id"], _phase(task)))})
                    continue
                ready, reason = _dependency_ready(plan, task)
                if not ready:
                    if execute and task["action"] == "evaluate" and reason == "refit_failed":
                        write_json(status_path(root / "jobs" / task["job_id"], "evaluate"), {"job_id": task["job_id"], "state": "blocked", "failure_class": "dependency_failed", "failure_origin": "slurm_dispatch", "error": reason})
                    continue
                deadline = plan.get("stop_before_utc")
                if deadline is None:
                    deadline = json.loads(Path(plan["manifest"]).read_text()).get("spec", {}).get("compute", {}).get("stop_before_utc")
                if deadline is not None and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
                    continue
                group = task["resource_class"]
                if active[group] >= plan["resource_config"][group]["max_concurrent"]: continue
                command = _sbatch(plan, task)
                if not execute:
                    launched.append({"task_id": task["task_id"], "state": "dry_run", "command": command}); active[group] += 1; continue
                _intent(plan, task, command)  # durable before a possibly accepted sbatch call
                answer = runner(command, check=True, text=True, capture_output=True).stdout.strip().split(";", 1)[0]
                if not answer.isdigit(): raise RuntimeError(f"sbatch did not return numeric job ID: {answer!r}")
                value = json.loads(receipt.read_text()); value.update(submitted_utc=_utc(), slurm_job_id=answer, scheduler_state="PENDING"); write_json(receipt, value)
                launched.append({"task_id": task["task_id"], "state": "submitted", "slurm_job_id": answer}); active[group] += 1
            if execute:
                write_json(root / "slurm" / "dispatch_state.json", {"utc": _utc(), "plan_sha256": plan["sha256"], "active": active, "launched": launched[-20:]})
            if not (watch and execute): break
            outstanding = any(not _receipt(plan, task).exists() for task in plan["tasks"])
            if not outstanding and not any(active.values()): break
            time.sleep(poll_seconds)
    return {"state": "dispatched" if execute else "dry_run", "launched": launched, "active": _active(plan)}


def submit_plan(plan_path: str | Path, *, execute: bool = False, runner: Callable = subprocess.run) -> dict[str, Any]:
    return dispatch(plan_path, execute=execute, watch=False, runner=runner)


def add_subparser(commands) -> None:
    p = commands.add_parser("slurm", help="Plan, dispatch, and reconcile frozen Slurm campaign tasks")
    p.add_argument("operation", choices=("plan", "dispatch", "reconcile", "submit")); p.add_argument("--manifest"); p.add_argument("--plan"); p.add_argument("--out")
    p.add_argument("--phase", choices=("fit", "evaluate", "all"), default="all"); p.add_argument("--runner", choices=("all", "nnstpp", "statstpp"), default="all")
    p.add_argument("--resources"); p.add_argument("--python", default="python"); p.add_argument("--working-directory"); p.add_argument("--container-prefix", nargs="*", default=[])
    p.add_argument("--execute", action="store_true"); p.add_argument("--watch", action="store_true"); p.add_argument("--poll-seconds", type=int, default=30)


def handle(args) -> dict[str, Any]:
    if args.operation == "plan":
        if not args.manifest or not args.out: raise ValueError("plan requires --manifest and --out")
        return build_plan(args.manifest, out=args.out, phase=args.phase, runner=args.runner, resources=args.resources, python=args.python, working_directory=args.working_directory, container_prefix=args.container_prefix)
    if not args.plan: raise ValueError(f"{args.operation} requires --plan")
    if args.operation == "reconcile": return reconcile(args.plan, execute=args.execute)
    return dispatch(args.plan, execute=args.execute, watch=args.watch, poll_seconds=args.poll_seconds)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); subs = parser.add_subparsers(dest="root", required=True); add_subparser(subs)
    print(json.dumps(handle(parser.parse_args(argv)), indent=2, sort_keys=True))

if __name__ == "__main__": main()
