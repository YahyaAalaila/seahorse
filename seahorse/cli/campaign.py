"""Public commands for preparing and operating a frozen Seahorse campaign."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import json
import time
from pathlib import Path


def add_subparser(sub) -> None:
    parser = sub.add_parser("campaign", help="Prepare, validate, and operate a frozen campaign")
    commands = parser.add_subparsers(dest="campaign_command", required=True)
    from seahorse.experiments.campaign_slurm import add_subparser as add_slurm_subparser
    add_slurm_subparser(commands)
    from seahorse.experiments.campaign_rehearse import add_subparser as add_rehearsal_subparser
    add_rehearsal_subparser(commands)

    prepare = commands.add_parser("prepare", help="Compile a new immutable campaign manifest")
    prepare.add_argument("--spec", required=True); prepare.add_argument("--out", required=True)
    prepare.add_argument("--repo", default=None); prepare.add_argument("--skip-data-verification", action="store_true")

    verify = commands.add_parser("verify", help="Verify a frozen campaign manifest")
    verify.add_argument("--manifest", required=True); verify.add_argument("--repo", default=None)

    preflight = commands.add_parser("preflight", help="Run bounded source-bound integration checks")
    preflight.add_argument("--manifest", required=True); preflight.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    preflight.add_argument("--runner", choices=("all", "nnstpp", "statstpp"), default="all")
    preflight.add_argument("--models", nargs="*"); preflight.add_argument("--skip-dataset-smokes", action="store_true")

    for name, help_text in (("run", "Run one preflight-authorized campaign fit cell"),
                            ("evaluate", "Evaluate one completed campaign refit cell")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--manifest", required=True); command.add_argument("--job", required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--retry", action="store_true", help="Archive a failed attempt and explicitly retry it")
        command.add_argument("--retry-reason", help="Required explanation for a retry")

    queue = commands.add_parser("queue", help="Inspect or advance the serialized campaign queue")
    queue.add_argument("--manifest", required=True); queue.add_argument("--execute", action="store_true")
    queue.add_argument("--stage", choices=("core", "expansion", "all"), default="core")
    queue.add_argument("--phase", choices=("fit", "evaluate", "all"), default="all")
    queue.add_argument("--runner", choices=("all", "nnstpp", "statstpp"), default="all",
                       help="Restrict queue inspection/execution to one runner family")
    queue.add_argument("--models", nargs="+", default=None,
                       help="Restrict queue execution to these frozen model IDs")
    queue.add_argument("--watch", action="store_true", help="After a GPU-memory wait, continue queueing in a detached 30-second watcher")
    queue.add_argument("--detach", action="store_true",
                       help="Run the checked queue command in a detached background process; logs beside the manifest")
    queue.add_argument("--watch-worker", action="store_true", help=argparse.SUPPRESS)
    queue.add_argument("--already-executed", type=int, default=0, help=argparse.SUPPRESS)
    queue.add_argument("--max-jobs", type=int, default=None)

    report = commands.add_parser("report", help="Build the read-only campaign ledger")
    report.add_argument("--manifest", required=True); report.add_argument("--out", required=True)
    report.add_argument("--replicates", type=int, default=2000); report.add_argument("--block-records", type=int, default=7)
    report.add_argument("--seed", type=int, default=20260910)


def _delegate(args, *command: str) -> None:
    """Delegate execution to the campaign module owning that operation."""
    subprocess.run([sys.executable, "-m", *command], check=True)


def _watch_queue(args) -> dict:
    """Retry only the documented shared-GPU-memory wait state."""
    from seahorse.experiments.campaign_run import queue
    executed_before = int(args.already_executed)
    # The parent invocation already observed the wait.  Avoid an immediate
    # second nvidia-smi/queue pass that would busy-loop on shared hardware.
    time.sleep(30)
    while True:
        remaining = None if args.max_jobs is None else max(0, args.max_jobs - executed_before)
        if remaining == 0:
            return {"state": "requested_action_limit", "executed": executed_before, "runner": args.runner}
        result = queue(args.manifest, execute=True, stage=args.stage, max_jobs=remaining,
                       phase=args.phase, runner=args.runner, models=getattr(args, "models", None))
        executed_before += int(result.get("executed", 0))
        if result.get("state") != "waiting_for_shared_gpu_memory":
            return result
        time.sleep(30)


def _start_detached_watcher(args, initial: dict) -> dict:
    from seahorse.experiments.campaign_run import _queue_selection_suffix
    suffix = _queue_selection_suffix(getattr(args, "models", None))
    log = Path(args.manifest).resolve().parent / f"queue_watch_{args.runner}{suffix}.log"
    command = [sys.executable, "-m", "seahorse", "campaign", "queue", "--manifest", str(Path(args.manifest).resolve()),
               "--execute", "--stage", args.stage, "--phase", args.phase, "--runner", args.runner,
               "--watch-worker", "--already-executed", str(initial.get("executed", 0))]
    if args.max_jobs is not None:
        command.extend(("--max-jobs", str(args.max_jobs)))
    if getattr(args, "models", None):
        command.extend(("--models", *args.models))
    with log.open("a") as handle:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)
    return {**initial, "watch_state": "detached_waiting_for_shared_gpu_memory", "watch_pid": proc.pid,
            "watch_log": str(log)}


def _queue_child_command(args) -> list[str]:
    manifest = str(Path(args.manifest).resolve())
    command = [sys.executable, "-m", "seahorse", "campaign", "queue", "--manifest", manifest,
               "--execute", "--stage", args.stage, "--phase", args.phase, "--runner", args.runner]
    if args.max_jobs is not None:
        command.extend(("--max-jobs", str(args.max_jobs)))
    if getattr(args, "models", None):
        command.extend(("--models", *args.models))
    if args.watch:
        # Keep the receipt PID alive through all retry sleeps.  A nested
        # watcher would outlive this child and permit a duplicate launcher.
        command.extend(("--watch-worker", "--already-executed", "0"))
    return command


def _live_pid(receipt: Path) -> bool:
    if not receipt.exists():
        return False
    try:
        pid = int(json.loads(receipt.read_text())["pid"])
        os.kill(pid, 0)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def _start_detached_queue(args, *, manifest: dict, summary: dict) -> dict:
    from seahorse.experiments.campaign_run import _queue_selection_suffix
    directory = Path(args.manifest).resolve().parent
    suffix = _queue_selection_suffix(getattr(args, "models", None))
    receipt = directory / f"queue_{args.runner}{suffix}.pid.json"
    if _live_pid(receipt):
        raise RuntimeError(f"A detached {args.runner} queue worker is already recorded at {receipt}.")
    log = directory / f"queue_{args.runner}{suffix}.log"
    source_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    with log.open("a") as handle:
        proc = subprocess.Popen(_queue_child_command(args), cwd=source_root, env=environment,
                                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)
    receipt.write_text(json.dumps({"pid": proc.pid, "manifest": str(Path(args.manifest).resolve()),
                                   "runner": args.runner, "models": getattr(args, "models", None),
                                   "command": _queue_child_command(args),
                                   "log": str(log)}, indent=2, sort_keys=True) + "\n")
    return {"state": "detached", "pid": proc.pid, "log": str(log), "receipt": str(receipt),
            "manifest": str(Path(args.manifest).resolve()), "runner": args.runner, "models": getattr(args, "models", None),
            "dry_run": summary}


def execute(args) -> None:
    command = args.campaign_command
    if command == "rehearse":
        from seahorse.experiments.campaign_rehearse import rehearse
        result = rehearse(args.manifest, out=args.out, dataset=args.dataset, models=getattr(args, "models", None), steps=args.steps, device=args.device)
        print(json.dumps(result, indent=2))
        if not result["pass"]: raise SystemExit(1)
    elif command == "slurm":
        from seahorse.experiments.campaign_slurm import handle
        print(json.dumps(handle(args), indent=2))
    elif command == "prepare":
        from seahorse.experiments.campaign import prepare
        print(json.dumps(prepare(args.spec, args.out, repo=args.repo, verify_data=not args.skip_data_verification), indent=2))
    elif command == "verify":
        from seahorse.experiments.campaign import verify
        manifest = verify(args.manifest, repo=args.repo)
        print(json.dumps({"verified": True, "manifest": str(Path(args.manifest).resolve()),
                          "sha256": manifest.get("sha256"), "datasets": len(manifest.get("datasets", {})),
                          "models": len(manifest.get("spec", {}).get("models", [])), "jobs": len(manifest.get("jobs", []))}, indent=2))
    elif command == "preflight":
        from seahorse.experiments.campaign_preflight import run_preflight
        print(json.dumps(run_preflight(args.manifest, device=args.device, models=getattr(args, "models", None),
                                       dataset_smokes=not args.skip_dataset_smokes, runner=args.runner), indent=2))
    elif command == "queue":
        from seahorse.experiments.campaign_run import queue
        if args.watch_worker:
            print(json.dumps(_watch_queue(args), indent=2))
            return
        if args.watch and not args.execute:
            raise ValueError("--watch requires --execute.")
        if args.detach and not args.execute:
            raise ValueError("--detach requires --execute.")
        if args.detach:
            from seahorse.experiments.campaign import verify
            from seahorse.experiments.campaign_run import _launch_evidence
            manifest = verify(args.manifest)
            _launch_evidence(manifest, Path(args.manifest).resolve().parent, runner=args.runner)
            summary = queue(args.manifest, execute=False, stage=args.stage, max_jobs=args.max_jobs,
                            phase=args.phase, runner=args.runner, models=getattr(args, "models", None))
            print(json.dumps(_start_detached_queue(args, manifest=manifest, summary=summary), indent=2))
            return
        result = queue(args.manifest, execute=args.execute, stage=args.stage, max_jobs=args.max_jobs,
                       phase=args.phase, runner=args.runner, models=getattr(args, "models", None))
        if args.watch and result.get("state") == "waiting_for_shared_gpu_memory":
            result = _start_detached_watcher(args, result)
        print(json.dumps(result, indent=2))
    elif command in {"run", "evaluate"}:
        from seahorse.experiments import campaign_run
        extra = (["--retry", "--retry-reason", args.retry_reason] if args.retry and args.retry_reason else ["--retry"] if args.retry else [])
        campaign_run.main([command, "--manifest", args.manifest, "--job", args.job, "--device", args.device, *extra])
    elif command == "report":
        _delegate(args, "seahorse.experiments.campaign_report", "--manifest", args.manifest, "--out", args.out,
                  "--replicates", str(args.replicates), "--block-records", str(args.block_records), "--seed", str(args.seed))
    else:  # argparse makes this unreachable.
        raise ValueError(f"Unknown campaign command {command!r}")
