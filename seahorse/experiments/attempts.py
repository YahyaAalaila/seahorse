"""Atomic campaign-attempt claims and receipt normalization.

Claims serialize one job phase on a shared filesystem.  They intentionally do
not allocate GPUs or reinterpret scheduler ownership.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
from typing import Any, Iterator

from seahorse.experiments.runtime import classify_exception, process_context


class AttemptClaimBusy(RuntimeError):
    """Another process currently owns this job phase."""


class AttemptAlreadyRecorded(RuntimeError):
    """A terminal or running receipt already exists and must be preserved."""


_STATUS_FILES = {"fit": "status.json", "evaluate": "evaluation_status.json"}


def status_path(output: str | Path, phase: str) -> Path:
    try:
        return Path(output) / _STATUS_FILES[phase]
    except KeyError as exc:
        raise ValueError("phase must be 'fit' or 'evaluate'.") from exc


def _readonly_tree(path: Path) -> None:
    for item in [path, *path.rglob("*")]:
        if item.is_file(): item.chmod(0o444)
    path.chmod(0o555)


def _archive_retry(output: Path, *, phase: str, receipt: Path, reason: str) -> dict[str, Any]:
    root = output / "attempts" / phase
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while (root / str(number)).exists(): number += 1
    archive = root / str(number); archive.mkdir()
    # Preserve only phase-owned receipts/logs.  Evaluation NLL receipts and
    # target artifacts are intentionally left in place so forecast-only retry
    # can resume from verified scoring work.
    names = [receipt.name]
    names += ["stdout.log", "stderr.log", "traceback.txt", "input_contract.json", "model"] if phase == "fit" else ["evaluation_stdout.log", "evaluation_stderr.log", "evaluation_traceback.txt"]
    moved = []
    for name in names:
        source = output / name
        if source.exists():
            shutil.move(str(source), str(archive / name)); moved.append(name)
    prior = json.loads((archive / receipt.name).read_text()) if (archive / receipt.name).exists() else {}
    retry_of = {"phase": phase, "attempt": number, "archive": str(archive), "reason": reason,
                "prior_state": prior.get("state"), "preserved_nll": bool((output / "nll_status.json").exists() and (output / "target_nll.npz").exists())}
    (archive / "retry.json").write_text(json.dumps(retry_of, indent=2, sort_keys=True) + "\n")
    _readonly_tree(archive)
    return retry_of


@contextmanager
def claim_job_phase(output: str | Path, *, phase: str, device: str | None = None,
                    retry: bool = False, retry_reason: str | None = None) -> Iterator[dict[str, Any]]:
    """Acquire a nonblocking per-job/phase lock and check receipts under it."""
    output = Path(output)
    receipt = status_path(output, phase)
    lock_path = output / f".{phase}.attempt.lock"
    output.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AttemptClaimBusy(f"{phase} attempt is already claimed: {output}") from exc
        try:
            retry_of = None
            if receipt.exists():
                if not retry:
                    raise AttemptAlreadyRecorded(f"Preserve existing {phase} receipt: {receipt}")
                if not isinstance(retry_reason, str) or not retry_reason.strip():
                    raise ValueError("An explicit nonempty retry_reason is required.")
                if json.loads(receipt.read_text()).get("state") == "completed":
                    raise AttemptAlreadyRecorded("Completed attempts cannot be retried; preserve their scientific outcome.")
                retry_of = _archive_retry(output, phase=phase, receipt=receipt, reason=retry_reason.strip())
            yield {"phase": phase, "status_path": str(receipt), "process": process_context(device=device),
                   "retry_of": retry_of}
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def failure_receipt(exc: BaseException, *, origin: str, device: str | None = None,
                    retry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create stable failure fields suitable for fit or evaluation status JSON."""
    value = {**classify_exception(exc, origin=origin), "process": process_context(device=device)}
    if retry is not None:
        value["retry"] = dict(retry)
    return value


def failure_from_status(status: dict[str, Any], *, default_origin: str) -> dict[str, Any] | None:
    """Preserve an existing receipt's taxonomy, with a stable legacy fallback."""
    if not status or status.get("state") not in {"failed", "timeout", "cancelled", "stale"}:
        return None
    failure_class = status.get("failure_class")
    if failure_class is None:
        failure_class = "timeout" if status.get("state") == "timeout" else "scheduler_stale" if status.get("state") == "stale" else "failed"
    return {key: status.get(key) for key in ("failure_class", "failure_origin", "error_type", "error", "process", "retry") if key in status} | {
        "failure_class": failure_class, "failure_origin": status.get("failure_origin", default_origin),
        "error_type": status.get("error_type"), "error": status.get("error"),
    }
