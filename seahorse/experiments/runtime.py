"""Small, dependency-light runtime evidence helpers for campaign attempts."""
from contextlib import contextmanager
import os
import platform
import resource
import signal
import sys
import time
from typing import Any


@contextmanager
def wall_time_limit(seconds):
    """Optionally interrupt a direct command on Unix.

    ``None`` deliberately installs no internal watchdog.  The caller then owns
    its stopping policy (for example a scheduler allocation); silently adding
    a substitute cap would make an attempt receipt misleading.
    """
    if seconds is None:
        yield
        return
    seconds = float(seconds)
    if seconds <= 0:
        raise ValueError("A wall-time budget must be positive.")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    def expired(_signal, _frame):
        raise TimeoutError(f"Campaign wall-time budget of {seconds:g} seconds expired.")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, previous_timer[0]) if previous_timer[0] else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(1e-6, previous_timer[0]-(time.monotonic()-started)), previous_timer[1])


def classify_exception(exc: BaseException, *, origin: str) -> dict[str, str]:
    """Return a stable, serializable failure taxonomy for an exception."""
    if not isinstance(origin, str) or not origin:
        raise ValueError("failure origin must be a nonempty string.")
    qualified_name = f"{type(exc).__module__}.{type(exc).__name__}"
    # Torch's public OOM classes have varied between releases.  Match only the
    # documented class name in a torch module, never incidental error text.
    if type(exc).__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"} and type(exc).__module__.startswith("torch"):
        failure_class = "cuda_oom"
    elif isinstance(exc, TimeoutError):
        failure_class = "timeout"
    elif isinstance(exc, MemoryError):
        failure_class = "host_oom"
    elif isinstance(exc, FloatingPointError):
        failure_class = "numerical"
    elif isinstance(exc, FileNotFoundError):
        failure_class = "input_missing"
    elif isinstance(exc, PermissionError):
        failure_class = "permission"
    elif isinstance(exc, KeyboardInterrupt):
        failure_class = "interrupted"
    else:
        failure_class = "exception"
    return {"failure_class": failure_class, "failure_origin": origin,
            "error_type": qualified_name, "error": str(exc)}


def process_context(*, device: str | None = None) -> dict[str, Any]:
    """Best-effort process/scheduler receipt data, without reserving hardware."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scheduler = {key.lower(): os.environ[key] for key in
                 ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_NAME", "SLURM_NODELIST") if key in os.environ}
    context = {
        "pid": os.getpid(), "ppid": os.getppid(), "hostname": platform.node(),
        "device_request": device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scheduler": scheduler or None,
        # Linux reports KiB; retain the platform label rather than silently
        # claiming a byte unit on macOS.
        "peak_rss_raw": int(usage.ru_maxrss), "peak_rss_unit": "KiB" if os.name == "posix" and platform.system() == "Linux" else "platform_dependent",
    }
    torch = sys.modules.get("torch")
    if torch is not None and str(device).startswith("cuda"):
        try:
            if torch.cuda.is_available():
                index = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(index)
                context["gpu"] = {"logical_index": index, "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory), "capability": [properties.major, properties.minor],
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                    "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
        except Exception as exc:
            context["gpu_probe_error"] = type(exc).__name__ + ": " + str(exc)
    return context


def process_exit_context(returncode: int | None, *, origin: str = "subprocess",
                         scheduler_state: str | None = None,
                         slurm_exit_code: str | None = None) -> dict[str, Any]:
    """Normalize a subprocess or scheduler terminal state into receipt fields."""
    state = str(scheduler_state).upper() if scheduler_state is not None else None
    scheduler_classes = {
        "OUT_OF_MEMORY": "scheduler_oom", "TIMEOUT": "timeout", "CANCELLED": "cancelled",
        "PREEMPTED": "preempted", "NODE_FAIL": "scheduler_stale",
    }
    if state in scheduler_classes:
        failure_class = scheduler_classes[state]
    elif returncode is None:
        failure_class = "scheduler_stale" if state else "unknown_exit"
    elif returncode == 0:
        failure_class = "success"
    elif returncode < 0:
        failure_class = "signal"
    else:
        failure_class = "process_exit"
    return {"failure_class": failure_class, "failure_origin": origin,
            "returncode": returncode, "scheduler_state": scheduler_state,
            "slurm_exit_code": slurm_exit_code,
            # A SIGKILL is intentionally only a signal. It can be an OOM kill,
            # an administrator action, or a scheduler action; Slurm state is
            # required before calling it scheduler_oom.
            "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None}
