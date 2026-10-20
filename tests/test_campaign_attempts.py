from __future__ import annotations

import threading
from pathlib import Path

import pytest

from seahorse.experiments.attempts import (
    AttemptAlreadyRecorded,
    AttemptClaimBusy,
    claim_job_phase,
    failure_receipt,
)
from seahorse.experiments.runtime import classify_exception, process_exit_context, wall_time_limit


def test_failure_taxonomy_and_no_watchdog_none():
    assert classify_exception(TimeoutError("late"), origin="fit") == {
        "failure_class": "timeout",
        "failure_origin": "fit",
        "error_type": "builtins.TimeoutError",
        "error": "late",
    }
    receipt = failure_receipt(MemoryError("full"), origin="evaluation", device="cuda")
    assert receipt["failure_class"] == "host_oom" and receipt["failure_origin"] == "evaluation"
    with wall_time_limit(None):
        pass


def test_claim_is_nonblocking_and_checks_existing_receipt(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def owner():
        with claim_job_phase(tmp_path, phase="fit"):
            started.set()
            release.wait(timeout=2)

    thread = threading.Thread(target=owner)
    thread.start()
    assert started.wait(timeout=2)
    with pytest.raises(AttemptClaimBusy):
        with claim_job_phase(tmp_path, phase="fit"):
            pass
    release.set()
    thread.join(timeout=2)
    (tmp_path / "status.json").write_text('{"state":"running"}')
    with pytest.raises(AttemptAlreadyRecorded):
        with claim_job_phase(tmp_path, phase="fit"):
            pass
    assert (
        process_exit_context(None, scheduler_state="NODE_FAIL")["failure_class"]
        == "scheduler_stale"
    )


def test_retry_archives_phase_receipt_but_preserves_evaluation_nll(tmp_path):
    (tmp_path / "evaluation_status.json").write_text('{"state":"timeout"}')
    (tmp_path / "evaluation_stdout.log").write_text("old log")
    (tmp_path / "nll_status.json").write_text('{"nll_status":"complete"}')
    (tmp_path / "target_nll.npz").write_bytes(b"verified")
    with claim_job_phase(
        tmp_path, phase="evaluate", retry=True, retry_reason="resume forecast"
    ) as claim:
        assert claim["retry_of"]["preserved_nll"] is True
        assert not Path(claim["status_path"]).exists()
    archived = tmp_path / "attempts" / "evaluate" / "1"
    assert (archived / "evaluation_status.json").is_file()
    assert (archived / "retry.json").is_file()
    assert (tmp_path / "nll_status.json").is_file() and (tmp_path / "target_nll.npz").is_file()


def test_explicit_oom_and_scheduler_taxonomy_do_not_guess_sigkill():
    torch_oom = type("OutOfMemoryError", (Exception,), {"__module__": "torch.cuda"})("OOM")
    assert classify_exception(torch_oom, origin="fit")["failure_class"] == "cuda_oom"
    assert classify_exception(MemoryError("OOM"), origin="fit")["failure_class"] == "host_oom"
    assert (
        classify_exception(FloatingPointError("nonfinite"), origin="fit")["failure_class"]
        == "numerical"
    )
    assert process_exit_context(-9)["failure_class"] == "signal"
    assert process_exit_context(-9)["signal"] == 9
    assert (
        process_exit_context(-9, scheduler_state="OUT_OF_MEMORY")["failure_class"]
        == "scheduler_oom"
    )
