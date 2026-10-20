from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from seahorse.experiments.campaign_report import build_report, moving_block_difference, write_report


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _complete(root: Path, job_id: str, values, *, ids=((0, 20), (1, 20))):
    out = root / "jobs" / job_id
    _write(out / "status.json", {"state": "completed"})
    _write(
        out / "evaluation.json",
        {
            "nll_status": "complete",
            "target_population": len(values),
            "predictive": {"status": "computed", "failures": []},
        },
    )
    np.savez_compressed(
        out / "target_nll.npz",
        nll=np.asarray(values, float),
        sequence_index=np.asarray([x[0] for x in ids]),
        target_event_index=np.asarray([x[1] for x in ids]),
    )


def test_report_keeps_unrun_and_rejects_partial_nll_coverage(tmp_path):
    jobs = [
        {
            "id": "d__poisson_gmm__seed1",
            "dataset": "d",
            "model": "poisson_gmm",
            "seed": 1,
            "action": "refit",
        },
        {
            "id": "d__poisson_gmm__seed2",
            "dataset": "d",
            "model": "poisson_gmm",
            "seed": 2,
            "action": "refit",
        },
    ]
    manifest = {"sha256": "x", "spec": {"seeds": [1, 2]}, "jobs": jobs}
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    _complete(tmp_path, jobs[0]["id"], [1.0, np.inf])
    report = build_report(path, replicates=10)
    states = {x["seed"]: x["state"] for x in report["ledger"]}
    assert states == {1: "eval_failed", 2: "not_run"}
    assert (
        report["within_dataset_model_summary"][0]["primary_nll_status"]
        == "incomplete_or_unavailable"
    )


def test_report_pairs_all_seeds_only_with_identical_target_ids(tmp_path):
    jobs = []
    for model in ("poisson_gmm", "candidate"):
        for seed in (1, 2):
            jobs.append(
                {
                    "id": f"d__{model}__seed{seed}",
                    "dataset": "d",
                    "model": model,
                    "seed": seed,
                    "action": "refit",
                }
            )
    path = tmp_path / "manifest.json"
    _write(path, {"sha256": "x", "spec": {"seeds": [1, 2]}, "jobs": jobs})
    for job in jobs:
        _complete(tmp_path, job["id"], [1.0, 2.0] if job["model"] == "poisson_gmm" else [0.5, 1.5])
    report = build_report(path, replicates=20, seed=3)
    comparison = report["paired_vs_poisson_gmm"][0]
    assert comparison["status"] == "complete"
    assert [x["paired_targets"] for x in comparison["per_seed"]] == [2, 2]
    assert all(
        x["mean_difference_candidate_minus_poisson_gmm"] == -0.5 for x in comparison["per_seed"]
    )

    candidate = {(0, 20): 0.5}
    baseline = {(0, 20): 1.0, (1, 20): 2.0}
    try:
        moving_block_difference(candidate, baseline, replicates=2)
    except ValueError as exc:
        assert "identical target IDs" in str(exc)
    else:
        raise AssertionError("mismatched target IDs must not be paired")


def test_report_keeps_nll_complete_but_refuses_partial_predictive_and_count_tables(tmp_path):
    jobs = [
        {
            "id": f"d__candidate__seed{seed}",
            "dataset": "d",
            "model": "candidate",
            "seed": seed,
            "action": "refit",
        }
        for seed in (1, 2, 3)
    ]
    manifest = {
        "sha256": "x",
        "spec": {
            "seeds": [1, 2, 3],
            "datasets": [{"id": "d", "observation_view": "continuous_fixed_window"}],
            "models": [
                {
                    "id": "candidate",
                    "preset": "demo",
                    "nll_kind": "exact",
                    "nll_path": "closed_form",
                    "provenance": "frozen-source",
                    "inference_objective": "mle",
                }
            ],
        },
        "jobs": jobs,
    }
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    predictive_summary = {
        "temporal_crps": 1.0,
        "spatial_energy_score": 2.0,
        "temporal_mae": 3.0,
        "spatial_mae": 4.0,
        "temporal_rmse": 5.0,
        "spatial_rmse": 6.0,
    }
    count_summary = {"count_crps": 7.0, "count_rmse": 8.0, "count_mae": 9.0}
    for job in jobs:
        out = tmp_path / "jobs" / job["id"]
        _write(out / "status.json", {"state": "completed"})
        count = {
            "status": "computed",
            "target_population": 4,
            "successful_targets": 4,
            "failures": [],
            "summary": count_summary,
        }
        if job["seed"] == 2:
            count = {
                "status": "computed",
                "target_population": 4,
                "successful_targets": 3,
                "failures": [{"target": 2}],
                "summary": None,
            }
        predictive = {
            "status": "computed",
            "target_population": 5,
            "successful_targets": 5,
            "failures": [],
            "summary": predictive_summary,
        }
        if job["seed"] == 3:
            predictive = {
                "status": "computed",
                "target_population": 5,
                "successful_targets": 4,
                "failures": [{"target": 1}],
                "summary": None,
            }
        _write(
            out / "evaluation.json",
            {
                "nll_status": "complete",
                "nll_path": "closed_form",
                "nll_kind": "exact",
                "target_population": 2,
                "predictive": predictive,
                "count_forecast": count,
            },
        )
        np.savez_compressed(
            out / "target_nll.npz",
            nll=np.asarray([1.0, 2.0]),
            sequence_index=np.asarray([0, 1]),
            target_event_index=np.asarray([20, 20]),
        )
    report = build_report(path, replicates=2)
    summary, row = report["within_dataset_model_summary"][0], report["wide_table"][0]
    assert summary["primary_nll_status"] == "complete"
    assert summary["predictive_table_status"] == "incomplete_or_failed"
    assert summary["count_forecast_table_status"] == "incomplete_or_failed"
    assert row["temporal_mae"] is None and row["count_rmse"] is None
    assert report["ledger"][1]["state"] == "complete"
    outputs = write_report(report, tmp_path / "report.json")
    assert Path(outputs["ledger"]).is_file() and Path(outputs["wide"]).is_file()


def test_report_retains_complete_nll_when_forecast_phase_times_out(tmp_path):
    job = {
        "id": "d__candidate__seed1",
        "dataset": "d",
        "model": "candidate",
        "seed": 1,
        "action": "refit",
    }
    path = tmp_path / "manifest.json"
    _write(path, {"sha256": "x", "spec": {"seeds": [1]}, "jobs": [job]})
    out = tmp_path / "jobs" / job["id"]
    _write(out / "status.json", {"state": "completed"})
    _write(
        out / "nll_status.json",
        {"nll_status": "complete", "target_population": 2, "nll_path": "closed_form"},
    )
    _write(
        out / "evaluation_status.json",
        {
            "state": "timeout",
            "failure_class": "timeout",
            "failure_origin": "forecast",
            "error_type": "TimeoutError",
            "error": "forecast budget expired",
            "log_path": "forecast.log",
        },
    )
    np.savez_compressed(
        out / "target_nll.npz",
        nll=np.asarray([1.0, 2.0]),
        sequence_index=np.asarray([0, 1]),
        target_event_index=np.asarray([20, 20]),
    )
    cell = build_report(path, replicates=2)["ledger"][0]
    assert cell["state"] == "complete" and cell["nll_status"] == "complete"
    assert cell["evaluation_status"] == "forecast_incomplete_after_nll"
    assert cell["evaluation_failure"]["failure_class"] == "timeout"


def test_report_exposes_stale_scheduler_evaluation_receipt(tmp_path):
    job = {
        "id": "d__candidate__seed1",
        "dataset": "d",
        "model": "candidate",
        "seed": 1,
        "action": "refit",
    }
    path = tmp_path / "manifest.json"
    _write(path, {"sha256": "x", "spec": {"seeds": [1]}, "jobs": [job]})
    out = tmp_path / "jobs" / job["id"]
    _write(out / "status.json", {"state": "completed"})
    _write(
        out / "evaluation_status.json",
        {
            "state": "stale",
            "failure_class": "scheduler_stale",
            "failure_origin": "scheduler",
            "error_type": "SchedulerState",
            "error": "NODE_FAIL",
            "log_path": "slurm.out",
            "retry": {"attempt": 1, "eligible": True},
            "slurm_exit_code": "0:9",
            "signal": 9,
            "elapsed_seconds": 123.0,
        },
    )
    cell = build_report(path, replicates=2)["ledger"][0]
    assert cell["state"] == "eval_failed"
    assert cell["evaluation_failure"]["failure_class"] == "scheduler_stale"
    assert cell["evaluation_attempt"]["retry"] == {"attempt": 1, "eligible": True}
    assert cell["evaluation_attempt"]["slurm_exit_code"] == "0:9"


def test_report_reads_next_event_progress_before_count_timeout(tmp_path):
    job = {
        "id": "d__candidate__seed1",
        "dataset": "d",
        "model": "candidate",
        "seed": 1,
        "action": "refit",
    }
    path = tmp_path / "manifest.json"
    _write(path, {"sha256": "x", "spec": {"seeds": [1]}, "jobs": [job]})
    out = tmp_path / "jobs" / job["id"]
    _write(out / "status.json", {"state": "completed"})
    _write(out / "nll_status.json", {"nll_status": "complete", "target_population": 2})
    _write(
        out / "evaluation_status.json",
        {"state": "timeout", "failure_class": "timeout", "failure_origin": "forecast"},
    )
    summary = {
        "temporal_crps": 1.0,
        "spatial_energy_score": 2.0,
        "temporal_mae": 3.0,
        "spatial_mae": 4.0,
        "temporal_rmse": 5.0,
        "spatial_rmse": 6.0,
    }
    _write(
        out / "predictive_status.json",
        {
            "predictive": {
                "status": "computed",
                "target_population": 2,
                "successful_targets": 2,
                "failures": [],
                "summary": summary,
            }
        },
    )
    np.savez_compressed(
        out / "target_nll.npz",
        nll=np.asarray([1.0, 2.0]),
        sequence_index=np.asarray([0, 1]),
        target_event_index=np.asarray([20, 20]),
    )
    cell = build_report(path, replicates=2)["ledger"][0]
    assert cell["state"] == "complete" and cell["predictive_status"] == "complete"
