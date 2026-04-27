from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.paper_readiness_report import render_report


def test_render_report_writes_expected_artifacts(tmp_path: Path) -> None:
    snapshot = {
        "created_at": "2026-04-27T08:00:00+00:00",
        "repo_root": "/tmp/repo",
        "expected_seeds": [42, 3, 555],
        "sacct_since": "2026-04-20T00:00:00",
        "live_jobs": [
            {
                "job_id": "1",
                "job_name": "suite3_job",
                "state": "RUNNING",
                "elapsed": "00:10:00",
                "reason_or_node": "nodeA",
            }
        ],
        "synthetic_training": [
            {
                "suite": "suite3_entanglement",
                "preset": "auto_stpp",
                "status": "done",
                "done_cells": 12,
                "expected_cells": 12,
                "missing_cells": [],
                "active_jobs": [],
                "latest_job_state": None,
                "configs": ["L0"],
                "expected_seeds": [42, 3, 555],
                "done_results": [
                    {"config_id": "L0", "seed": 42, "test_nll": 1.0, "val_objective": 0.9, "run_dir": "/tmp/a"},
                    {"config_id": "L0", "seed": 3, "test_nll": 1.2, "val_objective": 0.8, "run_dir": "/tmp/b"},
                    {"config_id": "L0", "seed": 555, "test_nll": 1.1, "val_objective": 0.85, "run_dir": "/tmp/c"},
                ],
            },
            {
                "suite": "suite4_heterogeneity",
                "preset": "neural_jumpcnf",
                "status": "partial",
                "done_cells": 2,
                "expected_cells": 12,
                "missing_cells": [{"config_id": "H1", "seed": 42}],
                "active_jobs": ["s4het_v2__neural_jumpcnf"],
                "latest_job_state": "FAILED",
                "configs": ["H1"],
                "expected_seeds": [42, 3, 555],
                "done_results": [
                    {"config_id": "H1", "seed": 3, "test_nll": 2.2, "val_objective": 2.0, "run_dir": "/tmp/d"},
                    {"config_id": "H1", "seed": 555, "test_nll": 2.0, "val_objective": 1.9, "run_dir": "/tmp/e"},
                ],
            },
        ],
        "realdata_training": [
            {
                "dataset": "covid-stpp",
                "preset": "hawkes_gmm",
                "status": "done",
                "done_seeds": [3, 42, 555],
                "missing_seeds": [],
                "active_jobs": [],
                "latest_job_state": None,
                "expected_seeds": [42, 3, 555],
                "campaigns": ["covid-stpp__fact__04231151"],
                "done_results": [
                    {"seed": 42, "test_nll": 0.5, "val_objective": 0.4, "run_dir": "/tmp/f", "campaign": "covid"},
                    {"seed": 3, "test_nll": 0.6, "val_objective": 0.5, "run_dir": "/tmp/g", "campaign": "covid"},
                    {"seed": 555, "test_nll": 0.4, "val_objective": 0.3, "run_dir": "/tmp/h", "campaign": "covid"},
                ],
            },
            {
                "dataset": "citibike-stpp",
                "preset": "neural_jumpcnf",
                "status": "missing",
                "done_seeds": [],
                "missing_seeds": [3, 42, 555],
                "active_jobs": [],
                "latest_job_state": "FAILED",
                "expected_seeds": [42, 3, 555],
                "campaigns": ["citibike-stpp__neural_jumpcnf__04261324"],
                "done_results": [],
            },
        ],
        "protocol_evals": [
            {
                "subject": "covid_seed42/hawkes_gmm/predictive",
                "profile": "predictive",
                "preset": "hawkes_gmm",
                "status": "done",
                "available_metrics": ["temporal_crps"],
                "unavailable_metrics": ["surface_metric"],
                "root": "/tmp/repo/runs/eval_protocol/covid_seed42/hawkes_gmm/predictive",
            }
        ],
        "predictive_kde_evals": [
            {
                "suite": "suite3_entanglement",
                "campaign": "s3ent_v2__gen__04241035",
                "preset": "smash",
                "status": "done",
                "worker_rows": 12,
                "metrics_json": True,
                "metrics_by_run_csv": True,
                "metrics_by_family_level_csv": True,
                "active_jobs": [],
                "latest_job_state": None,
                "root": "/tmp/repo/runs/hawkesnest_campaigns/suite3/smash",
            }
        ],
        "submission_reconciliation": [
            {
                "kind": "realdata_training_submission",
                "submitted_at": "2026-04-27 08:00:00",
                "job_id": "123",
                "job_name": "citibike-stpp__neural_jumpcnf__test",
                "scope": "citibike-stpp",
                "family_or_profile": "neural_jumpcnf",
                "presets": "neural_jumpcnf",
                "current_state": "FAILED",
                "current_bucket": "failed",
                "expected_outputs": 3,
                "done_outputs": 0,
                "failure_summary": "torch.OutOfMemoryError: CUDA out of memory",
                "out_path": "/tmp/repo/runs/exp1/citibike-stpp/bench/test",
            }
        ],
    }

    out_dir = tmp_path / "report"
    render_report(snapshot, out_dir)

    expected_files = [
        "cluster_status_snapshot.json",
        "paper_readiness_report.md",
        "synthetic_training_status.csv",
        "realdata_training_status.csv",
        "protocol_eval_status.csv",
        "predictive_kde_eval_status.csv",
        "synthetic_test_nll_cells.csv",
        "realdata_test_nll_cells.csv",
        "submission_reconciliation.csv",
        "synthetic_test_nll_tables.md",
        "realdata_test_nll_table.md",
        "missing_targets.csv",
        "live_jobs.csv",
    ]
    for name in expected_files:
        assert (out_dir / name).exists(), name

    snapshot_json = json.loads((out_dir / "cluster_status_snapshot.json").read_text())
    assert snapshot_json["repo_root"] == "/tmp/repo"

    report_text = (out_dir / "paper_readiness_report.md").read_text()
    assert "suite4_heterogeneity" in report_text
    assert "citibike-stpp" in report_text
    assert "covid_seed42/hawkes_gmm/predictive" in report_text
    assert "Test NLL Tables" in report_text

    with (out_dir / "missing_targets.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert any(row["scope"] == "suite4_heterogeneity" for row in rows)
    assert any(row["scope"] == "citibike-stpp" for row in rows)
