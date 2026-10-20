"""Read-only aggregation for a frozen Seahorse campaign.

The report is deliberately a ledger before it is a table: every planned final
seed remains visible, including unrun and failed cells.  It never pools raw
NLLs across datasets.  Model comparisons within a dataset use matched seeds
and a source-calendar moving-block bootstrap over target records.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from seahorse.experiments.attempts import failure_from_status


PRIMARY_UNAVAILABLE = {"none", "bound", "surrogate"}
PREDICTIVE_METRICS = (
    "temporal_crps",
    "spatial_energy_score",
    "temporal_mae",
    "spatial_mae",
    "temporal_rmse",
    "spatial_rmse",
)
COUNT_METRICS = ("count_crps", "count_rmse", "count_mae")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _nll_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"nll", "sequence_index", "target_event_index"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"target NLL artifact lacks {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in required}
    n = len(arrays["nll"])
    if any(len(value) != n for value in arrays.values()):
        raise ValueError("target NLL arrays have inconsistent lengths")
    return arrays


def _complete_metric_section(
    section: Any, metrics: tuple[str, ...]
) -> tuple[str, dict[str, float] | None, dict[str, Any]]:
    """Validate target coverage before exposing an evaluation summary."""
    if not isinstance(section, dict):
        return "missing", None, {"expected": None, "present": None, "failures": None}
    expected = section.get("target_population")
    present = section.get("successful_targets")
    failures = section.get("failures", [])
    summary = section.get("summary")
    complete = (
        section.get("status") == "computed"
        and isinstance(expected, int)
        and expected >= 0
        and present == expected
        and not failures
        and isinstance(summary, dict)
        and all(name in summary and np.isfinite(float(summary[name])) for name in metrics)
    )
    coverage = {
        "expected": expected,
        "present": present,
        "failures": len(failures) if isinstance(failures, list) else None,
    }
    if not complete:
        return "coverage_failed", None, coverage
    return "complete", {name: float(summary[name]) for name in metrics}, coverage


def _model_metadata(manifest: dict[str, Any], model_id: str) -> dict[str, Any]:
    entry = next(
        (m for m in manifest.get("spec", {}).get("models", []) if m.get("id") == model_id), {}
    )
    # Preserve frozen metadata verbatim where available: a reporting layer must
    # not infer provenance or numerical exactness from a friendly model name.
    return {
        "preset": entry.get("preset"),
        "nll_kind": entry.get("nll_kind"),
        "nll_path_declared": entry.get("nll_path"),
        "provenance": entry.get("provenance", entry.get("provenance_label")),
        "inference_objective": entry.get("inference_objective", entry.get("objective")),
        "computational_censorship_declared": entry.get("computational_censorship"),
    }


def _attempt_diagnostics(status: dict[str, Any] | None) -> dict[str, Any] | None:
    if not status:
        return None
    keys = (
        "log_path",
        "stdout_log",
        "stderr_log",
        "elapsed_seconds",
        "retry",
        "retry_of",
        "process",
        "scheduler_state",
        "slurm_exit_code",
        "signal",
        "returncode",
    )
    value = {key: status.get(key) for key in keys if key in status}
    return value or None


def _cell(manifest: dict[str, Any], job: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / "jobs" / job["id"]
    status_path = output / "status.json"
    cell = {
        "job_id": job["id"],
        "dataset": job["dataset"],
        "model": job["model"],
        "seed": job["seed"],
        "state": "not_run",
        "nll_status": "not_evaluated",
        "predictive_status": "not_evaluated",
        "count_forecast_status": "not_evaluated",
        "output": str(output),
        **_model_metadata(manifest, job["model"]),
    }
    if not status_path.exists():
        return cell
    status = _load_json(status_path)
    cell["fit_status"] = status.get("state")
    if status.get("state") != "completed":
        cell["state"] = "fit_failed"
        cell["failure"] = failure_from_status(status, default_origin="fit") or status.get("error")
        cell["fit_attempt"] = _attempt_diagnostics(status)
        return cell
    evaluation_path = output / "evaluation.json"
    nll_receipt_path = output / "nll_status.json"
    attempt_path = output / "evaluation_status.json"
    attempt = _load_json(attempt_path) if attempt_path.exists() else None
    if not evaluation_path.exists():
        # NLL is persisted before forecast sampling.  A later sampling timeout
        # must not erase a complete, finite all-target NLL receipt.
        if not nll_receipt_path.exists():
            evaluation_state = attempt.get("state") if attempt else None
            cell["state"] = (
                "eval_failed"
                if evaluation_state in {"failed", "timeout", "cancelled", "stale"}
                else "evaluation_pending"
            )
            if attempt:
                cell["evaluation_failure"] = failure_from_status(
                    attempt, default_origin="evaluation"
                )
                cell["evaluation_attempt"] = _attempt_diagnostics(attempt)
            return cell
        evaluation = _load_json(nll_receipt_path)
        # Forecast progress is separately atomic so a count-forecast timeout
        # cannot hide the already completed next-event metrics.
        predictive_receipt = output / "predictive_status.json"
        count_receipt = output / "count_forecast_status.json"
        if predictive_receipt.exists():
            progress = _load_json(predictive_receipt)
            if isinstance(progress.get("predictive"), dict):
                evaluation["predictive"] = progress["predictive"]
        if count_receipt.exists():
            progress = _load_json(count_receipt)
            if isinstance(progress.get("count_forecast"), dict):
                evaluation["count_forecast"] = progress["count_forecast"]
        partial_evaluation = True
    else:
        evaluation = _load_json(evaluation_path)
        partial_evaluation = False
    cell["nll_path"] = evaluation.get("nll_path", cell.get("nll_path_declared"))
    cell["nll_kind"] = evaluation.get("nll_kind", cell.get("nll_kind"))
    cell["nll_status"] = evaluation.get("nll_status", "failed")
    predictive = evaluation.get("predictive", {})
    count_forecast = evaluation.get("count_forecast", {})
    predictive_status, predictive_summary, predictive_coverage = _complete_metric_section(
        predictive, PREDICTIVE_METRICS
    )
    count_status, count_summary, count_coverage = _complete_metric_section(
        count_forecast, COUNT_METRICS
    )
    cell.update(
        predictive_status=predictive_status,
        predictive_summary=predictive_summary,
        predictive_coverage=predictive_coverage,
        count_forecast_status=count_status,
        count_forecast_summary=count_summary,
        count_forecast_coverage=count_coverage,
        predictive_failures=predictive_coverage["failures"],
        computational_censorship={
            "fit_budget_limited": bool(status.get("budget_limited", False)),
            "evaluation": evaluation.get("computational_censorship"),
        },
    )
    nll_path = output / "target_nll.npz"
    if cell["nll_status"] == "complete":
        try:
            payload = _nll_payload(nll_path)
            finite = np.isfinite(payload["nll"])
            expected = int(evaluation.get("target_population", len(payload["nll"])))
            if len(payload["nll"]) != expected or not finite.all():
                cell.update(
                    state="eval_failed",
                    nll_status="coverage_failed",
                    coverage={
                        "expected": expected,
                        "present": int(len(payload["nll"])),
                        "finite": int(finite.sum()),
                    },
                )
                return cell
            cell.update(
                state="complete",
                coverage={"expected": expected, "present": expected, "finite": expected},
                nll_path=str(nll_path),
                nll_mean=float(payload["nll"].mean()),
            )
            if partial_evaluation:
                cell["evaluation_status"] = "forecast_incomplete_after_nll"
                cell["evaluation_failure"] = (
                    failure_from_status(attempt, default_origin="forecast") if attempt else None
                )
                cell["evaluation_attempt"] = _attempt_diagnostics(attempt)
        except Exception as exc:
            cell.update(state="eval_failed", nll_status="coverage_failed", failure=str(exc))
    elif (
        evaluation.get("nll_path") in PRIMARY_UNAVAILABLE
        or cell["nll_status"] == "unavailable_by_design"
    ):
        predictive_complete = (
            predictive.get("status") == "computed"
            and predictive.get("successful_targets", 0) == predictive.get("target_population", -1)
            and not predictive.get("failures")
        )
        cell["state"] = "complete" if predictive_complete else "eval_failed"
    else:
        cell["state"] = "eval_failed"
    return cell


def _matched_payload(cell: dict[str, Any]) -> dict[tuple[int, int], float]:
    data = _nll_payload(Path(cell["nll_path"]))
    keys = list(
        zip(data["sequence_index"].astype(int), data["target_event_index"].astype(int), strict=True)
    )
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (source record, target event) in target NLL artifact")
    return dict(zip(keys, data["nll"].astype(float), strict=True))


def moving_block_difference(
    candidate: dict[tuple[int, int], float],
    baseline: dict[tuple[int, int], float],
    *,
    block_records: int = 7,
    replicates: int = 2000,
    seed: int = 20260910,
) -> dict[str, Any]:
    """Paired source-calendar moving-block bootstrap of candidate minus baseline.

    Records are ordered by observed source-sequence index.  A block contains
    adjacent *observed* records, not arithmetic index values; missing source
    records therefore remain a documented limitation rather than an iid fix.
    """
    if set(candidate) != set(baseline):
        raise ValueError("paired comparison requires identical target IDs")
    if block_records < 1 or replicates < 1:
        raise ValueError("block_records and replicates must be positive")
    records = sorted({key[0] for key in candidate})
    if not records:
        raise ValueError("paired comparison has no targets")
    record_index = {record: i for i, record in enumerate(records)}
    sums, counts = np.zeros(len(records)), np.zeros(len(records), dtype=np.int64)
    for key, value in candidate.items():
        i = record_index[key[0]]
        sums[i] += value - baseline[key]
        counts[i] += 1
    if not np.isfinite(sums).all():
        raise ValueError("paired comparison contains nonfinite scores")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    n_blocks = int(np.ceil(len(records) / block_records))
    for b in range(replicates):
        starts = rng.integers(0, len(records), size=n_blocks)
        chosen = ((starts[:, None] + np.arange(block_records)) % len(records)).ravel()[
            : len(records)
        ]
        # Record sums/counts give exactly the event-weighted mean obtained by
        # repeating all events in each sampled record, at O(records) cost.
        draws[b] = sums[chosen].sum() / counts[chosen].sum()
    return {
        "paired_targets": int(counts.sum()),
        "source_records": int(len(records)),
        "block_records": int(block_records),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_difference_candidate_minus_poisson_gmm": float(sums.sum() / counts.sum()),
        "bootstrap_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "source_record_index_policy": "sorted observed source-calendar records; source-index gaps (including gaps among same-calendar-day source records) are not treated as iid observations",
        "limitation": "This addresses local dependence across adjacent retained source records, not all temporal dependence; synthetic independent-window data needs a separately declared bootstrap.",
    }


def _all_seed_metric(
    cells: list[dict[str, Any]],
    seeds: list[int],
    *,
    status_key: str,
    summary_key: str,
    metrics: tuple[str, ...],
) -> tuple[str, dict[str, dict[str, float]] | None, dict[str, float] | None]:
    by_seed = {cell["seed"]: cell for cell in cells}
    complete = set(by_seed) == set(seeds) and all(
        by_seed[seed].get(status_key) == "complete" for seed in seeds
    )
    if not complete:
        return "incomplete_or_failed", None, None
    per_seed = {str(seed): by_seed[seed][summary_key] for seed in seeds}
    mean = {
        metric: float(np.mean([per_seed[str(seed)][metric] for seed in seeds]))
        for metric in metrics
    }
    return "complete", per_seed, mean


def _observation_view(manifest: dict[str, Any], dataset: str) -> Any:
    entry = next(
        (d for d in manifest.get("spec", {}).get("datasets", []) if d.get("id") == dataset), {}
    )
    return entry.get("observation_view", entry.get("observation_policy"))


def build_report(
    manifest_path: str | Path,
    *,
    replicates: int = 2000,
    block_records: int = 7,
    seed: int = 20260910,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = _load_json(manifest_path)
    refits = [job for job in manifest["jobs"] if job["action"] == "refit"]
    ledger = [_cell(manifest, job, root) for job in refits]
    seeds = sorted(manifest["spec"]["seeds"])
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in ledger:
        by_pair.setdefault((cell["dataset"], cell["model"]), []).append(cell)
    summaries = []
    wide_rows = []
    for (dataset, model), cells in sorted(by_pair.items()):
        ordered = sorted(cells, key=lambda c: c["seed"])
        primary = [c for c in ordered if c["state"] == "complete" and c["nll_status"] == "complete"]
        full = len(primary) == len(seeds) and {c["seed"] for c in primary} == set(seeds)
        predictive_status, predictive_seed_metrics, predictive_mean = _all_seed_metric(
            ordered,
            seeds,
            status_key="predictive_status",
            summary_key="predictive_summary",
            metrics=PREDICTIVE_METRICS,
        )
        count_status, count_seed_metrics, count_mean = _all_seed_metric(
            ordered,
            seeds,
            status_key="count_forecast_status",
            summary_key="count_forecast_summary",
            metrics=COUNT_METRICS,
        )
        metadata = _model_metadata(manifest, model)
        summary = {
            "dataset": dataset,
            "model": model,
            "planned_seeds": seeds,
            "cell_states": {str(c["seed"]): c["state"] for c in ordered},
            "nll_paths_by_seed": {str(c["seed"]): c.get("nll_path") for c in ordered},
            "nll_kinds_by_seed": {str(c["seed"]): c.get("nll_kind") for c in ordered},
            "computational_censorship_by_seed": {
                str(c["seed"]): c.get("computational_censorship") for c in ordered
            },
            "primary_nll_status": "complete" if full else "incomplete_or_unavailable",
            "primary_nll_seed_means": {str(c["seed"]): c["nll_mean"] for c in primary}
            if full
            else None,
            "primary_nll_mean_across_seeds": float(np.mean([c["nll_mean"] for c in primary]))
            if full
            else None,
            "predictive_statuses": {str(c["seed"]): c["predictive_status"] for c in ordered},
            "predictive_table_status": predictive_status,
            "predictive_seed_metrics": predictive_seed_metrics,
            "predictive_mean_across_seeds": predictive_mean,
            "count_forecast_statuses": {
                str(c["seed"]): c["count_forecast_status"] for c in ordered
            },
            "count_forecast_table_status": count_status,
            "count_forecast_seed_metrics": count_seed_metrics,
            "count_forecast_mean_across_seeds": count_mean,
            "observation_view": _observation_view(manifest, dataset),
            **metadata,
        }
        summaries.append(summary)
        wide_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "planned_seed_count": len(seeds),
                "primary_nll_status": summary["primary_nll_status"],
                "primary_nll_mean_across_seeds": summary["primary_nll_mean_across_seeds"],
                "predictive_table_status": predictive_status,
                "count_forecast_table_status": count_status,
                "temporal_mae": None
                if predictive_mean is None
                else predictive_mean["temporal_mae"],
                "temporal_rmse": None
                if predictive_mean is None
                else predictive_mean["temporal_rmse"],
                "temporal_crps": None
                if predictive_mean is None
                else predictive_mean["temporal_crps"],
                "spatial_mae": None if predictive_mean is None else predictive_mean["spatial_mae"],
                "spatial_rmse": None
                if predictive_mean is None
                else predictive_mean["spatial_rmse"],
                "spatial_energy_score": None
                if predictive_mean is None
                else predictive_mean["spatial_energy_score"],
                "count_crps": None if count_mean is None else count_mean["count_crps"],
                "count_rmse": None if count_mean is None else count_mean["count_rmse"],
                "count_mae": None if count_mean is None else count_mean["count_mae"],
                "nll_paths_by_seed": summary["nll_paths_by_seed"],
                "nll_kinds_by_seed": summary["nll_kinds_by_seed"],
                "computational_censorship_by_seed": summary["computational_censorship_by_seed"],
                "observation_view": summary["observation_view"],
                **metadata,
            }
        )
    comparisons = []
    for dataset in sorted({c["dataset"] for c in ledger}):
        dataset_entry = next(
            (d for d in manifest["spec"].get("datasets", []) if d["id"] == dataset), {}
        )
        dataset_blocks = int(dataset_entry.get("bootstrap_block_records", block_records))
        base = {
            c["seed"]: c
            for c in ledger
            if c["dataset"] == dataset
            and c["model"] == "poisson_gmm"
            and c["state"] == "complete"
            and c["nll_status"] == "complete"
        }
        for summary in [
            s for s in summaries if s["dataset"] == dataset and s["model"] != "poisson_gmm"
        ]:
            candidate = {
                c["seed"]: c
                for c in ledger
                if c["dataset"] == dataset
                and c["model"] == summary["model"]
                and c["state"] == "complete"
                and c["nll_status"] == "complete"
            }
            if set(base) != set(seeds) or set(candidate) != set(seeds):
                comparisons.append(
                    {
                        "dataset": dataset,
                        "model": summary["model"],
                        "baseline": "poisson_gmm",
                        "status": "incomplete_seed_pairing",
                    }
                )
                continue
            per_seed = []
            try:
                for s in seeds:
                    per_seed.append(
                        moving_block_difference(
                            _matched_payload(candidate[s]),
                            _matched_payload(base[s]),
                            block_records=dataset_blocks,
                            replicates=replicates,
                            seed=seed + int(s),
                        )
                    )
                comparisons.append(
                    {
                        "dataset": dataset,
                        "model": summary["model"],
                        "baseline": "poisson_gmm",
                        "status": "complete",
                        "per_seed": per_seed,
                    }
                )
            except Exception as exc:
                comparisons.append(
                    {
                        "dataset": dataset,
                        "model": summary["model"],
                        "baseline": "poisson_gmm",
                        "status": "pairing_failed",
                        "failure": str(exc),
                    }
                )
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.get("sha256"),
        "ledger": ledger,
        "within_dataset_model_summary": summaries,
        "wide_table": wide_rows,
        "paired_vs_poisson_gmm": comparisons,
        "aggregation_rule": "No raw NLL is pooled or ranked across datasets. Primary NLL requires all planned seeds and complete finite target coverage.",
    }


def write_report(report: dict[str, Any], out: str | Path) -> dict[str, str]:
    """Write JSON plus flat CSV ledger and within-dataset wide table."""
    out = Path(out)
    _write(out, report)
    outputs = {"json": str(out)}
    for label, rows in (("ledger", report["ledger"]), ("wide", report["wide_table"])):
        path = out.with_name(f"{out.stem}_{label}.csv")
        keys = sorted({key for row in rows for key in row})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                        for key, value in row.items()
                    }
                )
        outputs[label] = str(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--block-records", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    report = build_report(
        args.manifest, replicates=args.replicates, block_records=args.block_records, seed=args.seed
    )
    outputs = write_report(report, args.out)
    print(json.dumps({"out": outputs, "planned_refit_cells": len(report["ledger"])}, indent=2))


if __name__ == "__main__":
    main()
