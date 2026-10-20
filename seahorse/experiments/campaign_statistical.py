"""Statistical cells in the same frozen campaign and public runner lifecycle."""
from __future__ import annotations

import json
import math
import time
import traceback
from datetime import datetime, timezone

import numpy as np

from seahorse.config import STPPConfig
from seahorse.data.prepared_protocol import _copy_unmarked
from seahorse.experiments.campaign import write_json
from seahorse.runner import STPPRunner
from seahorse.runner.runner import RunnerRegistry
from seahorse.utils import deep_update


def raw_context(dataset, index, coord=None):
    """Expose the exact same target identity in source coordinates."""
    if hasattr(dataset, "raw_context"):
        return dataset.raw_context(index)
    # Independent synthetic realizations never inherit another realization's
    # past. Recover raw arrays before the neural float32 boundary.
    source, begin, end, _ = dataset._source_window(index)
    record = dataset.raw_sequences[source]
    times = np.asarray(record["times"], dtype=np.float64)[begin:end]
    locations = np.asarray(record["locations"], dtype=np.float64)[begin:end]
    offset = float(times[0])
    if coord is not None:
        offset = offset * coord.time_scale + coord.time_mean
        times = times * coord.time_scale + coord.time_mean
        locations = locations * np.asarray(coord.space_scale) + np.asarray(coord.space_offset)
    times = times - times[0]
    history = {"times": times[:-1], "locations": locations[:-1], "t0": 0., "t1": float(times[-2]),
               "calendar_offset_from_window_start": offset}
    for key in ("window_start", "time_unit", "time_axis", "source_timezone"):
        if key in record:
            history[key] = record[key]
    metadata = dataset[index]["observation_metadata"]
    return {"times": times, "locations": locations, "history": history,
            "next_time": float(times[-1]), "next_location": locations[-1],
            "observation_metadata": metadata}


def select_statistical_trial(manifest, job, campaign_dir):
    candidates = []
    for dependency in job["dependencies"]:
        path = campaign_dir / "jobs" / dependency / "status.json"
        if not path.exists():
            raise ValueError(f"HPO trial has not been attempted: {dependency}")
        status = json.loads(path.read_text())
        if status.get("state") not in {"completed", "failed", "timeout", "blocked"}:
            raise ValueError(f"HPO trial is unfinished: {dependency}")
        value = status.get("validation_objective")
        if status["state"] == "completed" and value is not None and math.isfinite(value):
            trial = next(j for j in manifest["jobs"] if j["id"] == dependency)
            candidates.append((value, trial["trial"], trial))
    if not candidates:
        raise ValueError("No successful statistical HPO trial.")
    return min(candidates, key=lambda item: item[:2])[2]


def _seasonal_smoke_records(path, model, history):
    """Keep a train-only prefix with observed exposure/events in every phase.

    A single calendar record cannot exercise a weekly seasonal estimator.
    Retain intervening empty records; this only bounds a preflight diagnostic,
    and never changes the records used by an actual campaign fit.
    """
    from seahorse.models.statistical.core import calendar_offset
    from seahorse.utils import _canonicalize_sequence_record
    exposure = np.zeros(model.n_phases)
    counts = np.zeros(model.n_phases)
    records = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = _canonicalize_sequence_record(json.loads(line))
            records.append(row)
            offset = calendar_offset(row)
            exposure += model._phase_exposure(float(row["t0"])+offset, float(row["t1"])+offset)
            counts += np.bincount(model._phase(np.asarray(row["times"])+offset), minlength=model.n_phases)
            if (exposure > 0).all() and (counts > 0).all() and counts.sum() > history:
                return records
    raise ValueError("Seasonal smoke training prefix lacks exposure/events in every declared phase.")


def _data(manifest, job, split, *, smoke=False, seasonal_smoke_model=None):
    from seahorse.experiments.campaign_run import read_records, context_dataset
    route = manifest["datasets"][job["dataset"]]
    history = manifest["spec"]["protocol"]["history_events"]
    if smoke and split == "train" and seasonal_smoke_model is not None:
        records = _copy_unmarked(_seasonal_smoke_records(route["splits"][split], seasonal_smoke_model, history))
    else:
        records = _copy_unmarked(read_records(route["splits"][split], sha256=route["split_sha256"][split],
                                            sample_history=history if smoke else None))
    dataset = context_dataset(records, manifest, job, None, split=split)
    return records, dataset


def _scores(runner, dataset, indices):
    if hasattr(runner.model, "score_conditional"):
        from itertools import islice
        iterator = iter(indices)
        batches = []
        while True:
            batch = list(islice(iterator, 8192))
            if not batch:
                break
            batches.append(runner.model.score_conditional(
                lambda: (raw_context(dataset, i) for i in batch), history_events=dataset.lookback,
                chunk_size=8192, cache_bytes=64 * 1024 * 1024))
        return np.concatenate(batches) if batches else np.empty(0, dtype=np.float64)
    values = []
    for index in indices:
        context = raw_context(dataset, index)
        values.append(runner.model.next_event_nll(context["history"], context["next_time"], context["next_location"]))
    return np.asarray(values, dtype=np.float64)


def fit_statistical(manifest, job, campaign_dir, *, smoke=False, claim=None):
    from seahorse.experiments.runtime import classify_exception, process_context
    from seahorse.experiments.campaign_run import _model_entry, _coordinate_entry, select_indices, clean_json
    out = campaign_dir / ("preflight" if smoke else "jobs") / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    if not smoke and status_path.exists():
        raise FileExistsError("Preserve previous attempts; use a versioned retry.")
    status = {"job_id": job["id"], "state": "running", "smoke_only": smoke, "runner": "statstpp",
              "manifest_sha256": manifest["sha256"], "started_utc": datetime.now(timezone.utc).isoformat(),
              "attempt": claim, "process": process_context(device="cpu"),
              "stdout_path": str(out / "stdout.log"), "traceback_path": str(out / "traceback.txt")}
    if claim and claim.get("retry_of"):
        status["retry_of"] = claim["retry_of"]
    write_json(status_path, status)
    started = time.monotonic()
    try:
        model, coord = _model_entry(manifest, job), _coordinate_entry(manifest, job)
        selected = select_statistical_trial(manifest, job, campaign_dir) if job["action"] == "refit" and not smoke else job
        values = STPPConfig.from_preset(model["preset"]).model_dump(mode="python")
        deep_update(values, model.get("config_overrides", {}))
        deep_update(values, model.get("dataset_config_overrides", {}).get(job["dataset"], {}))
        deep_update(values, selected.get("config_overrides", {}))
        deep_update(values, {"data": {"normalize": False, "protocol": "raw", "seed": int(job["seed"])},
                             "model": {"support": list(coord.spatial_bounds_lon_lat)},
                             "logging": {"out_dir": str(out / "fit"), "save_checkpoints": True}})
        cfg = STPPConfig.model_validate(values)
        phase_probe = cfg.model.build_model() if smoke and model["preset"] == "seasonal_poisson" else None
        train_records, train = _data(manifest, job, "train", smoke=smoke, seasonal_smoke_model=phase_probe)
        val_records, validation = _data(manifest, job, "val", smoke=smoke)
        protocol = manifest["spec"]["protocol"]
        val_ids = select_indices(len(validation), 2 if smoke else protocol["validation_targets"], protocol["validation_target_seed"])
        if not len(train) or not val_ids:
            raise ValueError("No shared train/validation targets under the frozen history policy.")
        runner = RunnerRegistry.create(cfg)
        conditional = bool(model.get("conditional_fit"))
        kwargs = {}
        if conditional:
            cfg.statistical["history_events"] = protocol["history_events"]
            train_ids = select_indices(len(train), 16, 0) if smoke else range(len(train))
            kwargs["conditional_contexts"] = lambda: (raw_context(train, i) for i in train_ids)
        from seahorse.experiments.runtime import wall_time_limit
        with wall_time_limit(120 if smoke else job["max_seconds"]):
            result = runner.fit(train_records, None, None, dataset_id=job["dataset"], **kwargs)
        scores = _scores(runner, validation, val_ids)
        if not scores.size or not np.isfinite(scores).all():
            raise ValueError("Statistical validation requires all selected common targets to have finite scores.")
        saved = runner.save(out / "model")
        restored = STPPRunner.load(saved)
        if runner.model.to_state() != restored.model.to_state():
            raise ValueError("Statistical checkpoint state differs after reload.")
        np.testing.assert_allclose(scores, _scores(restored, validation, val_ids), rtol=1e-12, atol=1e-12)
        write_json(out / "input_contract.json", clean_json({"coordinate_transform": coord.to_dict(),
            "model_input_space": "raw source units", "validation_target_indices": val_ids,
            "train_target_population": len(train), "filter_counts": train.sequence_filter_counts,
            "validation_target_policy": protocol, "sample_only": smoke,
            "fit_policy": "shared H-event conditional composite likelihood" if conditional else "analytic complete observed-window exposure fit",
            "observation_view": manifest["datasets"][job["dataset"]]}))
        status.update(state="completed", validation_objective=float(scores.mean()), validation_metric="raw_conditional_next_event_nll",
            chosen_hpo_trial=selected["id"] if selected is not job else None,
            checkpoint=str(saved), elapsed_seconds=time.monotonic()-started,
            checkpoint_semantic_parity="shared_validation_context_scores_and_full_native_state",
            training_audit={"fit_seconds": result.train_time_sec, "n_params": result.n_params,
                            "target_population": len(train), "diagnostics": result.extra_metrics["diagnostics"]},
            budget_limited=(120 if smoke else job["max_seconds"]) is not None and
                result.train_time_sec >= .95 * (120 if smoke else job["max_seconds"]),
            process=process_context(device="cpu"))
    except Exception as error:
        status.update(state="timeout" if isinstance(error, TimeoutError) else "failed",
                      **classify_exception(error, origin="fit"), process=process_context(device="cpu"),
                      elapsed_seconds=time.monotonic()-started, budget_limited=isinstance(error, TimeoutError))
        (out / "traceback.txt").write_text(traceback.format_exc())
    write_json(status_path, clean_json(status))
    return status


def evaluate_statistical(manifest, job, campaign_dir):
    from seahorse.experiments.campaign_run import _coordinate_entry, _model_entry, clean_json
    out = campaign_dir / "jobs" / job["id"]
    status = json.loads((out / "status.json").read_text())
    if status["state"] != "completed" or job["action"] != "refit":
        raise ValueError("Test evaluation requires a completed final-seed refit.")
    if (out / "evaluation.json").exists():
        raise FileExistsError("Test evaluation already recorded.")
    runner = STPPRunner.load(status["checkpoint"])
    _, dataset = _data(manifest, job, "test")
    from seahorse.experiments.campaign_progress import checkpoint_fingerprint
    from seahorse.experiments.campaign import digest_file
    binding = {"manifest_sha256": manifest["sha256"], "job_id": job["id"],
        "checkpoint_sha256": checkpoint_fingerprint(status["checkpoint"]),
        "test_split_sha256": manifest["datasets"][job["dataset"]]["split_sha256"]["test"]}
    if (out / "nll_status.json").exists():
        metrics = json.loads((out / "nll_status.json").read_text())
        if metrics.get("evaluation_binding") != binding or metrics.get("artifact_sha256") != digest_file(out / "target_nll.npz"):
            raise ValueError("Saved statistical NLL receipt is not bound to this checkpoint and target artifact.")
        from seahorse.experiments.campaign_forecast import evaluate_forecasts
        metrics.update(evaluate_forecasts(runner, dataset, _coordinate_entry(manifest, job), manifest, job, out, device="cpu"))
        write_json(out / "evaluation.json", clean_json(metrics))
        return metrics
    values = np.full(len(dataset), np.nan, dtype=np.float64)
    sequence_ids, event_ids = np.empty(len(dataset), np.int64), np.empty(len(dataset), np.int64)
    errors = []
    if hasattr(runner.model, "score_conditional"):
        values = _scores(runner, dataset, range(len(dataset)))
    for index in range(len(dataset)):
        context = raw_context(dataset, index)
        metadata = context["observation_metadata"]
        sequence_ids[index] = metadata["target_source_sequence_index"]
        event_ids[index] = metadata["target_global_event_index"]
        if hasattr(runner.model, "score_conditional"):
            continue
        try:
            values[index] = runner.model.next_event_nll(context["history"], context["next_time"], context["next_location"])
        except (TimeoutError, MemoryError):
            raise
        except Exception as error:
            if len(errors) < 100:
                errors.append({"target": index, "error": str(error)})
    np.savez_compressed(out / "target_nll.npz", nll=values, sequence_index=sequence_ids, target_event_index=event_ids)
    finite = np.isfinite(values)
    metrics = {"runner": "statstpp", "nll_path": _model_entry(manifest, job)["nll_path"],
               "target_population": len(dataset), "finite_targets": int(finite.sum()),
               "failed_targets": int((~finite).sum()), "failure_batches": errors,
               "raw_next_event_nll": float(values.mean()) if values.size and finite.all() else None,
               "nll_status": "complete" if values.size and finite.all() else "failed",
               "filter_counts": dataset.sequence_filter_counts,
               "nll_method": getattr(runner.model, "nll_method", "native_statistical_next_event"),
               "nll_kind": runner.capabilities.nll_kind,
               "nll_description": "Joint conditional next-event density after the shared H-event observed history, in raw source units.",
               "nll_footnote": runner.capabilities.nll_footnote,
               "observation": manifest["datasets"][job["dataset"]].get("observation")}
    write_json(out / "nll_status.json", clean_json({**metrics, "evaluation_binding": binding,
        "artifact_sha256": digest_file(out / "target_nll.npz"), "manifest_sha256": manifest["sha256"], "job_id": job["id"]}))
    from seahorse.experiments.campaign_forecast import evaluate_forecasts
    metrics.update(evaluate_forecasts(runner, dataset, _coordinate_entry(manifest, job), manifest, job, out, device="cpu"))
    write_json(out / "evaluation.json", clean_json(metrics))
    return metrics
