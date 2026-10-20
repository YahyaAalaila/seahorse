"""Shared next-event and fixed-horizon forecast evaluation for both runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from seahorse.evaluation.forecast import forecast_next, forecast_horizon, score_horizon_counts
from seahorse.experiments.campaign_statistical import raw_context
from seahorse.experiments.runtime import classify_exception


def _is_statistical(runner):
    return getattr(runner, "runner_name", None) == "statstpp"


def _input_history(context, dataset, index, coord, statistical):
    if statistical:
        return dict(context["history"])
    item = dataset[index]
    return {
        "times": item["times"].numpy().astype(np.float64)[:-1],
        "locations": item["locations"].numpy().astype(np.float64)[:-1],
    }


def _count_context(dataset, index, horizon, coord):
    if hasattr(dataset, "raw_count_context"):
        return dataset.raw_count_context(index, horizon)
    source, begin, end, _ = dataset._source_window(index)
    context = raw_context(dataset, index, coord)
    record = dataset.raw_sequences[source]
    times, locations = (
        np.asarray(record["times"], dtype=np.float64),
        np.asarray(record["locations"], dtype=np.float64),
    )
    if coord is not None:
        times = times * coord.time_scale + coord.time_mean
        locations = locations * np.asarray(coord.space_scale) + np.asarray(coord.space_offset)
    base = float(times[begin])
    origin = float(context["history"]["times"][-1])
    finish = origin + horizon
    if "t1" not in record or finish + base > float(record["t1"]) + 1e-12:
        raise ValueError("Count horizon extends beyond the independent observation window.")
    future_end = np.searchsorted(times, finish + base, side="left")
    return {
        "origin": origin,
        "end": finish,
        "history": context["history"],
        "true_future": {
            "times": times[end - 1 : future_end] - base,
            "locations": locations[end - 1 : future_end],
        },
        "observation_metadata": context["observation_metadata"],
    }


def _select_count_contexts(dataset, candidates, limit, horizon, coord, seed):
    # Candidates are sorted for stable target indexing. Taking the first
    # eligible ones would concentrate forecasts at the beginning of test.
    # A seeded permutation instead samples eligible origins across the split.
    selected = []
    for index in np.random.default_rng(seed).permutation(candidates).tolist():
        try:
            context = _count_context(dataset, index, horizon, coord)
        except ValueError:
            continue
        selected.append((index, context))
        if len(selected) == limit:
            break
    return sorted(selected, key=lambda item: item[0])


def _count_horizon(coord, specification):
    if "horizon_median_intervals" in specification:
        # Synthetic coordinates intentionally use identity scaling, so their
        # normalization scale is not a useful forecasting duration. This rule
        # uses the same train-only physical interval statistic for all corpora.
        horizon = (
            float(specification["horizon_median_intervals"]) * coord.median_positive_train_interval
        )
    else:
        horizon = float(specification["horizon_time_scales"]) * coord.time_scale
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Count forecasting requires a positive finite train-defined horizon.")
    return float(horizon)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _checkpoint_identity(out: Path) -> dict:
    status = out / "status.json"
    if not status.exists():
        return {"checkpoint": None}
    value = json.loads(status.read_text())
    checkpoint = value.get("checkpoint")
    digest = None
    if checkpoint is not None:
        path = Path(checkpoint)
        if path.exists():
            from seahorse.experiments.campaign_progress import checkpoint_fingerprint

            digest = checkpoint_fingerprint(path)
    # The path is retained as human-readable provenance; reuse is authorized by
    # contents when an actual checkpoint is present.
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": digest,
        "fit_manifest_sha256": value.get("manifest_sha256"),
    }


def _binding(manifest, job, out: Path, *, phase: str, selected_ids, protocol) -> dict:
    source = manifest.get("source", {})
    return {
        "schema_version": 1,
        "phase": phase,
        "job_id": job["id"],
        "manifest_sha256": manifest.get("sha256"),
        "source_sha256": source.get("sha256"),
        "checkpoint_identity": _checkpoint_identity(out),
        "selected_ids": [int(i) for i in selected_ids],
        "protocol_sha256": _canonical_hash(protocol),
    }


def _progress(path: Path, binding: dict) -> dict:
    if not path.exists():
        return {
            "binding": binding,
            "rows": {},
            "failures": {},
            "unsupported": None,
            "complete": False,
        }
    value = json.loads(path.read_text())
    if value.get("binding") != binding:
        raise ValueError(
            f"Forecast progress binding differs at {path}; refuse to mix manifest, checkpoint, targets, or protocol."
        )
    return value


def _resource_failure(exc: BaseException) -> bool:
    return classify_exception(exc, origin="forecast")["failure_class"] in {
        "timeout",
        "host_oom",
        "cuda_oom",
    }


def _predictive_summary(values: np.ndarray, chosen) -> dict:
    predictive = {
        "status": "computed",
        "reason": None,
        "backend": None,
        "target_population": len(chosen),
        "successful_targets": len(values),
        "failures": [],
        "columns": [
            "target_index",
            "temporal_crps_source_units",
            "spatial_energy_source_units",
            "temporal_pit",
            "temporal_mae_source_units",
            "spatial_mean_error_source_units",
            "temporal_squared_error_source_units",
            "spatial_squared_error_source_units",
        ],
        "summary": None,
    }
    if len(values) == len(chosen) and len(values):
        predictive["summary"] = {
            "temporal_crps": float(values[:, 1].mean()),
            "spatial_energy_score": float(values[:, 2].mean()),
            "temporal_mae": float(values[:, 4].mean()),
            "spatial_mae": float(values[:, 5].mean()),
            "temporal_rmse": float(np.sqrt(values[:, 6].mean())),
            "spatial_rmse": float(np.sqrt(values[:, 7].mean())),
        }
    return predictive


def evaluate_forecasts(runner, dataset, coord, manifest, job, out, *, device="cpu"):
    from seahorse.experiments.campaign_run import select_indices

    protocol = manifest["spec"]["protocol"]
    statistical = _is_statistical(runner)
    context_coord = None if statistical else coord
    out = Path(out)
    chosen = select_indices(
        len(dataset), protocol["predictive_targets"], protocol["predictive_target_seed"]
    )
    predictive_path = out / "predictive_status.json"
    predictive_binding = _binding(
        manifest,
        job,
        out,
        phase="predictive",
        selected_ids=chosen,
        protocol={
            key: protocol[key]
            for key in (
                "history_events",
                "predictive_targets",
                "predictive_target_seed",
                "predictive_samples",
            )
        },
    )
    predictive_progress = _progress(predictive_path, predictive_binding)
    backend = predictive_progress.get("backend")
    unsupported = predictive_progress.get("unsupported")
    for index in chosen:
        if (
            str(index) in predictive_progress["rows"]
            or str(index) in predictive_progress["failures"]
            or unsupported
        ):
            continue
        context = raw_context(dataset, index, context_coord)
        history = _input_history(context, dataset, index, coord, statistical)
        try:
            samples = forecast_next(
                runner,
                history,
                n_samples=protocol["predictive_samples"],
                seed=protocol["predictive_target_seed"] + index,
                device=device,
            )
            if samples["status"] == "unsupported":
                unsupported = samples["reason"]
                predictive_progress["unsupported"] = unsupported
                _atomic_json(predictive_path, predictive_progress)
                break
            backend = samples["backend"]
            if not samples["valid"].all():
                raise ValueError(
                    "Next-event sample metrics require finite valid draws; no-event mass is retained in count forecasts only."
                )
            times, locations = samples["times"], samples["locations"]
            if not statistical:
                times = times * coord.time_scale
                locations = locations * np.asarray(coord.space_scale) + np.asarray(
                    coord.space_offset
                )
            truth_t, truth_x = context["next_time"], context["next_location"]
            crps = (
                np.abs(times - truth_t).mean()
                - 0.5 * np.abs(times[:, None] - times[None, :]).mean()
            )
            energy = (
                np.linalg.norm(locations - truth_x, axis=-1).mean()
                - 0.5 * np.linalg.norm(locations[:, None] - locations[None, :], axis=-1).mean()
            )
            time_error = float(times.mean() - truth_t)
            space_error = float(np.linalg.norm(locations.mean(axis=0) - truth_x))
            predictive_progress["backend"] = backend
            predictive_progress["rows"][str(index)] = [
                index,
                float(crps),
                float(energy),
                float(np.mean(times <= truth_t)),
                float(abs(np.median(times) - truth_t)),
                space_error,
                time_error**2,
                space_error**2,
            ]
            _atomic_json(predictive_path, predictive_progress)
        except Exception as error:
            if _resource_failure(error):
                _atomic_json(predictive_path, predictive_progress)
                raise
            predictive_progress["failures"][str(index)] = {
                "target": index,
                "error": str(error),
                "error_type": type(error).__name__,
            }
            _atomic_json(predictive_path, predictive_progress)
    rows = [
        predictive_progress["rows"][str(index)]
        for index in chosen
        if str(index) in predictive_progress["rows"]
    ]
    errors = [
        predictive_progress["failures"][str(index)]
        for index in chosen
        if str(index) in predictive_progress["failures"]
    ]
    values = np.asarray(rows, dtype=np.float64).reshape(-1, 8)
    np.savez_compressed(
        out / "predictive_metrics.npz",
        rows=values,
        selected_targets=np.asarray(chosen, dtype=np.int64),
    )
    predictive = _predictive_summary(values, chosen)
    predictive.update(
        status="unsupported" if unsupported else "computed",
        reason=unsupported,
        backend=backend,
        failures=errors,
        successful_targets=len(rows),
        time_unit=coord.input_time_unit,
        spatial_units="simulation units"
        if manifest["datasets"][job["dataset"]].get("kind") == "synthetic"
        else "longitude/latitude degrees",
    )
    predictive_progress.update(complete=True, predictive=predictive)
    _atomic_json(predictive_path, predictive_progress)
    count_spec = protocol.get("count_forecast")
    if not count_spec:
        return {"predictive": predictive}
    horizon = _count_horizon(coord, count_spec)
    # Eligibility depends only on declared observation coverage, never on
    # realized future counts or a model's predictions.
    candidate_ids = select_indices(
        len(dataset), count_spec["candidate_origins"], count_spec["seed"]
    )
    selected_contexts = _select_count_contexts(
        dataset, candidate_ids, count_spec["origins"], horizon, context_coord, count_spec["seed"]
    )
    eligible = [index for index, _ in selected_contexts]
    count_path = out / "count_forecast_status.json"
    count_binding = _binding(
        manifest, job, out, phase="count_forecast", selected_ids=eligible, protocol=count_spec
    )
    count_progress = _progress(count_path, count_binding)
    for index, context in selected_contexts:
        if unsupported or count_progress.get("unsupported"):
            break
        if str(index) in count_progress["rows"] or str(index) in count_progress["failures"]:
            continue
        try:
            history = dict(context["history"])
            endpoint = context["end"]
            if not statistical:
                history = {
                    "times": np.asarray(history["times"]) / coord.time_scale,
                    "locations": (np.asarray(history["locations"]) - np.asarray(coord.space_offset))
                    / np.asarray(coord.space_scale),
                }
                endpoint /= coord.time_scale
            forecast = forecast_horizon(
                runner,
                history,
                t_end=endpoint,
                n_samples=count_spec["samples"],
                seed=count_spec["seed"] + index,
                device=device,
                history_limit=protocol["history_events"],
                event_cap=count_spec["event_cap"],
            )
            if forecast["status"] == "unsupported":
                unsupported = forecast["reason"]
                count_progress["unsupported"] = unsupported
                _atomic_json(count_path, count_progress)
                break
            if not statistical:
                forecast["sample_times"] = [
                    np.asarray(t) * coord.time_scale for t in forecast["sample_times"]
                ]
                forecast["sample_locations"] = [
                    np.asarray(x) * np.asarray(coord.space_scale) + np.asarray(coord.space_offset)
                    for x in forecast["sample_locations"]
                ]
            score = score_horizon_counts(
                forecast,
                context["true_future"]["times"],
                truth_locations=context["true_future"]["locations"],
                t_start=context["origin"],
                t_end=context["end"],
                bin_width=horizon / count_spec["time_bins"],
                spatial_bounds=coord.spatial_bounds_lon_lat,
                spatial_grid=tuple(count_spec["spatial_grid"]),
            )
            if not all(np.isfinite(value) for value in score.values()):
                raise ValueError(
                    "Count forecast has invalid or capped paths; truncated forecasts cannot be scored."
                )
            count_progress["rows"][str(index)] = [
                index,
                float(score["count_crps"]),
                float(score["count_rmse"]),
                float(score["count_mae"]),
            ]
            _atomic_json(count_path, count_progress)
        except Exception as error:
            if _resource_failure(error):
                _atomic_json(count_path, count_progress)
                raise
            count_progress["failures"][str(index)] = {
                "target": index,
                "error": str(error),
                "error_type": type(error).__name__,
            }
            _atomic_json(count_path, count_progress)
    count_rows = [
        count_progress["rows"][str(index)]
        for index in eligible
        if str(index) in count_progress["rows"]
    ]
    count_failures = [
        count_progress["failures"][str(index)]
        for index in eligible
        if str(index) in count_progress["failures"]
    ]
    count_values = np.asarray(count_rows, dtype=np.float64).reshape(-1, 4)
    np.savez_compressed(
        out / "count_forecast_metrics.npz",
        rows=count_values,
        selected_targets=np.asarray(eligible, dtype=np.int64),
    )
    counts = {
        "status": "unsupported" if unsupported else "computed",
        "reason": unsupported,
        "target_population": len(eligible),
        "successful_targets": len(count_rows),
        "failures": count_failures,
        "horizon_source_time_units": horizon,
        "origin_policy": "fixed selected observed-event origins; H events; fully observed future horizon",
        "time_bins": count_spec["time_bins"],
        "spatial_grid": count_spec["spatial_grid"],
        "spatial_overflow": "additional cell per time bin; generated locations outside domain are retained",
        "summary": None,
    }
    if len(count_rows) == len(eligible) and count_rows:
        counts["summary"] = {
            "count_crps": float(count_values[:, 1].mean()),
            "count_rmse": float(np.sqrt(np.mean(count_values[:, 2] ** 2))),
            "count_mae": float(count_values[:, 3].mean()),
        }
    count_progress.update(complete=True, count_forecast=counts)
    _atomic_json(count_path, count_progress)
    return {"predictive": predictive, "count_forecast": counts}
