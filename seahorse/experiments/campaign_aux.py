"""Freeze auxiliary campaign artifacts without fitting any model.

The main next-event matrix deliberately does not mix calendar-count likelihoods,
author fixed-window likelihoods, or synthetic controls into one metric.  This
module materializes the synthetic control corpora under the configured output
root and emits separately launchable count/external-baseline specifications
with source hashes and gates.

Locations come from the environment: ``SEAHORSE_PREPARED_DATA_ROOT`` (prepared
corpora), ``SEAHORSE_WSM_CHECKOUT`` and ``SEAHORSE_TRITPP_CHECKOUT`` (pinned
author checkouts), ``SEAHORSE_EXTERNAL_ROUTE_ROOT`` (fixed-window routes) and
``SEAHORSE_AUX_OUTPUT_ROOT`` (where corpora may be written).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from seahorse.data.prepared import resolve_prepared_dataset
from seahorse.experiments.synthetic_controls import SyntheticRecipe, generate_splits, ground_truth

PREPARED_ROOT_ENV = "SEAHORSE_PREPARED_DATA_ROOT"
WSM_CHECKOUT_ENV = "SEAHORSE_WSM_CHECKOUT"
TRITPP_CHECKOUT_ENV = "SEAHORSE_TRITPP_CHECKOUT"
EXTERNAL_ROUTE_ROOT_ENV = "SEAHORSE_EXTERNAL_ROUTE_ROOT"
AUX_OUTPUT_ROOT_ENV = "SEAHORSE_AUX_OUTPUT_ROOT"
AUTHOR_WSM_COMMIT = "e051e47ee39f8545f5d76154b7f240988546ffa0"
AUTHOR_TRITPP_COMMIT = "a384a50caf6f489335c6829707e2d7abddb44492"
EXTERNAL_MAX_STEPS = 5000
EXTERNAL_MAX_SECONDS = 600
AUX_SCHEMA_VERSION = 1

_COUNT_IDS = (
    "comcat_california", "geonet_nz", "citibike_nyc", "uber_nyc", "austin311",
    "chicago_crime", "la_crime", "nyc311", "nyc_collisions", "seattle_fire911",
    "fpa_california",
)
_NATIVE_ROUTE = {name: f"prepared:{name}:native" for name in _COUNT_IDS}
_STRICT_EXTERNAL = {
    "comcat_california": "prepared:comcat_california:native",
    "geonet_nz": "prepared:geonet_nz:jittered_seed20260910",
    "austin311": "prepared:austin311:jittered_seed20260910",
    "chicago_crime": "prepared:chicago_crime:jittered_seed20260910",
    "la_crime": "prepared:la_crime:jittered_seed20260910",
    "nyc_collisions": "prepared:nyc_collisions:jittered_seed20260910",
    "seattle_fire911": "prepared:seattle_fire911:jittered_seed20260910",
}
_WSM_IDS = tuple(name for name in _STRICT_EXTERNAL if name != "comcat_california")


def _required_env(name: str, purpose: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; it must point to {purpose}.")
    return value


def prepared_root() -> str:
    """Prepared-data root (``SEAHORSE_PREPARED_DATA_ROOT``)."""
    return _required_env(PREPARED_ROOT_ENV, "the directory holding the prepared corpora")


def author_wsm_checkout() -> str:
    return _required_env(WSM_CHECKOUT_ENV, f"a WSM_STPP checkout at commit {AUTHOR_WSM_COMMIT}")


def author_tritpp_checkout() -> str:
    return _required_env(TRITPP_CHECKOUT_ENV, f"a triangular-tpp checkout at commit {AUTHOR_TRITPP_COMMIT}")


def fixed_window_route_root() -> str:
    return _required_env(EXTERNAL_ROUTE_ROOT_ENV, "the directory holding the fixed-window external routes")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)
    return _sha256(path)


def _assert_output_root(out: Path) -> None:
    """Synthetic corpora may only be written under the declared output root."""
    if os.environ.get("SEAHORSE_AUX_ALLOW_LOCAL_FIXTURE") == "1":
        return
    root = os.environ.get(AUX_OUTPUT_ROOT_ENV)
    if not root:
        raise RuntimeError(
            f"{AUX_OUTPUT_ROOT_ENV} is not set; point it at the directory under which auxiliary "
            "corpora may be materialized (SEAHORSE_AUX_ALLOW_LOCAL_FIXTURE=1 for test fixtures)."
        )
    if not str(out.resolve()).startswith(str(Path(root).expanduser().resolve())):
        raise RuntimeError(f"Auxiliary corpora may only be materialized under {AUX_OUTPUT_ROOT_ENV}={root}.")


def _synthetic_record(record: dict[str, Any], *, family: str, split: str, index: int) -> dict[str, Any]:
    times = [float(x) for x in record["times"]]
    locations = [[float(a), float(b)] for a, b in record["locations"]]
    return {
        "times": times,
        "locations": locations,
        "event_ids": [f"{family}:{split}:{index}:{j}" for j in range(len(times))],
        "sequence_id": f"synthetic:{family}:{split}:{index}",
        "stream_id": f"synthetic:{family}:{split}",
        "window_start": "synthetic:0", "window_end": "synthetic:1",
        "t0": 0.0, "t1": 1.0, "time_unit": "synthetic_horizon",
        "observation_metadata": {
            "time_axis": "synthetic", "source_timezone": None,
            "timestamp_semantics": "simulated event time", "synthetic_family": family,
            "window_start": 0.0, "window_end": 1.0,
        },
    }


def _identity_coordinate_entry(*, train_records: list[dict[str, Any]]) -> dict[str, Any]:
    intervals = [np.diff(np.asarray(r["times"], dtype=float)) for r in train_records if len(r["times"]) > 1]
    positive = np.concatenate(intervals) if intervals else np.asarray([], dtype=float)
    return {
        "version": 1, "history": 20, "input_time_unit": "synthetic_horizon",
        "input_time_axis": "synthetic", "spatial_bounds_lon_lat": [0.0, 1.0, 0.0, 1.0],
        "time_mean": 0.0, "time_scale": 1.0, "space_offset": [0.0, 0.0],
        "space_scale": [1.0, 1.0], "log_raw_volume_per_normalized_volume": 0.0,
        "train_sequences": len(train_records), "train_events": int(sum(len(r["times"]) for r in train_records)),
        "positive_train_intervals": int((positive > 0).sum()), "nonpositive_train_intervals": int((positive <= 0).sum()),
        "median_positive_train_interval": float(np.median(positive)) if len(positive) else None,
    }


def _prepare_synthetics(spec: Mapping[str, Any], out: Path) -> dict[str, Any]:
    synth = spec["additional_tracks"]["non_hawkes_synthetic"]
    n_train, n_val, n_test = map(int, synth["split_windows"])
    seed, rate, horizon = int(synth["generator_seed"]), float(synth["rate"]), float(synth["horizon"])
    datasets: dict[str, Any] = {}
    for family in synth["families"]:
        recipe = SyntheticRecipe(family=family, rate=rate, horizon=horizon)
        generated = generate_splits(recipe, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed)
        route_dir = out / "synthetic" / family
        split_paths, hashes, records_by_split = {}, {}, {}
        for split, records in generated.items():
            converted = [_synthetic_record(row, family=family, split=split, index=i) for i, row in enumerate(records)]
            path = route_dir / f"{split}.jsonl"
            hashes[split] = _write_jsonl(path, converted)
            split_paths[split], records_by_split[split] = str(path), converted
        dataset_id = f"synthetic:{family}:rate{rate:g}:horizon{horizon:g}:seed{seed}"
        datasets[family] = {
            "dataset_id": dataset_id, "splits": split_paths, "split_sha256": hashes,
            "recipe": asdict(recipe), "ground_truth": ground_truth(recipe),
            "independent_windows": True, "include_empty_windows": True,
            "coordinate_transform": _identity_coordinate_entry(train_records=records_by_split["train"]),
            "calibration": {"input": "identity", "target": "analytic process law where available", "fit": "none"},
        }
    return datasets


def _rectangle(route: Mapping[str, Any]) -> list[float] | None:
    domain = route.get("observation", {}).get("spatial_domain", {})
    if domain.get("kind") not in {"declared_preprocessing_rectangle", "declared_city_validity_envelope"}:
        return None
    lon, lat = domain.get("longitude"), domain.get("latitude")
    if not (isinstance(lon, list) and isinstance(lat, list) and len(lon) == len(lat) == 2):
        return None
    return [float(lon[0]), float(lon[1]), float(lat[0]), float(lat[1])]


def _count_jobs(out: Path, routes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for dataset_id in _COUNT_IDS:
        route = routes[dataset_id]
        rectangle = _rectangle(route)
        command = ["python", "-m", "seahorse.experiments.count_baselines", "--dataset", route["dataset_id"],
                   "--prepared-root", prepared_root(), "--output", str(out / "count_results" / f"{dataset_id}.json"),
                   "--bin-width", "1", "--period-bins", "7"]
        # One preregistered 8x8 grid avoids invalid selection/ranking of PMFs
        # defined on different numbers of spatial cells.
        if rectangle is not None:
            command += ["--rectangle", *map(str, rectangle), "--grids", "8x8"]
        jobs.append({
            "id": f"count__{dataset_id}", "dataset": dataset_id, "route": route,
            "command": command, "status": "prepared_not_launched", "fit_stage": "single_cli_train_val_test",
            "models": ["constant_calendar_poisson", "seasonal_weekly_calendar_poisson", "calendar_month_poisson_if_daily", "spatial_cell_poisson_fixed_8x8" if rectangle else "spatial_not_eligible"],
            "metric_lane": "discrete_count_only", "include_empty_bins": True,
            "selection": "no grid selection; a single preregistered 8x8 grid is scored only when a declared rectangle exists",
            "warning": "Do not compare count NLL across different spatial grid dimensionalities, and do not compare this lane to continuous-time NLL.",
        })
    return jobs


def _external_specs(out: Path, routes: Mapping[str, Mapping[str, Any]], seeds: list[int], lr_multipliers: list[float]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    route_root = Path(fixed_window_route_root())
    for dataset_id, route in routes.items():
        rectangle = _rectangle(route)
        converter_path=route_root/dataset_id/"converter_manifest.json"
        shared = {
            "source_prepared_route": route, "fixed_window_converter": {
                "input": route["dataset_id"], "output_directory": str(route_root / dataset_id),
                "required": "all source calendar windows, including empty windows; preserve window_start/end,t0,t1,event IDs and source support",
                "time_map": "u=(t-t0)/(t1-t0) in [0,1]; record per-window +log(t1-t0) time Jacobian externally",
                "spatial_map": "source locations unchanged in route; author adapters may use a documented train-only normalization",
                "status": "materialized" if converter_path.exists() else "requires_materialization",
            },
            "converter_manifest_sha256": _sha256(converter_path) if converter_path.exists() else None,
            "splits": {split: str(route_root / dataset_id / f"{split}.jsonl") for split in ("train", "val", "test")},
        }
        if dataset_id in _WSM_IDS:
            if rectangle is None:
                raise ValueError(f"WSM cohort {dataset_id} lacks a declared rectangle.")
            for estimator in ("wsm_survival", "mle"):
                for seed in seeds:
                    for trial, multiplier in enumerate(lr_multipliers):
                        spec = {
                            "kind": "wsm", "authorcheckout": author_wsm_checkout(),
                            "author_commit_required": AUTHOR_WSM_COMMIT, "estimator": estimator, "seed": seed,
                            "epochs": 100, "batch": 64, "lr": 1e-3 * float(multiplier),
                            "max_steps": EXTERNAL_MAX_STEPS, "max_time_seconds": EXTERNAL_MAX_SECONDS,
                            "resource_policy": {"device": "cpu_author_adapter", "outer_watchdog_seconds": EXTERNAL_MAX_SECONDS + 60, "no_gpu_parity_claim": True},
                            "contract": {"horizon": 1.0, "spatial_lower": [rectangle[0], rectangle[2]], "spatial_upper": [rectangle[1], rectangle[3]],
                                         "backbone": "SMASH_decouple", "cond_dim": 48, "with_survival": True,
                                         "grid_t": 5, "grid_s": 5, "eval_grid_t": 25, "eval_grid_s": 10, "num_units": 16, "n_layers": 1},
                            "splits": shared["splits"], **shared,
                        }
                        path = out / "external_specs" / f"wsm__{dataset_id}__{estimator}__seed{seed}__lr{trial}.json"
                        _write_json(path, spec)
                        jobs.append({"id": path.stem, "kind": "wsm", "dataset": dataset_id, "estimator": estimator, "seed": seed, "lr_trial": trial,
                                     "spec": str(path), "fit_command": ["python", "-m", "seahorse.baselines.run", "--spec", str(path), "--out", str(out / "external_results" / path.stem), "--stage", "fit"],
                                     "status": "prepared_not_launched", "selection_group": f"wsm__{dataset_id}__{estimator}__seed{seed}",
                                     "nll_lane": "author_fixed_window_survival; source-coordinate mean log likelihood per observation window"})
        # TriTPP gets the same all-window fixed-horizon route including ComCat empties.
        for seed in seeds:
            for trial, multiplier in enumerate(lr_multipliers):
                spec = {"kind": "tritpp", "seed": seed, "epochs": 100, "batch": 64, "lr": 1e-2 * float(multiplier),
                        "authorcheckout": author_tritpp_checkout(), "author_commit_required": AUTHOR_TRITPP_COMMIT,
                        "max_steps": EXTERNAL_MAX_STEPS, "max_time_seconds": EXTERNAL_MAX_SECONDS,
                        "resource_policy": {"device": "cpu_author_adapter", "outer_watchdog_seconds": EXTERNAL_MAX_SECONDS + 60, "no_gpu_parity_claim": True},
                        "contract": {"t_max": 1.0, "spatial_head_name": "seahorse_gmm", "n_knots": 20, "spline_order": 2, "n_blocks": 4, "block_size": 16},
                        "splits": shared["splits"], **shared}
                path = out / "external_specs" / f"tritpp__{dataset_id}__seed{seed}__lr{trial}.json"
                _write_json(path, spec)
                jobs.append({"id": path.stem, "kind": "tritpp", "dataset": dataset_id, "seed": seed, "lr_trial": trial,
                             "spec": str(path), "fit_command": ["python", "-m", "seahorse.baselines.run", "--spec", str(path), "--out", str(out / "external_results" / path.stem), "--stage", "fit"],
                             "status": "prepared_not_launched", "selection_group": f"tritpp__{dataset_id}__seed{seed}",
                             "nll_lane": "author_fixed_window; source-coordinate log probability per observation window"})
    for job in jobs:
        job["spec_sha256"]=_sha256(Path(job["spec"]))
    return jobs


def prepare_auxiliary(spec: str | Path | Mapping[str, Any], out: str | Path) -> dict[str, Any]:
    """Materialize synthetic routes and freeze all auxiliary launch recipes; never fit."""
    if isinstance(spec, (str, Path)):
        spec_path = Path(spec)
        payload = json.loads(spec_path.read_text())
        spec_sha256 = _sha256(spec_path)
    else:
        payload = dict(spec)
        spec_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    out = Path(out).resolve()
    _assert_output_root(out)
    if (out / "auxiliary_manifest.json").exists():
        raise FileExistsError("Auxiliary campaign already frozen; choose a new output directory.")
    additional = payload.get("additional_tracks", {})
    if not {"native_counts", "author_fixed_window", "non_hawkes_synthetic"}.issubset(additional):
        raise ValueError("Spec lacks required auxiliary-track declarations.")
    root = prepared_root()
    routes = {name: resolve_prepared_dataset(route_id, root=root).machine_readable() for name, route_id in _NATIVE_ROUTE.items()}
    strict_routes = {name: resolve_prepared_dataset(route_id, root=root).machine_readable() for name, route_id in _STRICT_EXTERNAL.items()}
    synth = _prepare_synthetics(payload, out)
    count_jobs = _count_jobs(out, routes)
    external_jobs = _external_specs(out, strict_routes, list(payload["seeds"]), list(payload["hpo"]["lr_multipliers"]))
    selection_evaluations = []
    for group in sorted({job["selection_group"] for job in external_jobs}):
        options = [job for job in external_jobs if job["selection_group"] == group]
        selection_evaluations.append({
            "id": f"evaluate_selected__{group}", "selection_group": group,
            "selection": "minimum finite validation objective among the three declared LR trials; trial0 wins exact ties; test is unopened before selection",
            "source_fit_specs": [job["spec"] for job in sorted(options, key=lambda x: x["lr_trial"])],
            "command_template": ["python", "-m", "seahorse.baselines.run", "--spec", "<selected_validation_spec>", "--checkpoint", "<selected_validation_checkpoint>", "--out", str(out / "external_results" / group / "evaluate"), "--stage", "evaluate"],
            "status": "prepared_not_launched; requires selected checkpoint path from validation only",
        })
    manifest = {
        "schema_version": AUX_SCHEMA_VERSION, "campaign_id": payload.get("campaign_id"), "spec_sha256": spec_sha256, "state": "prepared_not_launched",
        "prepared_root": root, "synthetic": synth, "native_count_jobs": count_jobs,
        "external_jobs": external_jobs, "external_selected_evaluations": selection_evaluations,
        "blocked": [{"dataset": "comcat_california", "model": "wsm/mle", "reason": "Prepared ComCat contains empty observation windows and the pinned WSM adapter hard-gates them; no windows are dropped."}],
        "separation": "Synthetic controls, discrete count likelihoods, and author fixed-window metrics are auxiliary lanes; none is pooled/ranked with main conditional next-event NLL.",
    }
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _write_json(out / "auxiliary_manifest.json", manifest)
    return manifest


__all__ = ["prepare_auxiliary"]
