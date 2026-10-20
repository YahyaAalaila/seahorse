"""Compile and verify an immutable experiment campaign. Never trains.

Usage: python -m seahorse.experiments.campaign prepare --spec ... --out ...
The compiled matrix has explicit dependencies, seed/trial budgets and blocked
extensions. Execution is a separate command in campaign_run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def source_manifest(root):
    root = Path(root).resolve()
    files = [root / "pyproject.toml", root / "LICENSE", root / "NOTICE"]
    for folder in ("seahorse", "tests", "scripts", "experiments/campaign_2026"):
        files.extend(
            p
            for p in (root / folder).rglob("*")
            if p.is_file()
            and (
                p.suffix in {".py", ".yaml", ".json", ".sh", ".txt"}
                or p.name in {"LICENSE", "NOTICE.md"}
            )
            and "__pycache__" not in p.parts
        )
    hashes = {str(p.relative_to(root)): digest_file(p) for p in sorted(set(files)) if p.exists()}
    return {"files": hashes, "sha256": canonical_hash(hashes)}


def validate_spec(spec):
    if spec.get("schema_version") != 1:
        raise ValueError("Only campaign schema_version=1 is supported.")
    if len(set(spec["seeds"])) != len(spec["seeds"]) or len(spec["seeds"]) < 3:
        raise ValueError("At least three distinct final seeds are required.")
    if spec["protocol"]["history_events"] < 1:
        raise ValueError("history_events must be positive.")
    if spec["protocol"]["loss_target_scope"] != "last":
        raise ValueError("The fixed-context campaign scores one common final target.")
    for label in ("datasets", "models"):
        names = [x["id"] for x in spec[label]]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate {label} IDs.")
    for model in spec["models"]:
        if model.get("runner", "nnstpp") not in {"nnstpp", "statstpp"}:
            raise ValueError("Campaign model runner must be nnstpp or statstpp.")
        if model.get("runner") == "statstpp" and not model.get("trials"):
            raise ValueError("Statistical models require explicit inference/model trials.")
    if len(spec["hpo"]["lr_multipliers"]) < 2 or spec["hpo"]["lr_multipliers"][0] != 1.0:
        raise ValueError("HPO trial zero must be the declared baseline learning rate.")
    if spec["compute"]["max_concurrent_gpu_jobs"] < 1:
        raise ValueError(
            "GPU concurrency must be positive; the execution backend allocates devices."
        )
    for section in ("hpo", "fit"):
        seconds = spec[section].get("max_seconds")
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
            raise ValueError(
                "max_seconds must be positive or null for scheduler-managed execution."
            )
    return spec


def make_jobs(spec):
    """Expand every planned comparison; no results enter roster construction."""
    jobs = []
    for dataset in spec["datasets"]:
        for model in spec["models"]:
            if model["track"] != "conditional_next_event":
                continue
            stage = "core" if dataset["stage"] == model["stage"] == "core" else "expansion"
            pair = f"{dataset['id']}__{model['id']}"
            hpo_ids = []
            statistical = model.get("runner", "nnstpp") == "statstpp"
            trials = (
                model["trials"]
                if statistical
                else [{"lr_multiplier": x} for x in spec["hpo"]["lr_multipliers"]]
            )
            for trial, settings in enumerate(trials):
                job_id = f"{pair}__hpo{trial}"
                hpo_ids.append(job_id)
                jobs.append(
                    {
                        "id": job_id,
                        "dataset": dataset["id"],
                        "model": model["id"],
                        "action": "hpo",
                        "stage": stage,
                        "trial": trial,
                        "runner": model.get("runner", "nnstpp"),
                        **settings,
                        "seed": spec["hpo"]["seed"],
                        "dependencies": [],
                        "max_seconds": model.get("hpo_max_seconds", spec["hpo"]["max_seconds"]),
                        "max_steps": spec["hpo"]["max_steps"],
                    }
                )
            for seed in spec["seeds"]:
                jobs.append(
                    {
                        "id": f"{pair}__seed{seed}",
                        "dataset": dataset["id"],
                        "model": model["id"],
                        "action": "refit",
                        "stage": stage,
                        "seed": seed,
                        "dependencies": hpo_ids,
                        "runner": model.get("runner", "nnstpp"),
                        "max_seconds": model.get("fit_max_seconds", spec["fit"]["max_seconds"]),
                        "max_steps": spec["fit"]["max_steps"],
                    }
                )
    # All pairs receive trial zero before any pair receives additional tuning.
    jobs.sort(
        key=lambda j: (
            j["stage"] != "core",
            j["action"] != "hpo",
            j.get("trial", j["seed"]),
            j["dataset"],
            j["model"],
        )
    )
    return jobs


def validate_coordinates(spec, routes, coordinates):
    """Reject missing, stale, or internally inconsistent transforms at freeze."""
    from seahorse.experiments.coordinates import CoordinateTransform

    for dataset in spec["datasets"]:
        route = routes[dataset["id"]]
        entry = coordinates.get("routes", {}).get(route["dataset_id"])
        if entry is None:
            raise ValueError(f"Missing coordinate transform for {route['dataset_id']}")
        if entry["train_split"]["sha256"] != route["split_sha256"]["train"]:
            raise ValueError(f"Stale training coordinate hash for {dataset['id']}")
        transform = CoordinateTransform.from_dict(entry["transform"])
        lo, hi, bottom, top = transform.spatial_bounds_lon_lat
        numbers = [
            lo,
            hi,
            bottom,
            top,
            transform.time_scale,
            *transform.space_offset,
            *transform.space_scale,
            transform.log_raw_volume_per_normalized_volume,
        ]
        if not all(math.isfinite(float(x)) for x in numbers):
            raise ValueError(f"Nonfinite coordinate transform for {dataset['id']}")
        if transform.time_scale <= 0 or hi <= lo or top <= bottom:
            raise ValueError(f"Nonpositive coordinate scale for {dataset['id']}")
        expected = [lo, bottom, hi - lo, top - bottom]
        actual = [*transform.space_offset, *transform.space_scale]
        if not all(
            math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(actual, expected)
        ):
            raise ValueError(f"Inconsistent declared spatial domain for {dataset['id']}")
        log_volume = math.log(transform.time_scale * (hi - lo) * (top - bottom))
        if not math.isclose(
            log_volume, transform.log_raw_volume_per_normalized_volume, abs_tol=1e-10
        ):
            raise ValueError(f"Incorrect coordinate Jacobian for {dataset['id']}")
        if transform.nonpositive_train_intervals:
            raise ValueError(
                f"Nonpositive train intervals require a declared observation policy: {dataset['id']}"
            )
        if transform.history != spec["protocol"]["history_events"]:
            raise ValueError(
                f"Coordinate history scaling differs from the frozen protocol: {dataset['id']}"
            )


def validate_data_audit(spec, routes, audit, repo):
    for filename, expected in audit["relevant_code_sha256"].items():
        if digest_file(Path(repo) / filename) != expected:
            raise ValueError(
                f"Data audit is stale after a relevant implementation edit: {filename}"
            )
    for dataset in spec["datasets"]:
        route = routes[dataset["id"]]
        if route.get("kind") == "synthetic":
            continue
        evidence = audit.get("routes", {}).get(route["dataset_id"], {}).get("splits", {})
        for split in ("train", "val", "test"):
            check = evidence.get(split, {})
            if check.get("split_sha256") != route["split_sha256"][split]:
                raise ValueError(f"Missing or stale full-data audit: {dataset['id']}/{split}")
            if (
                not check.get("strict_global_times_within_segments")
                or check.get("contexts", 0) <= 0
            ):
                raise ValueError(f"Continuous-context audit did not pass: {dataset['id']}/{split}")


def prepare(spec_path, out, *, repo=None, verify_data=True):
    from seahorse.data.prepared import resolve_prepared_dataset
    from seahorse.models.configs import ConfigRegistry

    repo = Path(repo or Path(__file__).resolve().parents[2]).resolve()
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    out = Path(out).resolve()
    if (out / "manifest.json").exists():
        raise FileExistsError(
            "Campaign already frozen. Use a new version directory; never overwrite its definitions."
        )
    routes = {}
    for dataset in spec["datasets"]:
        route = resolve_prepared_dataset(dataset["route"], verify_hashes=verify_data)
        route.assert_continuous_time_eligible(exact=False)
        routes[dataset["id"]] = route.machine_readable()
    for model in spec["models"]:
        if model["track"] == "conditional_next_event" and not ConfigRegistry.is_registered(
            model["preset"]
        ):
            raise ValueError(f"Unregistered model {model['preset']}")
        if ConfigRegistry.runner_kind(model["preset"]) != model.get("runner", "nnstpp"):
            raise ValueError(f"Declared runner differs from preset registry: {model['preset']}")
    coordinate_path = Path(spec["coordinate_artifact"])
    if not coordinate_path.is_file():
        raise FileNotFoundError(f"Train-only coordinate artifact missing: {coordinate_path}")
    coordinates = json.loads(coordinate_path.read_text())
    auxiliary = None
    if spec.get("auxiliary_manifest") or spec.get("synthetic_manifest"):
        auxiliary_path = Path(spec.get("auxiliary_manifest") or spec["synthetic_manifest"])
        auxiliary = json.loads(auxiliary_path.read_text())
        if auxiliary.get("sha256") != canonical_hash(
            {k: v for k, v in auxiliary.items() if k != "sha256"}
        ):
            raise ValueError("Auxiliary manifest integrity check failed.")
        for job in auxiliary["external_jobs"] if spec.get("auxiliary_manifest") else []:
            if digest_file(job["spec"]) != job["spec_sha256"]:
                raise ValueError("External specification changed after auxiliary freeze.")
            external = json.loads(Path(job["spec"]).read_text())
            if not external.get("converter_manifest_sha256"):
                raise ValueError("External input conversion has not been frozen.")
        for family, entry in auxiliary["synthetic"].items():
            dataset_id = f"synthetic_{family}"
            spec["datasets"].append(
                {
                    "id": dataset_id,
                    "route": entry["dataset_id"],
                    "stage": "core",
                    "kind": "synthetic",
                    "bootstrap_block_records": 1,
                }
            )
            routes[dataset_id] = {**entry, "kind": "synthetic"}
            if verify_data:
                for split, path in entry["splits"].items():
                    if digest_file(path) != entry["split_sha256"][split]:
                        raise ValueError(
                            f"Synthetic bytes differ from prepared recipe: {family}/{split}"
                        )
            coordinates["routes"][entry["dataset_id"]] = {
                "transform": entry["coordinate_transform"],
                "train_split": {
                    "path": entry["splits"]["train"],
                    "sha256": entry["split_sha256"]["train"],
                },
            }
        if not spec.get("auxiliary_manifest"):
            auxiliary = {
                "synthetic": auxiliary["synthetic"],
                "source_manifest": str(auxiliary_path),
                "source_manifest_sha256": digest_file(auxiliary_path),
                "external_jobs": [],
                "external_execution": "excluded; native integrations only",
            }
    validate_coordinates(spec, routes, coordinates)
    data_audit = (
        json.loads(Path(spec["data_audit"]).read_text()) if spec.get("data_audit") else None
    )
    if data_audit is not None:
        validate_data_audit(spec, routes, data_audit, repo)
    source = source_manifest(repo)
    jobs = make_jobs(spec)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": spec["campaign_id"],
        "spec": spec,
        "datasets": routes,
        "coordinates": coordinates,
        "coordinate_artifact_sha256": digest_file(coordinate_path),
        "auxiliary": auxiliary,
        "data_audit": data_audit,
        "source": source,
        "jobs": jobs,
        "state": "prepared_not_launched",
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    manifest["sha256"] = canonical_hash(manifest)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    write_json(out / "spec.json", spec)
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=(out / "pip-freeze.txt").open("w"),
        check=True,
    )
    lines = ["job_id\tstage\taction\tdataset\tmodel\tseed\tmax_seconds"]
    lines.extend(
        "\t".join(
            str(j[k]) for k in ("id", "stage", "action", "dataset", "model", "seed", "max_seconds")
        )
        for j in jobs
    )
    (out / "jobs.tsv").write_text("\n".join(lines) + "\n")
    core = [j for j in jobs if j["stage"] == "core"]

    def capped_hours(selected):
        limits = [j["max_seconds"] for j in selected]
        return sum(limits) / 3600 if all(x is not None for x in limits) else None

    summary = {
        "jobs": len(jobs),
        "core_jobs": len(core),
        "expansion_jobs": len(jobs) - len(core),
        "datasets": len(routes),
        "models": len(spec["models"]),
        "refit_evaluations": sum(j["action"] == "refit" for j in jobs),
        "maximum_fit_gpu_hours_core": capped_hours([j for j in core if j["runner"] == "nnstpp"]),
        "maximum_fit_gpu_hours_all": capped_hours([j for j in jobs if j["runner"] == "nnstpp"]),
        "maximum_fit_cpu_hours_all": capped_hours([j for j in jobs if j["runner"] == "statstpp"]),
        "fit_overhead_watchdog_seconds_per_job": spec["compute"].get("fit_overhead_seconds", 900),
        "interpretation": "Configured upper bounds, not measured runtime estimates. Evaluation adds cost. Watchdog logs censored runs.",
    }
    write_json(out / "summary.json", summary)
    return summary


def verify(manifest_path, repo=None):
    manifest = json.loads(Path(manifest_path).read_text())
    expected = manifest.pop("sha256")
    if canonical_hash(manifest) != expected:
        raise ValueError("Campaign manifest has changed since freezing.")
    manifest["sha256"] = expected
    repo = Path(repo or Path(__file__).resolve().parents[2])
    actual = source_manifest(repo)
    if actual["sha256"] != manifest["source"]["sha256"]:
        changed = sorted(
            k
            for k in actual["files"].keys() | manifest["source"]["files"].keys()
            if actual["files"].get(k) != manifest["source"]["files"].get(k)
        )
        raise ValueError(f"Source differs from frozen campaign: {changed[:20]}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--spec", required=True)
    prep.add_argument("--out", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--manifest", required=True)
    args = p.parse_args()
    result = (
        prepare(args.spec, args.out)
        if args.command == "prepare"
        else {"verified": verify(args.manifest)["sha256"]}
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
