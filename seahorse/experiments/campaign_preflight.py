"""Run bounded integration checks and produce source-bound launch evidence.

Every trainer check is exactly one optimizer step on two actual contexts.
These checks establish executability, never performance or convergence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
import scipy

from seahorse.experiments.campaign import canonical_hash, verify, write_json
from seahorse.experiments.campaign_run import fit_one


def model_gate(manifest, readiness, model_id):
    if readiness.get("source_sha256") != manifest["source"]["sha256"]:
        return "readiness_source_mismatch"
    if readiness.get("manifest_sha256") != manifest["sha256"]:
        return "readiness_manifest_mismatch"
    evidence = readiness.get("models", {}).get(model_id, {})
    model = next(m for m in manifest["spec"]["models"] if m["id"] == model_id)
    if model.get("runner", "nnstpp") == "statstpp":
        # Statistical ports have no Lightning optimizer contract.  Their gate
        # is an actual public runner fit, persisted native state, reload, and
        # equality of validation-context scores after reload.
        required = [
            "native_fit_and_reload",
            "full_native_state_and_score_reload",
            *model.get("required_checks", []),
        ]
    else:
        required = ["optimizer_and_reload", "real_trainer", *model.get("required_checks", [])]
    missing = [check for check in required if evidence.get("checks", {}).get(check) != "pass"]
    return "missing_checks:" + ",".join(missing) if missing else None


def check_runtime(readiness, *, runner="all"):
    if runner not in {"all", "nnstpp", "statstpp"}:
        raise ValueError("runner must be all, nnstpp, or statstpp.")
    expected = readiness["runtime"]
    if expected.get("python_executable") != sys.executable:
        raise ValueError("Use the frozen campaign Python environment.")
    if runner in {"all", "nnstpp"}:
        if expected.get("device") != "cuda" or expected.get("torch") != torch.__version__:
            raise ValueError("GPU campaign evidence requires the same CUDA/PyTorch environment.")
        if not torch.cuda.is_available():
            raise ValueError("The neural campaign requires a working CUDA device.")
    if runner in {"all", "statstpp"}:
        if expected.get("numpy") != np.__version__ or expected.get("scipy") != scipy.__version__:
            raise ValueError(
                "Statistical campaign evidence requires the same NumPy/SciPy environment."
            )


def _write_evidence(path, evidence):
    """Persist a self-verifying checkpoint; callers hold the preflight lock."""
    evidence["sha256"] = canonical_hash({k: v for k, v in evidence.items() if k != "sha256"})
    write_json(path, evidence)


def _upgrade_runtime(runtime, *, device):
    """Keep compatible CPU evidence, upgrading it only after a CUDA check."""
    if (
        runtime.get("python_executable") != sys.executable
        or runtime.get("numpy") != np.__version__
        or runtime.get("scipy") != scipy.__version__
    ):
        raise ValueError(
            "Existing launch readiness was created in a different Python/NumPy/SciPy environment."
        )
    if device == "cuda":
        runtime.update(device="cuda", torch=torch.__version__, gpu=torch.cuda.get_device_name(0))


def _merge_dataset_smokes(previous, statuses, manifest, evidence):
    merged = dict(previous.get("runner_smokes", {}))
    merged.update(statuses)
    known_families = {
        m.get("runner", "nnstpp")
        for m in manifest["spec"]["models"]
        if m["id"] in evidence["models"]
    }
    return {
        "status": "pass"
        if known_families
        and all(merged.get(kind, {}).get("status") == "pass" for kind in known_families)
        else "fail",
        "runner_smokes": merged,
    }


def _full_dataset_smoke(previous, checks, manifest, evidence):
    """Merge source-bound model×dataset smoke evidence by runner family."""
    merged = dict(previous.get("runner_smokes", {}))
    for kind, model_checks in checks.items():
        prior = dict(merged.get(kind, {}).get("model_checks", {}))
        prior.update(model_checks)
        expected = {
            m["id"] for m in manifest["spec"]["models"] if m.get("runner", "nnstpp") == kind
        }
        merged[kind] = {
            "status": "pass"
            if expected and all(prior.get(name, {}).get("status") == "pass" for name in expected)
            else "fail",
            "model_checks": prior,
        }
    known = {
        m.get("runner", "nnstpp")
        for m in manifest["spec"]["models"]
        if m["id"] in evidence["models"]
    }
    return {
        "status": "pass"
        if known and all(merged.get(k, {}).get("status") == "pass" for k in known)
        else "fail",
        "runner_smokes": merged,
    }


def _run_preflight_locked(
    manifest_path, *, device="cuda", models=None, dataset_smokes=True, runner="all"
):
    manifest = verify(manifest_path)
    root = Path(manifest_path).resolve().parent
    from seahorse.experiments.model_preflight import preflight_one

    torch.set_num_threads(manifest["spec"]["compute"]["cpu_threads_per_job"])
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA preflight requested but unavailable.")
        torch.ones(4, device="cuda").sum().item()
    evidence_path = root / "launch_readiness.json"
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text())
        if (
            evidence["source_sha256"] != manifest["source"]["sha256"]
            or evidence["manifest_sha256"] != manifest["sha256"]
        ):
            raise ValueError("Existing evidence belongs to a different frozen campaign.")
        if evidence.get("sha256") != canonical_hash(
            {k: v for k, v in evidence.items() if k != "sha256"}
        ):
            raise ValueError("Existing launch readiness checkpoint has an invalid integrity hash.")
        runtime = evidence["runtime"]
        # A later neural check upgrades CPU statistical evidence to CUDA.  A
        # subsequent statstpp-only pass must never erase that GPU evidence.
        _upgrade_runtime(runtime, device=device)
    else:
        evidence = {
            "manifest_sha256": manifest["sha256"],
            "source_sha256": manifest["source"]["sha256"],
            "purpose": "one-step integration checks only; no campaign fit or test selection",
            "runtime": {
                "python_executable": sys.executable,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "device": device,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            },
            "models": {},
            "datasets": {},
            "ready_models": [],
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    evidence["ready"] = False
    _write_evidence(evidence_path, evidence)
    if runner not in {"all", "nnstpp", "statstpp"}:
        raise ValueError("runner must be all, nnstpp, or statstpp.")
    for entry in manifest["spec"]["models"]:
        entry_runner = entry.get("runner", "nnstpp")
        if runner != "all" and entry_runner != runner:
            continue
        if models and entry["id"] not in models and entry["preset"] not in models:
            continue
        job = next(
            j
            for j in manifest["jobs"]
            if j["model"] == entry["id"] and j["action"] == "hpo" and j["trial"] == 0
        )
        if entry_runner == "statstpp":
            trainer = fit_one(manifest, job, root, smoke=True, device="cpu")
            passed = trainer["state"] == "completed"
            checks = {
                "native_fit_and_reload": "pass" if passed else "fail",
                "full_native_state_and_score_reload": "pass"
                if passed
                and trainer.get("checkpoint_semantic_parity")
                == "shared_validation_context_scores_and_full_native_state"
                else "fail",
                "validation_only_evaluation": "pass" if passed else "fail",
            }
            evidence["models"][entry["id"]] = {
                "checks": checks,
                "trainer_check": trainer,
                "runtime_device": "cpu",
                "limitation": "Bounded validation-context scoring and checkpoint parity only; no test split is read or selected.",
            }
        else:
            unit = preflight_one(entry, torch.device(device))
            trainer = fit_one(manifest, job, root, smoke=True, device=device)
            checks = {
                "optimizer_and_reload": unit["status"],
                "real_trainer": "pass" if trainer["state"] == "completed" else "fail",
                "validation_only_evaluation": "pass" if trainer["state"] == "completed" else "fail",
            }
            if entry["preset"] == "dnsk":
                checks["positive_intensity"] = (
                    "pass" if unit.get("dnsk", {}).get("positive_query_grid") else "fail"
                )
                checks["quadrature_refinement"] = (
                    "pass" if unit.get("dnsk", {}).get("converged") else "fail"
                )
            evidence["models"][entry["id"]] = {
                "checks": checks,
                "roster_check": unit,
                "trainer_check": trainer,
                "runtime_device": device,
                "limitation": "Bounded validation-context scoring and checkpoint parity only; no test split is read or selected.",
            }
        evidence["ready_models"] = [
            m["id"]
            for m in manifest["spec"]["models"]
            if model_gate(manifest, evidence, m["id"]) is None
        ]
        verify(manifest_path)  # No evidence is valid after a concurrent source edit.
        _write_evidence(evidence_path, evidence)
        print(json.dumps({"model": entry["id"], "checks": checks}), flush=True)
    if dataset_smokes:
        full_pairs = bool(
            manifest["spec"].get("compute", {}).get("preflight_all_model_dataset_pairs", False)
        )
        available_models = [
            m
            for m in manifest["spec"]["models"]
            if runner == "all" or m.get("runner", "nnstpp") == runner
        ]
        if models:
            available_models = [
                m for m in available_models if m["id"] in models or m["preset"] in models
            ]
        pair_passed = pair_total = 0
        if full_pairs:
            for dataset in manifest["spec"]["datasets"]:
                checks = {}
                for entry in available_models:
                    kind = entry.get("runner", "nnstpp")
                    job = next(
                        j
                        for j in manifest["jobs"]
                        if j["dataset"] == dataset["id"]
                        and j["model"] == entry["id"]
                        and j["action"] == "hpo"
                        and j["trial"] == 0
                    )
                    path = root / "preflight" / job["id"] / "status.json"
                    status = json.loads(path.read_text()) if path.exists() else None
                    if status is None or status.get("manifest_sha256") != manifest["sha256"]:
                        status = fit_one(
                            manifest,
                            job,
                            root,
                            smoke=True,
                            device="cpu" if kind == "statstpp" else device,
                        )
                    passed = status.get("state") == "completed"
                    checks.setdefault(kind, {})[entry["id"]] = {
                        "status": "pass" if passed else "fail",
                        "trainer_check": str(path),
                    }
                    pair_total += 1
                    pair_passed += int(passed)
                evidence["datasets"][dataset["id"]] = _full_dataset_smoke(
                    evidence["datasets"].get(dataset["id"], {}), checks, manifest, evidence
                )
                verify(manifest_path)
                _write_evidence(evidence_path, evidence)
        else:
            probes = {}
            preferred = {"nnstpp": "poisson_gmm", "statstpp": "homogeneous_poisson"}
            for runner_kind in ("nnstpp", "statstpp"):
                candidate = next(
                    (
                        m
                        for m in available_models
                        if m["id"] == preferred[runner_kind]
                        or m["preset"] == preferred[runner_kind]
                    ),
                    None,
                )
                if candidate is not None:
                    probes[runner_kind] = candidate["id"]
            for dataset in manifest["spec"]["datasets"]:
                statuses = {}
                for runner_kind, model_id in probes.items():
                    job = next(
                        j
                        for j in manifest["jobs"]
                        if j["dataset"] == dataset["id"]
                        and j["model"] == model_id
                        and j["action"] == "hpo"
                        and j["trial"] == 0
                    )
                    path = root / "preflight" / job["id"] / "status.json"
                    status = (
                        json.loads(path.read_text())
                        if path.exists()
                        else fit_one(
                            manifest,
                            job,
                            root,
                            smoke=True,
                            device="cpu" if runner_kind == "statstpp" else device,
                        )
                    )
                    statuses[runner_kind] = {
                        "status": "pass" if status["state"] == "completed" else "fail",
                        "trainer_check": str(path),
                    }
                evidence["datasets"][dataset["id"]] = _merge_dataset_smokes(
                    evidence["datasets"].get(dataset["id"], {}), statuses, manifest, evidence
                )
                verify(manifest_path)
                _write_evidence(evidence_path, evidence)
    evidence["ready"] = (
        device == "cuda"
        and len(evidence["ready_models"]) == len(manifest["spec"]["models"])
        and all(
            evidence["datasets"].get(d["id"], {}).get("status") == "pass"
            for d in manifest["spec"]["datasets"]
        )
    )
    _write_evidence(evidence_path, evidence)
    return {
        "ready": evidence["ready"],
        "models_passed": len(evidence["ready_models"]),
        "models_total": len(manifest["spec"]["models"]),
        "pairs_passed": pair_passed if dataset_smokes and full_pairs else None,
        "pairs_total": pair_total if dataset_smokes and full_pairs else None,
        "path": str(evidence_path),
    }


def run_preflight(manifest_path, *, device="cuda", models=None, dataset_smokes=True, runner="all"):
    """Serialize mutable readiness checkpoints against the campaign queue."""
    import fcntl

    root = Path(manifest_path).resolve().parent
    with (root / "launch_readiness.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_preflight_locked(
            manifest_path,
            device=device,
            models=models,
            dataset_smokes=dataset_smokes,
            runner=runner,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--models", nargs="*")
    p.add_argument("--runner", choices=("all", "nnstpp", "statstpp"), default="all")
    p.add_argument("--skip-dataset-smokes", action="store_true")
    args = p.parse_args()
    print(
        json.dumps(
            run_preflight(
                args.manifest,
                device=args.device,
                models=args.models,
                runner=args.runner,
                dataset_smokes=not args.skip_dataset_smokes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
