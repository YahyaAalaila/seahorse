"""Campaign target fairness, immutable definitions, and a real fit/save/load."""

import json
from pathlib import Path

import numpy as np
import pytest

from seahorse.experiments.campaign import (
    make_jobs,
    validate_spec,
    digest_file,
    validate_coordinates,
)
from seahorse.experiments.campaign_run import (
    fit_one,
    selected_lr,
    select_indices,
    IndependentContextDataset,
)
from seahorse.experiments.campaign_preflight import model_gate
from seahorse.experiments.coordinates import fit_coordinates


SPEC = Path(__file__).resolve().parents[1] / "experiments/campaign_2026/spec_v1.json"


def test_every_main_pair_receives_equal_trials_seeds_and_budget():
    spec = validate_spec(json.loads(SPEC.read_text()))
    jobs = make_jobs(spec)
    assert len(jobs) == len(spec["datasets"]) * len(spec["models"]) * 6
    for dataset in spec["datasets"]:
        for model in spec["models"]:
            pair = [j for j in jobs if j["dataset"] == dataset["id"] and j["model"] == model["id"]]
            assert [j["lr_multiplier"] for j in pair if j["action"] == "hpo"] == [1.0, 0.3, 3.0]
            assert sorted(j["seed"] for j in pair if j["action"] == "refit") == [11, 22, 33]
            assert len({j["max_seconds"] for j in pair if j["action"] == "refit"}) == 1


def test_validation_target_selection_independent_of_model_seed():
    ids = select_indices(20000, 8192, 20260910)
    assert len(ids) == len(set(ids)) == 8192
    assert ids == select_indices(20000, 8192, 20260910)
    assert select_indices(4, 8192, 1) == [0, 1, 2, 3]


def test_hpo_selection_requires_all_trials_and_ignores_test_values(tmp_path):
    spec = json.loads(SPEC.read_text())
    manifest = {"jobs": make_jobs(spec)}
    job = next(j for j in manifest["jobs"] if j["action"] == "refit")
    with pytest.raises(ValueError, match="not been attempted"):
        selected_lr(manifest, job, tmp_path)
    for i, dep in enumerate(job["dependencies"]):
        p = tmp_path / "jobs" / dep / "status.json"
        p.parent.mkdir(parents=True)
        p.write_text(
            json.dumps(
                {
                    "state": "completed",
                    "validation_objective": [3.0, 1.0, 2.0][i],
                    "test_nll": [-100.0, 99.0, 0.0][i],
                }
            )
        )
    multiplier, selected = selected_lr(manifest, job, tmp_path)
    assert multiplier == 0.3 and selected == job["dependencies"][1]


def test_real_lightning_fit_validation_checkpoint_roundtrip(tmp_path):
    spec = json.loads(SPEC.read_text())
    spec["models"] = [m for m in spec["models"] if m["id"] == "poisson_gmm"]
    job = make_jobs(spec)[0]
    rng = np.random.default_rng(0)
    train = {
        "times": np.cumsum(rng.uniform(0.01, 0.05, 25)).tolist(),
        "locations": rng.uniform(0.1, 0.9, (25, 2)).tolist(),
        "window_start": "2020-01-01T00:00:00",
        "window_end": "2020-01-02T00:00:00",
        "t0": 0.0,
        "t1": 1.0,
    }
    splits, hashes = {}, {}
    for split in ("train", "val"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(train) + "\n")
        splits[split], hashes[split] = str(path), digest_file(path)
    route_id = "prepared:comcat_california:native"
    coordinate = fit_coordinates([train], (0.0, 1.0, 0.0, 1.0))
    manifest = {
        "sha256": "smoke-fixture-only",
        "spec": spec,
        "datasets": {
            job["dataset"]: {"dataset_id": route_id, "splits": splits, "split_sha256": hashes}
        },
        "coordinates": {
            "routes": {
                route_id: {
                    "transform": coordinate.to_dict(),
                    "train_split": {"sha256": hashes["train"]},
                }
            }
        },
    }
    status = fit_one(manifest, job, tmp_path / "campaign", smoke=True, device="cpu")
    assert status["state"] == "completed", status
    assert status["smoke_only"]
    assert status["training_audit"]["optimizer_steps"] == 1
    assert status["training_audit"]["target_presentations"] == 2
    assert Path(status["checkpoint"]).exists()
    # Reuse fixture validation bytes as a held-out API fixture; this is not a
    # campaign test set or an HPO result. Exercise the actual evaluator too.
    from seahorse.experiments.campaign_run import evaluate_one

    final_job = {**job, "action": "refit"}
    output = tmp_path / "campaign" / "jobs" / job["id"]
    output.mkdir(parents=True)
    (output / "status.json").write_text(json.dumps(status))
    splits["test"], hashes["test"] = splits["val"], hashes["val"]
    evaluation = evaluate_one(manifest, final_job, tmp_path / "campaign", device="cpu")
    assert evaluation["nll_status"] == "complete", evaluation
    assert evaluation["finite_targets"] == 5
    assert evaluation["predictive"]["successful_targets"] == 5


def test_declared_ready_model_cannot_bypass_required_evidence():
    spec = json.loads(SPEC.read_text())
    manifest = {"sha256": "manifest", "source": {"sha256": "source"}, "spec": spec}
    ready = {
        "manifest_sha256": "manifest",
        "source_sha256": "source",
        "ready_models": ["dnsk"],
        "models": {"dnsk": {"checks": {"optimizer_and_reload": "pass", "real_trainer": "pass"}}},
    }
    assert "positive_intensity" in model_gate(manifest, ready, "dnsk")
    ready["models"]["dnsk"]["checks"].update(
        positive_intensity="pass", quadrature_refinement="pass"
    )
    assert model_gate(manifest, ready, "dnsk") is None
    ready["source_sha256"] = "old-source"
    assert model_gate(manifest, ready, "dnsk") == "readiness_source_mismatch"


def test_coordinate_freeze_rejects_stale_train_and_wrong_jacobian():
    records = [{"times": [0.0, 0.1, 0.3], "locations": [[0.1, 0.2], [0.3, 0.2], [0.2, 0.4]]}]
    transform = fit_coordinates(records, (0.0, 1.0, 0.0, 1.0)).to_dict()
    spec = {"datasets": [{"id": "fixture"}], "protocol": {"history_events": 20}}
    routes = {"fixture": {"dataset_id": "fixture", "split_sha256": {"train": "train"}}}
    coords = {"routes": {"fixture": {"train_split": {"sha256": "old"}, "transform": transform}}}
    with pytest.raises(ValueError, match="Stale training"):
        validate_coordinates(spec, routes, coords)
    coords["routes"]["fixture"]["train_split"]["sha256"] = "train"
    validate_coordinates(spec, routes, coords)
    transform["log_raw_volume_per_normalized_volume"] += 1.0
    with pytest.raises(ValueError, match="Jacobian"):
        validate_coordinates(spec, routes, coords)


def test_synthetic_realizations_never_join_contexts():
    record = {"times": np.linspace(0.0, 1.0, 21), "locations": np.full((21, 2), 0.5)}
    dataset = IndependentContextDataset([record, record], lookback=20)
    assert len(dataset) == 2
    assert [dataset[i]["observation_metadata"]["target_global_event_index"] for i in range(2)] == [
        20,
        41,
    ]


def test_queue_dry_run_has_eval_dependencies_and_no_side_effects(tmp_path, monkeypatch):
    import seahorse.experiments.campaign_run as module

    spec = json.loads(SPEC.read_text())
    manifest = {"spec": spec, "jobs": make_jobs(spec)}
    monkeypatch.setattr(module, "verify", lambda _: manifest)
    result = module.queue(tmp_path / "manifest.json", stage="all")
    assert result["fit_actions"] == len(manifest["jobs"])
    assert result["evaluation_actions"] == sum(j["action"] == "refit" for j in manifest["jobs"])
    assert not list(tmp_path.iterdir())


def test_training_gpu_peak_survives_lightning_cpu_teardown(monkeypatch):
    from types import SimpleNamespace
    import time
    import torch
    from seahorse.experiments.campaign_run import TrainingAudit

    audit = TrainingAudit()
    audit._batch_start = time.monotonic()
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 123456)
    audit.on_train_batch_end(
        None,
        SimpleNamespace(device=torch.device("cuda:0")),
        None,
        {"lengths": torch.tensor([21, 21])},
        0,
    )
    runner = SimpleNamespace(
        _lightning_module=SimpleNamespace(device=torch.device("cpu"), global_step=1)
    )
    assert audit.summary(runner)["peak_allocated_gpu_bytes"] == 123456
