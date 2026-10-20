"""Actual shared-campaign fits, cross-day target parity and forecast artifacts."""
import json
from pathlib import Path

import numpy as np

from seahorse.experiments.campaign import digest_file, make_jobs
from seahorse.experiments.campaign_run import fit_one, evaluate_one, queue
from seahorse.experiments.coordinates import fit_coordinates


def _manifest(tmp_path):
    spec = json.loads((Path(__file__).resolve().parents[1] / "experiments/campaign_2026/spec_v1.json").read_text())
    spec["datasets"] = [{"id": "fixture", "stage": "core"}]
    spec["models"] = [next(m for m in spec["models"] if m["id"] == "poisson_gmm"),
        {"id": "homogeneous_poisson", "preset": "homogeneous_poisson", "runner": "statstpp", "stage": "core",
         "track": "conditional_next_event", "nll_path": "closed_form", "trials": [{"config_overrides": {}}]}]
    spec["protocol"].update(predictive_targets=4, predictive_samples=4,
        count_forecast={"horizon_time_scales": .2, "candidate_origins": 10, "origins": 2, "samples": 3,
                        "seed": 17, "time_bins": 2, "spatial_grid": [2, 2], "event_cap": 100})
    rng = np.random.default_rng(11)
    records = [{"times": np.linspace(.02, .96, 25).tolist(), "locations": rng.uniform(.1, .9, (25, 2)).tolist(),
                "window_start": "2020-01-01T00:00:00", "window_end": "2020-01-02T00:00:00", "t0": 0., "t1": 1.},
               {"times": [.1, .3, .5, .7, .9], "locations": rng.uniform(.1, .9, (5, 2)).tolist(),
                "window_start": "2020-01-02T00:00:00", "window_end": "2020-01-03T00:00:00", "t0": 0., "t1": 1.}]
    paths, hashes = {}, {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row)+"\n" for row in records))
        paths[split], hashes[split] = str(path), digest_file(path)
    coord = fit_coordinates(records, (0., 1., 0., 1.))
    return {"sha256": "fixture-only", "spec": spec, "jobs": make_jobs(spec),
        "datasets": {"fixture": {"dataset_id": "fixture", "splits": paths, "split_sha256": hashes}},
        "coordinates": {"routes": {"fixture": {"transform": coord.to_dict(), "train_split": {"sha256": hashes["train"]}}}}}


def test_actual_neural_statistical_campaign_same_cross_day_targets_and_count_table(tmp_path):
    manifest = _manifest(tmp_path)
    payloads = []
    for name in ("poisson_gmm", "homogeneous_poisson"):
        job = next(j for j in manifest["jobs"] if j["model"] == name and j.get("trial") == 0)
        result = fit_one(manifest, job, tmp_path / "campaign", smoke=True, device="cpu")
        assert result["state"] == "completed", result
        out = tmp_path / "campaign" / "jobs" / job["id"]
        out.mkdir(parents=True)
        (out / "status.json").write_text(json.dumps(result))
        evaluation = evaluate_one(manifest, {**job, "action": "refit"}, tmp_path / "campaign", device="cpu")
        assert evaluation["nll_status"] == "complete", evaluation
        assert evaluation["target_population"] == 10
        assert evaluation["predictive"]["summary"] is not None, evaluation
        assert evaluation["count_forecast"]["summary"] is not None, evaluation
        with np.load(out / "target_nll.npz") as arrays:
            payloads.append((arrays["sequence_index"].copy(), arrays["target_event_index"].copy()))
    for a, b in zip(*payloads, strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.any(payloads[0][0] == 1), "Targets in the second daily record must carry the first day's history."


def test_runner_filtered_dry_run_keeps_full_model_specific_trial_matrix(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    import seahorse.experiments.campaign_run as module
    monkeypatch.setattr(module, "verify", lambda _: manifest)
    neural = queue(tmp_path / "absent.json", stage="all", runner="nnstpp")
    statistical = queue(tmp_path / "absent.json", stage="all", runner="statstpp")
    assert neural["fit_actions"] == 6
    assert statistical["fit_actions"] == 4  # analytic MLE has no meaningful LR sweep
    assert neural["evaluation_actions"] == statistical["evaluation_actions"] == 3
    assert not (tmp_path / "jobs").exists()


def test_model_filtered_queue_counts_only_selected_model_cells(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    import seahorse.experiments.campaign_run as module
    monkeypatch.setattr(module, "verify", lambda _: manifest)
    result = queue(tmp_path / "absent.json", stage="all", runner="statstpp", models=["homogeneous_poisson"])
    selected = [job for job in manifest["jobs"] if job["model"] == "homogeneous_poisson"]
    assert result["models"] == ["homogeneous_poisson"]
    assert result["fit_actions"] == len(selected)
    assert result["evaluation_actions"] == sum(job["action"] == "refit" for job in selected)
    assert {entry["id"] for entry in result["next_jobs"]} <= {job["id"] for job in selected}
    with np.testing.assert_raises_regex(ValueError, "Unknown campaign model"):
        queue(tmp_path / "absent.json", models=["not-in-manifest"])


def test_count_origins_span_test_and_eligibility_uses_only_observation_coverage():
    from seahorse.experiments.campaign_forecast import _select_count_contexts

    class Coverage:
        def raw_count_context(self, index, horizon):
            if index % 3 == 0:
                raise ValueError("Declared horizon crosses an observation gap.")
            return {"origin": index, "horizon": horizon}

    candidates = list(range(100))
    selected = _select_count_contexts(Coverage(), candidates, 8, 1., None, 17)
    ids = [index for index, _ in selected]
    expected = sorted(i for i in np.random.default_rng(17).permutation(candidates).tolist() if i % 3)
    # Selection uses the random order before restoring chronological output.
    permutation = np.random.default_rng(17).permutation(candidates).tolist()
    assert ids == sorted([i for i in permutation if i % 3][:8])
    assert max(ids) > 50 and ids != expected[:8]
    assert selected == _select_count_contexts(Coverage(), candidates, 8, 1., None, 17)


def test_count_horizon_uses_training_intervals_even_for_identity_synthetic_coordinates():
    from types import SimpleNamespace
    from seahorse.experiments.campaign_forecast import _count_horizon
    coord = SimpleNamespace(time_scale=1., median_positive_train_interval=.008)
    assert _count_horizon(coord, {"horizon_median_intervals": 20}) == .16
    coord.time_scale = 999.  # normalization must not change a physical task
    assert _count_horizon(coord, {"horizon_median_intervals": 20}) == .16


def test_seasonal_smoke_retains_empty_calendar_exposure(tmp_path):
    from datetime import datetime, timedelta
    from seahorse.experiments.campaign_statistical import _seasonal_smoke_records
    from seahorse.models.statistical.poisson import PoissonSTPP

    records = []
    for index in range(14):
        day = datetime(2020, 1, 1) + timedelta(days=index)
        count = 0 if index == 2 else 25
        records.append({"times": np.linspace(.02, .95, count).tolist(),
                        "locations": [[.5, .5]] * count, "t0": 0., "t1": 1.,
                        "time_unit": "day", "window_start": day.isoformat()})
    path = tmp_path / "seasonal.jsonl"
    path.write_text("".join(json.dumps(record)+"\n" for record in records))
    model = PoissonSTPP((0., 1., 0., 1.), period=7., n_phases=7)
    selected = _seasonal_smoke_records(path, model, 20)
    assert len(selected) == 10  # the next observed instance of the empty weekday
    assert len(selected[2]["times"]) == 0
    diagnostics = model.fit(selected, seed=0)
    assert sum(diagnostics["exposure_by_phase"]) == 10.
    assert all(rate > 0 for rate in diagnostics["rates"])


def test_synthetic_seasonal_campaign_uses_declared_period_one(tmp_path):
    from seahorse.runner import STPPRunner
    manifest = _manifest(tmp_path)
    seasonal = {"id": "seasonal_poisson", "preset": "seasonal_poisson", "runner": "statstpp", "stage": "core",
                "track": "conditional_next_event", "nll_path": "closed_form", "trials": [{"config_overrides": {}}],
                "dataset_config_overrides": {"fixture": {"model": {"period": 1., "n_phases": 7}}}}
    manifest["spec"]["models"] = [seasonal]
    manifest["jobs"] = make_jobs(manifest["spec"])
    route = manifest["datasets"]["fixture"]
    route.update(independent_windows=True, kind="synthetic")
    for split, filename in route["splits"].items():
        path = Path(filename)
        row = json.loads(path.read_text().splitlines()[0])
        row.update(window_start="synthetic:0", window_end="synthetic:0", time_unit="synthetic_horizon")
        path.write_text(json.dumps(row)+"\n")
        route["split_sha256"][split] = digest_file(path)
    manifest["coordinates"]["routes"]["fixture"]["train_split"]["sha256"] = route["split_sha256"]["train"]
    job = manifest["jobs"][0]
    result = fit_one(manifest, job, tmp_path / "campaign", smoke=True, device="cpu")
    assert result["state"] == "completed", result
    restored = STPPRunner.load(result["checkpoint"])
    assert restored.model.period == 1. and restored.model.n_phases == 7
