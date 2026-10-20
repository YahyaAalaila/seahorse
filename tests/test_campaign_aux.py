from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from seahorse.experiments import campaign_aux


def _spec() -> dict:
    return json.loads(
        (Path(__file__).parents[1] / "experiments/campaign_2026/spec_v1.json").read_text()
    )


def _fake_route(route_id: str):
    name = route_id.split(":")[1]
    domains = {
        "fpa_california": {"kind": "declared_nonrectangular_filter"},
        "comcat_california": {
            "kind": "declared_preprocessing_rectangle",
            "longitude": [-130.0, -110.0],
            "latitude": [30.79, 44.45],
        },
    }
    domain = domains.get(
        name,
        {
            "kind": "declared_city_validity_envelope",
            "longitude": [-10.0, 10.0],
            "latitude": [-5.0, 5.0],
        },
    )
    return SimpleNamespace(
        machine_readable=lambda: {
            "dataset_id": route_id,
            "manifest_sha256": f"manifest-{name}",
            "split_sha256": {
                "train": f"train-{name}",
                "val": f"val-{name}",
                "test": f"test-{name}",
            },
            "splits": {
                split: f"/immutable/{name}/{split}.jsonl" for split in ("train", "val", "test")
            },
            "observation": {"spatial_domain": domain},
        }
    )


def _environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SEAHORSE_AUX_ALLOW_LOCAL_FIXTURE", "1")
    monkeypatch.setenv("SEAHORSE_PREPARED_DATA_ROOT", str(tmp_path / "prepared"))
    monkeypatch.setenv("SEAHORSE_WSM_CHECKOUT", str(tmp_path / "wsm_checkout"))
    monkeypatch.setenv("SEAHORSE_TRITPP_CHECKOUT", str(tmp_path / "tritpp_checkout"))
    monkeypatch.setenv("SEAHORSE_EXTERNAL_ROUTE_ROOT", str(tmp_path / "external_routes"))


def test_prepare_auxiliary_materializes_synthetic_and_freezes_separate_lanes(tmp_path, monkeypatch):
    _environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        campaign_aux, "resolve_prepared_dataset", lambda route_id, root: _fake_route(route_id)
    )
    manifest = campaign_aux.prepare_auxiliary(_spec(), tmp_path / "aux")
    assert manifest["state"] == "prepared_not_launched"
    assert len(manifest["native_count_jobs"]) == 11
    assert len(manifest["external_jobs"]) == 171  # 6*2*3*3 WSM/MLE + 7*3*3 TriTPP
    assert (tmp_path / "aux" / "auxiliary_manifest.json").is_file()

    synthetic = manifest["synthetic"]["gamma_renewal"]
    train = Path(synthetic["splits"]["train"])
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    assert len(rows) == 512
    assert {
        "times",
        "locations",
        "event_ids",
        "sequence_id",
        "stream_id",
        "t0",
        "t1",
        "window_start",
        "window_end",
    } <= rows[0].keys()
    assert synthetic["calibration"]["input"] == "identity"
    assert synthetic["split_sha256"]["train"]

    counts = {job["dataset"]: job for job in manifest["native_count_jobs"]}
    assert "--grids" in counts["austin311"]["command"]
    assert "8x8" in counts["austin311"]["command"]
    assert "16x16" not in counts["austin311"]["command"]
    assert "--rectangle" not in counts["fpa_california"]["command"]
    assert counts["fpa_california"]["metric_lane"] == "discrete_count_only"

    wsm = [j for j in manifest["external_jobs"] if j["kind"] == "wsm"]
    tri = [j for j in manifest["external_jobs"] if j["kind"] == "tritpp"]
    assert {j["dataset"] for j in wsm} == set(campaign_aux._WSM_IDS)
    assert "comcat_california" in {j["dataset"] for j in tri}
    assert all(j["status"] == "prepared_not_launched" for j in manifest["external_jobs"])
    assert manifest["blocked"][0]["dataset"] == "comcat_california"


def test_synthetic_identity_transform_is_coordinate_dataclass_compatible(tmp_path, monkeypatch):
    _environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        campaign_aux, "resolve_prepared_dataset", lambda route_id, root: _fake_route(route_id)
    )
    manifest = campaign_aux.prepare_auxiliary(_spec(), tmp_path / "aux")
    from seahorse.experiments.coordinates import CoordinateTransform

    transform = CoordinateTransform.from_dict(
        manifest["synthetic"]["homogeneous_poisson"]["coordinate_transform"]
    )
    assert transform.time_scale == 1.0
    assert len(manifest["external_selected_evaluations"]) == 57
    wsm_spec = json.loads(
        Path(next(j["spec"] for j in manifest["external_jobs"] if j["kind"] == "wsm")).read_text()
    )
    tri_spec = json.loads(
        Path(
            next(j["spec"] for j in manifest["external_jobs"] if j["kind"] == "tritpp")
        ).read_text()
    )
    assert (wsm_spec["max_steps"], wsm_spec["max_time_seconds"]) == (5000, 600)
    assert (tri_spec["max_steps"], tri_spec["max_time_seconds"]) == (5000, 600)
    assert tri_spec["authorcheckout"] == campaign_aux.author_tritpp_checkout()


def test_external_selection_uses_all_trials_and_validation_only(tmp_path):
    from seahorse.experiments.campaign_aux_run import select_trial

    jobs = [{"id": f"trial{i}", "kind": "wsm", "lr_trial": i} for i in range(3)]
    with pytest.raises(ValueError, match="All external"):
        select_trial(jobs, tmp_path)
    for i, job in enumerate(jobs):
        out = tmp_path / "external_results" / job["id"]
        out.mkdir(parents=True)
        checkpoint = out / "model.pt"
        checkpoint.write_bytes(b"fixture")
        (out / "status.json").write_text(json.dumps({"state": "completed"}))
        (out / "metrics.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "val_loglik": [-2.0, -1.0, -3.0][i],
                    "test_loglik": [100.0, -99.0, 200.0][i],
                }
            )
        )
    chosen, _, objective = select_trial(jobs, tmp_path)
    assert chosen["id"] == "trial1" and objective == 1.0
