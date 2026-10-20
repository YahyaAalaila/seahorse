import json
from pathlib import Path
import numpy as np

from seahorse.experiments.campaign import digest_file, make_jobs
from seahorse.experiments.coordinates import fit_coordinates


def test_full_batch_rehearsal_reads_only_train_validation(tmp_path, monkeypatch):
    import seahorse.experiments.campaign_rehearse as module
    source = Path(__file__).resolve().parents[1]
    spec = json.loads((source / "experiments/campaign_2026/spec_nnstat_pegasus_v1.json").read_text())
    spec["models"] = [m for m in spec["models"] if m["id"] == "poisson_gmm"]
    spec["datasets"] = spec["datasets"][:1]
    spec["protocol"]["validation_targets"] = 8
    spec["fit"]["batch_size"] = 8
    job = make_jobs(spec)[0]
    rng = np.random.default_rng(42)
    row = {"times": np.linspace(.01, .99, 100).tolist(), "locations": rng.uniform(.1, .9, (100, 2)).tolist(),
        "window_start": "2020-01-01", "window_end": "2020-01-02", "t0": 0., "t1": 1.}
    splits, hashes = {}, {}
    for split in ("train", "val"):
        path = tmp_path / (split + ".jsonl")
        path.write_text(json.dumps(row) + "\n")
        splits[split], hashes[split] = str(path), digest_file(path)
    route = "prepared:comcat_california:native"
    manifest = {"sha256": "fixture", "source": {"sha256": "fixture-source"}, "spec": spec, "jobs": [job],
        "datasets": {job["dataset"]: {"dataset_id": route, "splits": splits, "split_sha256": hashes}},
        "coordinates": {"routes": {route: {"train_split": {"sha256": hashes["train"]},
            "transform": fit_coordinates([row], (0., 1., 0., 1.)).to_dict()}}}}
    monkeypatch.setattr(module, "verify", lambda _: manifest)
    result = module.rehearse(tmp_path / "manifest.json", out=tmp_path / "rehearsal", dataset=job["dataset"], models=[job["model"]], steps=1, device="cpu")
    assert result["pass"], result
    assert result["rows"][0]["train_targets"] == 80
    assert result["rows"][0]["batch_size"] == 8
    assert result["rows"][0]["training_audit"]["optimizer_steps"] == 1
    assert len(result["rows"][0]["fitted_validation_batch_seconds"]) == 3
