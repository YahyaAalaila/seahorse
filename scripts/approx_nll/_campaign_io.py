"""Shared loading for the approximate-NLL scripts: one finished campaign job -> test targets.

Mirrors ``seahorse.experiments.campaign_run.evaluate_one``: manifest, coordinate transform,
hash-checked test records, context dataset, preset collate and sequential target order
(``sequence_index``, ``target_event_index`` from ``observation_metadata``). Only the scoring
step differs, so the per-target files align with the exact presets' ``target_nll.npz``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seahorse.data.prepared_protocol import _copy_unmarked
from seahorse.data.registry import DataRegistry
from seahorse.experiments.campaign_run import _coordinate_entry, context_dataset, read_records
from seahorse.experiments.coordinates import transform_sequences
from seahorse.models.configs import ConfigRegistry
from seahorse.models.targets import last_target_state
from seahorse.runner.runner import STPPRunner


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def remap(path: str, maps: list[tuple[str, str]]) -> str:
    """Rewrite a data path recorded in the manifest (``OLD=NEW`` prefix maps)."""
    for old, new in maps:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def add_common_args(parser) -> None:
    parser.add_argument("--manifest", required=True, help="campaign manifest.json")
    parser.add_argument("--job-dir", required=True, help="finished job directory (model/, status.json)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                        help="prefix remap for the data paths recorded in the manifest (repeatable)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ref", default=None,
                        help="target_nll.npz of an exact preset of the same dataset/seed: check target order")


@dataclass
class Job:
    job: dict[str, Any]
    job_dir: Path
    runner: Any
    model: Any
    coord: Any
    dataset: Any
    collate: Any
    test_path: str
    test_sha256: str


def load_job(args, *, model_kind: str) -> Job:
    manifest = json.loads(Path(args.manifest).read_text())
    job_dir = Path(args.job_dir)
    job = next(j for j in manifest["jobs"] if j["id"] == job_dir.name)
    status = json.loads((job_dir / "status.json").read_text())
    if job["model"] != model_kind:
        raise SystemExit(f"{job_dir.name}: model {job['model']!r}, expected {model_kind!r}")
    if status.get("state") != "completed":
        raise SystemExit(f"{job_dir.name}: job state {status.get('state')!r}, expected 'completed'")
    runner = STPPRunner.load(str(job_dir / "model"))
    model = runner.model.to(args.device).eval()
    if (runner.norm_stats or {}).get("normalize", False):
        raise SystemExit("dataset normalisation is on; the Jacobians below assume it is off")
    route = manifest["datasets"][job["dataset"]]
    coord = _coordinate_entry(manifest, job)
    contract = job_dir / "input_contract.json"
    if contract.exists():
        recorded = json.loads(contract.read_text())["coordinate_transform"]["log_raw_volume_per_normalized_volume"]
        if abs(recorded - coord.log_raw_volume_per_normalized_volume) > 1e-12:
            raise SystemExit("coordinate transform differs from the job's input contract")
    maps = [tuple(m.split("=", 1)) for m in args.path_map]
    test_path = remap(route["splits"]["test"], maps)
    test_sha = route["split_sha256"]["test"]
    records = _copy_unmarked(list(transform_sequences(read_records(test_path, sha256=test_sha), coord)))
    dataset = context_dataset(records, manifest, job, coord, split="test")
    collate = DataRegistry.get_collate(ConfigRegistry.get_collate_key(runner.config.model.preset))
    return Job(job, job_dir, runner, model, coord, dataset, collate, test_path, test_sha)


def last_event_state(model, batch, device):
    """State of the last event of every context window, and the batch tensors on ``device``."""
    sm = model.state_model
    kw = {k: batch[k].to(device) for k in ("times", "locations", "lengths")}
    state = last_target_state(sm.sequence_states(sm.encode_history(**kw), **kw), kw["lengths"])
    return state, kw


def target_ids(batch) -> tuple[list[int], list[int]]:
    md = batch["observation_metadata"]
    return ([m["target_source_sequence_index"] for m in md], [m["target_global_event_index"] for m in md])


def check_alignment(ref: str | None, seq_ids: np.ndarray, ev_ids: np.ndarray, positions=None) -> bool | None:
    if not ref:
        return None
    R = np.load(ref)
    rs, re_ = R["sequence_index"], R["target_event_index"]
    if positions is not None:
        rs, re_ = rs[positions], re_[positions]
    else:
        rs, re_ = rs[: len(seq_ids)], re_[: len(ev_ids)]
    return bool(np.array_equal(rs, seq_ids) and np.array_equal(re_, ev_ids))


def provenance(job: Job) -> dict[str, Any]:
    ckpt = job.job_dir / "model" / "model.ckpt"
    return {"job_id": job.job_dir.name, "dataset": job.job["dataset"], "model": job.job["model"],
            "checkpoint_sha256": sha256(ckpt) if ckpt.exists() else None,
            "test_split_sha256": job.test_sha256,
            "log_raw_volume_per_normalized_volume": job.coord.log_raw_volume_per_normalized_volume,
            "time_scale": job.coord.time_scale, "space_scale": list(job.coord.space_scale)}
