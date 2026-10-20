from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from seahorse.data.hub import download_dataset
from seahorse.data.prepared import (
    PREPARED_CORPORA,
    build_prepared_datasets,
    load_prepared_split_sample,
    load_prepared_splits,
    prepared_dataset_ids,
    resolve_prepared_dataset,
)
from seahorse.data.registry import DataRegistry


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(*, marks: list[int], sequence_id: str) -> dict:
    return {
        "sequence_id": sequence_id,
        "stream_id": "toy-stream",
        "window_start": "2023-01-01T00:00:00Z",
        "window_end": "2023-01-01T01:00:00Z",
        "time_axis": "utc",
        "source_timezone": "UTC",
        "t0": 0.0,
        "t1": 1.0,
        "times": [0.1, 0.2, 0.3],
        "locations": [[-1.0, 2.0], [-0.5, 2.5], [0.0, 3.0]],
        "event_ids": [f"{sequence_id}-{i}" for i in range(3)],
        "event_metadata": {"raw_label": ["a", "b", "c"]},
        "marks": marks,
        "event_covariates": [[1.0], [2.0], [3.0]],
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _prepared_root(tmp_path: Path) -> Path:
    root = tmp_path / "prepared"
    corpus = root / "urban_austin311_2023"
    native = corpus / "native"
    jitter = corpus / "jittered_seed20260910"
    native.mkdir(parents=True)
    jitter.mkdir()
    split_records = {
        "train": [_record(marks=[10, 30, 10], sequence_id="train")],
        "val": [_record(marks=[30, 99, 10], sequence_id="val")],
        "test": [_record(marks=[99, 30, 10], sequence_id="test")],
    }
    variants = []
    for name, directory, strict in (("native", native, False), ("jittered_seed20260910", jitter, True)):
        for split, records in split_records.items():
            _write_jsonl(directory / f"{split}.jsonl", records)
        variants.append(
            {
                "name": name,
                # Deliberately nonlocal: resolver must prefer relocated root.
                "path": f"/immutable/snapshot/{name}",
                "strictly_increasing_global_times": strict,
                "files_sha256": {f"{split}.jsonl": _sha(directory / f"{split}.jsonl") for split in split_records},
            }
        )
    manifest = {
        "dataset_name": "urban_austin311_2023",
        "source": {
            "time_axis": "utc",
            "source_timezone": "UTC",
            "timestamp_semantics": "synthetic test record",
            "spatial_support": "declared test support",
        },
        "cleaning": {"bounding_box_lon_lat": [-2.0, 2.0, 1.0, 4.0]},
        "native_qc": {"adjacent_tie_fraction": 0.5, "fraction_on_minute": 1.0},
        "variants": variants,
    }
    (corpus / "dataset_meta.json").write_text(json.dumps(manifest))
    return root


def test_catalog_declares_recommended_reserve_and_scale_routes():
    assert sum(spec.role == "recommended" for spec in PREPARED_CORPORA) == 11
    assert {spec.name for spec in PREPARED_CORPORA if spec.role == "reserve"} == {
        "foursquare_nyc",
        "foursquare_tokyo",
    }
    assert "prepared:fpa_us_scale:native" in prepared_dataset_ids()
    assert "prepared:austin311:jittered_seed20260910" in prepared_dataset_ids(roles=("recommended",))


def test_route_resolves_manifest_hashes_and_declared_spatial_rectangle(tmp_path):
    root = _prepared_root(tmp_path)
    route = resolve_prepared_dataset("prepared:austin311:jittered_seed20260910", root=root, verify_hashes=True)

    assert route.view_root == root / "urban_austin311_2023" / "jittered_seed20260910"
    assert route.strictly_increasing_global_times
    domain = route.observation["spatial_domain"]
    assert domain["kind"] == "declared_city_validity_envelope"
    assert domain["longitude"] == [-98.0, -97.4]
    assert domain["latitude"] == [30.0, 30.6]
    assert domain["builder_sha256"] == "adcd2d999268d6e0566bf21b41de6f79773b58382a8dc38a3fb25721a285c862"
    assert route.machine_readable()["continuous_time"] == {
        "exact_eligible": False,
        "approximate_observation_model": True,
    }


def test_hub_prepared_prefix_uses_environment_root(tmp_path, monkeypatch):
    root = _prepared_root(tmp_path)
    monkeypatch.setenv("SEAHORSE_PREPARED_DATA_ROOT", str(root))
    assert download_dataset("prepared:austin311:native") == root / "urban_austin311_2023" / "native"


def test_train_only_mark_map_preserves_source_indices_and_reserves_unknown(tmp_path):
    route = resolve_prepared_dataset("prepared:austin311:jittered_seed20260910", root=_prepared_root(tmp_path))
    prepared = load_prepared_splits(route, mark_policy="train_only")

    assert prepared.mark_map is not None
    assert prepared.mark_map.source_to_model == {10: 0, 30: 1}
    assert prepared.mark_map.unknown_index == 2
    assert prepared.train[0]["marks"] == [0, 1, 0]
    assert prepared.val[0]["marks"] == [1, 2, 0]
    assert prepared.val[0]["event_metadata"]["prepared_source_mark_index"] == [30, 99, 10]
    assert prepared.val[0]["event_ids"] == ["val-0", "val-1", "val-2"]
    assert prepared.val[0]["observation_metadata"]["prepared_route"]["split"] == "val"


def test_bounded_sample_is_a_complete_source_record_not_an_event_prefix(tmp_path):
    route = resolve_prepared_dataset("prepared:austin311:jittered_seed20260910", root=_prepared_root(tmp_path))
    sample = load_prepared_split_sample(route, "train", min_events=3)
    assert sample is not None
    assert sample["times"] == [0.1, 0.2, 0.3]
    assert sample["event_ids"] == ["train-0", "train-1", "train-2"]
    assert sample["observation_metadata"]["prepared_route"]["split_sha256"] == route.split_sha256["train"]


def test_exact_gate_blocks_native_ties_and_jittered_approximation_is_explicit(tmp_path):
    root = _prepared_root(tmp_path)
    native = load_prepared_splits(resolve_prepared_dataset("prepared:austin311:native", root=root))
    with pytest.raises(ValueError, match="exact continuous-time"):
        build_prepared_datasets(native, normalize_time=True, normalize_space=True, continuous_time_mode="exact")

    jittered = load_prepared_splits(resolve_prepared_dataset("prepared:austin311:jittered_seed20260910", root=root))
    with pytest.raises(ValueError, match="exact continuous-time"):
        build_prepared_datasets(jittered, normalize_time=True, normalize_space=True, continuous_time_mode="exact")
    train_ds, val_ds, test_ds = build_prepared_datasets(
        jittered,
        normalize_time=True,
        normalize_space=True,
        continuous_time_mode="approximate",
    )
    assert val_ds.time_mean == train_ds.time_mean
    assert test_ds.loc_mean is train_ds.loc_mean
    assert train_ds[0]["observation_metadata"]["prepared_route"]["approximate_observation_model"]


def test_prepared_protocol_defaults_unmarked_and_train_only_marks_are_train_fitted(tmp_path, monkeypatch):
    root = _prepared_root(tmp_path)
    monkeypatch.setenv("SEAHORSE_PREPARED_DATA_ROOT", str(root))
    raw = {split: [json.loads(line) for line in (root / "urban_austin311_2023" / "jittered_seed20260910" / f"{split}.jsonl").read_text().splitlines()] for split in ("train", "val", "test")}
    base = dict(dataset="prepared:austin311:jittered_seed20260910", normalize=True)
    unmarked = SimpleNamespace(**base, adapter_kwargs={"mark_policy": "unmarked"})
    train_ds, val_ds, _ = DataRegistry.build_datasets("prepared", unmarked, raw["train"], raw["val"], raw["test"])
    assert train_ds.prepared_n_marks == 0
    assert "marks" not in train_ds[0]
    assert train_ds[0]["observation_metadata"]["event_metadata"]["prepared_source_mark_index"] == [10, 30, 10]
    assert val_ds.time_mean == train_ds.time_mean

    marked = SimpleNamespace(**base, adapter_kwargs={"mark_policy": "train_only"})
    train_ds, val_ds, _ = DataRegistry.build_datasets("prepared", marked, raw["train"], raw["val"], raw["test"])
    assert train_ds.prepared_n_marks == 3
    assert train_ds[0]["marks"].tolist() == [0, 1, 0]
    assert val_ds[0]["marks"].tolist() == [1, 2, 0]


def test_prepared_lazy_window_protocol_preserves_target_and_does_not_materialize_windows(tmp_path, monkeypatch):
    root = _prepared_root(tmp_path)
    monkeypatch.setenv("SEAHORSE_PREPARED_DATA_ROOT", str(root))
    raw = [json.loads(line) for line in (root / "urban_austin311_2023" / "jittered_seed20260910" / "train.jsonl").read_text().splitlines()]
    data = SimpleNamespace(
        dataset="prepared:austin311:jittered_seed20260910", normalize=False,
        adapter_kwargs={"mark_policy": "unmarked", "training_view": "sliding_window", "lookback": 2, "lookahead": 1},
    )
    train_ds, _, _ = DataRegistry.build_datasets("prepared", data, raw, raw, raw)
    assert len(train_ds) == 1
    assert not hasattr(train_ds, "sequences")
    metadata = train_ds[0]["observation_metadata"]
    assert metadata["target_event_index"] == 2
    assert metadata["target_event_id"] == "train-2"
    assert metadata["event_ids"] == ["train-0", "train-1", "train-2"]
    assert metadata["window_is_not_parent_observation_exposure"] is True
    assert metadata["window_time_coordinate"] == "float64_rebased_to_first_history_event"
    assert train_ds[0]["times"].tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert train_ds.stat_sequences is train_ds.raw_sequences
    sampler = train_ds.batch_by_size(3)
    assert not isinstance(sampler, list)
    assert list(next(iter(sampler))) == [0]
