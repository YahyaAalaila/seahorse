"""Released-dataset loader: registry, card-hash parsing and verification (no network)."""

from __future__ import annotations

import hashlib
import json

import pytest

from seahorse.data import hf

CARD = """---
license: other
configs:
- config_name: native
  data_files:
  - split: train
    path: native/train.jsonl
seahorse_release:
  release: v1.0-iclr2027
  split_sha256:
    native:
      train: {train}
      val: {val}
      test: {test}
---
# Card body
"""


def _jsonl(records):
    return "".join(json.dumps(r) + "\n" for r in records).encode()


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Serve README.md and split files from a local directory instead of the Hub."""
    records = {
        "train": [{"times": [0.1, 0.5], "locations": [[0.1, 0.2], [0.3, 0.4]]}],
        "val": [{"times": [0.2], "locations": [[0.5, 0.5]]}],
        "test": [{"times": [0.3, 0.6, 0.9], "locations": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]}],
    }
    root = tmp_path / "hub"
    (root / "native").mkdir(parents=True)
    digests = {}
    for split, rows in records.items():
        payload = _jsonl(rows)
        (root / "native" / f"{split}.jsonl").write_bytes(payload)
        digests[split] = hashlib.sha256(payload).hexdigest()
    (root / "README.md").write_text(CARD.format(**digests))
    calls = []

    def fake_download(repo_id, filename, *, revision, cache_dir=None):
        calls.append((repo_id, filename, revision))
        return root / filename

    monkeypatch.setattr(hf, "_hub_download", fake_download)
    return root, digests, calls


def test_registry_lists_paper_and_script_only_repositories():
    names = hf.list_released_datasets(in_paper=True)
    assert {"comcat_california", "austin311", "citibike_nyc", "synthetic", "covid", "earthquakes"} <= set(names)
    assert hf.released_dataset("seahorse-stpp/austin_311_stpp").name == "austin311"
    assert hf.released_dataset("citibike_nyc").form == "script-only"
    assert hf.released_dataset("covid").split_path("default", "validation") == "val.jsonl"
    assert hf.released_dataset("austin311").split_path("native", "val") == "native/val.jsonl"
    for spec in hf.RELEASED_DATASETS.values():
        assert len(spec.revision) == 40 and spec.default_config in spec.configs


def test_card_hashes_are_read_from_the_front_matter():
    hashes = hf.card_split_hashes(CARD.format(train="a" * 64, val="b" * 64, test="c" * 64))
    assert hashes == {"native": {"train": "a" * 64, "val": "b" * 64, "test": "c" * 64}}
    assert hf.card_split_hashes("# no front matter\n") == {}


def test_load_verifies_every_split_against_the_card(fake_hub):
    root, digests, calls = fake_hub
    splits = hf.load_released_dataset("comcat_california", config="native")
    assert len(splits["train"]) == 1 and len(splits["test"]) == 3
    assert all(rec["verified"] for rec in splits["__provenance__"])
    assert {rec["sha256"] for rec in splits["__provenance__"]} == set(digests.values())
    spec = hf.released_dataset("comcat_california")
    assert all(call[2] == spec.revision for call in calls)


def test_hash_mismatch_is_an_error(fake_hub):
    root, _digests, _calls = fake_hub
    (root / "native" / "val.jsonl").write_bytes(_jsonl([{"times": [0.9], "locations": [[0.0, 0.0]]}]))
    with pytest.raises(hf.HashMismatchError):
        hf.load_released_dataset("comcat_california", config="native", splits=("val",))


def test_script_only_repositories_need_a_local_rebuild(fake_hub, tmp_path):
    root, digests, _calls = fake_hub
    with pytest.raises(hf.ScriptOnlyDatasetError):
        hf.load_released_dataset("seattle_fire911", config="native")
    local = tmp_path / "rebuilt"
    (local / "native").mkdir(parents=True)
    for split in ("train", "val", "test"):
        (local / "native" / f"{split}.jsonl").write_bytes((root / "native" / f"{split}.jsonl").read_bytes())
    splits = hf.load_released_dataset("seattle_fire911", config="native", local_root=local)
    assert all(rec["verified"] for rec in splits["__provenance__"])


def test_legacy_cards_without_hashes_load_unverified(tmp_path, monkeypatch):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "README.md").write_text("---\nlicense: cc-by-nc-4.0\n---\n")
    (root / "train.jsonl").write_bytes(_jsonl([{"times": [0.1], "locations": [[0.1, 0.1]]}]))
    monkeypatch.setattr(hf, "_hub_download", lambda repo_id, filename, *, revision, cache_dir=None: root / filename)
    splits = hf.load_released_dataset("covid", splits=("train",))
    assert len(splits["train"]) == 1
    assert splits["__provenance__"][0]["verified"] is False
