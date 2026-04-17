from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from unified_stpp.data import download_dataset, load_dataset
from unified_stpp.data.contract import (
    validate_sequence_record,
    validate_sequence_records,
)
from unified_stpp.data.hub import CuratedDatasetSpec


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def _toy_records() -> list[dict]:
    return [
        {
            "times": [0.1, 0.5],
            "locations": [[0.0, 1.0], [1.0, 0.0]],
        },
        {
            "times": [0.2],
            "locations": [[0.25, 0.75]],
            "marks": [1],
        },
    ]


class TestSequenceRecordValidation(unittest.TestCase):
    def test_validate_sequence_record_accepts_valid_record(self):
        validate_sequence_record(_toy_records()[0])

    def test_validate_sequence_record_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            validate_sequence_record(
                {"times": [0.1, 0.2], "locations": [[0.0, 1.0]]}
            )

    def test_validate_sequence_records_reports_source_and_index(self):
        with self.assertRaisesRegex(ValueError, "toy.jsonl\\[1\\]"):
            validate_sequence_records(
                [
                    _toy_records()[0],
                    {"times": [0.1], "locations": [[0.0, 1.0], [1.0, 0.0]]},
                ],
                source="toy.jsonl",
            )


class TestDatasetHub(unittest.TestCase):
    def test_load_dataset_from_local_directory_returns_available_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records = _toy_records()
            _write_jsonl(root / "train.jsonl", records)
            _write_jsonl(root / "val.jsonl", records[:1])
            _write_jsonl(root / "test.jsonl", records[:1])

            loaded = load_dataset(root)

        self.assertEqual(sorted(loaded), ["test", "train", "val"])
        self.assertEqual(len(loaded["train"]), 2)
        self.assertEqual(len(loaded["val"]), 1)
        self.assertEqual(len(loaded["test"]), 1)

    def test_load_dataset_from_local_file_returns_records(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "train.jsonl"
            records = _toy_records()
            _write_jsonl(path, records)

            loaded = load_dataset(path)

        self.assertEqual(loaded, records)

    def test_load_dataset_canonicalizes_t_x_y_aliases(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "train.jsonl"
            alias_records = [
                {
                    "t": [0.1, 0.5],
                    "x": [0.0, 1.0],
                    "y": [1.0, 0.0],
                }
            ]
            _write_jsonl(path, alias_records)

            loaded = load_dataset(path)

        self.assertEqual(
            loaded,
            [{"times": [0.1, 0.5], "locations": [[0.0, 1.0], [1.0, 0.0]], "t": [0.1, 0.5], "x": [0.0, 1.0], "y": [1.0, 0.0]}],
        )

    def test_load_dataset_canonicalizes_nested_events(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "train.jsonl"
            nested_records = [
                {
                    "sequence_id": "covid_001",
                    "events": [
                        {"t": 0.1, "x": 0.0, "y": 1.0},
                        {"t": 0.5, "x": 1.0, "y": 0.0},
                    ],
                }
            ]
            _write_jsonl(path, nested_records)

            loaded = load_dataset(path)

        self.assertEqual(loaded[0]["times"], [0.1, 0.5])
        self.assertEqual(loaded[0]["locations"], [[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(loaded[0]["sequence_id"], "covid_001")

    def test_load_dataset_rejects_invalid_records(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(
                root / "train.jsonl",
                [{"times": [0.1], "locations": [[0.0, 1.0], [1.0, 0.0]]}],
            )

            with self.assertRaisesRegex(ValueError, "train.jsonl\\[0\\]"):
                load_dataset(root, split="train")

    def test_download_dataset_prefers_curated_local_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            local_root = Path(td) / "toy_dataset"
            _write_jsonl(local_root / "train.jsonl", _toy_records())
            spec = CuratedDatasetSpec(
                name="toy_local",
                local_paths=(str(local_root),),
                repo_id="test/repo",
                repo_path="toy_local",
            )

            with patch("unified_stpp.data.hub._CURATED_DATASETS", {"toy_local": spec}):
                with patch("unified_stpp.data.hub._snapshot_download") as download_mock:
                    resolved = download_dataset("toy_local")

        self.assertEqual(resolved, local_root.resolve())
        download_mock.assert_not_called()

    def test_download_dataset_uses_hf_snapshot_cache_when_needed(self):
        with tempfile.TemporaryDirectory() as td:
            snapshot_root = Path(td) / "snapshot"
            remote_path = snapshot_root / "datasets" / "toy_remote"
            _write_jsonl(remote_path / "train.jsonl", _toy_records())
            spec = CuratedDatasetSpec(
                name="toy_remote",
                local_paths=(),
                repo_id="test/repo",
                repo_path="datasets/toy_remote",
            )

            with patch("unified_stpp.data.hub._CURATED_DATASETS", {"toy_remote": spec}):
                with patch(
                    "unified_stpp.data.hub._snapshot_download",
                    return_value=str(snapshot_root),
                ) as download_mock:
                    resolved = download_dataset("toy_remote", revision="branch-2")

        self.assertEqual(resolved, remote_path.resolve())
        download_mock.assert_called_once()
        kwargs = download_mock.call_args.kwargs
        self.assertEqual(kwargs["repo_id"], "test/repo")
        self.assertEqual(kwargs["repo_type"], "dataset")
        self.assertEqual(kwargs["revision"], "branch-2")
        self.assertEqual(kwargs["allow_patterns"], ["datasets/toy_remote/**"])
        self.assertFalse(kwargs["local_files_only"])
        self.assertFalse(kwargs["force_download"])

    def test_download_dataset_accepts_direct_hf_repo_id(self):
        with tempfile.TemporaryDirectory() as td:
            snapshot_root = Path(td) / "snapshot"
            _write_jsonl(snapshot_root / "train.jsonl", _toy_records())

            with patch(
                "unified_stpp.data.hub._snapshot_download",
                return_value=str(snapshot_root),
            ) as download_mock:
                resolved = download_dataset("owner/repo", revision="main")

        self.assertEqual(resolved, snapshot_root.resolve())
        kwargs = download_mock.call_args.kwargs
        self.assertEqual(kwargs["repo_id"], "owner/repo")
        self.assertEqual(kwargs["revision"], "main")
        self.assertIsNone(kwargs["allow_patterns"])

    def test_download_dataset_accepts_direct_hf_repo_subdir(self):
        with tempfile.TemporaryDirectory() as td:
            snapshot_root = Path(td) / "snapshot"
            remote_path = snapshot_root / "datasets" / "toy_remote"
            _write_jsonl(remote_path / "train.jsonl", _toy_records())

            with patch(
                "unified_stpp.data.hub._snapshot_download",
                return_value=str(snapshot_root),
            ) as download_mock:
                resolved = download_dataset("owner/repo/datasets/toy_remote")

        self.assertEqual(resolved, remote_path.resolve())
        kwargs = download_mock.call_args.kwargs
        self.assertEqual(kwargs["repo_id"], "owner/repo")
        self.assertEqual(kwargs["allow_patterns"], ["datasets/toy_remote/**"])

    def test_load_dataset_can_select_one_split_from_curated_spec(self):
        with tempfile.TemporaryDirectory() as td:
            local_root = Path(td) / "toy_dataset"
            records = _toy_records()
            _write_jsonl(local_root / "train.jsonl", records)
            _write_jsonl(local_root / "val.jsonl", records[:1])
            spec = CuratedDatasetSpec(
                name="toy_split",
                local_paths=(str(local_root),),
                repo_path="toy_split",
            )

            with patch("unified_stpp.data.hub._CURATED_DATASETS", {"toy_split": spec}):
                loaded = load_dataset("toy_split", split="val")

        self.assertEqual(loaded, records[:1])


if __name__ == "__main__":
    unittest.main()
