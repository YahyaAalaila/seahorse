"""Data-source resolution helpers for config-driven CLI flows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from seahorse.utils import load_jsonl, load_splits_dir

if TYPE_CHECKING:
    from seahorse.config.schema import DataConfig


@dataclass(frozen=True)
class ResolvedDataPaths:
    """Resolved single-dataset file paths for fit/tune style flows."""

    train_path: Path
    val_path: Path
    test_path: Path | None
    dataset_id: str
    source_root: Path | None = None


@dataclass(frozen=True)
class ResolvedBenchmarkData:
    """Resolved split collection for benchmark flows."""

    splits_dir: Path
    splits: dict[str, tuple[list, list, list]]
    files: dict[str, dict[str, Path | None]] = field(default_factory=dict)


def download_dataset(name: str, revision: str | None = None):
    """Late-import dataset download helper so tests can patch this symbol."""
    from seahorse.data.hub import download_dataset as _download_dataset

    return _download_dataset(name, revision=revision)


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(resolved),
    }


def _maybe_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _dataset_filter(data_config: "DataConfig") -> list[str] | None:
    datasets = list(getattr(data_config, "datasets", []) or [])
    return datasets or None


def _requested_dataset_id(data_config: "DataConfig", resolved_root: Path) -> str:
    requested = getattr(data_config, "dataset", None)
    if requested:
        parts = [part for part in str(requested).strip().strip("/").split("/") if part]
        if parts:
            return parts[-1]
    return resolved_root.name


def build_single_data_provenance(
    data_config: "DataConfig",
    resolved_data: ResolvedDataPaths,
) -> dict[str, Any]:
    """Return a canonical provenance manifest for one resolved dataset."""
    files: dict[str, dict[str, Any]] = {
        "train": _file_fingerprint(resolved_data.train_path),
        "val": _file_fingerprint(resolved_data.val_path),
    }
    if resolved_data.test_path is not None and resolved_data.test_path.exists():
        files["test"] = _file_fingerprint(resolved_data.test_path)

    manifest: dict[str, Any] = {
        "mode": "single",
        "requested": {
            "dataset": getattr(data_config, "dataset", None),
            "dataset_revision": getattr(data_config, "dataset_revision", None),
            "splits_dir": getattr(data_config, "splits_dir", None),
            "datasets": list(getattr(data_config, "datasets", []) or []),
            "train_path": getattr(data_config, "train_path", None),
            "val_path": getattr(data_config, "val_path", None),
            "test_path": getattr(data_config, "test_path", None),
        },
        "resolved": {
            "dataset_id": resolved_data.dataset_id,
            "source_root": (
                None
                if resolved_data.source_root is None
                else str(resolved_data.source_root.expanduser().resolve())
            ),
            "train_path": str(resolved_data.train_path.expanduser().resolve()),
            "val_path": str(resolved_data.val_path.expanduser().resolve()),
            "test_path": (
                None
                if resolved_data.test_path is None
                else str(resolved_data.test_path.expanduser().resolve())
            ),
        },
        "files": files,
    }
    manifest["source_fingerprint"] = hashlib.sha256(
        _canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    return manifest


def build_benchmark_data_provenance(
    data_config: "DataConfig",
    resolved_data: ResolvedBenchmarkData,
) -> dict[str, Any]:
    """Return a canonical provenance manifest for a benchmark split collection."""
    datasets: dict[str, dict[str, Any]] = {}
    for dataset_id, files in sorted(resolved_data.files.items()):
        file_manifest: dict[str, Any] = {
            split: _file_fingerprint(path)
            for split, path in files.items()
            if path is not None and path.exists()
        }
        datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "files": file_manifest,
            "source_fingerprint": hashlib.sha256(
                _canonical_json(file_manifest).encode("utf-8")
            ).hexdigest(),
        }

    manifest: dict[str, Any] = {
        "mode": "benchmark",
        "requested": {
            "dataset": getattr(data_config, "dataset", None),
            "dataset_revision": getattr(data_config, "dataset_revision", None),
            "splits_dir": getattr(data_config, "splits_dir", None),
            "datasets": list(getattr(data_config, "datasets", []) or []),
        },
        "resolved": {
            "splits_dir": str(resolved_data.splits_dir.expanduser().resolve()),
            "datasets": sorted(resolved_data.splits.keys()),
        },
        "datasets": datasets,
    }
    manifest["source_fingerprint"] = hashlib.sha256(
        _canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    return manifest


def _resolve_dataset_root(data_config: "DataConfig") -> Path:
    dataset = getattr(data_config, "dataset", None)
    if not dataset:
        raise ValueError("data.dataset must be set for named dataset resolution.")
    root = Path(
        download_dataset(
            dataset,
            revision=getattr(data_config, "dataset_revision", None),
        )
    )
    if not root.is_dir():
        raise ValueError(
            f"data.dataset={dataset!r} resolved to '{root}', which is not a directory. "
            "Use data.train_path/data.val_path for explicit file inputs."
        )
    return root


def _validate_single_source(data_config: "DataConfig") -> Literal["dataset", "explicit"]:
    has_dataset = bool(getattr(data_config, "dataset", None))
    has_splits_dir = bool(getattr(data_config, "splits_dir", None))
    train_path = getattr(data_config, "train_path", None)
    val_path = getattr(data_config, "val_path", None)
    test_path = getattr(data_config, "test_path", None)
    has_explicit = train_path is not None or val_path is not None or test_path is not None
    has_datasets_filter = bool(getattr(data_config, "datasets", None))

    if has_splits_dir:
        raise ValueError(
            "single-dataset resolution does not accept data.splits_dir; "
            "use data.dataset or explicit data.train_path/data.val_path."
        )
    if has_datasets_filter:
        raise ValueError(
            "single-dataset resolution does not accept data.datasets; "
            "that filter is only valid for benchmark split collections."
        )
    if has_dataset and has_explicit:
        raise ValueError(
            "data.dataset is mutually exclusive with explicit data.train_path/data.val_path/data.test_path."
        )
    if has_dataset:
        return "dataset"
    if train_path is None and val_path is None and test_path is None:
        raise ValueError(
            "data resolution requires either data.dataset or both data.train_path and data.val_path."
        )
    if train_path is None or val_path is None:
        raise ValueError(
            "explicit file resolution requires both data.train_path and data.val_path."
        )
    return "explicit"


def _validate_benchmark_source(data_config: "DataConfig") -> Literal["dataset", "splits_dir"]:
    has_dataset = bool(getattr(data_config, "dataset", None))
    has_splits_dir = bool(getattr(data_config, "splits_dir", None))
    has_explicit_files = any(
        getattr(data_config, field, None) is not None
        for field in ("train_path", "val_path", "test_path")
    )

    if has_explicit_files:
        raise ValueError(
            "benchmark resolution does not accept explicit data.train_path/data.val_path/data.test_path; "
            "use data.dataset or data.splits_dir."
        )
    if has_dataset and has_splits_dir:
        raise ValueError("data.dataset and data.splits_dir are mutually exclusive.")
    if has_dataset:
        return "dataset"
    if has_splits_dir:
        return "splits_dir"
    raise ValueError(
        "benchmark resolution requires either data.dataset or data.splits_dir."
    )


def _resolve_benchmark_files(
    splits_dir: Path,
    *,
    datasets: list[str] | None = None,
) -> dict[str, dict[str, Path | None]]:
    root = splits_dir.expanduser()
    if (root / "train.jsonl").exists():
        if datasets not in (None, [], [root.name]):
            raise ValueError(
                f"Requested datasets {datasets!r}, but '{root}' is a single-dataset splits directory."
            )
        test_path = root / "test.jsonl"
        return {
            root.name: {
                "train": root / "train.jsonl",
                "val": root / "val.jsonl",
                "test": test_path if test_path.exists() else None,
            }
        }

    names = datasets or sorted(
        ds_dir.name
        for ds_dir in root.iterdir()
        if ds_dir.is_dir() and (ds_dir / "train.jsonl").exists()
    )
    files: dict[str, dict[str, Path | None]] = {}
    missing: list[str] = []
    for name in names:
        ds_dir = root / name
        train_path = ds_dir / "train.jsonl"
        val_path = ds_dir / "val.jsonl"
        if not ds_dir.is_dir() or not train_path.exists() or not val_path.exists():
            missing.append(name)
            continue
        test_path = ds_dir / "test.jsonl"
        files[name] = {
            "train": train_path,
            "val": val_path,
            "test": test_path if test_path.exists() else None,
        }
    if missing:
        raise ValueError(f"Dataset directories not found under {root!r}: {missing}")
    return files


def resolve_single_data(
    data_config: "DataConfig",
    *,
    include_test: bool = True,
) -> ResolvedDataPaths:
    """Resolve one dataset worth of train/val/(optional)test files."""
    mode = _validate_single_source(data_config)

    if mode == "dataset":
        root = _resolve_dataset_root(data_config)
        train_path = root / "train.jsonl"
        val_path = root / "val.jsonl"
        test_path = root / "test.jsonl" if include_test else None
        if not train_path.exists():
            raise ValueError(
                f"data.dataset={data_config.dataset!r} resolved to '{root}' but train.jsonl was not found."
            )
        if not val_path.exists():
            raise ValueError(
                f"data.dataset={data_config.dataset!r} resolved to '{root}' but val.jsonl was not found."
            )
        if include_test and test_path is not None and not test_path.exists():
            test_path = None
        return ResolvedDataPaths(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            dataset_id=_requested_dataset_id(data_config, root),
            source_root=root,
        )

    train_path = _maybe_path(getattr(data_config, "train_path", None))
    val_path = _maybe_path(getattr(data_config, "val_path", None))
    test_path = _maybe_path(getattr(data_config, "test_path", None)) if include_test else None
    assert train_path is not None
    assert val_path is not None
    return ResolvedDataPaths(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        dataset_id=train_path.stem,
        source_root=train_path.parent,
    )


def resolve_benchmark_data(data_config: "DataConfig") -> ResolvedBenchmarkData:
    """Resolve a benchmark splits directory into loaded split tuples."""
    mode = _validate_benchmark_source(data_config)
    if mode == "dataset":
        splits_dir = _resolve_dataset_root(data_config)
        if (splits_dir / "train.jsonl").exists():
            dataset_id = _requested_dataset_id(data_config, splits_dir)
            datasets = _dataset_filter(data_config)
            if datasets not in (None, [], [dataset_id]):
                raise ValueError(
                    f"Requested datasets {datasets!r}, but '{data_config.dataset}' resolves to one dataset: {dataset_id!r}."
                )
            test_path = splits_dir / "test.jsonl"
            files = {
                dataset_id: {
                    "train": splits_dir / "train.jsonl",
                    "val": splits_dir / "val.jsonl",
                    "test": test_path if test_path.exists() else None,
                }
            }
            splits = {
                dataset_id: (
                    load_jsonl(splits_dir / "train.jsonl"),
                    load_jsonl(splits_dir / "val.jsonl"),
                    load_jsonl(test_path) if test_path.exists() else None,
                )
            }
            return ResolvedBenchmarkData(splits_dir=splits_dir, splits=splits, files=files)
    else:
        splits_dir = _maybe_path(getattr(data_config, "splits_dir", None))
        assert splits_dir is not None
    files = _resolve_benchmark_files(splits_dir, datasets=_dataset_filter(data_config))
    splits = load_splits_dir(str(splits_dir), datasets=_dataset_filter(data_config))
    return ResolvedBenchmarkData(splits_dir=splits_dir, splits=splits, files=files)


def resolve_data_source(
    data_config: "DataConfig",
    *,
    mode: Literal["single", "benchmark"] = "single",
    include_test: bool = True,
) -> ResolvedDataPaths | ResolvedBenchmarkData:
    """Resolve one of the supported data-source modes from a DataConfig."""
    if mode == "single":
        return resolve_single_data(data_config, include_test=include_test)
    if mode == "benchmark":
        return resolve_benchmark_data(data_config)
    raise ValueError(f"Unknown data resolution mode {mode!r}.")


__all__ = [
    "ResolvedBenchmarkData",
    "ResolvedDataPaths",
    "build_benchmark_data_provenance",
    "build_single_data_provenance",
    "resolve_benchmark_data",
    "resolve_data_source",
    "resolve_single_data",
]
