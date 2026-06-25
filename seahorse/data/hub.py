"""Minimal dataset hub for curated and local JSONL datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from seahorse.data.contract import validate_sequence_records
from seahorse.utils import load_jsonl


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HF_DATASET_REPO = os.environ.get("SEAHORSE_HF_DATASET_REPO")
_PROJECT_HF_DATASET_ORG = os.environ.get("SEAHORSE_HF_DATASET_ORG", "seahorse-stpp")
_SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class CuratedDatasetSpec:
    """Small frozen spec for one curated dataset path."""

    name: str
    local_paths: tuple[str, ...] = ()
    repo_path: str = ""
    repo_id: str | None = _DEFAULT_HF_DATASET_REPO
    revision: str | None = None
    aliases: tuple[str, ...] = ()


def _candidate_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _leaf_specs(
    collection_name: str,
    repo_dir: str,
    leaves: tuple[str, ...],
) -> list[CuratedDatasetSpec]:
    specs = [
        CuratedDatasetSpec(
            name=collection_name,
            repo_path=repo_dir,
            aliases=(repo_dir,),
        )
    ]
    for leaf in leaves:
        aliases: list[str] = [f"{repo_dir}/{leaf}"]
        if leaf.startswith("entrt_"):
            aliases.append(f"{collection_name}/{leaf.replace('entrt_', 'ent_rt_', 1)}")
        specs.append(
            CuratedDatasetSpec(
                name=f"{collection_name}/{leaf}",
                repo_path=f"{repo_dir}/{leaf}",
                aliases=tuple(aliases),
            )
        )
    return specs


def _build_catalog() -> dict[str, CuratedDatasetSpec]:
    specs = [
        CuratedDatasetSpec(
            name="sthp0",
            repo_path="sthp0",
        ),
    ]
    specs.extend(
        _leaf_specs(
            "hawkesnest_v20260409",
            "seahorse_hawkesnest_v20260409",
            tuple(
                [f"pulse_P{i}" for i in range(6)]
                + [f"echo_E{i}" for i in range(6)]
                + [f"bg_BG{i}" for i in range(6)]
            ),
        )
    )
    specs.extend(
        _leaf_specs(
            "hawkesnest_families_v1",
            "seahorse_hawkesnest_families_v1",
            tuple(
                [f"moving_M{i}" for i in range(6)]
                + [f"regime_R{i}" for i in range(6)]
                + [f"entrt_E{i}" for i in range(6)]
            ),
        )
    )
    specs.extend(
        _leaf_specs(
            "hawkesnest_hard_v2",
            "seahorse_hawkesnest_hard_v2",
            tuple(
                [f"pulse_P{i}" for i in range(6)]
                + [f"echo_E{i}" for i in range(6)]
                + [f"regime_R{i}" for i in range(6)]
                + [f"topology_T{i}" for i in range(6)]
            ),
        )
    )
    specs.extend(_project_dataset_specs())

    catalog: dict[str, CuratedDatasetSpec] = {}
    for spec in specs:
        for key in (spec.name, *spec.aliases):
            if key in catalog:
                raise ValueError(f"Duplicate curated dataset key '{key}'.")
            catalog[key] = spec
    return catalog


def _project_dataset_specs() -> list[CuratedDatasetSpec]:
    datasets = (
        (
            "austin_311_stpp",
            "806f06a83dd821fda1702f1aa71b1fbeadbff728",
            ("austin_311",),
        ),
        (
            "uber_pickups_nyc_stpp",
            "5de1bb62d750c7a225a736a3cb36674bb61e3f70",
            ("uber_pickups_nyc",),
        ),
        (
            "us_wildfires_stpp",
            "82e84d9b2d2d0197664e26b3e77d96084c75f368",
            ("us_wildfires",),
        ),
        (
            "gtd_stpp",
            "379d8d3de49fdecf8dee797fc3133245d6d8eeec",
            ("gtd",),
        ),
        (
            "us_accidents_stpp",
            "20ed5757a52f0b1d67c0ee9cf428a98fa8b1c886",
            ("us_accidents",),
        ),
        (
            "la_crime_stpp",
            "1225c94f924839f2e6216d28e35fcd9ddccae370",
            ("la_crime",),
        ),
        (
            "brightkite_checkins_stpp",
            "d2e9079a672fd8ed762f2816e2f73a4788b3f95c",
            ("brightkite_checkins",),
        ),
        (
            "chicago_crime_stpp",
            "e6dc2c9edc427fac98b61ef181c750ae0b2bb818",
            ("chicago_crime",),
        ),
        (
            "gowalla_checkins_stpp",
            "18b615fa840e6f92511c350522c762bcf351d0ec",
            ("gowalla_checkins",),
        ),
        (
            "earthquakes-stpp",
            "b8a95e944c696bfe15a9684cc8532d6f0598d648",
            ("earthquakes_stpp", "earthquakes"),
        ),
        (
            "citibike-stpp",
            "3e058400fd46eb0995e39d3acf038069d05d2122",
            ("citibike_stpp", "citibike"),
        ),
        (
            "covid-stpp",
            "2c235058853d4920a92c42dafb49e1188ff08f86",
            ("covid_stpp", "covid"),
        ),
        (
            "bold5000-stpp",
            "928e9cbc6df930e7d895844d64d3f273eed05b69",
            ("bold5000_stpp", "bold5000"),
        ),
    )
    return [
        CuratedDatasetSpec(
            name=name,
            repo_id=f"{_PROJECT_HF_DATASET_ORG}/{name}",
            revision=revision,
            aliases=aliases,
        )
        for name, revision, aliases in datasets
    ]


_CURATED_DATASETS = _build_catalog()


def _split_paths(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for split_name in _SPLIT_NAMES:
        split_path = root / f"{split_name}.jsonl"
        if split_path.exists():
            paths[split_name] = split_path
    return paths


def _normalize_split(split: str) -> str:
    value = str(split).strip().lower()
    if value not in _SPLIT_NAMES:
        raise ValueError(
            f"Unknown split '{split}'. Expected one of {list(_SPLIT_NAMES)}."
        )
    return value


def _resolve_local_input(name: str | os.PathLike[str]) -> Path | None:
    path = Path(name).expanduser()
    if path.exists():
        return path.resolve()
    return None


def _parse_hf_dataset_ref(name: str | os.PathLike[str]) -> tuple[str, str] | None:
    raw = os.fspath(name).strip().strip("/")
    parts = [part for part in raw.split("/") if part]
    if len(parts) < 2:
        return None
    repo_id = "/".join(parts[:2])
    repo_path = "/".join(parts[2:])
    return repo_id, repo_path


def _snapshot_repo_path(
    *,
    repo_id: str,
    repo_path: str = "",
    revision: str | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
) -> Path:
    allow_patterns = [f"{repo_path}/**"] if repo_path else None
    try:
        snapshot_root = Path(
            _snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=allow_patterns,
                force_download=force_download,
                local_files_only=local_files_only,
            )
        ).resolve()
    except Exception as exc:
        raise FileNotFoundError(
            f"Failed to resolve Hugging Face dataset repo '{repo_id}'."
        ) from exc

    resolved = snapshot_root / repo_path if repo_path else snapshot_root
    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset repo '{repo_id}' was downloaded but the resolved path does not exist: {resolved}"
        )
    return resolved


def _load_records(path: Path, *, validate: bool) -> list[dict]:
    records = load_jsonl(path)
    if validate:
        validate_sequence_records(records, source=str(path))
    return records


def _snapshot_download(**kwargs) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "download_dataset requires 'huggingface_hub' for Hugging Face-backed "
            "dataset downloads."
        ) from exc
    return snapshot_download(**kwargs)


def download_dataset(
    name: str | os.PathLike[str],
    *,
    revision: str | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
) -> Path:
    """Resolve a dataset path from a local path, curated fallback, or HF cache."""
    local_input = _resolve_local_input(name)
    if local_input is not None:
        return local_input

    key = os.fspath(name)
    spec = _CURATED_DATASETS.get(key)
    if spec is None:
        direct_hf = _parse_hf_dataset_ref(key)
        if direct_hf is not None:
            repo_id, repo_path = direct_hf
            return _snapshot_repo_path(
                repo_id=repo_id,
                repo_path=repo_path,
                revision=revision,
                force_download=force_download,
                local_files_only=local_files_only,
            )
        raise ValueError(
            f"Unknown dataset '{key}'. Pass a local file/directory path, a curated dataset name, "
            "or a Hugging Face dataset repo id like 'owner/repo[/subdir]'."
        )

    if not force_download:
        for candidate in spec.local_paths:
            path = _candidate_path(candidate)
            if path.exists():
                return path

    if spec.repo_id is None:
        raise FileNotFoundError(
            f"Dataset '{key}' was not found locally and no Hugging Face repo is configured "
            "for this curated entry."
        )

    return _snapshot_repo_path(
        repo_id=spec.repo_id,
        repo_path=spec.repo_path.strip("/"),
        revision=revision if revision is not None else spec.revision,
        force_download=force_download,
        local_files_only=local_files_only,
    )


def load_dataset(
    name: str | os.PathLike[str],
    *,
    split: str | None = None,
    validate: bool = True,
) -> list[dict] | dict[str, list[dict]]:
    """Load one split or all available splits from a local or curated dataset."""
    path = download_dataset(name)

    if path.is_file():
        resolved_split = _normalize_split(split) if split is not None else None
        if (
            resolved_split is not None
            and path.stem in _SPLIT_NAMES
            and path.stem != resolved_split
        ):
            raise ValueError(
                f"Requested split '{resolved_split}' does not match file path '{path.name}'."
            )
        return _load_records(path, validate=validate)

    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    split_paths = _split_paths(path)
    if not split_paths:
        child_datasets = sorted(
            child.name
            for child in path.iterdir()
            if child.is_dir() and (child / "train.jsonl").exists()
        )
        hint = ""
        if child_datasets:
            preview = ", ".join(child_datasets[:5])
            suffix = "..." if len(child_datasets) > 5 else ""
            hint = f" Available child datasets: {preview}{suffix}"
        raise ValueError(
            f"Dataset path '{path}' does not contain train/val/test split files.{hint}"
        )

    if split is not None:
        split_name = _normalize_split(split)
        split_path = split_paths.get(split_name)
        if split_path is None:
            raise FileNotFoundError(
                f"Dataset '{name}' does not provide split '{split_name}'."
            )
        return _load_records(split_path, validate=validate)

    return {
        split_name: _load_records(split_path, validate=validate)
        for split_name, split_path in split_paths.items()
    }


__all__ = ["CuratedDatasetSpec", "download_dataset", "load_dataset"]
