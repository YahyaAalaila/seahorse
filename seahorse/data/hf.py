"""Released Seahorse datasets on the Hugging Face Hub, with hash verification.

The benchmark datasets live under https://huggingface.co/seahorse-stpp. Each
repository is pinned here to the commit used for the paper release; the
``split_sha256`` block of a repository's dataset card is the source of truth
for the split hashes, and :func:`load_released_dataset` verifies every
downloaded split against it before parsing.

Two repositories are *script-only*: their licences do not permit hosting the
data, so the repository ships a rebuild script and the expected hashes instead.
:func:`load_released_dataset` raises :class:`ScriptOnlyDatasetError` for them
unless ``local_root`` points at a locally rebuilt copy, which is then verified
against the card in the same way.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from seahorse.utils import load_jsonl

HF_ORG = "seahorse-stpp"
_SPLIT_FILES = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}
_SPLIT_ALIASES = {"validation": "val", "valid": "val", "dev": "val"}


class ScriptOnlyDatasetError(RuntimeError):
    """The repository ships a rebuild script, not the data."""


class HashMismatchError(RuntimeError):
    """A downloaded split does not match the hash declared on the dataset card."""


@dataclass(frozen=True)
class ReleasedDataset:
    """One repository of the ``seahorse-stpp`` organisation."""

    name: str
    repo: str
    revision: str
    configs: tuple[str, ...]
    default_config: str
    license: str
    form: str = "data"  # "data" | "script-only"
    layout: str = "config_dirs"  # "config_dirs" (<config>/<split>.jsonl) | "root" (<split>.jsonl)
    legacy: bool = False
    in_paper: bool = True
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def repo_id(self) -> str:
        return f"{HF_ORG}/{self.repo}"

    def split_path(self, config: str, split: str) -> str:
        split = _SPLIT_ALIASES.get(split, split)
        if split not in _SPLIT_FILES:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(_SPLIT_FILES)}.")
        if config not in self.configs:
            raise ValueError(f"{self.name}: unknown config {config!r}; available: {list(self.configs)}.")
        if self.layout == "root":
            return _SPLIT_FILES[split]
        return f"{config}/{_SPLIT_FILES[split]}"


_PAPER_URBAN = ("jittered_seed20260910", "native")

RELEASED_DATASETS: dict[str, ReleasedDataset] = {
    # Table 3 (prepared real-data corpora of the camera-ready campaign)
    "comcat_california": ReleasedDataset(
        "comcat_california", "comcat_california_stpp", "ee4af920003787de209bb54f2beaf4b1b1807c6b",
        ("native",), "native", "public domain (USGS ANSS ComCat)",
        notes="EarthquakeNPP-derived California catalog; native time stamps only."),
    "geonet_nz": ReleasedDataset(
        "geonet_nz", "geonet_nz_stpp", "ce22fb703f0d7ee39f32ffd3451ab3de59dc8663",
        _PAPER_URBAN, "jittered_seed20260910", "CC BY 3.0 NZ (GeoNet)"),
    "austin311": ReleasedDataset(
        "austin311", "austin_311_stpp", "df368263d93c71cc5e86561054046edcc9cc7928",
        _PAPER_URBAN, "jittered_seed20260910", "public domain (City of Austin)"),
    "nyc311": ReleasedDataset(
        "nyc311", "nyc311_stpp", "03762e4bd7d199bb9cefd757a597771d20a1772e",
        _PAPER_URBAN, "jittered_seed20260910", "NYC Open Data terms of use"),
    "nyc_collisions": ReleasedDataset(
        "nyc_collisions", "nyc_collisions_stpp", "f2656bcef715e3bc960bd0f625a0b33e1a837410",
        _PAPER_URBAN, "jittered_seed20260910", "NYC Open Data terms of use"),
    "chicago_crime": ReleasedDataset(
        "chicago_crime", "chicago_crime_stpp", "84c304c901013c19214c8f210fe7d60d675989ea",
        _PAPER_URBAN, "jittered_seed20260910", "City of Chicago data terms of use"),
    "la_crime": ReleasedDataset(
        "la_crime", "la_crime_stpp", "3a24edf6e1f473de6d4de03315ae7fc8cb2c2da0",
        _PAPER_URBAN, "jittered_seed20260910", "CC0 1.0 (LAPD)"),
    "uber_nyc": ReleasedDataset(
        "uber_nyc", "uber_pickups_nyc_stpp", "cd0a0bdd2bc5acb37f01ad9886576be1fbd67916",
        _PAPER_URBAN, "jittered_seed20260910", "CC0 1.0 (FiveThirtyEight / NYC TLC FOIL)"),
    "seattle_fire911": ReleasedDataset(
        "seattle_fire911", "seattle_fire911_stpp", "739d52ceaf9547fcce6fb85b2ee35b967c115362",
        _PAPER_URBAN, "jittered_seed20260910", "script MIT; data public domain (City of Seattle), not hosted",
        form="script-only",
        notes="Address-level emergency call types: rebuild locally with the repository's rebuild/rebuild.py."),
    "citibike_nyc": ReleasedDataset(
        "citibike_nyc", "nyc_bikeshare_2019_stpp", "4c25097611ee3e6226a7c82343ed6a52ae689617",
        _PAPER_URBAN, "jittered_seed20260910", "script MIT; data under the Citi Bike data licence, not hosted",
        form="script-only",
        notes="Rebuild locally with the repository's rebuild/rebuild.py; verify with verify.py."),
    "synthetic": ReleasedDataset(
        "synthetic", "seahorse_synthetic_stpp", "4563558bae8ff0bb4349df001bb7df62d023b616",
        ("homogeneous_poisson", "moving_seasonal_poisson", "gamma_renewal"), "homogeneous_poisson",
        "CC BY 4.0 (Seahorse)", notes="Synthetic controls with analytic ground truth."),
    # Table 2 (legacy Neural-STPP benchmark splits)
    "covid": ReleasedDataset(
        "covid", "covid-stpp", "597ca57c801dbb87b9a903083ec45e1a0a23330d",
        ("default",), "default", "CC BY-NC 4.0 (The New York Times)", layout="root", legacy=True,
        notes="Official Neural-STPP split semantics; validation/test windows lie inside the training period."),
    "earthquakes": ReleasedDataset(
        "earthquakes", "earthquakes-stpp", "adb9ea0066f9d8d1bb861a47e53c2afa118d473b",
        ("default",), "default", "public domain (USGS)", layout="root", legacy=True,
        notes="Official Neural-STPP split semantics; some test windows overlap training windows in time."),
    "citibike_legacy": ReleasedDataset(
        "citibike_legacy", "nyc_bikeshare_legacy_stpp", "51adba249f62cce395c2a8f7930fa551c4776e17",
        ("citibike_stpp_days",), "citibike_stpp_days", "script MIT; data under the Citi Bike data licence, not hosted",
        form="script-only", legacy=True,
        notes="Legacy Neural-STPP bike-share benchmark reconstructed by the repository's rebuild script (times in days)."),
    # Released but outside the paper
    "bold5000": ReleasedDataset(
        "bold5000", "bold5000-stpp", "dd5787d12d81faa33a28d9f9d08bd8767857b98d",
        ("default",), "default", "CC0 1.0", layout="root", in_paper=False),
    "us_wildfires": ReleasedDataset(
        "us_wildfires", "us_wildfires_stpp", "ddae6ff72d1c22ab16120f7351c5a33cac9f6394",
        ("default",), "default", "USFS RDS-2013-0009.4 (no permission required)", layout="root", in_paper=False),
    "us_accidents": ReleasedDataset(
        "us_accidents", "us_accidents_stpp", "3b9693f8feda93a0a26bf13956def5c966a48949",
        ("default",), "default", "CC BY-NC-SA 4.0 (Moosavi et al.)", layout="root", in_paper=False),
}


def list_released_datasets(*, in_paper: bool | None = None) -> list[str]:
    """Short names of the released datasets (optionally only those used in the paper)."""
    return [n for n, d in RELEASED_DATASETS.items() if in_paper is None or d.in_paper == in_paper]


def released_dataset(name: str) -> ReleasedDataset:
    if name in RELEASED_DATASETS:
        return RELEASED_DATASETS[name]
    for spec in RELEASED_DATASETS.values():
        if name in (spec.repo, spec.repo_id):
            return spec
    raise KeyError(f"Unknown released dataset {name!r}; available: {sorted(RELEASED_DATASETS)}.")


# ---------------------------------------------------------------------------
# Card hashes
# ---------------------------------------------------------------------------


def sha256_of(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_card_front_matter(card_text: str) -> dict[str, Any]:
    """Return the YAML front matter of a dataset card as a dict (``{}`` if absent)."""
    match = re.match(r"^---\n(.*?)\n---", card_text, re.S)
    if not match:
        return {}
    loaded = yaml.safe_load(match.group(1))
    return loaded if isinstance(loaded, dict) else {}


def card_split_hashes(card_text: str) -> dict[str, dict[str, str]]:
    """``{config: {split: sha256}}`` declared under ``seahorse_release.split_sha256``."""
    meta = parse_card_front_matter(card_text)
    release = meta.get("seahorse_release") or {}
    hashes = release.get("split_sha256") or {}
    out: dict[str, dict[str, str]] = {}
    for config, splits in hashes.items():
        if isinstance(splits, dict):
            out[str(config)] = {str(k): str(v) for k, v in splits.items()}
    return out


def _hub_download(repo_id: str, filename: str, *, revision: str, cache_dir=None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision, cache_dir=cache_dir))


def fetch_card(spec: ReleasedDataset, *, revision: str | None = None, cache_dir=None) -> str:
    """Download the dataset card (README.md) at the pinned revision."""
    return _hub_download(spec.repo_id, "README.md", revision=revision or spec.revision, cache_dir=cache_dir).read_text()


def verify_split_file(path: os.PathLike[str] | str, expected_sha256: str | None, *, label: str) -> dict[str, Any]:
    """Compare a split file with the card hash; ``None`` means the card declares none."""
    actual = sha256_of(path)
    if expected_sha256 is None:
        return {"path": str(path), "sha256": actual, "verified": False, "reason": "card declares no split_sha256"}
    if actual != expected_sha256:
        raise HashMismatchError(f"{label}: sha256 {actual} does not match the card hash {expected_sha256}.")
    return {"path": str(path), "sha256": actual, "verified": True}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def download_released_split(
    name: str,
    split: str = "train",
    *,
    config: str | None = None,
    revision: str | None = None,
    cache_dir=None,
    local_root: os.PathLike[str] | str | None = None,
    verify: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Return the local path of one split and its verification record.

    ``local_root`` points at a locally rebuilt copy (script-only repositories)
    laid out as ``<local_root>/<config>/<split>.jsonl``; the card hashes are
    still fetched from the Hub for verification.
    """
    spec = released_dataset(name)
    config = config or spec.default_config
    relative = spec.split_path(config, split)
    split_key = _SPLIT_ALIASES.get(split, split)
    expected = None
    if verify:
        hashes = card_split_hashes(fetch_card(spec, revision=revision, cache_dir=cache_dir))
        expected = hashes.get(config, {}).get(split_key)
    if local_root is not None:
        path = Path(local_root) / (relative if spec.layout == "config_dirs" else _SPLIT_FILES[split_key])
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path} not found under local_root.")
    elif spec.form == "script-only":
        raise ScriptOnlyDatasetError(
            f"{name} ({spec.repo_id}) does not host data: {spec.notes} "
            "Then pass local_root=<directory of the rebuilt splits> to verify and load it."
        )
    else:
        path = _hub_download(spec.repo_id, relative, revision=revision or spec.revision, cache_dir=cache_dir)
    record = (
        verify_split_file(path, expected, label=f"{name}:{config}:{split_key}")
        if verify else {"path": str(path), "verified": False, "reason": "verification disabled"}
    )
    record.update({"dataset": name, "repo": spec.repo_id, "revision": revision or spec.revision,
                   "config": config, "split": split_key})
    return path, record


def load_released_dataset(
    name: str,
    *,
    config: str | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    revision: str | None = None,
    cache_dir=None,
    local_root: os.PathLike[str] | str | None = None,
    verify: bool = True,
) -> dict[str, list[dict]]:
    """Load the requested splits of a released dataset as lists of canonical sequence records.

    Every split is verified against the hash on the dataset card (raising
    :class:`HashMismatchError` on disagreement); legacy cards without hashes are
    loaded with ``verified=False`` in the returned ``__provenance__`` entry.
    """
    out: dict[str, Any] = {}
    provenance = []
    for split in splits:
        path, record = download_released_split(
            name, split, config=config, revision=revision, cache_dir=cache_dir, local_root=local_root, verify=verify,
        )
        out[record["split"]] = load_jsonl(path)
        provenance.append(record)
    out["__provenance__"] = provenance
    return out


__all__ = [
    "HF_ORG",
    "HashMismatchError",
    "RELEASED_DATASETS",
    "ReleasedDataset",
    "ScriptOnlyDatasetError",
    "card_split_hashes",
    "download_released_split",
    "fetch_card",
    "list_released_datasets",
    "load_released_dataset",
    "parse_card_front_matter",
    "released_dataset",
    "sha256_of",
    "verify_split_file",
]
