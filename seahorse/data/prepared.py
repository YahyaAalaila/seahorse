"""Adapter for the immutable, externally prepared real-data corpora.

The prepared corpora live outside the repository; their root directory is
given by ``SEAHORSE_PREPARED_DATA_ROOT`` or the ``root`` argument.  This
module resolves only their existing JSONL variants and manifests: it never
downloads raw data, rewrites a corpus, or creates event chunks.  In particular,
``native`` and the declared strict-order jittered view are separate dataset IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from seahorse.data.contract import validate_sequence_records
from seahorse.utils import load_jsonl


PREPARED_ID_PREFIX = "prepared:"
PREPARED_DATA_ROOT_ENV = "SEAHORSE_PREPARED_DATA_ROOT"
_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PreparedCorpusSpec:
    """Static campaign role and physical corpus name for a prepared dataset."""

    name: str
    corpus: str
    role: Literal["recommended", "reserve", "scale_companion"]
    views: tuple[str, ...]


# The recommended core deliberately excludes the two Foursquare corpora and
# full-US FPA.  They are named, resolvable reserve/scale routes rather than
# silently joining the core campaign.
PREPARED_CORPORA: tuple[PreparedCorpusSpec, ...] = (
    PreparedCorpusSpec("citibike_nyc", "citibike_nyc_2019_apr_aug", "recommended", ("native", "jittered_seed20260910")),
    PreparedCorpusSpec("comcat_california", "earthquakenpp_comcat_california_native", "recommended", ("native",)),
    PreparedCorpusSpec("geonet_nz", "geonet_nz_2018_2019_native", "recommended", ("native", "jittered_seed20260910")),
    PreparedCorpusSpec("fpa_california", "fpa_fod_ca_1992_2024_native_day", "recommended", ("native",)),
    PreparedCorpusSpec("uber_nyc", "uber_nyc_2014", "recommended", ("native", "jittered_seed20260910")),
    PreparedCorpusSpec("austin311", "urban_austin311_2023", "recommended", ("native", "jittered_seed20260910")),
    PreparedCorpusSpec("chicago_crime", "urban_chicago_crime_2023", "recommended", ("native", "jittered_seed20260910")),
    PreparedCorpusSpec("la_crime", "urban_la_crime_2023", "recommended", ("native", "jittered_seed20260910")),
    PreparedCorpusSpec("nyc311", "urban_nyc311_2023", "recommended", ("native", "jittered_seed20260910")),
    PreparedCorpusSpec("nyc_collisions", "urban_nyc_collisions_2023", "recommended", ("native", "jittered_seed20260910")),
    PreparedCorpusSpec("seattle_fire911", "urban_seattle_fire911_2023", "recommended", ("native", "jittered_seed20260910")),
    PreparedCorpusSpec("foursquare_nyc", "foursquare_nyc_2012_2013", "reserve", ("native", "jittered_seed20260910")),
    PreparedCorpusSpec("foursquare_tokyo", "foursquare_tky_2012_2013", "reserve", ("native", "jittered_seed20260910")),
    PreparedCorpusSpec("fpa_us_scale", "fpa_fod_us_1992_2024_native_day", "scale_companion", ("native",)),
)
_CORPUS_BY_NAME = {spec.name: spec for spec in PREPARED_CORPORA}

# Declared in the source-faithful civic builder as municipality validity
# envelopes, not fitted from event locations.  The checksum binds these values
# to the exact preprocessing script archived with the prepared corpora.
_URBAN_VALIDITY_ENVELOPES = {
    "urban_austin311_2023": (-98.0, -97.4, 30.0, 30.6),
    "urban_seattle_fire911_2023": (-122.50, -122.20, 47.45, 47.80),
    "urban_chicago_crime_2023": (-88.00, -87.40, 41.60, 42.10),
    "urban_la_crime_2023": (-118.75, -118.10, 33.65, 34.35),
    "urban_nyc_collisions_2023": (-74.30, -73.65, 40.45, 40.95),
    "urban_nyc311_2023": (-74.30, -73.65, 40.45, 40.95),
}
_URBAN_VALIDITY_ENVELOPE_BUILDER_SHA256 = (
    "adcd2d999268d6e0566bf21b41de6f79773b58382a8dc38a3fb25721a285c862"
)


def prepared_data_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Return the configured prepared-data root without creating it.

    The root is the ``root`` argument when given, otherwise the
    ``SEAHORSE_PREPARED_DATA_ROOT`` environment variable.
    """
    value = root if root is not None else os.environ.get(PREPARED_DATA_ROOT_ENV)
    if not value:
        raise RuntimeError(
            "Prepared datasets need a root directory: pass root=... or set "
            f"{PREPARED_DATA_ROOT_ENV} to the directory holding the prepared corpora."
        )
    return Path(value).expanduser()


def is_prepared_dataset_id(name: str | os.PathLike[str]) -> bool:
    return os.fspath(name).startswith(PREPARED_ID_PREFIX)


def parse_prepared_dataset_id(name: str | os.PathLike[str]) -> tuple[PreparedCorpusSpec, str]:
    """Parse ``prepared:<short-name>:<native|jittered_seed20260910>``."""
    raw = os.fspath(name)
    if not raw.startswith(PREPARED_ID_PREFIX):
        raise ValueError(
            f"Prepared dataset IDs must begin with {PREPARED_ID_PREFIX!r}, got {raw!r}."
        )
    parts = raw.split(":")
    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise ValueError(
            "Prepared dataset ID must be 'prepared:<cohort>:<view>'; "
            f"got {raw!r}."
        )
    spec = _CORPUS_BY_NAME.get(parts[1])
    if spec is None:
        raise ValueError(
            f"Unknown prepared cohort {parts[1]!r}. Available: {sorted(_CORPUS_BY_NAME)}"
        )
    view = parts[2]
    if view not in spec.views:
        raise ValueError(
            f"Prepared cohort {spec.name!r} does not expose view {view!r}; "
            f"available views: {list(spec.views)}."
        )
    return spec, view


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Prepared corpus manifest was not found: {path}") from None
    if not isinstance(data, dict):
        raise ValueError(f"Prepared corpus manifest must contain a JSON object: {path}")
    return data


def _manifest_variant(manifest: dict[str, Any], view: str) -> dict[str, Any]:
    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise ValueError("Prepared corpus manifest has no list-valued 'variants' field.")
    for variant in variants:
        if isinstance(variant, dict) and variant.get("name") == view:
            return variant
    names = [item.get("name") for item in variants if isinstance(item, dict)]
    raise ValueError(f"Prepared corpus manifest does not declare view {view!r}; found {names!r}.")


def _observation_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    cleaning = dict(manifest.get("cleaning") if isinstance(manifest.get("cleaning"), dict) else {})
    cleaning["_prepared_dataset_name"] = manifest.get("dataset_name")
    native_qc = manifest.get("native_qc") if isinstance(manifest.get("native_qc"), dict) else {}
    # These are direct manifest fields, not an inferred temporal resolution.
    return {
        "time_axis": source.get("time_axis"),
        "source_timezone": source.get("source_timezone"),
        "timestamp_semantics": source.get("timestamp_semantics"),
        "calendar_date_convention": source.get("calendar_date_convention"),
        "time_axis_warning": source.get("time_axis_warning"),
        "spatial_support": source.get("spatial_support"),
        "spatial_domain": _declared_spatial_domain(source, cleaning),
        "native_timestamp_qc": {
            key: native_qc.get(key)
            for key in (
                "adjacent_tie_fraction",
                "events_in_tied_groups_fraction",
                "fraction_with_subsecond",
                "fraction_on_minute",
                "fraction_on_hour",
                "fraction_on_local_midnight",
                "fraction_on_utc_midnight",
                "positive_gap_seconds_p05_p50_p95",
                "max_timestamp_multiplicity",
            )
            if key in native_qc
        },
    }


def _declared_spatial_domain(source: dict[str, Any], cleaning: dict[str, Any]) -> dict[str, Any]:
    """Return only a declared support/filter rectangle, never event extrema.

    The returned rectangle is a reproducible candidate for a finite-domain
    model when its ``kind`` says so.  It is intentionally separate from the
    source's physical observation footprint (for example, station or venue
    support) and must not be replaced with a train/test coordinate extent.
    """
    # Dataset name is injected by the caller after reading the manifest.
    dataset_name = cleaning.get("_prepared_dataset_name")
    urban_bounds = _URBAN_VALIDITY_ENVELOPES.get(dataset_name)
    if urban_bounds is not None:
        lo, hi, bottom, top = urban_bounds
        return {
            "kind": "declared_city_validity_envelope",
            "longitude": [lo, hi],
            "latitude": [bottom, top],
            "builder_sha256": _URBAN_VALIDITY_ENVELOPE_BUILDER_SHA256,
            "configuration": "Use only as the source-preprocessing municipality validity envelope, not as an inferred physical observation boundary or empirical event extent.",
        }
    bbox = cleaning.get("bounding_box_lon_lat")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return {
            "kind": "declared_preprocessing_rectangle",
            "longitude": [bbox[0], bbox[1]],
            "latitude": [bbox[2], bbox[3]],
            "configuration": "Use only as the declared preprocessing rectangle; retain source spatial_support semantics.",
        }
    filters = cleaning.get("filters")
    if isinstance(filters, dict):
        longitude = filters.get("longitude_unwrapped_degreesE", filters.get("longitude"))
        latitude = filters.get("latitude")
        if isinstance(longitude, (list, tuple)) and isinstance(latitude, (list, tuple)) and len(longitude) == len(latitude) == 2:
            return {
                "kind": "declared_preprocessing_rectangle",
                "longitude": list(longitude),
                "latitude": list(latitude),
                "longitude_coordinate_convention": source.get("coordinate_convention"),
                "configuration": "Use the declared preprocessing rectangle; where source.region names a polygon, do not claim this rectangle is that polygon.",
            }
    if cleaning.get("state_filter") is not None:
        return {
            "kind": "declared_nonrectangular_filter",
            "filter": {"state": cleaning.get("state_filter")},
            "configuration": "No rectangle is declared. A finite rectangular likelihood domain requires a separately versioned domain artifact; do not use empirical event extrema.",
        }
    return {
        "kind": "not_declared",
        "configuration": "No finite rectangle is recorded in the prepared manifest. Block finite-domain likelihood configuration until a separately versioned domain artifact is supplied; do not infer from train or test extrema.",
    }


@dataclass(frozen=True)
class PreparedDatasetRoute:
    """Resolved immutable prepared view and its provenance/campaign gates."""

    dataset_id: str
    spec: PreparedCorpusSpec
    view: str
    root: Path
    corpus_root: Path
    view_root: Path
    manifest_path: Path
    manifest_sha256: str
    split_paths: dict[str, Path]
    split_sha256: dict[str, str]
    strictly_increasing_global_times: bool
    observation: dict[str, Any]
    mark_mapping_provenance: str

    @property
    def approximate_observation_model(self) -> bool:
        return self.view != "native"

    @property
    def exact_continuous_time_eligible(self) -> bool:
        """Whether the manifest supports *exact* continuous-time use.

        A strict jittered ordering is an explicit approximation, never evidence
        that the native tied observations had exact times.  A local-civil axis
        with an unresolved source warning is likewise not a physical UTC axis.
        """
        return bool(
            self.view == "native"
            and self.strictly_increasing_global_times
            and self.observation.get("time_axis") == "utc"
            and not self.observation.get("time_axis_warning")
        )

    def assert_continuous_time_eligible(
        self,
        *,
        exact: bool,
    ) -> None:
        """Gate exact or approximate continuous-time paths explicitly."""
        if exact and not self.exact_continuous_time_eligible:
            raise ValueError(
                f"{self.dataset_id} is not eligible for an exact continuous-time "
                "likelihood: require native strictly increasing observations on an "
                "unambiguous UTC axis. Use an explicitly labelled approximate view "
                "only when that observation model is scientifically appropriate."
            )
        if not exact and not self.strictly_increasing_global_times:
            raise ValueError(
                f"{self.dataset_id} has tied/non-strict native times in its manifest and "
                "cannot enter a strict continuous-time loader. Resolve a declared "
                "strict-order variant explicitly; do not silently dequantize it."
            )

    def machine_readable(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "corpus": self.spec.corpus,
            "role": self.spec.role,
            "view": self.view,
            "view_path": str(self.view_root),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "splits": {key: str(value) for key, value in self.split_paths.items()},
            "split_sha256": dict(self.split_sha256),
            "strictly_increasing_global_times": self.strictly_increasing_global_times,
            "continuous_time": {
                "exact_eligible": self.exact_continuous_time_eligible,
                "approximate_observation_model": self.approximate_observation_model,
            },
            "observation": self.observation,
            "mark_mapping_provenance": self.mark_mapping_provenance,
        }


def resolve_prepared_dataset(
    name: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
    verify_hashes: bool = False,
) -> PreparedDatasetRoute:
    """Resolve an immutable view, optionally checking manifest split hashes."""
    spec, view = parse_prepared_dataset_id(name)
    root_path = prepared_data_root(root)
    corpus_root = root_path / spec.corpus
    manifest_path = corpus_root / "dataset_meta.json"
    manifest = _read_manifest(manifest_path)
    if manifest.get("dataset_name") not in (None, spec.corpus):
        raise ValueError(
            f"Prepared manifest dataset_name={manifest.get('dataset_name')!r} does not "
            f"match requested corpus {spec.corpus!r}."
        )
    variant = _manifest_variant(manifest, view)
    declared_path = variant.get("path")
    view_root = Path(declared_path) if declared_path else corpus_root / view
    # A copied/symlinked corpus root must remain relocatable; use its local
    # view if it exists instead of trusting an absolute path from the manifest.
    local_view_root = corpus_root / view
    if local_view_root.exists():
        view_root = local_view_root
    split_paths = {split: view_root / f"{split}.jsonl" for split in _SPLITS}
    missing = [split for split, path in split_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Prepared view {name!r} is missing declared split file(s) {missing} under {view_root}."
        )
    hashes = variant.get("files_sha256")
    if not isinstance(hashes, dict) or any(f"{split}.jsonl" not in hashes for split in _SPLITS):
        raise ValueError(f"Prepared manifest for {name!r} has incomplete files_sha256 data.")
    split_sha256 = {split: str(hashes[f"{split}.jsonl"]) for split in _SPLITS}
    if verify_hashes:
        mismatches = {
            split: {"expected": split_sha256[split], "actual": _sha256_file(path)}
            for split, path in split_paths.items()
            if _sha256_file(path) != split_sha256[split]
        }
        if mismatches:
            raise ValueError(f"Prepared split checksum mismatch for {name!r}: {mismatches!r}")
    mark_map = manifest.get("categorical_mark_mapping")
    return PreparedDatasetRoute(
        dataset_id=os.fspath(name),
        spec=spec,
        view=view,
        root=root_path,
        corpus_root=corpus_root,
        view_root=view_root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        split_paths=split_paths,
        split_sha256=split_sha256,
        strictly_increasing_global_times=bool(variant.get("strictly_increasing_global_times")),
        observation=_observation_fields(manifest),
        mark_mapping_provenance=(
            "manifest_categorical_mark_mapping" if isinstance(mark_map, dict) else "not_recorded_in_manifest"
        ),
    )


def prepared_dataset_ids(*, roles: tuple[str, ...] | None = None) -> list[str]:
    """List every declared resolvable prepared ID without touching the filesystem."""
    return [
        f"{PREPARED_ID_PREFIX}{spec.name}:{view}"
        for spec in PREPARED_CORPORA
        if roles is None or spec.role in roles
        for view in spec.views
    ]


def prepared_catalog(
    *,
    root: str | os.PathLike[str] | None = None,
    verify_hashes: bool = False,
) -> dict[str, dict[str, Any]]:
    """Resolve all declared routes into a machine-readable campaign catalog."""
    return {
        dataset_id: resolve_prepared_dataset(
            dataset_id, root=root, verify_hashes=verify_hashes
        ).machine_readable()
        for dataset_id in prepared_dataset_ids()
    }


def _augment_record_provenance(record: dict[str, Any], route: PreparedDatasetRoute, split: str) -> dict[str, Any]:
    """Copy one record and add route-level provenance without altering events."""
    copied = dict(record)
    existing_metadata = copied.get("observation_metadata")
    metadata = dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
    metadata["prepared_route"] = {
        "dataset_id": route.dataset_id,
        "corpus": route.spec.corpus,
        "role": route.spec.role,
        "view": route.view,
        "split": split,
        "manifest_sha256": route.manifest_sha256,
        "split_sha256": route.split_sha256[split],
        "strictly_increasing_global_times": route.strictly_increasing_global_times,
        "exact_continuous_time_eligible": route.exact_continuous_time_eligible,
        "approximate_observation_model": route.approximate_observation_model,
    }
    copied["observation_metadata"] = metadata
    return copied


@dataclass(frozen=True)
class PreparedMarkMap:
    """A train-only reindexing artifact for already integer-encoded marks."""

    source_to_model: dict[int, int]
    unknown_index: int
    source_label_provenance: str

    @property
    def n_marks(self) -> int:
        return self.unknown_index + 1


def fit_train_mark_map(train_records: list[dict[str, Any]]) -> PreparedMarkMap | None:
    """Fit a stable source-index map using *only* the training split.

    Prepared urban manifests presently expose integer mark indices but do not
    preserve original category-name mapping provenance.  We therefore retain
    the source index in event metadata during transformation and reserve one
    model index for val/test categories absent from train; no category names
    are inferred.
    """
    values: set[int] = set()
    saw_marks = False
    for record in train_records:
        marks = record.get("marks")
        if marks is None:
            continue
        saw_marks = True
        for value in marks:
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"Prepared marks must be non-negative integer indices, got {value!r}.")
            values.add(int(value))
    if not saw_marks:
        return None
    mapping = {source: index for index, source in enumerate(sorted(values))}
    return PreparedMarkMap(
        source_to_model=mapping,
        unknown_index=len(mapping),
        source_label_provenance="integer source indices retained; category-name map not recorded in manifest",
    )


def apply_train_mark_map(records: list[dict[str, Any]], mark_map: PreparedMarkMap | None) -> list[dict[str, Any]]:
    """Return copied records with train-only model marks and preserved source indices."""
    if mark_map is None:
        return list(records)
    transformed: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        marks = copied.get("marks")
        if marks is None:
            transformed.append(copied)
            continue
        source_marks = [int(value) for value in marks]
        copied["marks"] = [mark_map.source_to_model.get(value, mark_map.unknown_index) for value in source_marks]
        event_metadata = dict(copied.get("event_metadata") or {})
        event_metadata["prepared_source_mark_index"] = source_marks
        copied["event_metadata"] = event_metadata
        transformed.append(copied)
    return transformed


@dataclass(frozen=True)
class PreparedSplits:
    route: PreparedDatasetRoute
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    mark_map: PreparedMarkMap | None


def load_prepared_splits(
    route: PreparedDatasetRoute,
    *,
    validate: bool = True,
    mark_policy: Literal["source_indices", "train_only"] = "train_only",
) -> PreparedSplits:
    """Load fixed splits with no event thinning and an explicit mark policy."""
    if mark_policy not in {"source_indices", "train_only"}:
        raise ValueError("mark_policy must be 'source_indices' or 'train_only'.")
    records = {
        split: [_augment_record_provenance(record, route, split) for record in load_jsonl(path)]
        for split, path in route.split_paths.items()
    }
    if validate:
        for split, split_records in records.items():
            validate_sequence_records(split_records, source=str(route.split_paths[split]))
    mark_map = fit_train_mark_map(records["train"]) if mark_policy == "train_only" else None
    if mark_policy == "train_only":
        records = {split: apply_train_mark_map(split_records, mark_map) for split, split_records in records.items()}
    return PreparedSplits(
        route=route,
        train=records["train"],
        val=records["val"],
        test=records["test"],
        mark_map=mark_map,
    )


def load_prepared_split_sample(
    route: PreparedDatasetRoute,
    split: Literal["train", "val", "test"],
    *,
    min_events: int = 3,
    validate: bool = True,
) -> dict[str, Any] | None:
    """Read one complete qualifying record for a bounded loader preflight.

    This deliberately does not truncate events or stand in for campaign
    coverage.  It is suitable only for an integration probe that needs one
    actual calendar window without materialising a multi-million-event corpus.
    """
    if split not in _SPLITS:
        raise ValueError(f"Unknown prepared split {split!r}; expected one of {_SPLITS}.")
    if min_events < 0:
        raise ValueError("min_events must be non-negative.")
    path = route.split_paths[split]
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if len(record.get("times", [])) < min_events:
                continue
            augmented = _augment_record_provenance(record, route, split)
            if validate:
                validate_sequence_records([augmented], source=f"{path}[line {line_number}]")
            return augmented
    return None


def build_prepared_datasets(
    prepared: PreparedSplits,
    *,
    normalize_time: bool,
    normalize_space: bool,
    normalize_covariates: bool | None = None,
    continuous_time_mode: Literal["none", "exact", "approximate"] = "none",
):
    """Build canonical datasets with transform statistics fitted on train only.

    ``continuous_time_mode='exact'`` gates out jittered, tied, or ambiguous
    local-civil observations.  ``'approximate'`` permits only manifest-declared
    strictly ordered views, leaving the jitter observation model labelled in
    route metadata.  ``'none'`` is for inspection/preflight paths.
    """
    if continuous_time_mode not in {"none", "exact", "approximate"}:
        raise ValueError("continuous_time_mode must be 'none', 'exact', or 'approximate'.")
    if continuous_time_mode == "exact":
        prepared.route.assert_continuous_time_eligible(exact=True)
    elif continuous_time_mode == "approximate":
        prepared.route.assert_continuous_time_eligible(exact=False)

    from seahorse.data.dataset import STPPDataset

    train_ds = STPPDataset(
        prepared.train,
        normalize_time=normalize_time,
        normalize_space=normalize_space,
        normalize_covariates=normalize_covariates,
    )
    shared_kwargs = {
        "normalize_time": normalize_time,
        "normalize_space": normalize_space,
        "normalize_covariates": normalize_covariates,
        "cov_mean": train_ds.cov_mean,
        "cov_std": train_ds.cov_std,
    }
    val_ds = STPPDataset(prepared.val, **shared_kwargs)
    test_ds = STPPDataset(prepared.test, **shared_kwargs)
    # STPPDataset evaluates these attributes at __getitem__, so assigning the
    # train statistics here prevents split leakage without changing its API.
    for dataset in (val_ds, test_ds):
        dataset.time_mean = train_ds.time_mean
        dataset.time_std = train_ds.time_std
        dataset.loc_mean = train_ds.loc_mean
        dataset.loc_std = train_ds.loc_std
    return train_ds, val_ds, test_ds


__all__ = [
    "PREPARED_DATA_ROOT_ENV",
    "PREPARED_CORPORA",
    "PREPARED_ID_PREFIX",
    "PreparedCorpusSpec",
    "PreparedDatasetRoute",
    "PreparedMarkMap",
    "PreparedSplits",
    "apply_train_mark_map",
    "build_prepared_datasets",
    "fit_train_mark_map",
    "is_prepared_dataset_id",
    "load_prepared_splits",
    "load_prepared_split_sample",
    "parse_prepared_dataset_id",
    "prepared_catalog",
    "prepared_data_root",
    "prepared_dataset_ids",
    "resolve_prepared_dataset",
]
