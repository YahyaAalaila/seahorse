"""
Data-layer exports.

The live ``python -m unified_stpp`` CLI primarily uses the JSONL dataset and
batch-contract helpers exported directly below. Synthetic split-generation and
benchmark-reference helpers remain part of the current data workflow, but are
loaded lazily because they are not needed on every CLI path.
"""

from __future__ import annotations

from importlib import import_module

from .contract import (
    fingerprint_batch,
    validate_batch,
    validate_sequence_record,
    validate_sequence_records,
)
from .hub import CuratedDatasetSpec, download_dataset, load_dataset
from .dataset import (
    PaperSlidingWindowDataset,
    SlidingWindowSTPPDataset,
    STPPDataset,
    collate_fn,
)
from .transforms import (
    CoordinateTransformArtifact,
    IdentityTransformArtifact,
    PaperAffineTransformArtifact,
    ZScoreTransformArtifact,
    transform_from_spec,
)


_DIRECT_EXPORTS = {
    "STPPDataset": STPPDataset,
    "SlidingWindowSTPPDataset": SlidingWindowSTPPDataset,
    "collate_fn": collate_fn,
    "PaperSlidingWindowDataset": PaperSlidingWindowDataset,
    "validate_batch": validate_batch,
    "fingerprint_batch": fingerprint_batch,
    "validate_sequence_record": validate_sequence_record,
    "validate_sequence_records": validate_sequence_records,
    "CuratedDatasetSpec": CuratedDatasetSpec,
    "download_dataset": download_dataset,
    "load_dataset": load_dataset,
    "CoordinateTransformArtifact": CoordinateTransformArtifact,
    "IdentityTransformArtifact": IdentityTransformArtifact,
    "PaperAffineTransformArtifact": PaperAffineTransformArtifact,
    "ZScoreTransformArtifact": ZScoreTransformArtifact,
    "transform_from_spec": transform_from_spec,
}


__all__ = list(_DIRECT_EXPORTS)

def __getattr__(name: str):
    if name in _DIRECT_EXPORTS:
        return _DIRECT_EXPORTS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
