"""
Data-layer exports.

The live ``python -m unified_stpp`` CLI primarily uses the JSONL dataset and
batch-contract helpers exported directly below. Synthetic split-generation and
benchmark-reference helpers remain part of the current data workflow, but are
loaded lazily because they are not needed on every CLI path.
"""

from __future__ import annotations

from importlib import import_module

from .contract import fingerprint_batch, validate_batch
from .dataset import PaperSlidingWindowDataset, STPPDataset, collate_fn


_DIRECT_EXPORTS = {
    "STPPDataset": STPPDataset,
    "collate_fn": collate_fn,
    "PaperSlidingWindowDataset": PaperSlidingWindowDataset,
    "validate_batch": validate_batch,
    "fingerprint_batch": fingerprint_batch,
}


__all__ = list(_DIRECT_EXPORTS) 

def __getattr__(name: str):
    if name in _DIRECT_EXPORTS:
        return _DIRECT_EXPORTS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
