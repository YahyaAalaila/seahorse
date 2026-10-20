"""Bound, resumable per-target likelihood storage for campaign evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from seahorse.experiments.campaign import canonical_hash, digest_file, write_json


def checkpoint_fingerprint(path):
    path = Path(path)
    if path.is_file():
        return digest_file(path)
    files = {str(p.relative_to(path)): digest_file(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError("Checkpoint directory contains no files.")
    return canonical_hash(files)


class NLLProgress:
    """Flush arrays before committing a cursor; an interrupted batch is replayed.

    A cursor never advances beyond durable target values and metadata.  Batch
    random seeds are assigned by the caller from frozen target offsets, making
    a replay independent of random draws in preceding batches.
    """
    def __init__(self, directory, *, binding, size):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.binding = {**binding, "size": int(size)}
        self.path = self.directory / "progress.json"
        previous = json.loads(self.path.read_text()) if self.path.exists() else None
        if previous and previous["binding"] != self.binding:
            raise ValueError("NLL progress belongs to a different checkpoint, dataset, or evaluation policy.")
        self.position = int(previous["position"]) if previous else 0
        self.errors = list(previous["errors"]) if previous else []
        if not 0 <= self.position <= size:
            raise ValueError("Invalid saved NLL cursor.")
        self.arrays = {}
        for name, dtype in (("nll", np.float64), ("sequence_index", np.int64), ("target_event_index", np.int64)):
            path = self.directory / (name + ".npy")
            a = np.lib.format.open_memmap(path, mode="r+" if previous else "w+", dtype=dtype, shape=(size,))
            if a.shape != (size,) or a.dtype != np.dtype(dtype):
                raise ValueError("NLL progress array shape or dtype has changed.")
            if not previous:
                a.fill(np.nan if name == "nll" else -1)
            self.arrays[name] = a
        if not previous:
            self.commit(0)

    def commit(self, position):
        for a in self.arrays.values():
            a.flush()
        write_json(self.path, {"binding": self.binding, "position": int(position), "errors": self.errors})
        self.position = int(position)

    def finalize(self, path):
        if self.position != len(self.arrays["nll"]):
            raise ValueError("Cannot finalize incomplete NLL target coverage.")
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **self.arrays)
        temporary.replace(path)
        return digest_file(path)
