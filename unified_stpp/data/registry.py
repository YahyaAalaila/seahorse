"""
DataRegistry — registry-driven data-building components.

Three registries:
  _DATASET_BUILDERS      : protocol name → fn(data_config, train_seqs, val_seqs, test_seqs)
                            → (train_ds, val_ds, test_ds)
  _COLLATE_FNS           : name → collate callable
  _TRAIN_LOADER_BUILDERS : name → fn(train_dataset, **kwargs) → batch_sampler | None

Adding a new dataset protocol, collate, or loader policy requires only a
decorated function here — no branching anywhere else.

DataBundle is the plain data container returned by STPPConfig.build_data_bundle().
STPPDataModule receives a DataBundle and wraps it in the Lightning interface.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DataBundle:
    """Resolved data components produced by the data-building registry.

    This is what ``STPPConfig.build_data_bundle()`` returns.
    ``STPPDataModule`` receives a ``DataBundle`` and serves the Lightning
    DataLoader interface on top of it.

    train_batch_sampler: pre-computed ``list[list[int]]`` for
        ``DataLoader(batch_sampler=...)``, or ``None`` for a standard
        fixed-batch DataLoader with shuffle.
    """
    train_dataset: Any
    val_dataset: Any
    test_dataset: Optional[Any]
    collate_fn: Callable
    train_batch_sampler: Optional[Any]

# ---------------------------------------------------------------------------
# Internal dicts
# ---------------------------------------------------------------------------

_DATASET_BUILDERS: dict[str, Callable] = {}
_COLLATE_FNS: dict[str, Any] = {}
_TRAIN_LOADER_BUILDERS: dict[str, Callable] = {}


# ---------------------------------------------------------------------------
# Registration decorators
# ---------------------------------------------------------------------------

def register_dataset(name: str):
    """Register a dataset-builder function under *name* (= DataConfig.protocol)."""
    def decorator(fn: Callable) -> Callable:
        _DATASET_BUILDERS[name] = fn
        return fn
    return decorator


def register_collate(name: str):
    """Register a collate function under *name*."""
    def decorator(fn: Callable) -> Callable:
        _COLLATE_FNS[name] = fn
        return fn
    return decorator


def register_train_loader(name: str):
    """Register a train-loader builder under *name*."""
    def decorator(fn: Callable) -> Callable:
        _TRAIN_LOADER_BUILDERS[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DataRegistry:
    """Namespace for registry lookups used by STPPConfig.build_data_module()."""

    @classmethod
    def build_datasets(cls, protocol: str, data_config, train_seqs, val_seqs, test_seqs):
        """Build (train_ds, val_ds, test_ds) using the registered builder for *protocol*."""
        if protocol not in _DATASET_BUILDERS:
            raise KeyError(
                f"Unknown dataset protocol {protocol!r}. "
                f"Registered: {sorted(_DATASET_BUILDERS)}"
            )
        return _DATASET_BUILDERS[protocol](data_config, train_seqs, val_seqs, test_seqs)

    @classmethod
    def get_collate(cls, name: str):
        """Return the collate callable registered under *name*."""
        if name not in _COLLATE_FNS:
            raise KeyError(
                f"Unknown collate {name!r}. Registered: {sorted(_COLLATE_FNS)}"
            )
        return _COLLATE_FNS[name]

    @classmethod
    def build_train_loader(cls, name: str, train_dataset, **kwargs):
        """Call the registered train-loader builder and return batch_sampler or None."""
        if name not in _TRAIN_LOADER_BUILDERS:
            raise KeyError(
                f"Unknown train loader {name!r}. "
                f"Registered: {sorted(_TRAIN_LOADER_BUILDERS)}"
            )
        return _TRAIN_LOADER_BUILDERS[name](train_dataset, **kwargs)


# ---------------------------------------------------------------------------
# Built-in dataset builders
# ---------------------------------------------------------------------------

@register_dataset("standard")
def _build_standard_datasets(data_config, train_seqs, val_seqs, test_seqs):
    from unified_stpp.data.dataset import STPPDataset

    train_ds = STPPDataset(
        train_seqs,
        normalize_time=data_config.normalize,
        normalize_space=data_config.normalize,
    )
    val_ds = STPPDataset(
        val_seqs,
        normalize_time=data_config.normalize,
        normalize_space=data_config.normalize,
        cov_mean=train_ds.cov_mean,
        cov_std=train_ds.cov_std,
    )
    val_ds.time_mean = train_ds.time_mean
    val_ds.time_std  = train_ds.time_std
    val_ds.loc_mean  = train_ds.loc_mean
    val_ds.loc_std   = train_ds.loc_std

    test_ds = None
    if test_seqs is not None:
        test_ds = STPPDataset(
            test_seqs,
            normalize_time=data_config.normalize,
            normalize_space=data_config.normalize,
            cov_mean=train_ds.cov_mean,
            cov_std=train_ds.cov_std,
        )
        test_ds.time_mean = train_ds.time_mean
        test_ds.time_std  = train_ds.time_std
        test_ds.loc_mean  = train_ds.loc_mean
        test_ds.loc_std   = train_ds.loc_std

    return train_ds, val_ds, test_ds


@register_dataset("raw")
def _build_raw_datasets(data_config, train_seqs, val_seqs, test_seqs):
    from unified_stpp.data.dataset import STPPDataset

    train_ds = STPPDataset(
        train_seqs,
        normalize_time=False,
        normalize_space=False,
        normalize_covariates=False,
    )
    val_ds = STPPDataset(
        val_seqs,
        normalize_time=False,
        normalize_space=False,
        normalize_covariates=False,
    )
    test_ds = None
    if test_seqs is not None:
        test_ds = STPPDataset(
            test_seqs,
            normalize_time=False,
            normalize_space=False,
            normalize_covariates=False,
        )

    return train_ds, val_ds, test_ds


@register_dataset("sliding_window")
def _build_sliding_window_datasets(data_config, train_seqs, val_seqs, test_seqs):
    from sklearn.preprocessing import MinMaxScaler
    from unified_stpp.data.dataset import PaperSlidingWindowDataset

    all_seqs = list(train_seqs) + list(val_seqs) + (list(test_seqs) if test_seqs else [])
    times_abs = np.concatenate([np.asarray(s["times"]) for s in all_seqs]).astype(np.float64)
    locs = np.concatenate([np.asarray(s["locations"]) for s in all_seqs]).reshape(-1, 2)

    delta_t = np.diff(times_abs, prepend=times_abs[0])
    delta_t[0] = 0.0
    xyt = np.column_stack([locs[:, 0], locs[:, 1], delta_t]).astype(np.float32)

    scaler = MinMaxScaler()
    xyt_mm = scaler.fit_transform(xyt).astype(np.float32)

    lb = data_config.paper_lookback or 10
    la = data_config.paper_lookahead
    T = lb + la
    N = len(xyt_mm)
    W = max(0, N - T + 1)
    windows = np.stack([xyt_mm[i: i + T] for i in range(W)], axis=0)

    r0, r1, r2 = data_config.paper_split_ratio
    total_r = r0 + r1 + r2
    n_tr = int(W * r0 / total_r)
    n_va = int(W * r1 / total_r)

    mn = np.round(scaler.data_min_, 4).tolist()
    mx = np.round(scaler.data_max_, 4).tolist()
    print(
        f"[DataRegistry] protocol=sliding_window  scaler=MinMaxScaler"
        f"(min={mn}  max={mx})  time_format=delta_t→cumsum_mm  "
        f"window_shape=(B, {T}, 3)  "
        f"n_train={n_tr}  n_val={n_va}  n_test={W - n_tr - n_va}"
    )

    return (
        PaperSlidingWindowDataset(windows[:n_tr], scaler=scaler),
        PaperSlidingWindowDataset(windows[n_tr: n_tr + n_va], scaler=scaler),
        PaperSlidingWindowDataset(windows[n_tr + n_va:], scaler=scaler),
    )


# ---------------------------------------------------------------------------
# Built-in collate functions
# ---------------------------------------------------------------------------

@register_collate("canonical")
def _canonical_collate():
    from unified_stpp.data.dataset import collate_fn
    return collate_fn


# Eagerly resolve "canonical" so get_collate("canonical") returns the fn directly
_COLLATE_FNS["canonical"] = _canonical_collate()


# ---------------------------------------------------------------------------
# Built-in train loader builders
# ---------------------------------------------------------------------------

@register_train_loader("fixed_batch")
def _fixed_batch(train_dataset, **_):
    """Standard DataLoader + shuffle.  Return None so STPPDataModule uses its default path."""
    return None


@register_train_loader("batch_by_size")
def _batch_by_size(train_dataset, *, max_events: int = 3000, **_):
    """Token-count batching: each batch contains at most *max_events* total events."""
    return train_dataset.batch_by_size(max_events)
