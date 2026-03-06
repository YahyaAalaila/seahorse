"""
Canonical batch schema for the unified STPP framework.

Every DataLoader that feeds a UnifiedSTPP model MUST produce batches that pass
`validate_batch`.  The canonical fields are:

Required (always non-None tensors)
-----------------------------------
  times         (B, N)          float32 — event times, zero-padded past `lengths`
  locations     (B, N, d)       float32 — event locations, zero-padded
  lengths       (B,)            int64   — actual sequence lengths
  pad_mask      (B, N)          bool    — True for valid (non-padded) positions
  txys          (B, N, 1+d)     float32 — packed (t, x…) for compact access

Optional (present in dict but may be None)
-------------------------------------------
  marks             (B, N)      int64   — discrete mark indices
  event_covariates  (B, N, p)   float32 — per-event covariate features
  field_covariates  (B, N, r)   float32 — field covariates pre-evaluated at events

Invariants checked by `validate_batch`
---------------------------------------
  * pad_mask.sum(1) == lengths  (mask counts equal reported lengths)
  * txys[..., 0] == times        (field order: t first)
  * txys[..., 1:] == locations   (then spatial coords)
  * No NaN / Inf in times or locations
"""

from __future__ import annotations

import json
from typing import Any, Dict

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ("times", "locations", "lengths", "pad_mask", "txys")
OPTIONAL_FIELDS = ("marks", "event_covariates", "field_covariates")

_REQUIRED_DTYPES: Dict[str, torch.dtype] = {
    "times":     torch.float32,
    "locations": torch.float32,
    "lengths":   torch.int64,
    "pad_mask":  torch.bool,
    "txys":      torch.float32,
}


# ---------------------------------------------------------------------------
# validate_batch
# ---------------------------------------------------------------------------

def validate_batch(batch: Dict[str, Any]) -> None:
    """
    Validate that *batch* conforms to the canonical STPP batch schema.

    Raises
    ------
    KeyError   — required field missing
    TypeError  — field has wrong dtype
    ValueError — shape mismatch or failed invariant check
    """
    # --- presence ---
    for key in REQUIRED_FIELDS:
        if key not in batch:
            raise KeyError(
                f"validate_batch: required field '{key}' missing from batch. "
                f"Keys present: {list(batch.keys())}"
            )
        if batch[key] is None:
            raise ValueError(
                f"validate_batch: required field '{key}' must not be None."
            )

    times     = batch["times"]
    locations = batch["locations"]
    lengths   = batch["lengths"]
    pad_mask  = batch["pad_mask"]
    txys      = batch["txys"]

    # --- dtype checks ---
    for key, expected in _REQUIRED_DTYPES.items():
        t = batch[key]
        if not isinstance(t, Tensor):
            raise TypeError(
                f"validate_batch: field '{key}' must be a torch.Tensor, "
                f"got {type(t).__name__}."
            )
        if t.dtype != expected:
            raise TypeError(
                f"validate_batch: field '{key}' has dtype {t.dtype}, "
                f"expected {expected}."
            )

    # --- shape checks ---
    if times.ndim != 2:
        raise ValueError(
            f"validate_batch: 'times' must be 2-D (B, N), got shape {tuple(times.shape)}."
        )
    B, N = times.shape

    if locations.ndim != 3 or locations.shape[:2] != (B, N):
        raise ValueError(
            f"validate_batch: 'locations' must be (B={B}, N={N}, d), "
            f"got {tuple(locations.shape)}."
        )
    d = locations.shape[2]
    if d < 1:
        raise ValueError("validate_batch: spatial_dim d must be >= 1.")

    if lengths.shape != (B,):
        raise ValueError(
            f"validate_batch: 'lengths' must be (B={B},), got {tuple(lengths.shape)}."
        )

    if pad_mask.shape != (B, N):
        raise ValueError(
            f"validate_batch: 'pad_mask' must be ({B}, {N}), "
            f"got {tuple(pad_mask.shape)}."
        )

    if txys.shape != (B, N, 1 + d):
        raise ValueError(
            f"validate_batch: 'txys' must be ({B}, {N}, {1 + d}), "
            f"got {tuple(txys.shape)}."
        )

    # --- semantic invariants ---
    mask_counts = pad_mask.sum(dim=1).to(lengths.dtype)
    if not torch.all(mask_counts == lengths):
        bad = (mask_counts != lengths).nonzero(as_tuple=False).squeeze(-1).tolist()
        raise ValueError(
            f"validate_batch: pad_mask row-sum != lengths for "
            f"{len(bad)} sequence(s) (first few: {bad[:5]})."
        )

    if not torch.allclose(txys[..., 0], times, atol=1e-5):
        raise ValueError(
            "validate_batch: txys[..., 0] does not match 'times' "
            "(expected field order t, x, y, …)."
        )

    if not torch.allclose(txys[..., 1:], locations, atol=1e-5):
        raise ValueError(
            "validate_batch: txys[..., 1:] does not match 'locations'."
        )

    # --- numerical sanity ---
    if not torch.isfinite(times).all():
        raise ValueError("validate_batch: 'times' contains NaN or Inf.")
    if not torch.isfinite(locations).all():
        raise ValueError("validate_batch: 'locations' contains NaN or Inf.")

    # --- optional-field dtype checks (only when not None) ---
    if batch.get("marks") is not None:
        m = batch["marks"]
        if not isinstance(m, Tensor) or m.dtype != torch.int64:
            raise TypeError(
                f"validate_batch: 'marks' must be a int64 Tensor, got {type(m)} dtype={getattr(m,'dtype',None)}."
            )
        if m.shape != (B, N):
            raise ValueError(
                f"validate_batch: 'marks' shape {tuple(m.shape)} != ({B}, {N})."
            )

    for cov_key in ("event_covariates", "field_covariates"):
        cov = batch.get(cov_key)
        if cov is not None:
            if not isinstance(cov, Tensor):
                raise TypeError(
                    f"validate_batch: '{cov_key}' must be a Tensor or None."
                )
            if cov.dtype != torch.float32:
                raise TypeError(
                    f"validate_batch: '{cov_key}' must be float32, got {cov.dtype}."
                )
            if cov.ndim != 3 or cov.shape[:2] != (B, N):
                raise ValueError(
                    f"validate_batch: '{cov_key}' shape {tuple(cov.shape)} != "
                    f"({B}, {N}, *)."
                )


# ---------------------------------------------------------------------------
# fingerprint_batch
# ---------------------------------------------------------------------------

def fingerprint_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a compact, JSON-serialisable summary of *batch*.

    Useful for logging / reproducibility bookkeeping.  Does NOT call
    `validate_batch`; the caller should do that separately.

    Returns
    -------
    dict with keys:
      n_seq           — batch size B
      n_events_max    — padded sequence length N
      spatial_dim     — d (from locations)
      total_events    — sum of sequence lengths
      shapes          — {field: [dim, …]} for all Tensor fields
      dtypes          — {field: dtype_str} for all Tensor fields
      stats           — {times: {min,mean,max}, locations: {min,mean,max}}
    """
    fp: Dict[str, Any] = {}

    # Shapes and dtypes of every non-None tensor
    shapes: Dict[str, list] = {}
    dtypes: Dict[str, str] = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            shapes[k] = list(v.shape)
            dtypes[k] = str(v.dtype)

    times     = batch.get("times")
    locations = batch.get("locations")
    lengths   = batch.get("lengths")

    fp["n_seq"] = int(shapes["times"][0]) if "times" in shapes else None
    fp["n_events_max"] = int(shapes["times"][1]) if "times" in shapes else None
    fp["spatial_dim"] = int(shapes["locations"][2]) if "locations" in shapes else None
    fp["total_events"] = int(lengths.sum().item()) if lengths is not None else None

    fp["shapes"] = shapes
    fp["dtypes"] = dtypes

    stats: Dict[str, Any] = {}
    if times is not None:
        valid = times[batch["pad_mask"]] if "pad_mask" in batch and batch["pad_mask"] is not None else times.flatten()
        stats["times"] = {
            "min":  float(valid.min().item()),
            "mean": float(valid.float().mean().item()),
            "max":  float(valid.max().item()),
        }
    if locations is not None:
        pad_mask = batch.get("pad_mask")
        if pad_mask is not None:
            # locations[pad_mask] → (total_events, d)
            valid_locs = locations[pad_mask]
        else:
            valid_locs = locations.reshape(-1, locations.shape[-1])
        stats["locations"] = {
            "min":  float(valid_locs.min().item()),
            "mean": float(valid_locs.float().mean().item()),
            "max":  float(valid_locs.max().item()),
        }
    fp["stats"] = stats

    return fp
