"""Explicit fitted coordinate transforms for STPP model families.

The data layer stays raw-first. Families that need transformed coordinates own
an explicit fitted transform artifact, serialized into config / run artifacts
and reconstructed inside model components.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch import Tensor


def _as_float_tuple(value: Iterable[float], *, dim: int | None = None) -> tuple[float, ...]:
    arr = np.asarray(list(value), dtype=np.float32).reshape(-1)
    if dim is not None and arr.size != dim:
        raise ValueError(f"Expected {dim} values, got {arr.size}.")
    return tuple(float(x) for x in arr.tolist())


class CoordinateTransformArtifact(ABC):
    """Base interface for fitted reversible coordinate transforms."""

    kind: str = "base"
    is_identity: bool = False
    is_exactly_invertible: bool = True
    supports_raw_reporting: bool = False

    def temporal_reporting_correction(self, ref: Tensor | None = None) -> Tensor:
        if ref is None:
            return torch.tensor(0.0, dtype=torch.float32)
        return ref.new_zeros(())

    def spatial_reporting_correction(self, ref: Tensor | None = None) -> Tensor:
        if ref is None:
            return torch.tensor(0.0, dtype=torch.float32)
        return ref.new_zeros(())

    def forward_times(self, times_raw: Tensor, lengths: Tensor | None = None) -> Tensor:
        del lengths
        return times_raw

    def forward_locations(self, locs_raw: Tensor, lengths: Tensor | None = None) -> Tensor:
        del lengths
        return locs_raw

    def forward_covariates(self, covs_raw: Tensor, lengths: Tensor | None = None) -> Tensor:
        del lengths
        return covs_raw

    def inverse_times(self, times_native: Tensor, lengths: Tensor | None = None) -> Tensor:
        del lengths
        return times_native

    def inverse_locations(self, locs_native: Tensor, lengths: Tensor | None = None) -> Tensor:
        del lengths
        return locs_native

    def reporting_correction(
        self,
        *,
        times_raw: Tensor | None = None,
        locs_raw: Tensor | None = None,
        lengths: Tensor | None = None,
        ref: Tensor | None = None,
    ) -> Tensor:
        del times_raw, locs_raw, lengths
        return self.temporal_reporting_correction(ref) + self.spatial_reporting_correction(ref)

    def describe_space(self) -> str:
        return self.kind

    def serialize(self) -> Dict[str, Any]:
        return {"type": self.kind}


@dataclass(frozen=True)
class IdentityTransformArtifact(CoordinateTransformArtifact):
    kind: str = "identity"
    is_identity: bool = True
    supports_raw_reporting: bool = True

    def describe_space(self) -> str:
        return "raw/original coordinates"


@dataclass(frozen=True)
class ZScoreTransformArtifact(CoordinateTransformArtifact):
    """Affine z-score transform over time and/or space."""

    normalize_time: bool = True
    normalize_space: bool = True
    time_mean: float = 0.0
    time_std: float = 1.0
    loc_mean: tuple[float, ...] = (0.0, 0.0)
    loc_std: tuple[float, ...] = (1.0, 1.0)
    kind: str = "zscore"
    supports_raw_reporting: bool = True

    def forward_times(self, times_raw: Tensor, lengths: Tensor | None = None) -> Tensor:
        del lengths
        if not self.normalize_time:
            return times_raw
        mean = times_raw.new_tensor(float(self.time_mean))
        std = times_raw.new_tensor(max(float(self.time_std), 1e-8))
        return (times_raw - mean) / std

    def forward_locations(self, locs_raw: Tensor, lengths: Tensor | None = None) -> Tensor:
        del lengths
        if not self.normalize_space:
            return locs_raw
        mean = locs_raw.new_tensor(self.loc_mean).view(*([1] * (locs_raw.ndim - 1)), -1)
        std = locs_raw.new_tensor(self.loc_std).view(*([1] * (locs_raw.ndim - 1)), -1).clamp(min=1e-8)
        return (locs_raw - mean) / std

    def inverse_times(self, times_native: Tensor, lengths: Tensor | None = None) -> Tensor:
        del lengths
        if not self.normalize_time:
            return times_native
        mean = times_native.new_tensor(float(self.time_mean))
        std = times_native.new_tensor(max(float(self.time_std), 1e-8))
        return times_native * std + mean

    def inverse_locations(self, locs_native: Tensor, lengths: Tensor | None = None) -> Tensor:
        del lengths
        if not self.normalize_space:
            return locs_native
        mean = locs_native.new_tensor(self.loc_mean).view(*([1] * (locs_native.ndim - 1)), -1)
        std = locs_native.new_tensor(self.loc_std).view(*([1] * (locs_native.ndim - 1)), -1).clamp(min=1e-8)
        return locs_native * std + mean

    def temporal_reporting_correction(self, ref: Tensor | None = None) -> Tensor:
        scalar = float(np.log(max(self.time_std, 1e-8))) if self.normalize_time else 0.0
        if ref is None:
            return torch.tensor(scalar, dtype=torch.float32)
        return ref.new_tensor(scalar)

    def spatial_reporting_correction(self, ref: Tensor | None = None) -> Tensor:
        scalar = 0.0
        if self.normalize_space:
            scalar = float(
                np.log(np.maximum(np.asarray(self.loc_std, dtype=np.float32), 1e-8)).sum()
            )
        if ref is None:
            return torch.tensor(scalar, dtype=torch.float32)
        return ref.new_tensor(scalar)

    def reporting_correction(
        self,
        *,
        times_raw: Tensor | None = None,
        locs_raw: Tensor | None = None,
        lengths: Tensor | None = None,
        ref: Tensor | None = None,
    ) -> Tensor:
        del times_raw, locs_raw, lengths
        return self.temporal_reporting_correction(ref) + self.spatial_reporting_correction(ref)

    def describe_space(self) -> str:
        if self.normalize_time and self.normalize_space:
            return "z-score standardized time + space"
        if self.normalize_space:
            return "raw time + z-score standardized space"
        if self.normalize_time:
            return "z-score standardized time + raw space"
        return "raw/original coordinates"

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": self.kind,
            "normalize_time": bool(self.normalize_time),
            "normalize_space": bool(self.normalize_space),
            "time_mean": float(self.time_mean),
            "time_std": float(self.time_std),
            "loc_mean": list(self.loc_mean),
            "loc_std": list(self.loc_std),
        }


@dataclass(frozen=True)
class PaperAffineTransformArtifact(CoordinateTransformArtifact):
    """Auto/Deep paper-space affine stats over raw delta-times and locations."""

    input_normalized: bool = False
    input_time_mean: float = 0.0
    input_time_std: float = 1.0
    input_loc_mean: tuple[float, ...] = (0.0, 0.0)
    input_loc_std: tuple[float, ...] = (1.0, 1.0)
    paper_dt_min: float = 0.0
    paper_dt_range: float = 1.0
    paper_loc_min: tuple[float, ...] = (0.0, 0.0)
    paper_loc_range: tuple[float, ...] = (1.0, 1.0)
    kind: str = "paper_affine"
    supports_raw_reporting: bool = True

    def inverse_times(self, times_native: Tensor, lengths: Tensor | None = None) -> Tensor:
        del lengths
        if not self.input_normalized:
            return times_native
        mean = times_native.new_tensor(float(self.input_time_mean))
        std = times_native.new_tensor(max(float(self.input_time_std), 1e-8))
        return times_native * std + mean

    def inverse_locations(self, locs_native: Tensor, lengths: Tensor | None = None) -> Tensor:
        del lengths
        if not self.input_normalized:
            return locs_native
        mean = locs_native.new_tensor(self.input_loc_mean).view(*([1] * (locs_native.ndim - 1)), -1)
        std = locs_native.new_tensor(self.input_loc_std).view(*([1] * (locs_native.ndim - 1)), -1).clamp(min=1e-8)
        return locs_native * std + mean

    def scale_delta_times(self, delta_t_raw: Tensor) -> Tensor:
        vmin = delta_t_raw.new_tensor(float(self.paper_dt_min))
        vrange = delta_t_raw.new_tensor(max(float(self.paper_dt_range), 1e-8))
        return (delta_t_raw - vmin) / vrange

    def inverse_delta_times(self, delta_t_native: Tensor) -> Tensor:
        vmin = delta_t_native.new_tensor(float(self.paper_dt_min))
        vrange = delta_t_native.new_tensor(max(float(self.paper_dt_range), 1e-8))
        return delta_t_native * vrange + vmin

    def scale_locations(self, locs_raw: Tensor) -> Tensor:
        vmin = locs_raw.new_tensor(self.paper_loc_min).view(*([1] * (locs_raw.ndim - 1)), -1)
        vrange = locs_raw.new_tensor(self.paper_loc_range).view(*([1] * (locs_raw.ndim - 1)), -1).clamp(min=1e-8)
        return (locs_raw - vmin) / vrange

    def inverse_scaled_locations(self, locs_native: Tensor) -> Tensor:
        vmin = locs_native.new_tensor(self.paper_loc_min).view(*([1] * (locs_native.ndim - 1)), -1)
        vrange = locs_native.new_tensor(self.paper_loc_range).view(*([1] * (locs_native.ndim - 1)), -1).clamp(min=1e-8)
        return locs_native * vrange + vmin

    def temporal_reporting_correction(self, ref: Tensor | None = None) -> Tensor:
        scalar = float(np.log(max(self.paper_dt_range, 1e-8)))
        if ref is None:
            return torch.tensor(scalar, dtype=torch.float32)
        return ref.new_tensor(scalar)

    def spatial_reporting_correction(self, ref: Tensor | None = None) -> Tensor:
        scalar = float(
            np.log(np.maximum(np.asarray(self.paper_loc_range, dtype=np.float32), 1e-8)).sum()
        )
        if ref is None:
            return torch.tensor(scalar, dtype=torch.float32)
        return ref.new_tensor(scalar)

    def reporting_correction(
        self,
        *,
        times_raw: Tensor | None = None,
        locs_raw: Tensor | None = None,
        lengths: Tensor | None = None,
        ref: Tensor | None = None,
    ) -> Tensor:
        del times_raw, locs_raw, lengths
        return self.temporal_reporting_correction(ref) + self.spatial_reporting_correction(ref)

    def describe_space(self) -> str:
        return "paper MinMax space over raw delta-time + raw locations"

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": self.kind,
            "input_normalized": bool(self.input_normalized),
            "input_time_mean": float(self.input_time_mean),
            "input_time_std": float(self.input_time_std),
            "input_loc_mean": list(self.input_loc_mean),
            "input_loc_std": list(self.input_loc_std),
            "paper_dt_min": float(self.paper_dt_min),
            "paper_dt_range": float(self.paper_dt_range),
            "paper_loc_min": list(self.paper_loc_min),
            "paper_loc_range": list(self.paper_loc_range),
        }


def transform_from_spec(spec: Dict[str, Any] | None) -> CoordinateTransformArtifact | None:
    """Rebuild a fitted transform artifact from its serialized dict."""
    if not spec:
        return None
    kind = str(spec.get("type", "")).strip().lower()
    if kind in {"", "none"}:
        return None
    if kind == "identity":
        return IdentityTransformArtifact()
    if kind == "zscore":
        return ZScoreTransformArtifact(
            normalize_time=bool(spec.get("normalize_time", True)),
            normalize_space=bool(spec.get("normalize_space", True)),
            time_mean=float(spec.get("time_mean", 0.0)),
            time_std=float(spec.get("time_std", 1.0)),
            loc_mean=_as_float_tuple(spec.get("loc_mean", [0.0, 0.0])),
            loc_std=_as_float_tuple(spec.get("loc_std", [1.0, 1.0])),
        )
    if kind == "paper_affine":
        return PaperAffineTransformArtifact(
            input_normalized=bool(spec.get("input_normalized", False)),
            input_time_mean=float(spec.get("input_time_mean", 0.0)),
            input_time_std=float(spec.get("input_time_std", 1.0)),
            input_loc_mean=_as_float_tuple(spec.get("input_loc_mean", [0.0, 0.0])),
            input_loc_std=_as_float_tuple(spec.get("input_loc_std", [1.0, 1.0])),
            paper_dt_min=float(spec.get("paper_dt_min", 0.0)),
            paper_dt_range=float(spec.get("paper_dt_range", 1.0)),
            paper_loc_min=_as_float_tuple(spec.get("paper_loc_min", [0.0, 0.0])),
            paper_loc_range=_as_float_tuple(spec.get("paper_loc_range", [1.0, 1.0])),
        )
    raise ValueError(f"Unknown transform artifact type {kind!r}.")


__all__ = [
    "CoordinateTransformArtifact",
    "IdentityTransformArtifact",
    "PaperAffineTransformArtifact",
    "ZScoreTransformArtifact",
    "transform_from_spec",
]
