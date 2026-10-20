"""Construction configuration for the independent GLIDE integration.

Available preset; not benchmarked in the Seahorse paper. GLIDE declares
``nll_kind="none"``: its diffusion objective is not a point-process likelihood.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Dict

import numpy as np

from .base import BaseModelConfig, ConfigRegistry


def _tuple(value: Any, dim: int, default: float) -> tuple[float, ...]:
    if value is None:
        return (default,) * dim
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"Expected {dim} values, received {arr.size}.")
    return tuple(float(x) for x in arr)


def _sequence(value: Any, default: tuple, cast):
    """Accept comma-delimited YAML scalars so config HPO sanitization cannot
    mistake fixed architectural vectors for a choice-list search space."""
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if isinstance(value, (tuple, list)):
        return tuple(cast(item) for item in value)
    raise ValueError(
        "Fixed GLIDE architecture vectors must be comma-delimited strings or sequences."
    )


def _stats(train_dataset: Any, spatial_dim: int) -> dict[str, Any]:
    """Fit all GLIDE token/history statistics exclusively on train sequences."""
    ts, ls, dts, dls = [], [], [], []
    for seq in getattr(train_dataset, "sequences", []):
        t = np.asarray(seq["times"], dtype=np.float64).reshape(-1)
        x = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, spatial_dim)
        if t.size:
            ts.append(t)
            ls.append(x)
        if t.size > 1:
            # Author preprocessing inserts a leading zero interval before fitting
            # time-difference statistics; location-difference statistics exclude it.
            dts.append(np.concatenate((np.zeros(1, dtype=np.float64), np.diff(t))))
            dls.append(np.diff(x, axis=0))
    if not ts:
        return {}
    t, x = np.concatenate(ts), np.concatenate(ls)
    dt = np.concatenate(dts) if dts else np.zeros(1)
    dx = np.concatenate(dls) if dls else np.zeros((1, spatial_dim))
    return {
        "abs_time_min": float(t.min()),
        "abs_time_range": float(max(t.max() - t.min(), 1e-8)),
        "delta_mean": float(dt.mean()),
        "delta_std": float(max(dt.std(), 1e-8)),
        "delta_min": float(dt.min()),
        "delta_range": float(max(dt.max() - dt.min(), 1e-8)),
        "loc_min": tuple(float(v) for v in x.min(0)),
        "loc_range": tuple(float(v) for v in np.maximum(x.max(0) - x.min(0), 1e-8)),
        "loc_delta_mean": tuple(float(v) for v in dx.mean(0)),
        "loc_delta_std": tuple(float(v) for v in np.maximum(dx.std(0), 1e-8)),
    }


@ConfigRegistry.register("glide")
@dataclasses.dataclass
class GLIDEConfig(BaseModelConfig):
    _STATE_MODEL: ClassVar[str] = "glide"
    _EVENT_MODEL: ClassVar[str] = "glide"
    _SUPPORTED_PROTOCOLS: ClassVar[frozenset] = frozenset({"raw", "standard"})

    d_model: int = 64
    hidden_units: int = 64
    temporal_windows: tuple[int, ...] = (64, 32, 8, 2)
    spatial_windows: tuple[int, ...] = (64, 32, 8, 2)
    spatial_grid_sizes: tuple[float, ...] = (0.2, 0.1, 0.02, 0.005)
    graph_layers: int = 2
    graph_time_scales: tuple[int, ...] = (0, 0, 0)
    graph_neighbors: tuple[int, ...] = (0, 0, 0)
    num_time_harmonics: int = 8
    timesteps: int = 500
    start_ratio: float = 0.5
    guidance: float = 0.2
    ema_decay: float = 0.999
    input_normalized: bool = False
    input_time_mean: float = 0.0
    input_time_std: float = 1.0
    input_loc_mean: tuple[float, ...] = (0.0, 0.0)
    input_loc_std: tuple[float, ...] = (1.0, 1.0)
    abs_time_min: float = 0.0
    abs_time_range: float = 1.0
    delta_mean: float = 0.0
    delta_std: float = 1.0
    delta_min: float = 0.0
    delta_range: float = 1.0
    loc_min: tuple[float, ...] = (0.0, 0.0)
    loc_range: tuple[float, ...] = (1.0, 1.0)
    loc_delta_mean: tuple[float, ...] = (0.0, 0.0)
    loc_delta_std: tuple[float, ...] = (1.0, 1.0)

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        *,
        hidden_dim: int = 128,
        spatial_dim: int = 2,
        event_cov_dim: int = 0,
        field_cov_dim: int = 0,
        n_marks: int = 0,
    ) -> "GLIDEConfig":
        del event_cov_dim, field_cov_dim
        if spatial_dim != 2 or n_marks:
            raise ValueError(
                "GLIDE supports unmarked planar events only (spatial_dim=2, n_marks=0)."
            )
        enc, dec = dict(d.get("encoder", {})), dict(d.get("decoder", {}))
        if enc.pop("type", "glide") != "glide" or dec.pop("type", "glide") != "glide":
            raise ValueError("GLIDE preset requires encoder.type=decoder.type='glide'.")
        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            d_model=int(enc.pop("d_model", 64)),
            temporal_windows=_sequence(enc.pop("temporal_windows", None), (64, 32, 8, 2), int),
            spatial_windows=_sequence(enc.pop("spatial_windows", None), (64, 32, 8, 2), int),
            spatial_grid_sizes=_sequence(
                enc.pop("spatial_grid_sizes", None), (0.2, 0.1, 0.02, 0.005), float
            ),
            graph_layers=int(enc.pop("graph_layers", 2)),
            graph_time_scales=_sequence(enc.pop("graph_time_scales", None), (0, 0, 0), int),
            graph_neighbors=_sequence(enc.pop("graph_neighbors", None), (0, 0, 0), int),
            num_time_harmonics=int(enc.pop("num_time_harmonics", 8)),
            hidden_units=int(dec.pop("hidden_units", 64)),
            timesteps=int(dec.pop("timesteps", 500)),
            start_ratio=float(dec.pop("start_ratio", 0.5)),
            guidance=float(dec.pop("guidance", 0.2)),
            ema_decay=float(dec.pop("ema_decay", 0.999)),
            input_normalized=bool(d.get("input_normalized", False)),
            input_time_mean=float(d.get("input_time_mean", 0.0)),
            input_time_std=float(d.get("input_time_std", 1.0)),
            input_loc_mean=_tuple(d.get("input_loc_mean"), spatial_dim, 0.0),
            input_loc_std=_tuple(d.get("input_loc_std"), spatial_dim, 1.0),
            abs_time_min=float(d.get("abs_time_min", 0.0)),
            abs_time_range=float(d.get("abs_time_range", 1.0)),
            delta_mean=float(d.get("delta_mean", 0.0)),
            delta_std=float(d.get("delta_std", 1.0)),
            delta_min=float(d.get("delta_min", 0.0)),
            delta_range=float(d.get("delta_range", 1.0)),
            loc_min=_tuple(d.get("loc_min"), spatial_dim, 0.0),
            loc_range=_tuple(d.get("loc_range"), spatial_dim, 1.0),
            loc_delta_mean=_tuple(d.get("loc_delta_mean"), spatial_dim, 0.0),
            loc_delta_std=_tuple(d.get("loc_delta_std"), spatial_dim, 1.0),
        )

    def _state_kwargs(self) -> dict:
        return {
            name: getattr(self, name)
            for name in (
                "spatial_dim",
                "d_model",
                "temporal_windows",
                "spatial_windows",
                "spatial_grid_sizes",
                "graph_layers",
                "graph_time_scales",
                "graph_neighbors",
                "num_time_harmonics",
                "input_normalized",
                "input_time_mean",
                "input_time_std",
                "input_loc_mean",
                "input_loc_std",
                "abs_time_min",
                "abs_time_range",
                "delta_mean",
                "delta_std",
                "delta_min",
                "delta_range",
                "loc_min",
                "loc_range",
                "loc_delta_mean",
                "loc_delta_std",
            )
        }

    def _event_kwargs(self) -> dict:
        return {
            name: getattr(self, name)
            for name in (
                "hidden_units",
                "spatial_dim",
                "timesteps",
                "start_ratio",
                "guidance",
                "ema_decay",
            )
        } | {"cond_dim": self.d_model}

    @classmethod
    def data_init_overrides(cls, dm) -> dict:
        bundle = getattr(dm, "_bundle", None)
        train = getattr(bundle, "train_dataset", None)
        if train is None:
            return {}
        first = next(iter(getattr(train, "sequences", [])), None)
        dim = int(np.asarray(first["locations"]).shape[-1]) if first is not None else 2
        result = _stats(train, dim)
        result.update(
            {
                "input_normalized": bool(
                    getattr(train, "normalize_time", False)
                    or getattr(train, "normalize_space", False)
                ),
                "input_time_mean": float(getattr(train, "time_mean", 0.0)),
                "input_time_std": float(getattr(train, "time_std", 1.0)),
                "input_loc_mean": tuple(
                    float(v) for v in np.asarray(getattr(train, "loc_mean", np.zeros(dim)))
                ),
                "input_loc_std": tuple(
                    float(v) for v in np.asarray(getattr(train, "loc_std", np.ones(dim)))
                ),
            }
        )
        return result
