"""DeepSTPPConfig — construction config for the deep_stpp family."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, ClassVar, Dict

import numpy as np

from seahorse.data.transforms import PaperAffineTransformArtifact
from .base import BaseModelConfig, ConfigRegistry


def _as_float_tuple(
    value: Any,
    *,
    dim: int,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    if value is None:
        return default
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"Expected {dim} values, got {arr.size}.")
    return tuple(float(x) for x in arr.tolist())


def _compute_paper_stats_from_dataset(dataset: Any, spatial_dim: int) -> dict[str, Any]:
    dt_chunks: list[np.ndarray] = []
    loc_chunks: list[np.ndarray] = []

    for seq in getattr(dataset, "sequences", []):
        times = np.asarray(seq["times"], dtype=np.float64).reshape(-1)
        locs = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, spatial_dim)
        if times.size > 0:
            delta_t = np.empty_like(times, dtype=np.float64)
            delta_t[0] = times[0]
            if times.size > 1:
                delta_t[1:] = np.diff(times)
            dt_chunks.append(delta_t.astype(np.float32))
        if locs.size > 0:
            loc_chunks.append(locs.astype(np.float32))

    if dt_chunks:
        dt_all = np.concatenate(dt_chunks).astype(np.float32)
        dt_min = float(dt_all.min())
        dt_range = float(max(dt_all.max() - dt_min, 1e-8))
    else:
        dt_min = 0.0
        dt_range = 1.0

    if loc_chunks:
        loc_all = np.concatenate(loc_chunks, axis=0).astype(np.float32)
        loc_min = loc_all.min(axis=0)
        loc_range = np.maximum(loc_all.max(axis=0) - loc_min, 1e-8)
    else:
        loc_min = np.zeros(spatial_dim, dtype=np.float32)
        loc_range = np.ones(spatial_dim, dtype=np.float32)

    return {
        "paper_dt_min": dt_min,
        "paper_dt_range": dt_range,
        "paper_loc_min": tuple(float(x) for x in loc_min.tolist()),
        "paper_loc_range": tuple(float(x) for x in loc_range.tolist()),
    }


@ConfigRegistry.register("deep_stpp")
@dataclasses.dataclass
class DeepSTPPConfig(BaseModelConfig):
    _STATE_MODEL: ClassVar[str] = "deep_stpp"
    _EVENT_MODEL: ClassVar[str] = "deep_stpp"
    _SUPPORTED_PROTOCOLS: ClassVar[frozenset] = frozenset({"raw", "standard"})

    # Encoder params
    enc_num_heads: int = 2
    enc_num_layers: int = 3
    enc_dropout: float = 0.0
    # Decoder / paper-window params
    seq_len: int = 20
    lookahead: int = 1
    num_points: int = 20
    sigma_min: float = 1e-4
    dec_n_layers: int = 3
    constrain_b: str | bool = False
    b_max: float = 20.0
    s_max: float | None = None
    # Dimension params (not from YAML; set by registry at instantiation)
    event_cov_dim: int = 0
    field_cov_dim: int = 0
    # Optional
    vae: bool = False
    expose_decoded_params: bool = False
    # Input and paper-space normalization stats
    input_normalized: bool = False
    input_time_mean: float = 0.0
    input_time_std: float = 1.0
    input_loc_mean: tuple[float, ...] = (0.0, 0.0)
    input_loc_std: tuple[float, ...] = (1.0, 1.0)
    paper_dt_min: float = 0.0
    paper_dt_range: float = 1.0
    paper_loc_min: tuple[float, ...] = (0.0, 0.0)
    paper_loc_range: tuple[float, ...] = (1.0, 1.0)
    input_transform: Dict[str, Any] = dataclasses.field(default_factory=dict)

    _enc_cfg: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)
    _dec_cfg: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

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
    ) -> "DeepSTPPConfig":
        del n_marks

        enc_cfg = copy.deepcopy(d.get("encoder", {}))
        enc_type = enc_cfg.pop("type", "transformer")
        if enc_type != "transformer":
            raise ValueError(
                f"DeepSTPP build expects encoder.type='transformer', got '{enc_type}'."
            )

        dec_cfg = copy.deepcopy(d.get("decoder", {}))
        dec_type = dec_cfg.pop("type", "deep_stpp")
        if dec_type != "deep_stpp":
            raise ValueError(
                f"DeepSTPP build expects decoder.type='deep_stpp', got '{dec_type}'."
            )

        s_max_value = dec_cfg.pop("s_max", None)

        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            event_cov_dim=event_cov_dim,
            field_cov_dim=field_cov_dim,
            enc_num_heads=int(enc_cfg.pop("num_heads", 2)),
            enc_num_layers=int(enc_cfg.pop("num_layers", 3)),
            enc_dropout=float(enc_cfg.pop("dropout", 0.0)),
            seq_len=int(dec_cfg.pop("seq_len", 20)),
            lookahead=int(dec_cfg.pop("lookahead", 1)),
            num_points=int(dec_cfg.pop("num_points", 20)),
            sigma_min=float(dec_cfg.pop("sigma_min", 1e-4)),
            dec_n_layers=int(dec_cfg.pop("n_layers", 3)),
            constrain_b=dec_cfg.pop("constrain_b", False),
            b_max=float(dec_cfg.pop("b_max", 20.0)),
            s_max=None if s_max_value is None else float(s_max_value),
            vae=bool(d.get("vae", False)),
            expose_decoded_params=bool(d.get("deep_stpp_expose_decoded_params", False)),
            input_normalized=bool(d.get("input_normalized", False)),
            input_time_mean=float(d.get("input_time_mean", 0.0)),
            input_time_std=float(d.get("input_time_std", 1.0)),
            input_loc_mean=_as_float_tuple(
                d.get("input_loc_mean"),
                dim=spatial_dim,
                default=tuple(0.0 for _ in range(spatial_dim)),
            ),
            input_loc_std=_as_float_tuple(
                d.get("input_loc_std"),
                dim=spatial_dim,
                default=tuple(1.0 for _ in range(spatial_dim)),
            ),
            paper_dt_min=float(d.get("paper_dt_min", 0.0)),
            paper_dt_range=float(d.get("paper_dt_range", 1.0)),
            paper_loc_min=_as_float_tuple(
                d.get("paper_loc_min"),
                dim=spatial_dim,
                default=tuple(0.0 for _ in range(spatial_dim)),
            ),
            paper_loc_range=_as_float_tuple(
                d.get("paper_loc_range"),
                dim=spatial_dim,
                default=tuple(1.0 for _ in range(spatial_dim)),
            ),
            input_transform=copy.deepcopy(d.get("input_transform", {})),
            _enc_cfg=enc_cfg,
            _dec_cfg=dec_cfg,
        )

    def _state_kwargs(self) -> dict:
        return dict(
            hidden_dim=self.hidden_dim,
            spatial_dim=self.spatial_dim,
            event_cov_dim=self.event_cov_dim,
            seq_len=self.seq_len,
            lookahead=self.lookahead,
            enc_num_heads=self.enc_num_heads,
            enc_num_layers=self.enc_num_layers,
            enc_dropout=self.enc_dropout,
            vae=self.vae,
            input_normalized=self.input_normalized,
            input_time_mean=self.input_time_mean,
            input_time_std=self.input_time_std,
            input_loc_mean=self.input_loc_mean,
            input_loc_std=self.input_loc_std,
            paper_dt_min=self.paper_dt_min,
            paper_dt_range=self.paper_dt_range,
            paper_loc_min=self.paper_loc_min,
            paper_loc_range=self.paper_loc_range,
            input_transform=self.input_transform,
            **self._enc_cfg,
        )

    def _event_kwargs(self) -> dict:
        return dict(
            hidden_dim=self.hidden_dim,
            spatial_dim=self.spatial_dim,
            field_cov_dim=self.field_cov_dim,
            seq_len=self.seq_len,
            lookahead=self.lookahead,
            num_points=self.num_points,
            sigma_min=self.sigma_min,
            n_layers=self.dec_n_layers,
            constrain_b=self.constrain_b,
            b_max=self.b_max,
            s_max=self.s_max,
            expose_decoded_params=self.expose_decoded_params,
            **self._dec_cfg,
        )

    @classmethod
    def data_init_overrides(cls, dm) -> dict:
        bundle = getattr(dm, "_bundle", None)
        if bundle is None:
            return {}

        train_ds = getattr(bundle, "train_dataset", None)
        if train_ds is None:
            return {}
        if getattr(train_ds, "coordinate_space", None) == "raw":
            return {}

        first_seq = next(iter(getattr(train_ds, "sequences", [])), None)
        if first_seq is not None:
            spatial_dim = int(np.asarray(first_seq["locations"]).shape[-1])
        else:
            spatial_dim = int(
                np.asarray(
                    getattr(train_ds, "loc_mean", [0.0, 0.0]),
                    dtype=np.float32,
                ).shape[-1]
            )

        overrides = _compute_paper_stats_from_dataset(train_ds, spatial_dim=spatial_dim)
        overrides.update(
            {
                "input_normalized": bool(
                    getattr(train_ds, "normalize_time", False)
                    or getattr(train_ds, "normalize_space", False)
                ),
                "input_time_mean": float(getattr(train_ds, "time_mean", 0.0)),
                "input_time_std": float(getattr(train_ds, "time_std", 1.0)),
                "input_loc_mean": tuple(
                    float(x)
                    for x in np.asarray(
                        getattr(train_ds, "loc_mean", np.zeros(spatial_dim, dtype=np.float32)),
                        dtype=np.float32,
                    ).tolist()
                ),
                "input_loc_std": tuple(
                    float(x)
                    for x in np.asarray(
                        getattr(train_ds, "loc_std", np.ones(spatial_dim, dtype=np.float32)),
                        dtype=np.float32,
                    ).tolist()
                ),
            }
        )
        return overrides

    @classmethod
    def fit_transform_artifact(cls, dm):
        bundle = getattr(dm, "_bundle", None)
        if bundle is None:
            return None

        train_ds = getattr(bundle, "train_dataset", None)
        if train_ds is None:
            return None
        if getattr(train_ds, "coordinate_space", None) != "raw":
            return None

        first_seq = next(iter(getattr(train_ds, "sequences", [])), None)
        if first_seq is not None:
            spatial_dim = int(np.asarray(first_seq["locations"]).shape[-1])
        else:
            spatial_dim = int(
                np.asarray(
                    getattr(train_ds, "loc_mean", [0.0, 0.0]),
                    dtype=np.float32,
                ).shape[-1]
            )

        paper_stats = _compute_paper_stats_from_dataset(train_ds, spatial_dim=spatial_dim)
        return PaperAffineTransformArtifact(
            input_normalized=bool(
                getattr(train_ds, "normalize_time", False)
                or getattr(train_ds, "normalize_space", False)
            ),
            input_time_mean=float(getattr(train_ds, "time_mean", 0.0)),
            input_time_std=float(getattr(train_ds, "time_std", 1.0)),
            input_loc_mean=tuple(
                float(x)
                for x in np.asarray(
                    getattr(train_ds, "loc_mean", np.zeros(spatial_dim, dtype=np.float32)),
                    dtype=np.float32,
                ).tolist()
            ),
            input_loc_std=tuple(
                float(x)
                for x in np.asarray(
                    getattr(train_ds, "loc_std", np.ones(spatial_dim, dtype=np.float32)),
                    dtype=np.float32,
                ).tolist()
            ),
            paper_dt_min=float(paper_stats["paper_dt_min"]),
            paper_dt_range=float(paper_stats["paper_dt_range"]),
            paper_loc_min=tuple(paper_stats["paper_loc_min"]),
            paper_loc_range=tuple(paper_stats["paper_loc_range"]),
        )
