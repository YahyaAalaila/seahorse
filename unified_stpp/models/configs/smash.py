"""SMASHConfig — construction config for the smash family."""

from __future__ import annotations

import copy
import dataclasses
import numpy as np
from typing import Any, ClassVar, Dict

from .base import BaseModelConfig, ConfigRegistry


def _as_float_tuple(value: Any, *, dim: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"Expected {dim} values, got {arr.size}.")
    return tuple(float(x) for x in arr.tolist())


def _compute_smash_token_stats(datasets: list[Any], spatial_dim: int) -> dict[str, Any]:
    event_time_raw_chunks: list[np.ndarray] = []
    event_time_log_chunks: list[np.ndarray] = []
    loc_chunks: list[np.ndarray] = []
    mark_min: int | None = None
    mark_max: int | None = None

    for ds in datasets:
        if ds is None:
            continue
        for seq in getattr(ds, "sequences", []):
            times = np.asarray(seq["times"], dtype=np.float64).reshape(-1)
            locs = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, spatial_dim)
            if times.size > 0:
                event_time = np.empty_like(times, dtype=np.float64)
                event_time[0] = times[0]
                if times.size > 1:
                    event_time[1:] = times[1:] - times[:-1]
                event_time_raw_chunks.append(event_time.astype(np.float32))
                event_time_log_chunks.append(
                    np.log(np.maximum(event_time, 1e-4)).astype(np.float32)
                )
            if locs.size > 0:
                loc_chunks.append(locs.astype(np.float32))
            marks = seq.get("marks")
            if marks is not None:
                arr = np.asarray(marks, dtype=np.int64).reshape(-1)
                if arr.size > 0:
                    cur_min = int(arr.min())
                    cur_max = int(arr.max())
                    mark_min = cur_min if mark_min is None else min(mark_min, cur_min)
                    mark_max = cur_max if mark_max is None else max(mark_max, cur_max)

    def _min_range(chunks: list[np.ndarray]) -> tuple[float, float]:
        if not chunks:
            return 0.0, 1.0
        values = np.concatenate(chunks).astype(np.float32)
        vmin = float(values.min())
        vrange = float(max(values.max() - vmin, 1e-8))
        return vmin, vrange

    def _axis_min_range(chunks: list[np.ndarray]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if not chunks:
            zeros = tuple(0.0 for _ in range(spatial_dim))
            ones = tuple(1.0 for _ in range(spatial_dim))
            return zeros, ones
        values = np.concatenate(chunks, axis=0).astype(np.float32)
        vmin = values.min(axis=0)
        vrange = np.maximum(values.max(axis=0) - vmin, 1e-8)
        return tuple(float(x) for x in vmin.tolist()), tuple(float(x) for x in vrange.tolist())

    time_min_raw, time_range_raw = _min_range(event_time_raw_chunks)
    time_min_log, time_range_log = _min_range(event_time_log_chunks)
    loc_min, loc_range = _axis_min_range(loc_chunks)

    out: dict[str, Any] = {
        "token_time_min_raw": time_min_raw,
        "token_time_range_raw": time_range_raw,
        "token_time_min_log": time_min_log,
        "token_time_range_log": time_range_log,
        "token_loc_min": loc_min,
        "token_loc_range": loc_range,
    }
    if mark_max is not None:
        num_types = int(mark_max if (mark_min is not None and mark_min >= 1) else mark_max + 1)
        out["decoder"] = {"num_types": max(1, num_types)}
    return out


@ConfigRegistry.register("smash")
@dataclasses.dataclass
class SMASHConfig(BaseModelConfig):
    _STATE_MODEL: ClassVar[str] = "smash"
    _EVENT_MODEL: ClassVar[str] = "smash"
    _SUPPORTED_PROTOCOLS: ClassVar[frozenset] = frozenset({"raw", "standard"})

    # Encoder (TransformerST) params
    d_model: int = 64
    d_rnn: int = 256
    d_inner: int = 128
    enc_n_layers: int = 4
    n_head: int = 4
    d_k: int = 16
    d_v: int = 16
    enc_dropout: float = 0.1
    CosSin: bool = True
    # Decoder params
    sigma_time: float = 0.05
    sigma_loc: float = 0.05
    num_noise: int = 50
    sampling_timesteps: int = 500
    langevin_step: float = 0.005
    n_samples: int = 100
    sampling_method: str = "normal"
    loss_lambda: float = 0.5
    loss_lambda2: float = 1.0
    smooth: float = 0.0
    log_normalization: bool = True
    minmax_normalize_time: bool = True
    minmax_normalize_loc: bool = True
    mark_shift: int = 1
    # Marks
    num_types: int = 1
    # Input de-normalization stats for reconstructing raw coordinates locally.
    input_time_normalized: bool = False
    input_space_normalized: bool = False
    input_time_mean: float = 0.0
    input_time_std: float = 1.0
    input_loc_mean: tuple[float, ...] = (0.0, 0.0)
    input_loc_std: tuple[float, ...] = (1.0, 1.0)
    # Fixed global min-max stats for upstream-faithful SMASH tokens.
    token_time_min_raw: float = 0.0
    token_time_range_raw: float = 1.0
    token_time_min_log: float = 0.0
    token_time_range_log: float = 1.0
    token_loc_min: tuple[float, ...] = (0.0, 0.0)
    token_loc_range: tuple[float, ...] = (1.0, 1.0)

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
    ) -> "SMASHConfig":
        if spatial_dim != 2:
            raise ValueError(f"SMASH currently supports spatial_dim=2, got {spatial_dim}.")

        enc_cfg = copy.deepcopy(d.get("encoder", {}))
        enc_type = enc_cfg.pop("type", "smash_transformer")
        if enc_type not in {"smash_transformer", "smash_upstream_transformer"}:
            raise ValueError(
                "SMASH build expects encoder.type='smash_transformer' "
                f"(or 'smash_upstream_transformer'), got '{enc_type}'."
            )

        dec_cfg = copy.deepcopy(d.get("decoder", {}))
        dec_type = dec_cfg.pop("type", "smash")
        if dec_type != "smash":
            raise ValueError(f"SMASH build expects decoder.type='smash', got '{dec_type}'.")

        d_model = int(enc_cfg.get("d_model", 64))
        num_types = int(dec_cfg.pop("num_types", max(1, n_marks)))

        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            d_model=d_model,
            d_rnn=int(enc_cfg.get("d_rnn", 256)),
            d_inner=int(enc_cfg.get("d_inner", 128)),
            enc_n_layers=int(enc_cfg.get("n_layers", 4)),
            n_head=int(enc_cfg.get("n_head", 4)),
            d_k=int(enc_cfg.get("d_k", 16)),
            d_v=int(enc_cfg.get("d_v", 16)),
            enc_dropout=float(enc_cfg.get("dropout", 0.1)),
            CosSin=bool(enc_cfg.get("CosSin", True)),
            sigma_time=float(dec_cfg.pop("sigma_time", 0.05)),
            sigma_loc=float(dec_cfg.pop("sigma_loc", 0.05)),
            num_noise=int(dec_cfg.pop("num_noise", 50)),
            sampling_timesteps=int(dec_cfg.pop("samplingsteps", 500)),
            langevin_step=float(dec_cfg.pop("langevin_step", 0.005)),
            n_samples=int(dec_cfg.pop("n_samples", 100)),
            sampling_method=str(dec_cfg.pop("sampling_method", "normal")),
            loss_lambda=float(dec_cfg.pop("loss_lambda", 0.5)),
            loss_lambda2=float(dec_cfg.pop("loss_lambda2", 1.0)),
            smooth=float(dec_cfg.pop("smooth", 0.0)),
            log_normalization=bool(dec_cfg.pop("log_normalization", True)),
            minmax_normalize_time=bool(dec_cfg.pop("minmax_normalize_time", True)),
            minmax_normalize_loc=bool(dec_cfg.pop("minmax_normalize_loc", True)),
            mark_shift=int(dec_cfg.pop("mark_shift", 1)),
            num_types=num_types,
            input_time_normalized=bool(d.get("input_time_normalized", False)),
            input_space_normalized=bool(d.get("input_space_normalized", False)),
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
            token_time_min_raw=float(d.get("token_time_min_raw", 0.0)),
            token_time_range_raw=float(d.get("token_time_range_raw", 1.0)),
            token_time_min_log=float(d.get("token_time_min_log", 0.0)),
            token_time_range_log=float(d.get("token_time_range_log", 1.0)),
            token_loc_min=_as_float_tuple(
                d.get("token_loc_min"),
                dim=spatial_dim,
                default=tuple(0.0 for _ in range(spatial_dim)),
            ),
            token_loc_range=_as_float_tuple(
                d.get("token_loc_range"),
                dim=spatial_dim,
                default=tuple(1.0 for _ in range(spatial_dim)),
            ),
        )

    def _state_kwargs(self) -> dict:
        from unified_stpp.models.history_encoders import SMASHUpstreamTransformerST
        loc_dim = 3 if self.num_types > 1 else 2
        transformer = SMASHUpstreamTransformerST(
            d_model=self.d_model,
            d_rnn=self.d_rnn,
            d_inner=self.d_inner,
            n_layers=self.enc_n_layers,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            dropout=self.enc_dropout,
            device=None,
            loc_dim=loc_dim,
            CosSin=self.CosSin,
            num_types=self.num_types,
        )
        return dict(
            transformer=transformer,
            loc_dim=loc_dim,
            num_types=self.num_types,
            log_normalization=self.log_normalization,
            minmax_normalize_time=self.minmax_normalize_time,
            minmax_normalize_loc=self.minmax_normalize_loc,
            mark_shift=self.mark_shift,
            input_time_normalized=self.input_time_normalized,
            input_space_normalized=self.input_space_normalized,
            input_time_mean=self.input_time_mean,
            input_time_std=self.input_time_std,
            input_loc_mean=self.input_loc_mean,
            input_loc_std=self.input_loc_std,
            token_time_min_raw=self.token_time_min_raw,
            token_time_range_raw=self.token_time_range_raw,
            token_time_min_log=self.token_time_min_log,
            token_time_range_log=self.token_time_range_log,
            token_loc_min=self.token_loc_min,
            token_loc_range=self.token_loc_range,
        )

    def _event_kwargs(self) -> dict:
        from unified_stpp.models.event_models.smash_event import ScoreNet
        loc_dim = 3 if self.num_types > 1 else 2
        score_net = ScoreNet(
            dim=1 + loc_dim,
            condition=True,
            cond_dim=self.d_model,
            num_types=self.num_types,
        )
        return dict(
            score_net=score_net,
            sigma_time=self.sigma_time,
            sigma_loc=self.sigma_loc,
            seq_length=1 + loc_dim,
            num_noise=self.num_noise,
            sampling_timesteps=self.sampling_timesteps,
            langevin_step=self.langevin_step,
            n_samples=self.n_samples,
            sampling_method=self.sampling_method,
            num_types=self.num_types,
            loss_lambda=self.loss_lambda,
            loss_lambda2=self.loss_lambda2,
            smooth=self.smooth,
        )

    @classmethod
    def data_init_overrides(cls, dm) -> dict:
        """Compute upstream-faithful SMASH preprocessing stats from raw splits."""
        bundle = getattr(dm, "_bundle", None)
        if bundle is None:
            return {}

        train_ds = getattr(bundle, "train_dataset", None)
        val_ds = getattr(bundle, "val_dataset", None)
        test_ds = getattr(bundle, "test_dataset", None)
        if train_ds is None:
            return {}

        first_seq = next(iter(getattr(train_ds, "sequences", [])), None)
        if first_seq is not None:
            spatial_dim = int(np.asarray(first_seq["locations"]).shape[-1])
        else:
            spatial_dim = int(
                np.asarray(
                    getattr(train_ds, "loc_mean", np.zeros(2, dtype=np.float32)),
                    dtype=np.float32,
                ).shape[-1]
            )

        overrides = _compute_smash_token_stats(
            [train_ds, val_ds, test_ds],
            spatial_dim=spatial_dim,
        )
        overrides.update(
            {
                "input_time_normalized": bool(getattr(train_ds, "normalize_time", False)),
                "input_space_normalized": bool(getattr(train_ds, "normalize_space", False)),
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
