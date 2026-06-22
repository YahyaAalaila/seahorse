"""DiffusionSTPPConfig — construction config for the diffusion_stpp family."""

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


def _compute_token_stats_from_datasets(datasets: list[Any], spatial_dim: int) -> dict[str, Any]:
    delta_chunks: list[np.ndarray] = []
    loc_chunks: list[np.ndarray] = []

    for ds in datasets:
        if ds is None:
            continue
        for seq in getattr(ds, "sequences", []):
            times = np.asarray(seq["times"], dtype=np.float64).reshape(-1)
            locs = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, spatial_dim)
            if times.size > 1:
                delta_chunks.append(np.diff(times))
            if locs.size > 0:
                loc_chunks.append(locs)

    if delta_chunks:
        delta_all = np.concatenate(delta_chunks).astype(np.float32)
        delta_min = float(delta_all.min())
        delta_range = float(max(delta_all.max() - delta_min, 1e-8))
    else:
        delta_min = 0.0
        delta_range = 1.0

    if loc_chunks:
        loc_all = np.concatenate(loc_chunks, axis=0).astype(np.float32)
        loc_min = loc_all.min(axis=0)
        loc_range = np.maximum(loc_all.max(axis=0) - loc_min, 1e-8)
    else:
        loc_min = np.zeros(spatial_dim, dtype=np.float32)
        loc_range = np.ones(spatial_dim, dtype=np.float32)

    return {
        "token_delta_t_min": delta_min,
        "token_delta_t_range": delta_range,
        "token_loc_min": tuple(float(x) for x in loc_min.tolist()),
        "token_loc_range": tuple(float(x) for x in loc_range.tolist()),
    }


@ConfigRegistry.register("diffusion_stpp")
@dataclasses.dataclass
class DiffusionSTPPConfig(BaseModelConfig):
    _STATE_MODEL: ClassVar[str] = "diffusion_stpp"
    _EVENT_MODEL: ClassVar[str] = "diffusion_stpp"
    _SUPPORTED_PROTOCOLS: ClassVar[frozenset] = frozenset({"raw", "standard"})

    # Encoder params
    d_model: int = 64
    d_rnn: int = 256
    d_inner: int = 128
    enc_n_layers: int = 4
    n_head: int = 4
    d_k: int = 16
    d_v: int = 16
    enc_dropout: float = 0.1
    CosSin: bool = True
    # Diffusion denoising network params
    hidden_units: int = 64
    # GaussianDiffusionST params
    timesteps: int = 100
    sampling_timesteps: int = 100
    objective: str = "pred_noise"
    beta_schedule: str = "cosine"
    loss_type: str = "l2"
    # Normalisation for diff_img tokens
    minmax_normalize_time: bool = True
    minmax_normalize_loc: bool = True
    # Input de-normalization stats for reconstructing raw coordinates before
    # diffusion token construction.
    input_normalized: bool = False
    input_time_mean: float = 0.0
    input_time_std: float = 1.0
    input_loc_mean: tuple[float, ...] = (0.0, 0.0)
    input_loc_std: tuple[float, ...] = (1.0, 1.0)
    # Fixed global min-max stats for the diffusion token space.
    token_delta_t_min: float = 0.0
    token_delta_t_range: float = 1.0
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
    ) -> "DiffusionSTPPConfig":
        if spatial_dim != 2:
            raise ValueError(
                f"diffusion_stpp currently supports spatial_dim=2, got {spatial_dim}."
            )
        if n_marks > 0:
            raise ValueError(
                "diffusion_stpp is unmarked only (n_marks must be 0)."
            )

        enc_cfg = copy.deepcopy(d.get("encoder", {}))
        enc_type = enc_cfg.pop("type", "dstpp_transformer")
        if enc_type not in {"dstpp_transformer", "smash_transformer"}:
            raise ValueError(
                "diffusion_stpp expects encoder.type='dstpp_transformer' "
                f"(compatibility alias 'smash_transformer' also accepted), got {enc_type!r}."
            )

        dec_cfg = copy.deepcopy(d.get("decoder", {}))
        dec_type = dec_cfg.pop("type", "diffusion_stpp")
        if dec_type != "diffusion_stpp":
            raise ValueError(
                f"diffusion_stpp expects decoder.type='diffusion_stpp', got {dec_type!r}."
            )

        d_model = int(enc_cfg.get("d_model", 64))

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
            hidden_units=int(dec_cfg.pop("hidden_units", 64)),
            timesteps=int(dec_cfg.pop("timesteps", 100)),
            sampling_timesteps=int(dec_cfg.pop("sampling_timesteps", 100)),
            objective=str(dec_cfg.pop("objective", "pred_noise")),
            beta_schedule=str(dec_cfg.pop("beta_schedule", "cosine")),
            loss_type=str(dec_cfg.pop("loss_type", "l2")),
            minmax_normalize_time=bool(d.get("minmax_normalize_time", True)),
            minmax_normalize_loc=bool(d.get("minmax_normalize_loc", True)),
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
            token_delta_t_min=float(d.get("token_delta_t_min", 0.0)),
            token_delta_t_range=float(d.get("token_delta_t_range", 1.0)),
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
        from seahorse.models.history_encoders.dstpp_transformer import DSTPPTransformerST
        transformer = DSTPPTransformerST(
            d_model=self.d_model,
            d_rnn=self.d_rnn,
            d_inner=self.d_inner,
            n_layers=self.enc_n_layers,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            dropout=self.enc_dropout,
            device=None,
            loc_dim=self.spatial_dim,
            CosSin=self.CosSin,
        )
        return dict(
            transformer=transformer,
            spatial_dim=self.spatial_dim,
            minmax_normalize_time=self.minmax_normalize_time,
            minmax_normalize_loc=self.minmax_normalize_loc,
            input_normalized=self.input_normalized,
            input_time_mean=self.input_time_mean,
            input_time_std=self.input_time_std,
            input_loc_mean=self.input_loc_mean,
            input_loc_std=self.input_loc_std,
            token_delta_t_min=self.token_delta_t_min,
            token_delta_t_range=self.token_delta_t_range,
            token_loc_min=self.token_loc_min,
            token_loc_range=self.token_loc_range,
        )

    def _event_kwargs(self) -> dict:
        from seahorse.models.event_models.diffusion import STDiffusionNet
        cond_dim = self.d_model
        seq_length = 1 + self.spatial_dim  # delta_time + spatial coords
        denoising_model = STDiffusionNet(
            n_steps=self.timesteps,
            dim=seq_length,
            num_units=self.hidden_units,
            condition=True,
            cond_dim=cond_dim,
        )
        return dict(
            denoising_model=denoising_model,
            seq_length=seq_length,
            timesteps=self.timesteps,
            sampling_timesteps=self.sampling_timesteps,
            objective=self.objective,
            beta_schedule=self.beta_schedule,
            loss_type=self.loss_type,
        )

    @classmethod
    def data_init_overrides(cls, dm) -> dict:
        """Compute fixed global token stats from raw split sequences.

        Diffusion tokens use a stable [0,1] coordinate system built from raw
        delta-times and locations across the available splits. The state model
        reconstructs raw coordinates from the repository-standard tensors and
        feeds the encoder its expected input space internally.
        """
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
            spatial_dim = int(np.asarray(getattr(train_ds, "loc_mean", [0.0, 0.0])).shape[-1])

        overrides = _compute_token_stats_from_datasets(
            [train_ds, val_ds, test_ds],
            spatial_dim=spatial_dim,
        )
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
