"""DiffusionSTPPConfig — construction config for the diffusion_stpp family."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict

from .base import BaseModelConfig, ConfigRegistry


@ConfigRegistry.register("diffusion_stpp")
@dataclasses.dataclass
class DiffusionSTPPConfig(BaseModelConfig):
    # Encoder (TransformerST) params — same architecture as SMASH
    d_model: int = 128
    d_rnn: int = 64
    d_inner: int = 256
    enc_n_layers: int = 4
    n_head: int = 4
    d_k: int = 16
    d_v: int = 16
    enc_dropout: float = 0.1
    CosSin: bool = True
    # Diffusion denoising network params
    hidden_units: int = 64
    # GaussianDiffusionST params
    timesteps: int = 1000
    sampling_timesteps: int = 50
    objective: str = "pred_x0"
    beta_schedule: str = "cosine"
    loss_type: str = "l2"
    # Time normalisation
    minmax_normalize_time: bool = True

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
        enc_type = enc_cfg.pop("type", "smash_transformer")
        if enc_type != "smash_transformer":
            raise ValueError(
                f"diffusion_stpp expects encoder.type='smash_transformer', got {enc_type!r}."
            )

        dec_cfg = copy.deepcopy(d.get("decoder", {}))
        dec_type = dec_cfg.pop("type", "diffusion_stpp")
        if dec_type != "diffusion_stpp":
            raise ValueError(
                f"diffusion_stpp expects decoder.type='diffusion_stpp', got {dec_type!r}."
            )

        d_model = int(enc_cfg.get("d_model", hidden_dim))

        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            d_model=d_model,
            d_rnn=int(enc_cfg.get("d_rnn", d_model // 2)),
            d_inner=int(enc_cfg.get("d_inner", d_model * 2)),
            enc_n_layers=int(enc_cfg.get("n_layers", 4)),
            n_head=int(enc_cfg.get("n_head", 4)),
            d_k=int(enc_cfg.get("d_k", 16)),
            d_v=int(enc_cfg.get("d_v", 16)),
            enc_dropout=float(enc_cfg.get("dropout", 0.1)),
            CosSin=bool(enc_cfg.get("CosSin", True)),
            hidden_units=int(dec_cfg.pop("hidden_units", 64)),
            timesteps=int(dec_cfg.pop("timesteps", 1000)),
            sampling_timesteps=int(dec_cfg.pop("sampling_timesteps", 50)),
            objective=str(dec_cfg.pop("objective", "pred_x0")),
            beta_schedule=str(dec_cfg.pop("beta_schedule", "cosine")),
            loss_type=str(dec_cfg.pop("loss_type", "l2")),
            minmax_normalize_time=bool(d.get("minmax_normalize_time", True)),
        )

    def build_model(self):
        from unified_stpp.models.history_encoders.transformer_st import TransformerST
        from unified_stpp.models.event_models.diffusion_event import (
            DiffusionEventModel,
            STDiffusionNet,
        )
        from unified_stpp.models.state_models.diffusion_state import DiffusionStateModel
        from unified_stpp.models.unified_model import UnifiedSTPP

        loc_dim = self.spatial_dim  # 2 for unmarked

        transformer = TransformerST(
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
            num_types=1,
        )

        # TransformerST concatenates 3 streams → output dim = 3 * d_model
        cond_dim = 3 * self.d_model
        seq_length = 1 + self.spatial_dim  # delta_time + spatial coords

        denoising_model = STDiffusionNet(
            seq_length=seq_length,
            hidden_units=self.hidden_units,
            condition=True,
            cond_dim=cond_dim,
        )

        state_model = DiffusionStateModel(
            transformer=transformer,
            spatial_dim=self.spatial_dim,
            minmax_normalize_time=self.minmax_normalize_time,
        )

        event_model = DiffusionEventModel(
            denoising_model=denoising_model,
            seq_length=seq_length,
            timesteps=self.timesteps,
            sampling_timesteps=self.sampling_timesteps,
            objective=self.objective,
            beta_schedule=self.beta_schedule,
            loss_type=self.loss_type,
        )

        return UnifiedSTPP(
            state_model=state_model,
            event_model=event_model,
            hidden_dim=self.hidden_dim,
        )
