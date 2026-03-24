"""SMASHConfig — construction config for the smash family."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict

from .base import BaseModelConfig, ConfigRegistry


@ConfigRegistry.register("smash")
@dataclasses.dataclass
class SMASHConfig(BaseModelConfig):
    # Encoder (TransformerST) params
    d_model: int = 128
    d_rnn: int = 512
    d_inner: int = 256
    enc_n_layers: int = 4
    n_head: int = 4
    d_k: int = 16
    d_v: int = 16
    enc_dropout: float = 0.1
    CosSin: bool = True
    # Decoder params
    sigma_time: float = 0.1
    sigma_loc: float = 0.1
    num_noise: int = 50
    sampling_timesteps: int = 500
    langevin_step: float = 0.05
    n_samples: int = 300
    sampling_method: str = "normal"
    loss_lambda: float = 1.0
    loss_lambda2: float = 1.0
    smooth: float = 0.0
    log_normalization: bool = False
    minmax_normalize_time: bool = True
    mark_shift: int = 1
    # Marks
    num_types: int = 1

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
        if enc_type != "smash_transformer":
            raise ValueError(
                f"SMASH build expects encoder.type='smash_transformer', got '{enc_type}'."
            )

        dec_cfg = copy.deepcopy(d.get("decoder", {}))
        dec_type = dec_cfg.pop("type", "smash")
        if dec_type != "smash":
            raise ValueError(f"SMASH build expects decoder.type='smash', got '{dec_type}'.")

        d_model = int(enc_cfg.get("d_model", hidden_dim))
        num_types = int(dec_cfg.pop("num_types", max(1, n_marks)))

        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            d_model=d_model,
            d_rnn=int(enc_cfg.get("d_rnn", d_model * 4)),
            d_inner=int(enc_cfg.get("d_inner", d_model * 2)),
            enc_n_layers=int(enc_cfg.get("n_layers", 4)),
            n_head=int(enc_cfg.get("n_head", 4)),
            d_k=int(enc_cfg.get("d_k", 16)),
            d_v=int(enc_cfg.get("d_v", 16)),
            enc_dropout=float(enc_cfg.get("dropout", 0.1)),
            CosSin=bool(enc_cfg.get("CosSin", True)),
            sigma_time=float(dec_cfg.pop("sigma_time", 0.1)),
            sigma_loc=float(dec_cfg.pop("sigma_loc", 0.1)),
            num_noise=int(dec_cfg.pop("num_noise", 50)),
            sampling_timesteps=int(dec_cfg.pop("samplingsteps", 500)),
            langevin_step=float(dec_cfg.pop("langevin_step", 0.05)),
            n_samples=int(dec_cfg.pop("n_samples", 300)),
            sampling_method=str(dec_cfg.pop("sampling_method", "normal")),
            loss_lambda=float(dec_cfg.pop("loss_lambda", 1.0)),
            loss_lambda2=float(dec_cfg.pop("loss_lambda2", 1.0)),
            smooth=float(dec_cfg.pop("smooth", 0.0)),
            log_normalization=bool(dec_cfg.pop("log_normalization", False)),
            minmax_normalize_time=bool(dec_cfg.pop("minmax_normalize_time", True)),
            mark_shift=int(dec_cfg.pop("mark_shift", 1)),
            num_types=num_types,
        )

    def build_model(self):
        from unified_stpp.models.history_encoders import TransformerST
        from unified_stpp.models.event_models.smash_event import ScoreNet, SMASHEventModel
        from unified_stpp.models.state_models import SMASHStateModel
        from unified_stpp.models.unified_model import UnifiedSTPP

        is_marked = self.num_types > 1
        loc_dim = 3 if is_marked else 2

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
            num_types=self.num_types,
        )

        score_net = ScoreNet(
            dim=1 + loc_dim,
            condition=True,
            cond_dim=self.d_model,
            num_types=self.num_types,
        )

        state_model = SMASHStateModel(
            transformer=transformer,
            loc_dim=loc_dim,
            num_types=self.num_types,
            log_normalization=self.log_normalization,
            minmax_normalize_time=self.minmax_normalize_time,
            mark_shift=self.mark_shift,
        )

        event_model = SMASHEventModel(
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

        return UnifiedSTPP(
            state_model=state_model,
            event_model=event_model,
            hidden_dim=self.hidden_dim,
        )
