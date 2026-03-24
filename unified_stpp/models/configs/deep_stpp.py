"""DeepSTPPConfig — construction config for the deep_stpp family."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict

from .base import BaseModelConfig, ConfigRegistry


@ConfigRegistry.register("deep_stpp")
@dataclasses.dataclass
class DeepSTPPConfig(BaseModelConfig):
    # Encoder params
    enc_num_heads: int = 2
    enc_num_layers: int = 3
    enc_dropout: float = 0.0
    # Decoder params
    seq_len: int = 20
    num_points: int = 20
    sigma_min: float = 1e-4
    dec_n_layers: int = 3
    # Dimension params (not from YAML; set by registry at instantiation)
    event_cov_dim: int = 0
    field_cov_dim: int = 0
    # Optional
    vae: bool = False
    expose_decoded_params: bool = False

    # Carry the raw sub-dicts so the build step has access to any extra keys
    # that are not explicitly declared above (forward-compat).
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

        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            event_cov_dim=event_cov_dim,
            field_cov_dim=field_cov_dim,
            enc_num_heads=int(enc_cfg.pop("num_heads", 2)),
            enc_num_layers=int(enc_cfg.pop("num_layers", 3)),
            enc_dropout=float(enc_cfg.pop("dropout", 0.0)),
            seq_len=int(dec_cfg.pop("seq_len", 20)),
            num_points=int(dec_cfg.pop("num_points", 20)),
            sigma_min=float(dec_cfg.pop("sigma_min", 1e-4)),
            dec_n_layers=int(dec_cfg.pop("n_layers", 3)),
            vae=bool(d.get("vae", False)),
            expose_decoded_params=bool(d.get("deep_stpp_expose_decoded_params", False)),
            _enc_cfg=enc_cfg,   # residual keys (forward-compat)
            _dec_cfg=dec_cfg,
        )

    def build_model(self):
        from unified_stpp.models.event_models import DeepSTPPEventModel
        from unified_stpp.models.state_models import DeepSTPPStateModel
        from unified_stpp.models.unified_model import UnifiedSTPP

        state_model = DeepSTPPStateModel(
            hidden_dim=self.hidden_dim,
            spatial_dim=self.spatial_dim,
            event_cov_dim=self.event_cov_dim,
            enc_num_heads=self.enc_num_heads,
            enc_num_layers=self.enc_num_layers,
            enc_dropout=self.enc_dropout,
            vae=self.vae,
            **self._enc_cfg,
        )
        event_model = DeepSTPPEventModel(
            hidden_dim=self.hidden_dim,
            spatial_dim=self.spatial_dim,
            field_cov_dim=self.field_cov_dim,
            seq_len=self.seq_len,
            num_points=self.num_points,
            sigma_min=self.sigma_min,
            n_layers=self.dec_n_layers,
            expose_decoded_params=self.expose_decoded_params,
            **self._dec_cfg,
        )
        return UnifiedSTPP(
            state_model=state_model,
            event_model=event_model,
            hidden_dim=self.hidden_dim,
        )
