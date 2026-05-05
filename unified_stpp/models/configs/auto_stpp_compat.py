"""Construction config for the compatibility AutoSTPP preset."""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Dict

import numpy as np

from .base import BaseModelConfig, ConfigRegistry

if TYPE_CHECKING:
    from unified_stpp.training.data_module import STPPDataModule

# Percentile margins used when estimating the bounding box from data.
_PERCENTILE_LO = 1
_PERCENTILE_HI = 99
_MARGIN = 0.5
_MAX_BATCHES = 20
_FALLBACK_BOX = {"x_lo": -2.0, "x_hi": 2.0, "y_lo": -2.0, "y_hi": 2.0}


def _bbox_from_dm(dm: "STPPDataModule") -> dict:
    """Estimate spatial bounding box from normalized training data."""
    locs = []
    loader = dm.train_dataloader()
    for i, batch in enumerate(loader):
        if i >= _MAX_BATCHES:
            break
        t = batch["locations"]   # (B, T, 2)
        lens = batch["lengths"]  # (B,)
        for b in range(t.shape[0]):
            valid = t[b, : int(lens[b].item())]
            locs.append(valid.detach().cpu().numpy())

    if not locs:
        return dict(_FALLBACK_BOX)

    pts = np.concatenate(locs, axis=0)  # (N, 2)
    return {
        "x_lo": float(np.percentile(pts[:, 0], _PERCENTILE_LO)) - _MARGIN,
        "x_hi": float(np.percentile(pts[:, 0], _PERCENTILE_HI)) + _MARGIN,
        "y_lo": float(np.percentile(pts[:, 1], _PERCENTILE_LO)) - _MARGIN,
        "y_hi": float(np.percentile(pts[:, 1], _PERCENTILE_HI)) + _MARGIN,
    }


@ConfigRegistry.register("auto_stpp_legacy", status="legacy")
@dataclasses.dataclass
class AutoSTPPCompatConfig(BaseModelConfig):
    _STATE_MODEL: ClassVar[str] = "auto_stpp_legacy"
    _EVENT_MODEL: ClassVar[str] = "auto_stpp_legacy"
    _SUPPORTED_PROTOCOLS: ClassVar[frozenset] = frozenset({"raw", "standard", "sliding_window"})

    # Encoder params
    enc_num_heads: int = 2
    enc_num_layers: int = 3
    enc_dropout: float = 0.1
    # Decoder (AutoInt) params
    n_components: int = 8
    n_layers: int = 2
    internal_dim: int = 64
    x_lo: float = -3.5
    x_hi: float = 3.5
    y_lo: float = -3.5
    y_hi: float = 3.5
    # Dimension params
    event_cov_dim: int = 0
    field_cov_dim: int = 0

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
    ) -> "AutoSTPPCompatConfig":
        enc_cfg = copy.deepcopy(d.get("encoder", {}))
        enc_type = enc_cfg.pop("type", "transformer")
        if enc_type != "transformer":
            raise ValueError(
                f"Compatibility AutoSTPP build expects encoder.type='transformer', got '{enc_type}'."
            )

        dec_cfg = copy.deepcopy(d.get("decoder", {}))
        dec_type = dec_cfg.pop("type", "autoint")
        if dec_type != "autoint":
            raise ValueError(
                f"Compatibility AutoSTPP build expects decoder.type='autoint', got '{dec_type}'."
            )

        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            event_cov_dim=event_cov_dim,
            field_cov_dim=field_cov_dim,
            enc_num_heads=int(enc_cfg.pop("num_heads", 2)),
            enc_num_layers=int(enc_cfg.pop("num_layers", 3)),
            enc_dropout=float(enc_cfg.pop("dropout", 0.1)),
            n_components=int(dec_cfg.pop("n_components", 8)),
            n_layers=int(dec_cfg.pop("n_layers", 2)),
            internal_dim=int(dec_cfg.pop("internal_dim", 64)),
            x_lo=float(dec_cfg.pop("x_lo", -3.5)),
            x_hi=float(dec_cfg.pop("x_hi", 3.5)),
            y_lo=float(dec_cfg.pop("y_lo", -3.5)),
            y_hi=float(dec_cfg.pop("y_hi", 3.5)),
            _enc_cfg=enc_cfg,
            _dec_cfg=dec_cfg,
        )

    def _state_kwargs(self) -> dict:
        return dict(
            hidden_dim=self.hidden_dim,
            spatial_dim=self.spatial_dim,
            event_cov_dim=self.event_cov_dim,
            enc_num_heads=self.enc_num_heads,
            enc_num_layers=self.enc_num_layers,
            enc_dropout=self.enc_dropout,
            **self._enc_cfg,
        )

    def _event_kwargs(self) -> dict:
        return dict(
            hidden_dim=self.hidden_dim,
            spatial_dim=self.spatial_dim,
            field_cov_dim=self.field_cov_dim,
            n_components=self.n_components,
            n_layers=self.n_layers,
            internal_dim=self.internal_dim,
            x_lo=self.x_lo,
            x_hi=self.x_hi,
            y_lo=self.y_lo,
            y_hi=self.y_hi,
            **self._dec_cfg,
        )

    @classmethod
    def data_init_overrides(cls, dm) -> dict:
        """Compute spatial bounding box from normalized training data."""
        bbox = _bbox_from_dm(dm)
        return {"decoder": bbox}
