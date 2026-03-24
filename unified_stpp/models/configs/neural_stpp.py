"""NeuralSTPPConfig — construction config for the neural_stpp family.

Covers both neural_stpp_jump_sc and neural_stpp_attn_sc presets.
The spatial decoder type is determined by decoder.spatial.type in the
merged config dict and is dispatched via the same registry as before.
"""

from __future__ import annotations

import copy
import dataclasses
import warnings
from typing import Any, ClassVar, Dict

from .base import BaseModelConfig, ConfigRegistry

_SPATIAL_DECODER_REGISTRY: Dict[str, Any] = {}  # populated lazily


def _get_spatial_registry() -> Dict[str, Any]:
    if not _SPATIAL_DECODER_REGISTRY:
        from unified_stpp.models.spatial_models import JumpCNFSpatial, SelfAttentiveCNFSpatial
        _SPATIAL_DECODER_REGISTRY.update({
            "jump_cnf": JumpCNFSpatial,
            "self_attentive_cnf": SelfAttentiveCNFSpatial,
        })
    return _SPATIAL_DECODER_REGISTRY


@dataclasses.dataclass
class NeuralSTPPConfig(BaseModelConfig):
    # Defaults for backbone and spatial decoder — overridden by each subclass.
    _BACKBONE_DEFAULTS: ClassVar[dict] = {}
    _SPATIAL_DEFAULTS: ClassVar[dict] = {}

    # Backbone params (carried as raw dict for full forward-compat)
    backbone_cfg: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)
    # Spatial decoder params
    spatial_type: str = "jump_cnf"
    spatial_cfg: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)
    # Dimension params
    field_cov_dim: int = 0

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
    ) -> "NeuralSTPPConfig":
        bb_cfg = copy.deepcopy(cls._BACKBONE_DEFAULTS)
        bb_cfg.update(d.get("backbone", {}))

        spat_cfg = copy.deepcopy(cls._SPATIAL_DEFAULTS)
        spat_cfg.update(d.get("decoder", {}).get("spatial", {}))
        spat_type = spat_cfg.pop("type")

        registry = _get_spatial_registry()
        if spat_type not in registry:
            raise ValueError(
                f"Unsupported neural spatial decoder '{spat_type}'. "
                f"Available: {sorted(registry)}"
            )

        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            field_cov_dim=field_cov_dim,
            backbone_cfg=bb_cfg,
            spatial_type=spat_type,
            spatial_cfg=spat_cfg,
        )

    def build_model(self):
        from unified_stpp.models.event_models import NeuralSTPPEventModel
        from unified_stpp.models.state_models import NeuralSTPPStateModel
        from unified_stpp.models.unified_model import UnifiedSTPP

        state_model = NeuralSTPPStateModel(
            hidden_dim=self.hidden_dim,
            spatial_dim=self.spatial_dim,
            **self.backbone_cfg,
        )
        event_model = NeuralSTPPEventModel(
            hidden_dim=self.hidden_dim,
            spatial_dim=self.spatial_dim,
            field_cov_dim=self.field_cov_dim,
            spatial_type=self.spatial_type,
            **self.spatial_cfg,
        )
        return UnifiedSTPP(
            state_model=state_model,
            event_model=event_model,
            hidden_dim=self.hidden_dim,
        )


def _neural_stpp_resolve_accelerator(requested: str) -> str:
    """Fall back to CPU when MPS is requested: torchdiffeq requires float64."""
    if requested != "auto":
        return requested
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
    except AttributeError:
        mps_available = False
    if mps_available:
        warnings.warn(
            "MPS detected but torchdiffeq requires float64, which MPS does not support. "
            "Falling back to CPU for this preset.",
            UserWarning,
            stacklevel=3,
        )
        return "cpu"
    return requested


_NEURAL_STPP_BACKBONE_DEFAULTS: dict = {
    "update_type": "gru",
    "solver": "dopri5",
    "atol": 1e-4,
    "rtol": 1e-4,
    "use_adjoint": False,
    "energy_regularization": 1e-4,
}


@ConfigRegistry.register("neural_stpp_jump_sc")
@dataclasses.dataclass
class NeuralSTPPJumpSCConfig(NeuralSTPPConfig):
    _BACKBONE_DEFAULTS: ClassVar[dict] = _NEURAL_STPP_BACKBONE_DEFAULTS
    _SPATIAL_DEFAULTS: ClassVar[dict] = {"type": "jump_cnf", "n_flows": 4}

    @classmethod
    def resolve_accelerator(cls, requested: str) -> str:
        return _neural_stpp_resolve_accelerator(requested)


@ConfigRegistry.register("neural_stpp_attn_sc")
@dataclasses.dataclass
class NeuralSTPPAttnSCConfig(NeuralSTPPConfig):
    _BACKBONE_DEFAULTS: ClassVar[dict] = _NEURAL_STPP_BACKBONE_DEFAULTS
    _SPATIAL_DEFAULTS: ClassVar[dict] = {
        "type": "self_attentive_cnf",
        "num_heads": 4,
        "n_hidden_layers": 2,
        "solver": "dopri5",
        "atol": 1e-4,
        "rtol": 1e-4,
        "otreg_strength": 1e-4,
    }

    @classmethod
    def resolve_accelerator(cls, requested: str) -> str:
        return _neural_stpp_resolve_accelerator(requested)
