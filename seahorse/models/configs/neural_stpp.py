"""Construction config for the shared-hidden Neural STPP family."""

from __future__ import annotations

import copy
import dataclasses
import warnings
from typing import Any, ClassVar, Dict

import numpy as np

from seahorse.data.transforms import ZScoreTransformArtifact
from .base import BaseModelConfig, ConfigRegistry


@dataclasses.dataclass
class NeuralSTPPConfig(BaseModelConfig):
    # Defaults for backbone and spatial decoder — overridden by each subclass.
    _BACKBONE_DEFAULTS: ClassVar[dict] = {}
    _SPATIAL_DEFAULTS: ClassVar[dict] = {}
    _STATE_MODEL: ClassVar[str] = "neural_stpp"
    _EVENT_MODEL: ClassVar[str] = "neural_stpp"

    # Data-registry declarations
    _TRAIN_LOADER: ClassVar[str] = "batch_by_size"
    _SUPPORTED_PROTOCOLS: ClassVar[frozenset] = frozenset({"raw", "standard"})

    # Backbone params (carried as raw dict for full forward-compat)
    backbone_cfg: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)
    # Spatial decoder params
    spatial_type: str = "jump_cnf"
    spatial_cfg: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)
    # Dimension params
    field_cov_dim: int = 0
    input_transform: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

    @staticmethod
    def _parse_temporal_hidden_dims(cfg: Dict[str, Any], fallback_hidden_dim: int) -> tuple[int, ...]:
        from seahorse.models.temporal_models.neural_point_process import _normalize_hidden_dims

        return tuple(
            _normalize_hidden_dims(
                cfg.get("tpp_hidden_dims"),
                fallback=fallback_hidden_dim,
            )
        )

    def _resolved_temporal_hidden_dim(self) -> int:
        return int(self._parse_temporal_hidden_dims(self.backbone_cfg, self.hidden_dim)[0])

    def _resolved_temporal_hdim(self) -> int:
        temporal_hidden_dim = self._resolved_temporal_hidden_dim()
        return int(self.backbone_cfg.get("temporal_hdim", temporal_hidden_dim // 2))

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
        bb_cfg["tpp_hidden_dims"] = list(cls._parse_temporal_hidden_dims(bb_cfg, hidden_dim))

        spat_cfg = copy.deepcopy(cls._SPATIAL_DEFAULTS)
        spat_cfg.update(d.get("decoder", {}).get("spatial", {}))
        spat_type = spat_cfg.pop("type")

        # Validate spatial_type early — get_spatial_cls raises ValueError for unknown keys
        from seahorse.models.model_registry import get_spatial_cls
        get_spatial_cls(spat_type)

        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            field_cov_dim=field_cov_dim,
            backbone_cfg=bb_cfg,
            spatial_type=spat_type,
            spatial_cfg=spat_cfg,
            input_transform=copy.deepcopy(d.get("input_transform", {})),
        )

    def _state_kwargs(self) -> dict:
        return dict(
            hidden_dim=self._resolved_temporal_hidden_dim(),
            spatial_dim=self.spatial_dim,
            input_transform=self.input_transform,
            **self.backbone_cfg,
        )

    def _event_kwargs(self) -> dict:
        temporal_hidden_dim = self._resolved_temporal_hidden_dim()
        temporal_hdim = self._resolved_temporal_hdim()
        return dict(
            hidden_dim=temporal_hidden_dim,
            spatial_dim=self.spatial_dim,
            field_cov_dim=self.field_cov_dim,
            spatial_type=self.spatial_type,
            temporal_hdim=temporal_hdim,
            spatial_aux_dim=max(0, temporal_hidden_dim - temporal_hdim),
            **self.spatial_cfg,
        )

    def build_model(self):
        """Build the shared Neural STPP family with optional runtime wiring.

        JumpCNF's faithful ``solve_reverse`` path needs access to the shared
        temporal hidden-state dynamics object. That dependency belongs here in
        the model-family construction layer, not in the CLI or runner.
        """
        from seahorse.models.model_registry import get_event_cls
        from seahorse.models.unified_model import UnifiedSTPP

        state_model = self.build_state_model()
        event_kwargs = dict(self._event_kwargs())
        if (
            self.spatial_type == "neural_jumpcnf"
            and bool(event_kwargs.get("solve_reverse", False))
        ):
            event_kwargs["aux_odefunc"] = state_model.temporal_core.hidden_state_dynamics
        event_model = get_event_cls(self._EVENT_MODEL)(**event_kwargs)
        return UnifiedSTPP(
            state_model=state_model,
            event_model=event_model,
            hidden_dim=self.hidden_dim,
        )

    @classmethod
    def data_init_overrides(cls, dm) -> dict:
        ds = getattr(dm, "train_dataset", None)
        if ds is None:
            return {}
        if getattr(ds, "coordinate_space", None) == "raw":
            return {}
        return {
            "backbone": {
                "normalize_time_inputs": bool(getattr(ds, "normalize_time", False)),
                "normalize_space_inputs": bool(getattr(ds, "normalize_space", False)),
                "time_mean": float(getattr(ds, "time_mean", 0.0)),
                "time_std": float(getattr(ds, "time_std", 1.0)),
            }
        }

    @classmethod
    def fit_transform_artifact(cls, dm):
        ds = getattr(dm, "train_dataset", None)
        if ds is None:
            return None
        if getattr(ds, "coordinate_space", None) != "raw":
            return None
        sequences = list(getattr(ds, "sequences", []))
        first_seq = next(iter(sequences), None)
        if first_seq is not None:
            spatial_dim = int(np.asarray(first_seq["locations"]).shape[-1])
            all_locs = np.concatenate(
                [np.asarray(seq["locations"], dtype=np.float32).reshape(-1, spatial_dim) for seq in sequences],
                axis=0,
            )
            loc_mean_arr = all_locs.mean(axis=0).astype(np.float32)
            loc_std_arr = (all_locs.std(axis=0) + 1e-8).astype(np.float32)
        else:
            spatial_dim = 2
            loc_mean_arr = np.zeros(spatial_dim, dtype=np.float32)
            loc_std_arr = np.ones(spatial_dim, dtype=np.float32)
        return ZScoreTransformArtifact(
            normalize_time=False,
            normalize_space=True,
            time_mean=0.0,
            time_std=1.0,
            loc_mean=tuple(float(x) for x in loc_mean_arr.tolist()),
            loc_std=tuple(float(x) for x in loc_std_arr.tolist()),
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
    "tpp_hidden_dims": [32, 32],
    "tpp_cond": True,
    "tpp_style": "gru",
    "share_hidden": True,
    "tpp_actfn": "softplus",
    "update_type": "gru",
    "solver": "dopri5",
    "atol": 1e-4,
    "rtol": 1e-4,
    "use_adjoint": False,
    "energy_regularization": 1e-4,
    "normalize_time_inputs": False,
    "normalize_space_inputs": False,
    "time_mean": 0.0,
    "time_std": 1.0,
}


@ConfigRegistry.register("neural_stpp_jump_sc", status="legacy")
@dataclasses.dataclass
class NeuralSTPPJumpSCConfig(NeuralSTPPConfig):
    _BACKBONE_DEFAULTS: ClassVar[dict] = _NEURAL_STPP_BACKBONE_DEFAULTS
    _SPATIAL_DEFAULTS: ClassVar[dict] = {"type": "jump_cnf", "n_flows": 4}

    @classmethod
    def resolve_accelerator(cls, requested: str) -> str:
        return _neural_stpp_resolve_accelerator(requested)


@ConfigRegistry.register("neural_stpp_attn_sc", status="legacy")
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


@ConfigRegistry.register("njsde", status="canonical")
@dataclasses.dataclass
class NeuralSTPPSharedCondGMMConfig(NeuralSTPPConfig):
    _BACKBONE_DEFAULTS: ClassVar[dict] = _NEURAL_STPP_BACKBONE_DEFAULTS
    _SPATIAL_DEFAULTS: ClassVar[dict] = {
        "type": "conditional_gmm",
        "hidden_dims": [64, 64, 64],
        "n_mixtures": 5,
        "actfn": "softplus",
    }

    @classmethod
    def resolve_accelerator(cls, requested: str) -> str:
        return _neural_stpp_resolve_accelerator(requested)


@ConfigRegistry.register("neural_jumpcnf", status="canonical")
@dataclasses.dataclass
class NeuralSTPPSharedJumpCNFConfig(NeuralSTPPConfig):
    _BACKBONE_DEFAULTS: ClassVar[dict] = _NEURAL_STPP_BACKBONE_DEFAULTS
    _SPATIAL_DEFAULTS: ClassVar[dict] = {
        "type": "neural_jumpcnf",
        "hidden_dims": [64, 64, 64],
        "layer_type": "concat",
        "actfn": "swish",
        "zero_init": True,
        "tol": 1e-4,
        "otreg_strength": 1e-4,
        "use_adjoint": False,
        "solve_reverse": True,
        "n_flows": 4,
    }

    @classmethod
    def resolve_accelerator(cls, requested: str) -> str:
        return _neural_stpp_resolve_accelerator(requested)


@ConfigRegistry.register("neural_attncnf", status="canonical")
@dataclasses.dataclass
class NeuralSTPPSharedAttnCNFConfig(NeuralSTPPConfig):
    _BACKBONE_DEFAULTS: ClassVar[dict] = _NEURAL_STPP_BACKBONE_DEFAULTS
    _SPATIAL_DEFAULTS: ClassVar[dict] = {
        "type": "neural_attncnf",
        "hidden_dims": [64, 64, 64],
        "layer_type": "concat",
        "actfn": "swish",
        "zero_init": True,
        "l2_attn": True,
        "naive_hutch": False,
        "tol": 1e-4,
        "otreg_strength": 1e-4,
        "nblocks": 2,
        "num_heads": 4,
    }

    @classmethod
    def resolve_accelerator(cls, requested: str) -> str:
        return _neural_stpp_resolve_accelerator(requested)


ConfigRegistry.register_alias("neural_cond_gmm", "njsde", status="deprecated")
ConfigRegistry.register_alias("neural_stpp_shared_cond_gmm", "njsde", status="deprecated")
ConfigRegistry.register_alias("neural_stpp_shared_jumpcnf", "neural_jumpcnf", status="deprecated")
ConfigRegistry.register_alias("neural_stpp_shared_attncnf", "neural_attncnf", status="deprecated")
