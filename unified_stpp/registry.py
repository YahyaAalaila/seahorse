"""
Registry and factory for building UnifiedSTPP models.

The repository now treats the coarse StateModel/EventModel path as the
single framework. Only actively maintained presets are exposed:
  - neural_stpp_attn_sc
  - neural_stpp_jump_sc
  - deep_stpp
  - auto_stpp
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from .models.decoders import (
    AutoIntDecoder,
    DeepSTPPDecoder,
    JumpCNFSpatial,
    SelfAttentiveCNFSpatial,
)
from .models.dynamics import IdentityDynamics
from .models.encoders import TransformerEncoder
from .models.event_models import (
    AutoSTPPEventModel,
    DeepSTPPEventModel,
    NeuralSTPPSequenceEventModel,
)
from .models.neural_tpp_backbone import NeuralTPPBackbone
from .models.state_models import (
    AutoSTPPStateModel,
    DeepSTPPStateModel,
    NeuralTPPBackboneStateModel,
)
from .models.unified_model import UnifiedSTPP


_NEURAL_SPATIAL_DECODER_REGISTRY = {
    "jump_cnf": JumpCNFSpatial,
    "self_attentive_cnf": SelfAttentiveCNFSpatial,
}


PRESETS: Dict[str, Dict[str, Any]] = {
    # Sequence-coupled JumpCNF with faithful NeuralPointProcess backbone.
    "neural_stpp_jump_sc": {
        "backbone": {
            "update_type": "gru",
            "solver": "dopri5",
            "atol": 1e-4,
            "rtol": 1e-4,
            "use_adjoint": False,
            "energy_regularization": 1e-4,
        },
        "decoder": {
            "spatial": {"type": "jump_cnf", "n_flows": 4},
        },
    },
    # Sequence-coupled SelfAttentiveCNF with faithful NeuralPointProcess backbone.
    "neural_stpp_attn_sc": {
        "backbone": {
            "update_type": "gru",
            "solver": "dopri5",
            "atol": 1e-4,
            "rtol": 1e-4,
            "use_adjoint": False,
            "energy_regularization": 1e-4,
        },
        "decoder": {
            "spatial": {
                "type": "self_attentive_cnf",
                "num_heads": 4,
                "n_hidden_layers": 2,
                "solver": "dopri5",
                "atol": 1e-4,
                "rtol": 1e-4,
                "otreg_strength": 1e-4,
            },
        },
    },
    "deep_stpp": {
        "encoder": {
            "type": "transformer",
            "num_heads": 2,
            "num_layers": 3,
            "dropout": 0.0,
        },
        "decoder": {
            "type": "deep_stpp",
            "seq_len": 20,
            "num_points": 20,
            "sigma_min": 1e-4,
            "n_layers": 3,
        },
    },
    "auto_stpp": {
        "encoder": {
            "type": "transformer",
            "num_heads": 2,
            "num_layers": 3,
            "dropout": 0.1,
        },
        "decoder": {
            "type": "autoint",
            "n_components": 8,
            "n_layers": 2,
            "internal_dim": 64,
            "x_lo": -3.5,
            "x_hi": 3.5,
            "y_lo": -3.5,
            "y_hi": 3.5,
        },
    },
}


def build_model(
    config: Dict[str, Any],
    spatial_dim: int = 2,
    hidden_dim: int = 128,
    event_cov_dim: int = 0,
    field_cov_dim: int = 0,
    preset: Optional[str] = None,
    n_marks: int = 0,
) -> UnifiedSTPP:
    """
    Build a UnifiedSTPP model from a configuration dict.

    Only active coarse-framework presets are supported.
    """
    del n_marks

    cfg = copy.deepcopy(config)
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available presets: {sorted(PRESETS)}"
            )
        base_config = copy.deepcopy(PRESETS[preset])
        _deep_update(base_config, cfg)
        cfg = base_config

    if "backbone" in cfg:
        return _build_neural_stpp_backbone_model(
            cfg=cfg,
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            field_cov_dim=field_cov_dim,
        )

    dec_type = cfg.get("decoder", {}).get("type")
    if dec_type == "deep_stpp":
        return _build_deep_stpp_model(
            cfg=cfg,
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            event_cov_dim=event_cov_dim,
            field_cov_dim=field_cov_dim,
        )
    if dec_type == "autoint":
        return _build_auto_stpp_model(
            cfg=cfg,
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            event_cov_dim=event_cov_dim,
            field_cov_dim=field_cov_dim,
        )

    raise ValueError(
        "Unsupported configuration: only neural_stpp_attn_sc, "
        "neural_stpp_jump_sc, deep_stpp, and auto_stpp are available."
    )


def _build_neural_stpp_backbone_model(
    *,
    cfg: Dict[str, Any],
    hidden_dim: int,
    spatial_dim: int,
    field_cov_dim: int,
) -> UnifiedSTPP:
    bb_cfg = copy.deepcopy(cfg["backbone"])
    backbone = NeuralTPPBackbone(
        hidden_dim=hidden_dim,
        spatial_dim=spatial_dim,
        **bb_cfg,
    )

    spat_cfg = copy.deepcopy(cfg["decoder"]["spatial"])
    spat_type = spat_cfg.pop("type")
    if spat_type not in _NEURAL_SPATIAL_DECODER_REGISTRY:
        raise ValueError(
            f"Unsupported neural spatial decoder '{spat_type}'. "
            f"Available: {sorted(_NEURAL_SPATIAL_DECODER_REGISTRY)}"
        )
    backbone_spatial = _NEURAL_SPATIAL_DECODER_REGISTRY[spat_type](
        spatial_dim=spatial_dim,
        hidden_dim=hidden_dim,
        field_cov_dim=field_cov_dim,
        **spat_cfg,
    )

    state_model = NeuralTPPBackboneStateModel(
        sequence_nll_and_states_fn=backbone.sequence_nll_and_states,
    )
    event_model = NeuralSTPPSequenceEventModel(
        spatial_sequence_nll_fn=backbone_spatial.sequence_nll,
        spatial_regularization_fn=lambda: getattr(backbone_spatial, "_energy_reg", 0.0),
    )

    return UnifiedSTPP(
        backbone=backbone,
        backbone_spatial=backbone_spatial,
        state_model=state_model,
        event_model=event_model,
        use_state_event_path=True,
        hidden_dim=hidden_dim,
    )


def _build_transformer_encoder(
    *,
    cfg: Dict[str, Any],
    hidden_dim: int,
    spatial_dim: int,
    event_cov_dim: int,
) -> TransformerEncoder:
    enc_cfg = copy.deepcopy(cfg.get("encoder", {}))
    enc_type = enc_cfg.pop("type", "transformer")
    if enc_type != "transformer":
        raise ValueError(
            f"Only Transformer encoder is supported in the coarse framework; got '{enc_type}'."
        )
    return TransformerEncoder(
        input_dim=1 + spatial_dim,
        hidden_dim=hidden_dim,
        event_cov_dim=event_cov_dim,
        **enc_cfg,
    )


def _build_deep_stpp_model(
    *,
    cfg: Dict[str, Any],
    hidden_dim: int,
    spatial_dim: int,
    event_cov_dim: int,
    field_cov_dim: int,
) -> UnifiedSTPP:
    encoder = _build_transformer_encoder(
        cfg=cfg,
        hidden_dim=hidden_dim,
        spatial_dim=spatial_dim,
        event_cov_dim=event_cov_dim,
    )
    dynamics = IdentityDynamics(hidden_dim=hidden_dim)

    dec_cfg = copy.deepcopy(cfg["decoder"])
    dec_type = dec_cfg.pop("type", "deep_stpp")
    if dec_type != "deep_stpp":
        raise ValueError(f"DeepSTPP build expects decoder.type='deep_stpp', got '{dec_type}'.")
    decoder = DeepSTPPDecoder(
        hidden_dim=hidden_dim,
        spatial_dim=spatial_dim,
        field_cov_dim=field_cov_dim,
        **dec_cfg,
    )

    model = UnifiedSTPP(
        encoder=encoder,
        dynamics=dynamics,
        decoder=decoder,
        vae=bool(cfg.get("vae", False)),
        hidden_dim=hidden_dim,
        use_state_event_path=True,
    )

    model.state_model = DeepSTPPStateModel(
        encode_fn=model._encode_legacy_history,
        vae_stats_fn=model._vae_project if model.vae else None,
        vae_reparameterize_fn=model._vae_reparameterize if model.vae else None,
    )
    model.event_model = DeepSTPPEventModel(
        decode_fn=model.decoder._decode,
        temporal_log_fn=model.decoder._log_ft,
        spatial_log_fn=model.decoder._log_s_intensity,
        background_fn=lambda: model.decoder.background,
        seq_len=model.decoder.seq_len,
        num_points=model.decoder.num_points,
        spatial_dim=spatial_dim,
        expose_decoded_params=bool(cfg.get("deep_stpp_expose_decoded_params", False)),
    )
    return model


def _build_auto_stpp_model(
    *,
    cfg: Dict[str, Any],
    hidden_dim: int,
    spatial_dim: int,
    event_cov_dim: int,
    field_cov_dim: int,
) -> UnifiedSTPP:
    encoder = _build_transformer_encoder(
        cfg=cfg,
        hidden_dim=hidden_dim,
        spatial_dim=spatial_dim,
        event_cov_dim=event_cov_dim,
    )
    dynamics = IdentityDynamics(hidden_dim=hidden_dim)

    dec_cfg = copy.deepcopy(cfg["decoder"])
    dec_type = dec_cfg.pop("type", "autoint")
    if dec_type != "autoint":
        raise ValueError(f"AutoSTPP build expects decoder.type='autoint', got '{dec_type}'.")
    decoder = AutoIntDecoder(
        hidden_dim=hidden_dim,
        spatial_dim=spatial_dim,
        field_cov_dim=field_cov_dim,
        **dec_cfg,
    )

    model = UnifiedSTPP(
        encoder=encoder,
        dynamics=dynamics,
        decoder=decoder,
        hidden_dim=hidden_dim,
        use_state_event_path=True,
    )

    model.state_model = AutoSTPPStateModel(
        encode_fn=model._encode_legacy_history,
        mark_embed_fn=None,
    )
    model.event_model = AutoSTPPEventModel(
        nll_fn=model.decoder.nll,
        log_prob_fn=model.decoder.log_prob,
        compensator_fn=model.decoder._compensator,
        mu_fn=model.decoder._mu,
    )
    return model


def _deep_update(base: dict, override: dict):
    """Recursively update base dict with override dict."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_update(base[key], val)
        else:
            base[key] = val
