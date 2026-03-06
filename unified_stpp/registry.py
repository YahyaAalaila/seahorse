"""
Registry and factory for building unified STPP models from configuration.
"""

from typing import Dict, Any, Optional

from .models.base import Encoder, Dynamics, Updater, Decoder
from .models.encoders import GRUEncoder, AttentionEncoder, TransformerEncoder
from .models.dynamics import IdentityDynamics, NeuralODEDynamics, AugmentedODEFunc
from .models.updaters import GRUJumpUpdater, AttentionUpdater
from .models.decoders import (
    FactorizedDecoder,
    CumulativeHazardTemporal,
    LogNormalMixtureTemporal,
    CNFSpatial,
    GaussianMixtureSpatial,
    DataCenteredGaussianSpatial,
    DiffusionDecoder,
    MLPMarkDecoder,
    AttentionMarkDecoder,
    AutoIntDecoder,
)
from .models.covariates import LiftingMap, MarkEmbedding
from .models.unified_model import UnifiedSTPP


ENCODER_REGISTRY = {
    "gru": GRUEncoder,
    "attention": AttentionEncoder,
    "transformer": TransformerEncoder,
}

DYNAMICS_REGISTRY = {
    "identity": IdentityDynamics,
    "neural_ode": NeuralODEDynamics,
}

UPDATER_REGISTRY = {
    "gru_jump": GRUJumpUpdater,
    "attention": AttentionUpdater,
}

TEMPORAL_DECODER_REGISTRY = {
    "cumulative_hazard": CumulativeHazardTemporal,
    "lognormal_mixture": LogNormalMixtureTemporal,
}

SPATIAL_DECODER_REGISTRY = {
    "cnf": CNFSpatial,
    "gaussian_mixture": GaussianMixtureSpatial,
    "data_centered_gaussian": DataCenteredGaussianSpatial,
}

MARK_DECODER_REGISTRY = {
    "mlp": MLPMarkDecoder,
    "attention": AttentionMarkDecoder,
}


# ============================================================================
# Predefined configurations matching published methods
# ============================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    "neural_stpp": {
        "encoder": {"type": "gru", "num_layers": 1},
        "dynamics": {
            "type": "neural_ode",
            "solver": "dopri5",
            # Jointly integrate [z(t), Λ(t)] in one ODE solve so the compensator
            # uses the actually-evolved state z(τ) at each quadrature point τ,
            # not z frozen at the event time (which causes NLL explosion).
            "augmented": True,
            # Standard backprop through ODE steps rather than adjoint; avoids the
            # second backward ODE pass, halving training time on CPU.
            "use_adjoint": False,
        },
        "updater": {"type": "gru_jump"},
        "decoder": {
            "type": "factorized",
            "temporal": {"type": "cumulative_hazard", "n_quad_points": 20},
            "spatial": {"type": "cnf", "solver": "dopri5"},
        },
    },
    # DeepSTPP in the unified framework (Lin et al. 2021):
    #   TransformerEncoder (sinusoidal time PE) +
    #   LogNormalMixtureTemporal +
    #   DataCenteredGaussianSpatial (event-location Gaussians + background anchors).
    "deep_stpp": {
        "encoder": {"type": "transformer", "num_heads": 2, "num_layers": 3, "dropout": 0.1},
        "dynamics": {"type": "identity"},
        "updater": {"type": "attention", "num_heads": 2},
        "decoder": {
            "type": "factorized",
            "temporal": {"type": "lognormal_mixture", "n_components": 16},
            "spatial": {
                "type": "data_centered_gaussian",
                "seq_len": 20,
                "num_points": 20,
                "sigma_min": 0.3,
            },
        },
    },
    # Free-GMM variant: attention encoder + LogNormal mixture + unconstrained GMM spatial.
    "deep_stpp_free": {
        "encoder": {"type": "attention", "num_heads": 4, "num_layers": 2},
        "dynamics": {"type": "identity"},
        "updater": {"type": "attention", "num_heads": 4},
        "decoder": {
            "type": "factorized",
            "temporal": {"type": "lognormal_mixture", "n_components": 16},
            "spatial": {"type": "gaussian_mixture", "n_components": 16},
        },
    },
    "dstpp": {
        "encoder": {"type": "transformer", "num_heads": 4, "num_layers": 2},
        "dynamics": {"type": "identity"},
        "updater": {"type": "attention", "num_heads": 4},
        "decoder": {
            "type": "diffusion",
            "n_noise_levels": 50,
            "sigma_min": 0.01,
            "sigma_max": 5.0,
        },
    },
    "auto_stpp": {
        "encoder": {"type": "gru", "num_layers": 1},
        "dynamics": {"type": "identity"},
        "updater": {"type": "gru_jump"},
        "decoder": {
            "type": "autoint",
            "n_components": 8,
            "n_layers": 2,
            "internal_dim": 64,
            # Bounding box in z-scored coordinates (STPPDataModule normalize=True).
            # After z-scoring, data has mean≈0, std≈1; ±3.5σ covers >99.9% of
            # the distribution without inflating the compensator integral.
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

    Args:
        config: Configuration dict (or empty if using preset).
        spatial_dim: Dimension of spatial coordinates.
        hidden_dim: Latent state dimension.
        event_cov_dim: Dimension of event-level covariates.
        field_cov_dim: Dimension of field covariates.
        preset: Name of a preset configuration (e.g., "neural_stpp").
        n_marks: Number of discrete mark types (0 = unmarked, no mark components).
    """
    if preset is not None:
        base_config = PRESETS[preset].copy()
        # Deep merge: config overrides preset
        _deep_update(base_config, config)
        config = base_config

    input_dim = 1 + spatial_dim  # time + space

    # ------------------------------------------------------------------ #
    # Mark embedding: increases effective event_cov_dim seen by encoder   #
    # and updater so mark types are visible without changing their APIs.  #
    # ------------------------------------------------------------------ #
    mark_embedding = None
    mark_decoder = None
    effective_event_cov_dim = event_cov_dim

    if n_marks > 0:
        embed_dim = min(n_marks * 2, hidden_dim // 2)
        mark_embedding = MarkEmbedding(n_marks=n_marks, embed_dim=embed_dim)
        effective_event_cov_dim = event_cov_dim + embed_dim

        # Mark decoder: default to MLP unless overridden in config
        mark_dec_cfg = config.get("mark_decoder", {"type": "mlp", "n_layers": 2}).copy()
        mark_dec_type = mark_dec_cfg.pop("type", "mlp")
        mark_decoder = MARK_DECODER_REGISTRY[mark_dec_type](
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            n_marks=n_marks,
            field_cov_dim=field_cov_dim,
            **mark_dec_cfg,
        )

    # Build encoder (uses effective_event_cov_dim to see mark embeddings)
    enc_cfg = config["encoder"]
    
    enc_type = enc_cfg.pop("type")
    print(f"Building encoder of type '{enc_type}' for preset = {preset}, ")
    encoder = ENCODER_REGISTRY[enc_type](
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        event_cov_dim=effective_event_cov_dim,
        **enc_cfg,
    )
    enc_cfg["type"] = enc_type  # restore

    # Build dynamics
    dyn_cfg = config["dynamics"]
    dyn_type = dyn_cfg.pop("type")
    dynamics = DYNAMICS_REGISTRY[dyn_type](
        hidden_dim=hidden_dim,
        field_cov_dim=field_cov_dim,
        **dyn_cfg,
    )
    dyn_cfg["type"] = dyn_type

    # Build updater (uses effective_event_cov_dim to see mark embeddings)
    upd_cfg = config["updater"]
    upd_type = upd_cfg.pop("type")
    updater = UPDATER_REGISTRY[upd_type](
        hidden_dim=hidden_dim,
        spatial_dim=spatial_dim,
        event_cov_dim=effective_event_cov_dim,
        field_cov_dim=field_cov_dim,
        **upd_cfg,
    )
    upd_cfg["type"] = upd_type

    # Build decoder
    dec_cfg = config["decoder"]
    dec_type = dec_cfg.get("type", "factorized")

    if dec_type == "factorized":
        # Build temporal and spatial sub-decoders
        temp_cfg = dec_cfg["temporal"].copy()
        temp_type = temp_cfg.pop("type")
        temporal = TEMPORAL_DECODER_REGISTRY[temp_type](
            hidden_dim=hidden_dim,
            field_cov_dim=field_cov_dim,
            **temp_cfg,
        )

        spat_cfg = dec_cfg["spatial"].copy()
        spat_type = spat_cfg.pop("type")
        spatial = SPATIAL_DECODER_REGISTRY[spat_type](
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            field_cov_dim=field_cov_dim,
            **spat_cfg,
        )

        decoder = FactorizedDecoder(
            temporal_decoder=temporal,
            spatial_decoder=spatial,
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
        )

        # Wire up augmented ODE: if dynamics requested augmented mode but was
        # built before the temporal decoder existed, connect intensity_fn now.
        if (
            isinstance(dynamics, NeuralODEDynamics)
            and dynamics.augmented
            and dynamics.aug_func is None
            and hasattr(temporal, '_intensity')
        ):
            dynamics.intensity_fn = temporal._intensity
            dynamics.aug_func = AugmentedODEFunc(dynamics.func, temporal._intensity)
    elif dec_type == "diffusion":
        decoder = DiffusionDecoder(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            field_cov_dim=field_cov_dim,
            n_noise_levels=dec_cfg.get("n_noise_levels", 50),
            sigma_min=dec_cfg.get("sigma_min", 0.01),
            sigma_max=dec_cfg.get("sigma_max", 5.0),
        )
    elif dec_type == "autoint":
        autoint_cfg = {k: v for k, v in dec_cfg.items() if k != "type"}
        decoder = AutoIntDecoder(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            field_cov_dim=field_cov_dim,
            **autoint_cfg,
        )
    else:
        raise ValueError(f"Unknown decoder type: {dec_type}")

    # Build lifting map if needed
    lifting_map = None
    if event_cov_dim > 0 and field_cov_dim == 0:
        lifting_map = LiftingMap(
            event_cov_dim=event_cov_dim,
            output_dim=event_cov_dim,
            spatial_dim=spatial_dim,
        )

    model = UnifiedSTPP(
        encoder=encoder,
        dynamics=dynamics,
        updater=updater,
        decoder=decoder,
        lifting_map=lifting_map,
        mark_decoder=mark_decoder,
        mark_embedding=mark_embedding,
    )

    return model


def _deep_update(base: dict, override: dict):
    """Recursively update base dict with override dict."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
