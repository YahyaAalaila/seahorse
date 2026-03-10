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
    DeepSTPPDecoder,
    DiffusionDecoder,
    MLPMarkDecoder,
    AttentionMarkDecoder,
    AutoIntDecoder,
    JumpCNFSpatial,
    SelfAttentiveCNFSpatial,
)
from .models.covariates import LiftingMap, MarkEmbedding
from .models.unified_model import UnifiedSTPP
from .models.neural_tpp_backbone import NeuralTPPBackbone


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
    "jump_cnf": JumpCNFSpatial,
    "self_attentive_cnf": SelfAttentiveCNFSpatial,
}

MARK_DECODER_REGISTRY = {
    "mlp": MLPMarkDecoder,
    "attention": AttentionMarkDecoder,
}


# ============================================================================
# Predefined configurations matching published methods
# ============================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    # SelfAttentiveCNF variant — main model from Chen et al. 2021 (NeuralSTPP).
    # Key paper faithfulness choices:
    #   - AttentionEncoder (not GRU) for self-attention over event history.
    #   - AttentionUpdater at each event arrival.
    #   - ConcatSquash velocity field (layer_type="concat", 3 hidden layers).
    #   - Self-attentive base N(μ_attn, I) for the CNF (base_type="self_attentive").
    #   - ODE tolerance 1e-4 (paper default --tol 1e-4).
    "neural_stpp": {
        "encoder": {"type": "attention", "num_heads": 4, "num_layers": 2},
        "dynamics": {
            "type": "neural_ode",
            "solver": "dopri5",
            "atol": 1e-4,
            "rtol": 1e-4,
            # Jointly integrate [z(t), Λ(t)] so the compensator uses the
            # actually-evolved state z(τ) at each quadrature point τ.
            "augmented": True,
            # Standard backprop through ODE steps; avoids the second backward
            # ODE pass, halving training time on CPU.
            "use_adjoint": False,
        },
        "updater": {"type": "attention", "num_heads": 4},
        "decoder": {
            "type": "factorized",
            "temporal": {"type": "cumulative_hazard", "n_quad_points": 20},
            "spatial": {
                "type": "cnf",
                "solver": "dopri5",
                "atol": 1e-4,
                "rtol": 1e-4,
                "layer_type": "concat",      # ConcatSquash — paper default
                "n_hidden_layers": 3,         # hdims="64-64-64"
                "base_type": "self_attentive",  # SelfAttentiveCNF key contribution
                "history_k": 20,
            },
        },
    },
    # JumpCNF variant — second model from Chen et al. 2021.
    # GRU encoder + GRU jump update; plain N(0,I) base with ConcatSquash flow.
    "neural_stpp_jump": {
        "encoder": {"type": "gru", "num_layers": 1},
        "dynamics": {
            "type": "neural_ode",
            "solver": "dopri5",
            "atol": 1e-4,
            "rtol": 1e-4,
            "augmented": True,
            "use_adjoint": False,
        },
        "updater": {"type": "gru_jump"},
        "decoder": {
            "type": "factorized",
            "temporal": {"type": "cumulative_hazard", "n_quad_points": 20},
            "spatial": {
                "type": "cnf",
                "solver": "dopri5",
                "atol": 1e-4,
                "rtol": 1e-4,
                "layer_type": "concat",
                "n_hidden_layers": 3,
                "base_type": "standard",  # JumpCNF uses standard N(0,I) base
            },
        },
    },
    # DeepSTPP faithful to Lin et al. 2021.
    #
    # Original pipeline:
    #   TransformerEncoder (sinusoidal time PE on cumsum(Δt)) → VAE → z
    #   z → w_dec / b_dec / s_dec → M = seq_len + num_points Hawkes kernels
    #   λ(s,t) = λ_t(t) · f(s|t)
    #   λ_t(t) = Σ w_i exp(−b_i (t−tᵢ))             [Hawkes temporal, closed compensator]
    #   f(s|t) = Σ (vᵢ/Σvⱼ) N(s; sᵢ, diag(σᵢ))    [GMM weights tied to temporal vᵢ]
    #
    # The coupled decoder (DeepSTPPDecoder) replaces the previous factorized
    # LogNormalMixtureTemporal + DataCenteredGaussianSpatial pair, which was
    # decoupled and used the wrong temporal model.
    "deep_stpp": {
        "encoder": {"type": "transformer", "num_heads": 2, "num_layers": 3, "dropout": 0.0},
        "dynamics": {"type": "identity"},
        "updater": {"type": "attention", "num_heads": 2},
        "decoder": {
            "type": "deep_stpp",   # → DeepSTPPDecoder (coupled Hawkes + GMM)
            "seq_len": 20,         # paper: seq_len=20
            "num_points": 20,      # paper: num_points=20
            "sigma_min": 1e-4,     # paper: s_min=1e-4 (in MinMax [0,1] space)
            "n_layers": 3,         # paper: decoder_n_layer=3
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
        # Use the same Transformer/attention backbone family as deep_stpp;
        # only the decoder differs (AutoInt vs factorized lognormal+spatial).
        "encoder": {"type": "transformer", "num_heads": 2, "num_layers": 3, "dropout": 0.1},
        "dynamics": {"type": "identity"},
        "updater": {"type": "attention", "num_heads": 2},
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
    # Sequence-coupled JumpCNF with faithful NeuralPointProcess backbone.
    #
    # Temporal: NeuralTPPBackbone owns the full temporal process:
    #   - learnable _init_state (no encoder)
    #   - time-independent ODE dh/dt = tanh(net(h)), zero-init last layer
    #   - intensity sigmoid(linear(h) − 2) × 50  (IntensityODEFunc faithful)
    #   - joint [h, Λ] ODE; jump update h+ = GRUCell(s, h) spatial-only
    # Spatial:  JumpCNFSpatial — backward-chained radial flows, O(T²).
    "neural_stpp_jump_sc": {
        "backbone": {
            "update_type": "gru",
            "solver": "dopri5",
            "atol": 1e-4,
            "rtol": 1e-4,
            "use_adjoint": False,
            "energy_regularization": 1e-4,  # --tpp_otreg_strength 1e-4
        },
        "decoder": {
            "spatial": {"type": "jump_cnf", "n_flows": 4},
        },
    },
    # Sequence-coupled SelfAttentiveCNF with faithful NeuralPointProcess backbone.
    #
    # Same temporal backbone as neural_stpp_jump_sc.
    # Spatial: SelfAttentiveCNFSpatial — cross-event attention context then
    #   L independent ConcatSquash CNFs in one batched ODE solve.
    "neural_stpp_attn_sc": {
        "backbone": {
            "update_type": "gru",
            "solver": "dopri5",
            "atol": 1e-4,
            "rtol": 1e-4,
            "use_adjoint": False,
            "energy_regularization": 1e-4,  # --tpp_otreg_strength 1e-4
        },
        "decoder": {
            "spatial": {
                "type": "self_attentive_cnf",
                "num_heads": 4,
                "n_hidden_layers": 2,
                "solver": "dopri5",
                "atol": 1e-4,
                "rtol": 1e-4,
                "otreg_strength": 1e-4,  # --otreg_strength 1e-4
            },
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

    # ------------------------------------------------------------------ #
    # Backbone path: NeuralTPP faithful presets (neural_stpp_jump_sc, _attn_sc)
    # These bypass encoder / dynamics / updater / temporal-decoder entirely.
    # ------------------------------------------------------------------ #
    if "backbone" in config:
        bb_cfg = config["backbone"].copy()
        backbone = NeuralTPPBackbone(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            **bb_cfg,
        )

        spat_cfg = config["decoder"]["spatial"].copy()
        spat_type = spat_cfg.pop("type")
        backbone_spatial = SPATIAL_DECODER_REGISTRY[spat_type](
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            field_cov_dim=field_cov_dim,
            **spat_cfg,
        )

        return UnifiedSTPP(
            backbone=backbone,
            backbone_spatial=backbone_spatial,
            hidden_dim=hidden_dim,
        )

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
            dynamics.aug_func = AugmentedODEFunc(
                dynamics.func, temporal._intensity, intensity_module=temporal
            )
    elif dec_type == "deep_stpp":
        dec_cfg_clean = {k: v for k, v in dec_cfg.items() if k != "type"}
        decoder = DeepSTPPDecoder(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            field_cov_dim=field_cov_dim,
            **dec_cfg_clean,
        )
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
        vae=config.get("vae", False),
        hidden_dim=hidden_dim,
    )

    return model


def _deep_update(base: dict, override: dict):
    """Recursively update base dict with override dict."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
