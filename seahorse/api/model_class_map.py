"""Friendly model-name mapping for the lightweight estimator API."""

from __future__ import annotations


FRIENDLY_TO_PRESET: dict[str, str] = {
    "PoissonGMM": "poisson_gmm",
    "PoissonCNF": "poisson_cnf",
    "PoissonTVCNF": "poisson_tvcnf",
    "HawkesGMM": "hawkes_gmm",
    "HawkesCNF": "hawkes_cnf",
    "HawkesTVCNF": "hawkes_tvcnf",
    "SelfCorrectingGMM": "selfcorrecting_gmm",
    "SelfCorrectingCNF": "selfcorrecting_cnf",
    "SelfCorrectingTVCNF": "selfcorrecting_tvcnf",
    "RMTPPGMM": "rmtpp_gmm",
    "THPGMM": "thp_gmm",
    "NeuralSTPP": "neural_stpp_attn_sc",
    "NeuralJumpSC": "neural_stpp_jump_sc",
    "NeuralAttnSC": "neural_stpp_attn_sc",
    "NeuralJumpCNF": "neural_jumpcnf",
    "NeuralAttnCNF": "neural_attncnf",
    "NeuralCondGMM": "neural_cond_gmm",
    "NJSDE": "njsde",
    "DeepSTPP": "deep_stpp",
    "AutoSTPP": "auto_stpp",
    "SMASH": "smash",
    "DiffusionSTPP": "diffusion_stpp",
    "NSMPP": "nsmpp",
}


def _registry_names() -> set[str]:
    from seahorse.models.configs import ConfigRegistry

    return set(ConfigRegistry.accepted_preset_names())


def available_friendly_names() -> dict[str, str]:
    """Return friendly aliases whose target presets exist in this build."""
    accepted = _registry_names()
    return {
        friendly: preset
        for friendly, preset in FRIENDLY_TO_PRESET.items()
        if preset in accepted
    }


def resolve_preset(model_class: str) -> str:
    """Resolve a friendly model name or accepted preset name to a preset."""
    aliases = available_friendly_names()
    if model_class in aliases:
        return aliases[model_class]

    from seahorse.models.configs import ConfigRegistry

    if ConfigRegistry.is_registered(model_class):
        return ConfigRegistry.resolve_name(model_class)

    available = sorted(set(aliases) | set(ConfigRegistry.accepted_preset_names()))
    raise ValueError(f"Unknown model class {model_class!r}. Available: {available}")


def list_available_models() -> list[str]:
    """List friendly aliases and accepted preset names for this build."""
    return sorted(set(available_friendly_names()) | _registry_names())


def friendly_name_for_preset(preset: str) -> str:
    """Return a friendly display name for a preset when one is known."""
    for friendly, mapped in available_friendly_names().items():
        if mapped == preset:
            return friendly
    return preset
