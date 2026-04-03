"""
Archived compatibility wrapper for model construction.

The primary live construction path is ``ConfigRegistry.build(...)`` in the
active model-config layer. This module preserves the older top-level
``build_model(...)`` convenience surface for compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Importing configs triggers @ConfigRegistry.register decorators.
from unified_stpp.models.configs import (  # noqa: F401
    AutoSTPPConfig,
    ConfigRegistry,
    DeepSTPPConfig,
    DiffusionSTPPConfig,
    NeuralSTPPAttnSCConfig,
    NeuralSTPPJumpSCConfig,
    SMASHConfig,
)
from unified_stpp.models.unified_model import UnifiedSTPP


def build_model(
    config: Dict[str, Any],
    spatial_dim: int = 2,
    hidden_dim: int = 128,
    event_cov_dim: int = 0,
    field_cov_dim: int = 0,
    preset: Optional[str] = None,
    n_marks: int = 0,
) -> UnifiedSTPP:
    """Build a UnifiedSTPP from a configuration dict and optional preset name."""
    if preset is None:
        raise ValueError(
            "build_model() requires a preset name. "
            f"Available: {sorted(ConfigRegistry.preset_names())}"
        )
    return ConfigRegistry.build(
        preset,
        overrides=config,
        hidden_dim=hidden_dim,
        spatial_dim=spatial_dim,
        event_cov_dim=event_cov_dim,
        field_cov_dim=field_cov_dim,
        n_marks=n_marks,
    )


# Backward-compat shim used by some utilities/tests.
PRESETS: Dict[str, Dict[str, Any]] = {name: {} for name in ConfigRegistry.preset_names()}

__all__ = ["build_model", "PRESETS"]
