"""
Registry and factory for building UnifiedSTPP models.

Preset registration is self-contained: each config class registers itself
via @ConfigRegistry.register("preset_name") in models/configs/*.py.
Importing this module triggers those registrations by importing the config
package.

Actively maintained presets:
  - neural_stpp_attn_sc
  - neural_stpp_jump_sc
  - deep_stpp
  - auto_stpp
  - smash
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Importing configs triggers @ConfigRegistry.register decorators.
from .models.configs import (  # noqa: F401
    AutoSTPPConfig,
    ConfigRegistry,
    DeepSTPPConfig,
    DiffusionSTPPConfig,
    NeuralSTPPAttnSCConfig,
    NeuralSTPPJumpSCConfig,
    SMASHConfig,
)
from .models.unified_model import UnifiedSTPP


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Backward-compat shim (used by config/schema.py validator)
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {name: {} for name in ConfigRegistry.preset_names()}
