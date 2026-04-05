"""Compatibility shim for the historical top-level model factory.

The primary live construction path is ``ConfigRegistry.build(...)`` in the
active model-config layer. This module preserves ``build_model(...)`` and
``PRESETS`` at their historical import path for tests and older utilities
without depending on the archived package tree.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Importing ``unified_stpp.models.configs`` triggers preset registration via the
# ConfigRegistry decorators defined across the family config modules.
from unified_stpp.models.configs import ConfigRegistry
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
    """Build a ``UnifiedSTPP`` from overrides and a registered preset name."""
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


PRESETS: Dict[str, Dict[str, Any]] = {name: {} for name in ConfigRegistry.preset_names()}

__all__ = ["build_model", "PRESETS"]
