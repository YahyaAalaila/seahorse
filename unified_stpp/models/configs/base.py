"""Abstract base for model-family construction configs.

A ModelFamilyConfig is a plain dataclass that owns:
  - the constructor parameters for one model family
  - parameter validation (__post_init__)
  - build_model() → fully wired UnifiedSTPP

It is NOT a Pydantic model and knows nothing about training, data, or devices.
Data-derived overrides (e.g. AutoSTPP bbox) are resolved by PresetDescriptor
*before* from_dict() is called and arrive as part of the merged dict.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Dict

if TYPE_CHECKING:
    from unified_stpp.models.unified_model import UnifiedSTPP


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

class ConfigRegistry:
    """Maps preset names to config classes.  Config classes self-register via
    the @ConfigRegistry.register("preset_name") decorator.
    """

    _registry: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator that registers a config class under a preset name."""
        def decorator(config_cls):
            cls._registry[name] = config_cls
            return config_cls
        return decorator

    @classmethod
    def build(cls, name: str, overrides: dict, **dims) -> "UnifiedSTPP":
        """Build a UnifiedSTPP from a preset name and override dict."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown preset {name!r}. Known presets: {sorted(cls._registry)}"
            )
        cfg_cls = cls._registry[name]
        return cfg_cls.from_dict(copy.deepcopy(overrides), **dims).build_model()

    @classmethod
    def resolve_accelerator(cls, name: str, requested: str) -> str:
        """Delegate accelerator resolution to the config class."""
        cfg_cls = cls._registry.get(name)
        if cfg_cls is None:
            return requested
        return cfg_cls.resolve_accelerator(requested)

    @classmethod
    def preset_names(cls) -> list[str]:
        return list(cls._registry)

    @classmethod
    def data_init_overrides(cls, name: str, dm) -> dict:
        """Return data-dependent config overrides for *name* (e.g. bbox for auto_stpp).

        Delegates to the config class's ``data_init_overrides(dm)`` classmethod.
        Returns ``{}`` for presets that have no data-dependent init.
        """
        cfg_cls = cls._registry.get(name)
        if cfg_cls is None:
            return {}
        return cfg_cls.data_init_overrides(dm)


# ---------------------------------------------------------------------------
# Base config class
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BaseModelConfig:
    """Abstract base. Subclasses implement build_model()."""

    hidden_dim: int = 128
    spatial_dim: int = 2

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
    ) -> "BaseModelConfig":
        """Instantiate from an already-merged config dict.

        Subclasses override this to extract their parameters from *d*.
        The keyword arguments carry the positional dimensions that callers
        (i.e. registry.build_model) supply alongside the YAML/override dict.
        """
        raise NotImplementedError

    def build_model(self) -> "UnifiedSTPP":
        """Return a fully wired UnifiedSTPP.

        All construction parameters are already on self — no arguments needed.
        """
        raise NotImplementedError

    @classmethod
    def resolve_accelerator(cls, requested: str) -> str:
        """Return the accelerator that should actually be used.

        Override in subclasses that have hardware constraints (e.g. ODE-based
        models that require float64 and cannot run on MPS).
        """
        return requested

    @classmethod
    def data_init_overrides(cls, dm) -> dict:
        """Return config overrides derived from the fitted data module.

        Override in subclasses that require data-dependent initialization
        (e.g. spatial bounding box computed from training statistics).
        Returns ``{}`` by default.
        """
        return {}
