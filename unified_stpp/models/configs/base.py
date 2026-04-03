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

from unified_stpp.data.transforms import CoordinateTransformArtifact
from unified_stpp.utils import deep_update

if TYPE_CHECKING:
    from unified_stpp.models.abstractions import EventModel, StateModel
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
        overrides = copy.deepcopy(cfg_cls.data_init_overrides(dm) or {})
        artifact = cfg_cls.fit_transform_artifact(dm)
        if artifact is not None:
            deep_update(overrides, cfg_cls.transform_init_overrides(artifact))
        return overrides

    @classmethod
    def get_collate_key(cls, name: str) -> str:
        """Return the collate-function registry key declared by preset *name*."""
        cfg_cls = cls._registry.get(name)
        if cfg_cls is None:
            return "canonical"
        return cfg_cls._COLLATE

    @classmethod
    def get_train_loader_key(cls, name: str) -> str:
        """Return the train-loader-builder registry key declared by preset *name*."""
        cfg_cls = cls._registry.get(name)
        if cfg_cls is None:
            return "fixed_batch"
        return cfg_cls._TRAIN_LOADER

    @classmethod
    def get_supported_protocols(cls, name: str) -> "frozenset[str]":
        """Return the set of dataset protocols supported by preset *name*.

        An empty frozenset means all protocols are allowed.
        """
        cfg_cls = cls._registry.get(name)
        if cfg_cls is None:
            return frozenset()
        return cfg_cls._SUPPORTED_PROTOCOLS


# ---------------------------------------------------------------------------
# Base config class
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BaseModelConfig:
    """Abstract base for model-family construction configs.

    Subclasses must implement:
    - ``from_dict(d, *, hidden_dim, spatial_dim, ...)`` — parse a merged dict
    - ``_STATE_MODEL`` ClassVar — registry key for the state model class
    - ``_EVENT_MODEL`` ClassVar — registry key for the event model class
    - ``_state_kwargs()`` — kwargs forwarded to the state model constructor
    - ``_event_kwargs()`` — kwargs forwarded to the event model constructor

    The base ``build_model()`` uses these to assemble a ``UnifiedSTPP``.
    Subclasses with non-trivial construction (e.g. constructor injection of
    pre-built modules) may override ``build_model()`` directly as an escape hatch.
    """

    hidden_dim: int = 128
    spatial_dim: int = 2

    # Model-registry keys — must be overridden in subclasses.
    _STATE_MODEL: ClassVar[str] = ""
    _EVENT_MODEL: ClassVar[str] = ""

    # Data-registry keys — override in subclasses that need non-default behaviour.
    _COLLATE: ClassVar[str] = "canonical"
    _TRAIN_LOADER: ClassVar[str] = "fixed_batch"
    _SUPPORTED_PROTOCOLS: ClassVar[frozenset] = frozenset()  # empty = all allowed

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

    # ------------------------------------------------------------------
    # Subclass contract: declare what to build and with which kwargs
    # ------------------------------------------------------------------

    def _state_kwargs(self) -> dict:
        """Return kwargs for the state model constructor.

        Subclasses that declare ``_STATE_MODEL`` must implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _state_kwargs() "
            "or override build_model() directly."
        )

    def _event_kwargs(self) -> dict:
        """Return kwargs for the event model constructor.

        Subclasses that declare ``_EVENT_MODEL`` must implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _event_kwargs() "
            "or override build_model() directly."
        )

    # ------------------------------------------------------------------
    # Base-owned assembly
    # ------------------------------------------------------------------

    def build_state_model(self) -> "StateModel":
        """Build and return the state model via registry lookup."""
        from unified_stpp.models.model_registry import get_state_cls
        return get_state_cls(self._STATE_MODEL)(**self._state_kwargs())

    def build_event_model(self) -> "EventModel":
        """Build and return the event model via registry lookup."""
        from unified_stpp.models.model_registry import get_event_cls
        return get_event_cls(self._EVENT_MODEL)(**self._event_kwargs())

    def build_model(self) -> "UnifiedSTPP":
        """Return a fully wired UnifiedSTPP.

        Uses ``build_state_model()`` and ``build_event_model()`` which delegate
        to the model registry.  Subclasses with non-standard construction
        (e.g. constructor-injection of pre-built modules) may override this.
        """
        from unified_stpp.models.unified_model import UnifiedSTPP
        return UnifiedSTPP(
            state_model=self.build_state_model(),
            event_model=self.build_event_model(),
            hidden_dim=self.hidden_dim,
        )

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

    @classmethod
    def fit_transform_artifact(
        cls,
        dm,
    ) -> "CoordinateTransformArtifact | None":
        """Return a fitted coordinate transform artifact for this preset.

        Default: no explicit transform artifact. Families that consume raw
        canonical batches but need their own reversible coordinate transform
        override this hook.
        """
        del dm
        return None

    @classmethod
    def transform_init_overrides(
        cls,
        artifact: CoordinateTransformArtifact,
    ) -> dict:
        """Map a fitted transform artifact into model construction overrides."""
        return {"input_transform": artifact.serialize()}
