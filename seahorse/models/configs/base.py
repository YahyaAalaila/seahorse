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

from seahorse.data.transforms import CoordinateTransformArtifact
from seahorse.utils import deep_update

if TYPE_CHECKING:
    from seahorse.models.abstractions import EventModel, StateModel
    from seahorse.models.unified_model import UnifiedSTPP


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PresetDescriptor:
    """Resolved metadata for an accepted preset identifier."""

    name: str
    canonical_name: str
    status: str
    canonical_status: str
    config_cls: type
    is_alias: bool


class ConfigRegistry:
    """Maps preset names to config classes.  Config classes self-register via
    the @ConfigRegistry.register("preset_name") decorator.
    """

    _registry: ClassVar[dict[str, type]] = {}
    _canonical_name: ClassVar[dict[str, str]] = {}
    _status: ClassVar[dict[str, str]] = {}
    _is_alias: ClassVar[dict[str, bool]] = {}
    _VALID_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"canonical", "deprecated", "legacy"}
    )

    @classmethod
    def _validate_status(cls, status: str) -> str:
        if status not in cls._VALID_STATUSES:
            raise ValueError(
                f"Unknown preset status {status!r}. Valid: {sorted(cls._VALID_STATUSES)}"
            )
        return status

    @classmethod
    def register(cls, name: str, *, status: str = "canonical"):
        """Decorator that registers a config class under a preset name."""
        status = cls._validate_status(status)

        def decorator(config_cls):
            if name in cls._registry:
                raise ValueError(f"Preset {name!r} already registered.")
            cls._registry[name] = config_cls
            cls._canonical_name[name] = name
            cls._status[name] = status
            cls._is_alias[name] = False
            return config_cls
        return decorator

    @classmethod
    def register_alias(cls, alias: str, target: str, *, status: str = "deprecated") -> None:
        """Register *alias* as an accepted preset identifier for *target*."""
        status = cls._validate_status(status)
        canonical = cls.resolve_name(target)
        config_cls = cls._registry.get(canonical)
        if config_cls is None:
            raise ValueError(f"Cannot register alias {alias!r}: unknown target {target!r}.")
        if alias in cls._registry:
            raise ValueError(f"Preset alias {alias!r} already registered.")
        cls._registry[alias] = config_cls
        cls._canonical_name[alias] = canonical
        cls._status[alias] = status
        cls._is_alias[alias] = True

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry

    @classmethod
    def resolve_name(cls, name: str) -> str:
        """Return the canonical preset name for *name* when registered."""
        return cls._canonical_name.get(name, name)

    @classmethod
    def describe(cls, name: str) -> PresetDescriptor:
        """Return registry metadata for *name*."""
        canonical = cls.resolve_name(name)
        config_cls = cls._registry.get(canonical)
        if config_cls is None:
            raise ValueError(
                f"Unknown preset {name!r}. Known presets: {sorted(cls._registry)}"
            )
        lookup = name if name in cls._registry else canonical
        return PresetDescriptor(
            name=lookup,
            canonical_name=canonical,
            status=cls._status[lookup],
            canonical_status=cls._status[canonical],
            config_cls=config_cls,
            is_alias=cls._is_alias[lookup],
        )

    @classmethod
    def preset_status(cls, name: str) -> str:
        return cls.describe(name).status

    @classmethod
    def canonical_status(cls, name: str) -> str:
        return cls.describe(name).canonical_status

    @classmethod
    def build(cls, name: str, overrides: dict, **dims) -> "UnifiedSTPP":
        """Build a UnifiedSTPP from a preset name and override dict."""
        canonical = cls.resolve_name(name)
        if canonical not in cls._registry:
            raise ValueError(
                f"Unknown preset {name!r}. Known presets: {sorted(cls._registry)}"
            )
        cfg_cls = cls._registry[canonical]
        return cfg_cls.from_dict(copy.deepcopy(overrides), **dims).build_model()

    @classmethod
    def resolve_accelerator(cls, name: str, requested: str) -> str:
        """Delegate accelerator resolution to the config class."""
        cfg_cls = cls._registry.get(cls.resolve_name(name))
        if cfg_cls is None:
            return requested
        return cfg_cls.resolve_accelerator(requested)

    @classmethod
    def preset_names(cls, *, include_aliases: bool = False) -> list[str]:
        if include_aliases:
            return list(cls._registry)
        return [name for name in cls._registry if not cls._is_alias.get(name, False)]

    @classmethod
    def accepted_preset_names(cls) -> list[str]:
        """Return all accepted preset identifiers, including aliases."""
        return cls.preset_names(include_aliases=True)

    @classmethod
    def canonical_preset_names(cls) -> list[str]:
        return cls.preset_names()

    @classmethod
    def data_init_overrides(cls, name: str, dm) -> dict:
        """Return data-dependent config overrides for *name* (e.g. bbox for auto_stpp).

        Delegates to the config class's ``data_init_overrides(dm)`` classmethod.
        Returns ``{}`` for presets that have no data-dependent init.
        """
        cfg_cls = cls._registry.get(cls.resolve_name(name))
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
        cfg_cls = cls._registry.get(cls.resolve_name(name))
        if cfg_cls is None:
            return "canonical"
        return cfg_cls._COLLATE

    @classmethod
    def get_train_loader_key(cls, name: str) -> str:
        """Return the train-loader-builder registry key declared by preset *name*."""
        cfg_cls = cls._registry.get(cls.resolve_name(name))
        if cfg_cls is None:
            return "fixed_batch"
        return cfg_cls._TRAIN_LOADER

    @classmethod
    def get_supported_protocols(cls, name: str) -> "frozenset[str]":
        """Return the set of dataset protocols supported by preset *name*.

        An empty frozenset means all protocols are allowed.
        """
        cfg_cls = cls._registry.get(cls.resolve_name(name))
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
        from seahorse.models.model_registry import get_state_cls
        return get_state_cls(self._STATE_MODEL)(**self._state_kwargs())

    def build_event_model(self) -> "EventModel":
        """Build and return the event model via registry lookup."""
        from seahorse.models.model_registry import get_event_cls
        return get_event_cls(self._EVENT_MODEL)(**self._event_kwargs())

    def build_model(self) -> "UnifiedSTPP":
        """Return a fully wired UnifiedSTPP.

        Uses ``build_state_model()`` and ``build_event_model()`` which delegate
        to the model registry.  Subclasses with non-standard construction
        (e.g. constructor-injection of pre-built modules) may override this.
        """
        from seahorse.models.unified_model import UnifiedSTPP
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
