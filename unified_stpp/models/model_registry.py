"""Model-class registry for state models, event models, and spatial decoders.

All three model registries follow the same lazy-dict + decorator pattern:

1. Models self-register by decorating their class definition with
   ``@register_state``, ``@register_event``, or ``@register_spatial``.

2. Registration happens at module import time, so the registry is empty
   until the relevant package is first imported.

3. ``get_state_cls`` / ``get_event_cls`` / ``get_spatial_cls`` trigger that
   import automatically the first time an unknown key is requested, so callers
   never need to worry about import order.

This file replaces the duplicate ``_get_spatial_registry()`` functions that
previously appeared in both ``models/configs/neural_stpp.py`` and
``models/event_models/neural_stpp_event.py``.
"""

from __future__ import annotations

_STATE_REGISTRY: dict[str, type] = {}
_EVENT_REGISTRY: dict[str, type] = {}
_SPATIAL_REGISTRY: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Decorator factories
# ---------------------------------------------------------------------------

def register_state(key: str):
    """Decorator: register a StateModel class under *key*."""
    def decorator(cls):
        _STATE_REGISTRY[key] = cls
        return cls
    return decorator


def register_event(key: str):
    """Decorator: register an EventModel class under *key*."""
    def decorator(cls):
        _EVENT_REGISTRY[key] = cls
        return cls
    return decorator


def register_spatial(key: str):
    """Decorator: register a spatial decoder class under *key*."""
    def decorator(cls):
        _SPATIAL_REGISTRY[key] = cls
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Lookup helpers (trigger lazy import on first miss)
# ---------------------------------------------------------------------------

def get_state_cls(key: str) -> type:
    """Return the StateModel class registered under *key*.

    Imports ``unified_stpp.models.state_models`` on the first call if the key
    is not yet present, which triggers all ``@register_state`` decorators.
    """
    if key not in _STATE_REGISTRY:
        import unified_stpp.models.state_models  # noqa: F401
    if key not in _STATE_REGISTRY:
        raise ValueError(
            f"Unknown state model key {key!r}. "
            f"Registered: {sorted(_STATE_REGISTRY)}"
        )
    return _STATE_REGISTRY[key]


def get_event_cls(key: str) -> type:
    """Return the EventModel class registered under *key*."""
    if key not in _EVENT_REGISTRY:
        import unified_stpp.models.event_models  # noqa: F401
    if key not in _EVENT_REGISTRY:
        raise ValueError(
            f"Unknown event model key {key!r}. "
            f"Registered: {sorted(_EVENT_REGISTRY)}"
        )
    return _EVENT_REGISTRY[key]


def get_spatial_cls(key: str) -> type:
    """Return the spatial decoder class registered under *key*."""
    if key not in _SPATIAL_REGISTRY:
        import unified_stpp.models.spatial_models  # noqa: F401
    if key not in _SPATIAL_REGISTRY:
        raise ValueError(
            f"Unknown spatial decoder key {key!r}. "
            f"Registered: {sorted(_SPATIAL_REGISTRY)}"
        )
    return _SPATIAL_REGISTRY[key]
