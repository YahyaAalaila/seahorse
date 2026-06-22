"""Global metric registry and register_metric decorator."""

from __future__ import annotations

from typing import Type

from .result import Metric

_REGISTRY: dict[str, Type[Metric]] = {}


def register_metric(cls: Type[Metric]) -> Type[Metric]:
    """Class decorator that registers a Metric subclass by its .name attribute.

    Usage::

        @register_metric
        class TemporalCRPS(Metric):
            name = "temporal_crps"
            ...
    """
    if not hasattr(cls, "name") or not isinstance(getattr(cls, "name", None), str):
        raise TypeError(
            f"{cls.__qualname__} must define a 'name: str' class attribute before @register_metric"
        )
    existing = _REGISTRY.get(cls.name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Metric name {cls.name!r} is already registered by {existing.__qualname__}. "
            "Each metric must have a unique name."
        )
    _REGISTRY[cls.name] = cls
    return cls


def all_metric_instances() -> list[Metric]:
    """Return one freshly-instantiated instance of every registered metric class."""
    return [cls() for cls in _REGISTRY.values()]


def metric_by_name(name: str) -> Metric:
    """Instantiate and return the registered metric with the given name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        available = sorted(_REGISTRY)
        raise KeyError(f"No metric named {name!r}. Available: {available}")
    return cls()
