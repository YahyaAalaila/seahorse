"""Lightweight Python-first API for ``seahorse``."""

from seahorse.data import load_dataset
from seahorse.utils import load_jsonl

from .estimator import STPPEstimator
from .model_class_map import FRIENDLY_TO_PRESET, list_available_models, resolve_preset
from .viz import STPPPlotter


def _make_estimator_alias(model_class: str):
    class FriendlyEstimator(STPPEstimator):
        def __init__(
            self,
            config_overrides: dict | None = None,
            device: str = "auto",
            seed: int = 42,
        ) -> None:
            super().__init__(
                model_class,
                config_overrides=config_overrides,
                device=device,
                seed=seed,
            )

    FriendlyEstimator.__name__ = model_class
    FriendlyEstimator.__qualname__ = model_class
    FriendlyEstimator.__doc__ = f"STPPEstimator alias for the {model_class} preset."
    return FriendlyEstimator


FRIENDLY_MODEL_CLASSES = {
    name: _make_estimator_alias(name)
    for name in FRIENDLY_TO_PRESET
}
globals().update(FRIENDLY_MODEL_CLASSES)

__all__ = [
    "STPPEstimator",
    "STPPPlotter",
    "list_available_models",
    "load_dataset",
    "load_jsonl",
    "resolve_preset",
    *FRIENDLY_MODEL_CLASSES,
]
