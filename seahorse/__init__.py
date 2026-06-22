"""Public package exports for Seahorse / ``seahorse-stpp``."""

from .config import STPPConfig, DataConfig, ModelConfig, TrainingConfig, LoggingConfig
from .runner import STPPRunner, RunResult
from .benchmark import Benchmark, BenchmarkTable
from .api import (
    FRIENDLY_MODEL_CLASSES,
    STPPEstimator,
    STPPPlotter,
    list_available_models,
    load_jsonl,
    resolve_preset,
)

globals().update(FRIENDLY_MODEL_CLASSES)

__all__ = [
    "STPPConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "LoggingConfig",
    "STPPRunner",
    "RunResult",
    "Benchmark",
    "BenchmarkTable",
    "STPPEstimator",
    "STPPPlotter",
    "list_available_models",
    "load_jsonl",
    "resolve_preset",
    *FRIENDLY_MODEL_CLASSES,
]
