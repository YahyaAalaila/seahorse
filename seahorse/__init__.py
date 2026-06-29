"""Public package exports for Seahorse / ``seahorse-stpp``."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("seahorse-stpp")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

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
    "__version__",
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
