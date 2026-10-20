"""Public package exports for Seahorse / ``seahorse-stpp``."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("seahorse-stpp")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

from importlib import import_module
from ._friendly_names import FRIENDLY_TO_PRESET

# Author ETAS and data-preparation environments need not install PyTorch.
# Preserve the public API, loading its heavier dependencies only when used.
_EXPORT_MODULES = {
    **dict.fromkeys(("STPPConfig", "DataConfig", "ModelConfig", "TrainingConfig", "LoggingConfig"), ".config"),
    **dict.fromkeys(("STPPRunner", "RunResult"), ".runner"),
    **dict.fromkeys(("Benchmark", "BenchmarkTable"), ".benchmark"),
    **dict.fromkeys(("STPPEstimator", "STPPPlotter", "list_available_models", "load_jsonl", "resolve_preset", "FRIENDLY_MODEL_CLASSES", *FRIENDLY_TO_PRESET), ".api"),
}


def __getattr__(name):
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORT_MODULES))

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
    *FRIENDLY_TO_PRESET,
]
