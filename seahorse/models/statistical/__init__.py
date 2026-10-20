"""NumPy/SciPy statistical spatio-temporal point-process models.

These models deliberately have no torch or Lightning dependency.  They are
consumed by the ``statstpp`` runner (``seahorse.runner.statistical``) through
the protocol in :mod:`seahorse.models.statistical.core`.
"""

from .registry import STATISTICAL_PRESETS, build_statistical_model, model_from_state

__all__ = ["STATISTICAL_PRESETS", "build_statistical_model", "model_from_state"]
