"""Metric module package.

Importing this package registers all metrics into the global registry.
Each submodule applies @register_metric at class-definition time.
"""

from . import generative, grid, nll, predictive

__all__ = ["nll", "predictive", "generative", "grid"]
