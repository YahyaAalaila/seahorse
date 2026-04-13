"""Metric helpers over frozen evaluation bundles."""

from __future__ import annotations

import numpy as np

from unified_stpp.evaluation.predictive_compare import PredictiveComparisonResult


def predictive_surface_rmse(
    result: PredictiveComparisonResult,
    *,
    model_label: str,
    reference_surfaces: np.ndarray,
) -> np.ndarray:
    model = next(model for model in result.models if model.label == model_label)
    pred = np.stack([frame.derived_kde_rate_surface for frame in model.frames], axis=0)
    ref = np.asarray(reference_surfaces, dtype=np.float32)
    if pred.shape != ref.shape:
        raise ValueError(f"RMSE expects matching shapes, got {pred.shape} vs {ref.shape}")
    return np.sqrt(np.mean((pred - ref) ** 2, axis=(1, 2))).astype(np.float32)


def predictive_surface_mae(
    result: PredictiveComparisonResult,
    *,
    model_label: str,
    reference_surfaces: np.ndarray,
) -> np.ndarray:
    model = next(model for model in result.models if model.label == model_label)
    pred = np.stack([frame.derived_kde_rate_surface for frame in model.frames], axis=0)
    ref = np.asarray(reference_surfaces, dtype=np.float32)
    if pred.shape != ref.shape:
        raise ValueError(f"MAE expects matching shapes, got {pred.shape} vs {ref.shape}")
    return np.mean(np.abs(pred - ref), axis=(1, 2)).astype(np.float32)
