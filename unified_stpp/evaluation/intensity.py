"""Intensity evaluation utilities built on StateModel/EventModel capabilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from unified_stpp.models.sampling import IntensityEvaluator, supports_intensity_query
from unified_stpp.models.unified_model import UnifiedSTPP


def eval_intensity(
    model: UnifiedSTPP,
    t_query: float,
    s_grid: np.ndarray,
    history_times: np.ndarray,
    history_locs: np.ndarray,
    t_bias: float,
    t_scale: float,
    s_bias: np.ndarray,
    s_scale: np.ndarray,
    device: Optional[torch.device] = None,
    correct_for_normalization: bool = True,
) -> np.ndarray:
    """Evaluate conditional intensity lambda*(t_query, s_grid | history)."""
    if not supports_intensity_query(model):
        name = type(getattr(model, "event_model", None)).__name__
        raise NotImplementedError(
            f"EventModel '{name}' does not expose intensity queries."
        )

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()

    s_grid = np.asarray(s_grid, dtype=np.float32)
    M, d = s_grid.shape
    history_times = np.asarray(history_times, dtype=np.float32).reshape(-1)
    history_locs = np.asarray(history_locs, dtype=np.float32).reshape(-1, d)
    if history_times.size < 1:
        raise ValueError("eval_intensity requires at least one history event.")

    t_bias_f = float(t_bias)
    t_scale_f = float(t_scale)
    s_bias_arr = np.asarray(s_bias, dtype=np.float32)
    s_scale_arr = np.asarray(s_scale, dtype=np.float32)

    times_norm = (history_times - t_bias_f) / t_scale_f
    locs_norm = (history_locs - s_bias_arr) / s_scale_arr

    t_query_norm = (float(t_query) - t_bias_f) / t_scale_f
    s_grid_norm = (s_grid - s_bias_arr) / s_scale_arr

    with torch.enable_grad():
        history_times_t = torch.tensor(times_norm, dtype=torch.float32, device=device).unsqueeze(0)
        history_locs_t = torch.tensor(locs_norm, dtype=torch.float32, device=device).unsqueeze(0)
        history_lengths_t = torch.tensor([history_times.size], dtype=torch.long, device=device)

        evaluator = IntensityEvaluator(
            model,
            history_times=history_times_t,
            history_locations=history_locs_t,
            history_lengths=history_lengths_t,
        )

        t_q_batch = torch.full((M, 1), float(t_query_norm), dtype=torch.float32, device=device)
        s_q_batch = torch.tensor(s_grid_norm, dtype=torch.float32, device=device)
        lamb = evaluator.intensity(t_q_batch, s_q_batch)

    intensity = lamb.detach().cpu().numpy().astype(np.float32)

    if correct_for_normalization:
        scale_factor = float(t_scale_f * np.prod(s_scale_arr))
        intensity = intensity / scale_factor

    return intensity


def calc_lamb(
    model: UnifiedSTPP,
    history_times: np.ndarray,
    history_locs: np.ndarray,
    t_bias: float,
    t_scale: float,
    s_bias: np.ndarray,
    s_scale: np.ndarray,
    x_range: np.ndarray,
    y_range: np.ndarray,
    t_range: np.ndarray,
    device: Optional[torch.device] = None,
    correct_for_normalization: bool = True,
) -> np.ndarray:
    """Evaluate lambda*(t, x, y | H) over a (T, X, Y) grid."""
    x_range = np.asarray(x_range, dtype=np.float32)
    y_range = np.asarray(y_range, dtype=np.float32)
    t_range = np.asarray(t_range, dtype=np.float32)

    X = len(x_range)
    Y = len(y_range)
    T = len(t_range)

    xx, yy = np.meshgrid(x_range, y_range, indexing="ij")
    s_grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    lamb = np.zeros((T, X, Y), dtype=np.float32)

    for i, t_q in enumerate(t_range):
        vals = eval_intensity(
            model=model,
            t_query=float(t_q),
            s_grid=s_grid,
            history_times=history_times,
            history_locs=history_locs,
            t_bias=t_bias,
            t_scale=t_scale,
            s_bias=s_bias,
            s_scale=s_scale,
            device=device,
            correct_for_normalization=correct_for_normalization,
        )
        lamb[i] = vals.reshape(X, Y)

    return lamb
