"""
Intensity evaluation utilities for spatial visualization of STPP models.

Provides a unified interface for evaluating conditional intensity
λ*(t, s | H) on spatial grids with proper coordinate de-normalization.

Mathematical background
-----------------------
Supported decoders produce values in *normalized* coordinates
(after z-score normalization of times and locations):

  DeepSTPPDecoder:
      decoder.log_prob → log f*_norm(t', s')

  AutoIntDecoder:
      decoder.log_prob → log λ*_norm(t', s')

After z-score normalization (t' = (t − μ_t)/σ_t, s' = (s − μ_s)/σ_s)
the density/intensity in original coordinates is:

    q_orig(t, s) = q_norm(t', s') / (σ_t · σ_x · σ_y)

The same Jacobian factor applies in both cases, so
`correct_for_normalization=True` divides exp(log_prob_norm) by
(t_scale · prod(s_scale)) to approximate the quantity in original
coordinates.
"""

import torch
import numpy as np
from typing import Optional

from unified_stpp.models.unified_model import UnifiedSTPP
from unified_stpp.models.dynamics.identity import IdentityDynamics


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
    """
    Evaluate the conditional intensity λ*(t, s | H) at a fixed time ``t_query``
    over a spatial grid, using ``history_times`` / ``history_locs`` as the
    conditioning history H.

    Works for DeepSTPPDecoder and AutoIntDecoder. For IdentityDynamics the
    encoder output is used directly; for non-identity dynamics, the state is
    evolved to ``t_query`` via the dynamics module.

    Args:
        model: Trained UnifiedSTPP in eval mode (or will be switched to eval).
        t_query: Query time in *original* (un-normalized) coordinates.
            Must be ≥ history_times[-1].
        s_grid: (M, d) array of spatial query points in original coordinates.
        history_times: (N,) array of past event times in original coordinates,
            sorted ascending.
        history_locs: (N, d) array of past event locations in original coords.
        t_bias: Time normalization mean  (= dataset.time_mean from train split).
        t_scale: Time normalization std  (= dataset.time_std from train split).
        s_bias: (d,) spatial normalization means  (= dataset.loc_mean).
        s_scale: (d,) spatial normalization stds  (= dataset.loc_std).
        device: Torch device; defaults to the device of the model's first param.
        correct_for_normalization: If True (default), divide the raw decoder
            output by (t_scale · ∏ s_scale) so the values approximate the
            intensity in original coordinates.  Set to False to get the raw
            value in normalized space.

    Returns:
        intensity: (M,) float32 array of λ*(t_query, s_grid[i] | H) for each i.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()

    N = len(history_times)
    s_grid = np.asarray(s_grid, dtype=np.float32)
    M, d = s_grid.shape

    t_bias_f = float(t_bias)
    t_scale_f = float(t_scale)
    s_bias_arr = np.asarray(s_bias, dtype=np.float32)    # (d,)
    s_scale_arr = np.asarray(s_scale, dtype=np.float32)  # (d,)

    # ------------------------------------------------------------------
    # Normalize history and query to the space the model was trained in
    # ------------------------------------------------------------------
    times_norm = (history_times.astype(np.float32) - t_bias_f) / t_scale_f
    locs_norm  = (history_locs.astype(np.float32)  - s_bias_arr) / s_scale_arr

    t_query_norm = (float(t_query) - t_bias_f) / t_scale_f
    s_grid_norm  = (s_grid - s_bias_arr) / s_scale_arr           # (M, d)

    t_prev_norm = float(times_norm[-1]) if N > 0 else 0.0

    # ------------------------------------------------------------------
    # Encode history → conditioning state z
    # ------------------------------------------------------------------
    with torch.enable_grad():  # ProdNet.intensity needs autograd internally
        times_t = torch.tensor(
            times_norm, dtype=torch.float32, device=device
        ).unsqueeze(0)                                              # (1, N)
        locs_t = torch.tensor(
            locs_norm, dtype=torch.float32, device=device
        ).unsqueeze(0)                                              # (1, N, d)
        lengths = torch.tensor([N], dtype=torch.int64, device=device)  # (1,)

        events = torch.cat(
            [times_t.unsqueeze(-1), locs_t], dim=-1
        )                                                           # (1, N, 1+d)
        _z_final, all_states = model.encode(events, lengths)        # (1, N, h)

        # State after the last history event conditions the next prediction
        z_hist = all_states[:, N - 1, :]                           # (1, h)

        # For ODE-based dynamics, evolve state from last event time to t_query
        if not isinstance(model.dynamics, IdentityDynamics):
            dt_val = max(t_query_norm - t_prev_norm, 1e-6)
            dt = torch.tensor(
                [[dt_val]], dtype=torch.float32, device=device
            )                                                       # (1, 1)
            z_hist = model.dynamics(z_hist, dt, None).squeeze(1)   # (1, h)

        # ------------------------------------------------------------------
        # Evaluate decoder.log_prob for all M grid points in one batched call
        # ------------------------------------------------------------------
        z_batch = z_hist.expand(M, -1)                             # (M, h)
        t_q_batch = torch.full(
            (M, 1), t_query_norm, dtype=torch.float32, device=device
        )
        s_q_batch = torch.tensor(
            s_grid_norm, dtype=torch.float32, device=device
        )                                                           # (M, d)
        t_p_batch = torch.full(
            (M, 1), t_prev_norm, dtype=torch.float32, device=device
        )

        log_vals = model.decoder.log_prob(
            z_batch, t_q_batch, s_q_batch, t_p_batch
        )                                                           # (M,)

    intensity = torch.exp(log_vals).detach().cpu().numpy().astype(np.float32)

    if correct_for_normalization:
        # Undo z-score Jacobian: q_orig = q_norm / (σ_t · σ_x · σ_y · …)
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
    """
    Evaluate λ*(t, x, y | H) over a 3-D spatiotemporal grid.

    Each time in ``t_range`` is evaluated over the full
    (x_range × y_range) spatial grid, producing a ``(T, X, Y)`` array
    suitable for plt.imshow / plt.contourf visualizations.

    The conditioning history H (history_times / history_locs) is fixed
    across all time slices: the hidden state is that of the model after
    seeing the complete history, not updated as t advances through
    ``t_range``.

    Args:
        model: Trained UnifiedSTPP.
        history_times: (N,) past event times in original coordinates.
        history_locs: (N, d) past event locations in original coordinates.
        t_bias: Time normalization mean.
        t_scale: Time normalization std.
        s_bias: (d,) spatial normalization means.
        s_scale: (d,) spatial normalization stds.
        x_range: (X,) x-axis grid values in original coordinates.
        y_range: (Y,) y-axis grid values in original coordinates.
        t_range: (T,) query times in original coordinates.
        device: Torch device.
        correct_for_normalization: Passed through to eval_intensity.

    Returns:
        lamb: (T, X, Y) float32 array — intensity at each (t, x, y).
    """
    x_range = np.asarray(x_range, dtype=np.float32)
    y_range = np.asarray(y_range, dtype=np.float32)
    t_range = np.asarray(t_range, dtype=np.float32)

    X = len(x_range)
    Y = len(y_range)
    T = len(t_range)

    # Build the full 2-D spatial grid  shape (X*Y, 2)
    xx, yy = np.meshgrid(x_range, y_range, indexing="ij")   # each (X, Y)
    s_grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)     # (X*Y, 2)

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
        )                                                      # (X*Y,)
        lamb[i] = vals.reshape(X, Y)

    return lamb
