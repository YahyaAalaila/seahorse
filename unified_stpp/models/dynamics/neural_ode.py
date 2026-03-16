"""
Lightweight ODE utilities used by active coarse-framework models.

Only ``euler_solve`` is retained for NeuralTPP backbone fallback integration.
"""

from __future__ import annotations

import torch


def euler_solve(func, z0, t_span, n_steps: int = 50):
    """Simple Euler solver fallback when torchdiffeq is unavailable."""
    dt = (t_span[-1] - t_span[0]) / n_steps
    z = z0
    t = t_span[0]
    trajectory = [z0]
    target_times = t_span[1:]  # skip initial
    target_idx = 0
    for _ in range(n_steps):
        z = z + dt * func(t, z)
        t = t + dt
        while target_idx < len(target_times) and t >= target_times[target_idx] - 1e-6:
            trajectory.append(z)
            target_idx += 1
    while len(trajectory) < len(t_span):
        trajectory.append(z)
    return torch.stack(trajectory, dim=0)  # (T, B, h)
