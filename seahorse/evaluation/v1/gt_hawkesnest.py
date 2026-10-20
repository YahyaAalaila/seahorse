"""Ground-truth conditional intensity for HawkesNest Suite 3 (traveling wave).

    lambda*(t, s | H_t) = mu + alpha * sum_{t_j < t, t - t_j <= tau_max}
                          exp(-(t - t_j)/beta) exp(-||s - s_j - v e (t - t_j)||^2 / (2 sigma^2))

with ``e = (cos theta, sin theta)``. Parameters come from
``ground_truth/L*_params.json``. As in the stored reference grids
(``L*_intensity_grid_r0.npz``, which this reproduces to 1e-6 at interior
times), the history is the *test window only* (the same history the models
condition on), not the full simulated realisation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class TravelingWaveGT:
    def __init__(self, params: dict):
        k = params["kernel"]
        if k.get("type") != "traveling_wave":
            raise NotImplementedError(k.get("type"))
        bg = params["background"]
        if bg.get("type") != "constant":
            raise NotImplementedError(bg.get("type"))
        self.mu = float(bg["rate"])
        self.alpha = float(params["adjacency"])
        self.beta = float(k["temporal_scale"])
        self.sigma = float(k["sigma"])
        self.tau_max = float(k.get("tau_max", np.inf))
        self.v = float(k["v"])
        self.ex = math.cos(float(k["theta_wave"]))
        self.ey = math.sin(float(k["theta_wave"]))
        self.x_range = tuple(params.get("x_range", (0.0, 1.0)))
        self.y_range = tuple(params.get("y_range", (0.0, 1.0)))

    @classmethod
    def from_file(cls, path: str | Path) -> "TravelingWaveGT":
        return cls(json.loads(Path(path).read_text()))

    def intensity(self, t: float, pts: np.ndarray, hist_t: np.ndarray, hist_s: np.ndarray) -> np.ndarray:
        """lambda*(t, pts) given history arrays; pts (M, 2)."""
        m = (hist_t < t) & (t - hist_t <= self.tau_max)
        out = np.full(pts.shape[0], self.mu)
        if not m.any():
            return out
        tau = t - hist_t[m]
        cx = hist_s[m, 0] + self.v * self.ex * tau
        cy = hist_s[m, 1] + self.v * self.ey * tau
        dx = pts[:, None, 0] - cx[None]
        dy = pts[:, None, 1] - cy[None]
        w = self.alpha * np.exp(-tau / self.beta)
        out += (w[None] * np.exp(-0.5 * (dx * dx + dy * dy) / self.sigma**2)).sum(-1)
        return out

    def ground_intensity_in_domain(self, t, hist_t, hist_s, n: int = 64) -> float:
        c = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(
            self.x_range[0] + c * (self.x_range[1] - self.x_range[0]),
            self.y_range[0] + c * (self.y_range[1] - self.y_range[0]),
            indexing="ij",
        )
        pts = np.stack([X.ravel(), Y.ravel()], -1)
        area = (self.x_range[1] - self.x_range[0]) * (self.y_range[1] - self.y_range[0])
        return float(self.intensity(t, pts, hist_t, hist_s).mean() * area)
