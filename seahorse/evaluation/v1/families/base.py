"""Common interface for the v1.0 family adapters.

An adapter wraps one trained preset and exposes, for a *batch of equal-length
event sequences given in raw units*, the quantities the v1.0 metrics need:

* ``event_terms``      per-event ``log lambda``, ``dLambda``, ``log p(s|t)``
                       (teacher forcing over one sequence; NLL path).
* ``cum_hazard``       ``Lambda_k(tau) = int_{t_k}^{t_k+tau} int_S lambda`` on a
                       waiting-time grid, for chosen contexts ``(b, k)``.
* ``sample_space``     exact draws of ``s ~ p(s | t, H_k)`` at given times, plus
                       their raw log-densities (needed by the HPD-PIT).
* ``spatial_logpdf``   raw ``log p(s | t, H_k)`` at given points.
* ``ground_intensity`` raw ``lambda_g(t | H_k) = int_S lambda(t, s) ds``.

All inputs/outputs are raw units. Contexts are given as integer arrays
``(b_idx, k_idx)``: history = events ``0..k`` of sequence ``b``.

The adapters never change model code; they call the models' own functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class Prepared:
    """A batch of B equal-length raw sequences plus model-specific caches."""

    times: np.ndarray  # (B, T) raw
    locs: np.ndarray  # (B, T, 2) raw
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def B(self) -> int:
        return int(self.times.shape[0])

    @property
    def T(self) -> int:
        return int(self.times.shape[1])


@dataclass
class EventTerms:
    """Per-event NLL-path quantities for ONE sequence (length T).

    Entry ``i`` refers to event ``i`` as a *target* (history = events < i).
    ``defined[i]`` is False where the preset's likelihood does not define the
    event (e.g. AutoSTPP/DeepSTPP need a full 20-event window).
    """

    log_lambda: np.ndarray  # raw joint log-intensity log lambda(t_i, s_i | H_i)
    d_lambda: np.ndarray  # compensator increment over (t_{i-1}, t_i]
    log_ps: np.ndarray  # raw log p(s_i | t_i, H_i)
    defined: np.ndarray  # bool
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def nll(self) -> np.ndarray:
        return -self.log_lambda + self.d_lambda


class FamilyAdapter:
    """Base class. Subclasses set the class attributes and implement methods."""

    family: str = "base"
    #: exact | exact_ode | numerical_quadrature | framework_approx | none
    nll_kind: str = "exact"
    #: 'R2' or 'support_box' -- the spatial domain S of the NLL compensator
    spatial_domain: str = "R2"
    #: how next-event spatial samples are drawn
    spatial_sampler: str = "exact"
    #: minimum history length (number of events) for a defined conditional
    min_history: int = 1
    has_intensity: bool = True

    def __init__(self, runner, device: torch.device):
        self.runner = runner
        self.model = runner.model
        self.device = device
        self.preset = runner.config.model.preset

    # -- required -------------------------------------------------------
    def prepare(self, times: np.ndarray, locs: np.ndarray) -> Prepared:
        raise NotImplementedError

    def event_terms(self, prep: Prepared) -> EventTerms:
        raise NotImplementedError

    def cum_hazard(
        self, prep: Prepared, b_idx: np.ndarray, k_idx: np.ndarray, tau_grid: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def sample_space(
        self,
        prep: Prepared,
        b_idx: np.ndarray,
        k_idx: np.ndarray,
        t_query: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw one location per row at raw time ``t_query`` (rows = contexts).

        Returns ``(locs_raw (R, 2), logp_raw (R,))``.
        """
        raise NotImplementedError

    # -- rollouts ------------------------------------------------------------
    def rollout_init(self, seq_prep: Prepared, k0: int, M: int) -> Prepared:
        """Each sequence of ``seq_prep`` (history 0..k0) repeated M times (row-major by sequence)."""
        t = np.repeat(seq_prep.times[:, : k0 + 1], M, 0)
        s = np.repeat(seq_prep.locs[:, : k0 + 1], M, 0)
        return self.prepare(t, s)

    def extend(self, prep: Prepared, t_new: np.ndarray, s_new: np.ndarray) -> Prepared:
        """Append one event per row (default: re-encode the extended histories)."""
        t = np.concatenate([prep.times, np.asarray(t_new, float)[:, None]], 1)
        s = np.concatenate([prep.locs, np.asarray(s_new, float)[:, None, :]], 1)
        return self.prepare(t, s)

    def sample_space_k(self, prep, b_idx, k_idx, t_query, K, rng):
        """K draws per context at a common time per context; (R,K,2), (R,K)."""
        kc = np.repeat(np.asarray(k_idx), K)
        bc = np.repeat(np.asarray(b_idx), K)
        tc = np.repeat(np.asarray(t_query, float), K)
        x, lp = self.sample_space(prep, bc, kc, tc, rng)
        return x.reshape(len(k_idx), K, 2), lp.reshape(len(k_idx), K)

    def spatial_logpdf(
        self,
        prep: Prepared,
        b_idx: np.ndarray,
        k_idx: np.ndarray,
        t_query: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """Raw ``log p(points[r, m] | t_query[r], H_{b,k})``, shape (R, M)."""
        raise NotImplementedError

    def intensity(
        self,
        prep: Prepared,
        b_idx: np.ndarray,
        k_idx: np.ndarray,
        t_query: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """Raw joint intensity ``lambda(t_query[r], points[r, m] | H_{b,k})``."""
        lg = self.ground_intensity(prep, b_idx, k_idx, t_query)
        lp = self.spatial_logpdf(prep, b_idx, k_idx, t_query, points)
        return lg[:, None] * np.exp(lp)

    def ground_intensity(
        self, prep: Prepared, b_idx: np.ndarray, k_idx: np.ndarray, t_query: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------
    def require_identity_runner_norm(self) -> None:
        """Raise unless the runner-level normalisation is the identity map."""
        ns = self.runner.norm_stats or {}
        if not bool(ns.get("normalize", False)):
            return
        tm = float(ns.get("time_mean", 0.0))
        ts = float(ns.get("time_std", 1.0))
        lm = np.asarray(ns.get("loc_mean", [0.0, 0.0]), float)
        ls = np.asarray(ns.get("loc_std", [1.0, 1.0]), float)
        if tm != 0.0 or ts != 1.0 or np.any(lm != 0.0) or np.any(ls != 1.0):
            raise NotImplementedError(f"{self.family} adapter assumes identity runner norm_stats")

    def t(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def describe(self) -> dict:
        return {
            "family": self.family,
            "preset": self.preset,
            "nll_kind": self.nll_kind,
            "spatial_domain": self.spatial_domain,
            "spatial_sampler": self.spatial_sampler,
            "min_history": self.min_history,
            "has_intensity": self.has_intensity,
        }


def chunked(n: int, size: int):
    for start in range(0, n, size):
        yield slice(start, min(n, start + size))


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu", torch.float64).numpy()
