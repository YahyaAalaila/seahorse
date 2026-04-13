"""Reference surface providers for surface visualization comparisons.

The ``ReferenceSurfaceProvider`` Protocol defines the interface for any
external reference surface (e.g. ground-truth intensity, empirical KDE).

Built-in implementations
------------------------
CallableGroundTruthProvider
    Exact GT for synthetic data where the true intensity function is known.
    History-aware: intensity_fn receives the full observed history so that
    history-dependent processes (e.g. Hawkes) are conditioned correctly.

EmpiricalKDEProvider
    Marginal spatial KDE from a fixed set of event locations.
    History/time-independent — appropriate as a spatial marginal proxy.

STHPGroundTruthProvider
    True conditional intensity λ*(t,s|H) for an STHP model, evaluated via
    STHPDataset.lamb_st_grid.  Loads parameters from a dataset_meta.json
    file written by gen_sthp_splits.py.

All providers return ``SurfaceResult`` objects with correct ``surface_type``,
``comparable``, ``label``, and ``unit`` fields. Plotting code must not
contain any model- or provider-specific logic; all metadata lives in the result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from unified_stpp.evaluation.surface_query import SurfaceResult


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class ReferenceSurfaceProvider:
    """Interface for external reference surfaces.

    ``compute()`` receives the same (xs, ys) grid as the model SurfaceResult,
    in original (un-normalized) space, so outputs align for direct comparison.
    History is always passed to support non-stationary / history-dependent GT.

    Implementors must return a ``SurfaceResult`` with an appropriate
    ``surface_type``, ``comparable``, ``label``, and ``unit``.
    """

    def compute(
        self,
        history_times: np.ndarray,   # (L,) original space
        history_locs:  np.ndarray,   # (L, d) original space
        t_query:       float,
        xs:            np.ndarray,   # (n_grid,) from model result (original space)
        ys:            np.ndarray,   # (n_grid,)
    ) -> SurfaceResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in: Callable ground-truth provider
# ---------------------------------------------------------------------------

@dataclass
class CallableGroundTruthProvider(ReferenceSurfaceProvider):
    """Exact GT for synthetic data where the true intensity function is known.

    ``intensity_fn(history_times, history_locs, t_query, X, Y) -> (n_grid, n_grid)``

    where ``X, Y = np.meshgrid(xs, ys, indexing='ij')``.
    History is passed so the GT can be conditioned on the observed past
    (e.g. Hawkes process intensity depends on the event history).

    Returns ``SurfaceResult(surface_type='intensity', comparable=True)``.
    """

    intensity_fn: Callable[
        [np.ndarray, np.ndarray, float, np.ndarray, np.ndarray], np.ndarray
    ]
    label: str = "Ground truth λ_true(t,s|H)"
    unit: str = "events / (unit_time × unit_area)"

    def compute(
        self,
        history_times: np.ndarray,
        history_locs: np.ndarray,
        t_query: float,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> SurfaceResult:
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        values = self.intensity_fn(history_times, history_locs, t_query, X, Y)
        return SurfaceResult(
            surface_type="intensity",
            values=np.asarray(values, dtype=np.float32),
            xs=xs.astype(np.float32),
            ys=ys.astype(np.float32),
            t_query=t_query,
            label=self.label,
            unit=self.unit,
            comparable=True,
        )


# ---------------------------------------------------------------------------
# Built-in: Empirical KDE provider
# ---------------------------------------------------------------------------

@dataclass
class EmpiricalKDEProvider(ReferenceSurfaceProvider):
    """KDE-based reference from event locations.

    Marginal spatial KDE computed from the provided event locations.
    History and query time are ignored — this is a time-marginal proxy.

    Returns ``SurfaceResult(surface_type='proxy_kde', comparable=False)``.
    """

    event_locs: np.ndarray        # (N, d) all event locations, original space
    bandwidth: Optional[float] = None  # None → scipy Scott's rule
    label: str = "Empirical spatial KDE"
    unit: str = "proxy (not comparable)"

    def compute(
        self,
        history_times: np.ndarray,
        history_locs: np.ndarray,
        t_query: float,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> SurfaceResult:
        try:
            from scipy.stats import gaussian_kde
        except ImportError as exc:
            raise ImportError(
                "scipy is required for EmpiricalKDEProvider. "
                "Install with: pip install scipy"
            ) from exc

        locs = np.asarray(self.event_locs, dtype=np.float64)
        if locs.ndim == 1:
            locs = locs.reshape(-1, 1)

        d = locs.shape[1]
        kde_kws = {} if self.bandwidth is None else {"bw_method": self.bandwidth}

        if d >= 2:
            kde = gaussian_kde(locs[:, :2].T, **kde_kws)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            pts = np.stack([X.ravel(), Y.ravel()]).astype(np.float64)
            values = kde(pts).reshape(len(xs), len(ys)).astype(np.float32)
        else:
            kde = gaussian_kde(locs[:, 0], **kde_kws)
            values = kde(xs.astype(np.float64)).astype(np.float32)

        return SurfaceResult(
            surface_type="proxy_kde",
            values=values,
            xs=xs.astype(np.float32),
            ys=ys.astype(np.float32) if ys.size > 0 else np.zeros(0, dtype=np.float32),
            t_query=t_query,
            label=self.label,
            unit=self.unit,
            comparable=False,
        )


# ---------------------------------------------------------------------------
# Built-in: STHP ground-truth provider
# ---------------------------------------------------------------------------

class STHPGroundTruthProvider(ReferenceSurfaceProvider):
    """True conditional intensity λ*(t,s|H) for an STHP model.

    Evaluates the exact Hawkes intensity surface via
    ``STHPDataset.lamb_st_grid`` at each query time, conditioned on the
    observed event history.  Parameters are loaded from the
    ``dataset_meta.json`` written by ``gen_sthp_splits.py``.

    Returns ``SurfaceResult(surface_type='intensity', comparable=True)``.

    Notes
    -----
    ``history_times`` and ``xs``/``ys`` must all be in original (un-normalized)
    space — the same coordinate system used when generating the data.
    ``lamb_st_grid`` applies an internal ``< t_query`` filter, so passing the
    full fixed-history window is correct under both ``fixed`` and ``rolling``
    history modes.
    """

    def __init__(self, params: dict) -> None:
        """
        Parameters
        ----------
        params : dict with keys alpha, beta, mu, g0_cov, g2_cov, s_mu
            (as stored in dataset_meta.json["params"])
        """
        self._params = params

    @classmethod
    def from_meta_file(cls, path: str) -> "STHPGroundTruthProvider":
        """Load STHP parameters from a ``dataset_meta.json`` file."""
        with open(path) as f:
            meta = json.load(f)
        p = meta["params"]
        return cls({
            "alpha":  p["alpha"],
            "beta":   p["beta"],
            "mu":     p["mu"],
            "g0_cov": np.array(p["g0_cov"], dtype=np.float64),
            "g2_cov": np.array(p["g2_cov"], dtype=np.float64),
            "s_mu":   np.array(p["s_mu"],   dtype=np.float64),
        })

    def compute(
        self,
        history_times: np.ndarray,   # (L,) original space
        history_locs:  np.ndarray,   # (L, 2) original space
        t_query:       float,
        xs:            np.ndarray,   # (n_grid,) original space
        ys:            np.ndarray,   # (n_grid,)
    ) -> SurfaceResult:
        from unified_stpp.data.synthetic import STHPDataset

        p = self._params
        gen = STHPDataset(
            s_mu=p["s_mu"],
            g0_cov=p["g0_cov"],
            g2_cov=p["g2_cov"],
            alpha=p["alpha"],
            beta=p["beta"],
            mu=p["mu"],
        )
        # Inject observed history; lamb_st_grid applies < t_query internally
        gen.his_t = np.asarray(history_times, dtype=np.float64)
        gen.his_s = np.asarray(history_locs,  dtype=np.float64).reshape(-1, 2)

        values = gen.lamb_st_grid(
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            float(t_query),
        )   # (n_x, n_y)

        return SurfaceResult(
            surface_type="intensity",
            values=values.astype(np.float32),
            xs=xs.astype(np.float32),
            ys=ys.astype(np.float32),
            t_query=t_query,
            label="STHP ground truth λ*(t,s|H)",
            unit="events / (unit_time × unit_area)",
            comparable=True,
        )
