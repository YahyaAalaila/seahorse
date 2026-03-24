"""Unified surface-evaluation layer for all STPP model families.

Architecture
------------
``SurfaceEvalSpec``
    Pure evaluation parameters (history policy, query times, spatial grid).
    No rendering or visualization fields.  Usable standalone for benchmarks.

``SurfaceEvaluator``
    Model-agnostic engine.  Owns normalization, grid construction, and dispatch.
    Calls ``event_model.query_surface(state, grid_times, grid_locs)`` — the only
    model-specific call.  All other logic is shared.

``SurfaceResult``
    Carrier for one evaluated surface frame.  All coordinates in original space.
    ``comparable=True`` for intensity/density; ``False`` for proxy_kde.

``SurfaceQuery``
    Backward-compatibility shim wrapping ``SurfaceEvaluator``.
    New code should use ``SurfaceEvaluator`` directly.

Surface types (``surface_type``)
---------------------------------
``"intensity"``  — λ*(t,s|H).  Units: events / (unit_time × unit_area).
``"density"``    — p(s|t,H).   Units: probability / unit_area.
``"proxy_kde"``  — KDE from samples.  Unscaled; not comparable across models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from unified_stpp.runner.runner import STPPRunner

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SurfaceEvalSpec
# ---------------------------------------------------------------------------

@dataclass
class SurfaceEvalSpec:
    """Evaluation-only parameters for a surface query run.

    No rendering or visualization fields.  Pass to ``SurfaceEvaluator`` or
    embed inside ``SurfaceVizConfig`` (composition, not inheritance).
    """

    # Data selection
    split: str = "val"
    seq_idx: int = 0

    # History policy
    history_mode: Literal["fixed", "rolling"] = "fixed"
    # fixed   : select history once using history_anchor at the start
    # rolling : for each frame t_i, use all events strictly before t_i
    history_anchor: Literal["last_n", "first_n", "from_event"] = "last_n"
    history_anchor_event_idx: Optional[int] = None   # for "from_event" mode
    history_length: int = 0    # 0 = use all available events (no cap)

    # Query times
    t_query_mode: Literal["explicit", "after_history", "uniform"] = "uniform"
    t_queries: Optional[list] = None   # explicit mode only
    n_time_steps: int = 3
    horizon: float = 1.0               # for "after_history" mode (original space)

    # Spatial domain in original space.
    # None = auto-compute from history ± 0.5σ.
    # Provide as ((x_lo, x_hi), (y_lo, y_hi)) for 2-D or ((x_lo, x_hi),) for 1-D.
    spatial_domain: Optional[Tuple] = None

    # Grid resolution
    n_grid: int = 50

    # For proxy-KDE models only
    n_samples: int = 500


# ---------------------------------------------------------------------------
# SurfaceResult
# ---------------------------------------------------------------------------

@dataclass
class SurfaceResult:
    """A spatial surface evaluated at a fixed query time.

    All coordinates and values are in original (un-normalized) space.
    """

    surface_type: Literal["intensity", "density", "proxy_kde"]
    """Scientific type of the surface values."""

    values: np.ndarray
    """(n_grid, n_grid) or (n_grid,) surface values, float32."""

    xs: np.ndarray
    """(n_grid,) x-axis grid in original space."""

    ys: np.ndarray
    """(n_grid,) y-axis grid in original space.  Empty array for 1-D spatial models."""

    t_query: float
    """Query time in original (un-normalized) space."""

    label: str
    """Human-readable label for figure titles / colorbars."""

    unit: str
    """Physical unit string."""

    comparable: bool
    """True if this surface can be meaningfully compared to surfaces of the
    same type from other models (requires identical normalization and same
    surface_type).  Always False for proxy_kde surfaces."""

    n_samples: Optional[int] = None
    """Number of samples used for proxy_kde; None otherwise."""

    model_name: Optional[str] = None
    """Model that produced this surface, set externally (e.g. by SurfaceBenchmark)."""

    history_times: Optional[np.ndarray] = None
    """(T,) history event times in original space used for this frame (optional)."""

    history_locs: Optional[np.ndarray] = None
    """(T, d) history event locations in original space used for this frame (optional)."""


# ---------------------------------------------------------------------------
# Label / unit constants
# ---------------------------------------------------------------------------

_SURFACE_LABELS = {
    "intensity": ("Conditional intensity λ*(t,s|H)", "events / (unit_time × unit_area)"),
    "density":   ("Conditional spatial density p(s|t,H)", "probability / unit_area"),
    "proxy_kde": ("Spatial proxy KDE (not comparable)", "proxy (not comparable)"),
}


# ---------------------------------------------------------------------------
# SurfaceEvaluator
# ---------------------------------------------------------------------------

class SurfaceEvaluator:
    """Model-agnostic surface evaluation engine.

    Owns normalization, grid construction, and dispatch to
    ``event_model.query_surface()``.  No model-specific logic.

    Parameters
    ----------
    runner : STPPRunner
        A fitted runner (``runner.fit()`` must have been called).
    """

    def __init__(self, runner: "STPPRunner"):
        if runner._data_module is None or runner._lightning_module is None:
            raise RuntimeError(
                "SurfaceEvaluator requires a fitted runner. Call runner.fit() first."
            )
        self._runner = runner

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def evaluate_frame(
        self,
        history_times: np.ndarray,
        history_locs: np.ndarray,
        t_query: float,
        spatial_domain: Optional[Tuple] = None,
        n_grid: int = 50,
        n_samples: int = 500,
    ) -> SurfaceResult:
        """Evaluate one surface frame.

        Parameters
        ----------
        history_times : (T,) array, original space
        history_locs  : (T, d) array, original space
        t_query       : query time in original space
        spatial_domain: ((x_lo, x_hi), (y_lo, y_hi)) in original space;
                        None → auto from history ± 0.5σ
        n_grid        : grid resolution per axis
        n_samples     : samples for proxy_kde path

        Returns
        -------
        SurfaceResult with all coordinates in original space.
        """
        model = self._runner.model
        model.eval()
        ds = self._ds
        device = next(model.parameters()).device

        # 1. Normalize history
        t_norm, s_norm = self._normalize_history(history_times, history_locs)
        t_q_norm = self._normalize_t(t_query)

        # 2. Grid bounds in normalized space
        s_min_arr, s_max_arr = self._grid_bounds_norm(s_norm, spatial_domain)
        d = s_norm.shape[-1] if len(s_norm) > 0 else history_locs.shape[-1]

        # 3. Build flat grid in normalized space
        if d >= 2:
            x_norm = np.linspace(s_min_arr[0], s_max_arr[0], n_grid, dtype=np.float32)
            y_norm = np.linspace(s_min_arr[1], s_max_arr[1], n_grid, dtype=np.float32)
            xx, yy = np.meshgrid(x_norm, y_norm, indexing="ij")
            s_flat_np = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (G, 2)
        else:
            x_norm = np.linspace(s_min_arr[0], s_max_arr[0], n_grid, dtype=np.float32)
            s_flat_np = x_norm.reshape(-1, 1)
            y_norm = None

        G = s_flat_np.shape[0]

        # 4. Tensors
        N = len(t_norm)
        history_times_t = torch.tensor(t_norm, device=device).unsqueeze(0)   # (1, T)
        history_locs_t  = torch.tensor(s_norm, device=device).unsqueeze(0)   # (1, T, d)
        history_lengths = torch.tensor([N], dtype=torch.long, device=device)  # (1,)
        grid_times_t    = torch.full((G,), t_q_norm, dtype=torch.float32, device=device)
        grid_locs_t     = torch.tensor(s_flat_np, device=device)              # (G, d)

        # 5. Encode history
        with torch.no_grad():
            state_ctx = model.state_model.encode_history(
                times=history_times_t,
                locations=history_locs_t,
                lengths=history_lengths,
            )

        # 6. Query surface via model contract
        # Note: query_surface() for proxy_kde models (SMASH) may need
        # enable_grad internally — each model handles its own grad context.
        values_norm = model.event_model.query_surface(
            state=state_ctx,
            grid_times=grid_times_t,
            grid_locs=grid_locs_t,
            n_samples=n_samples,
        )

        # 7. Denormalize values
        surface_type = model.event_model.surface_query_type
        values_flat = values_norm.cpu().detach().numpy()

        if surface_type == "intensity":
            scale = float(max(ds.time_std, 1e-8) * np.prod(np.maximum(ds.loc_std, 1e-8)))
            values_out = (values_flat / scale).astype(np.float32)
        elif surface_type == "density":
            s_scale = float(np.prod(np.maximum(ds.loc_std, 1e-8)))
            values_out = (values_flat / s_scale).astype(np.float32)
        else:  # proxy_kde — no calibrated denormalization
            values_out = values_flat.astype(np.float32)

        if d >= 2:
            values_out = values_out.reshape(n_grid, n_grid)

        # 8. Denormalize grid axes
        xs_orig = x_norm * ds.loc_std[0] + ds.loc_mean[0]
        ys_orig = y_norm * ds.loc_std[1] + ds.loc_mean[1] if y_norm is not None else np.zeros(0)

        label, unit = _SURFACE_LABELS[surface_type]
        comparable = (surface_type != "proxy_kde")
        n_samp = n_samples if surface_type == "proxy_kde" else None

        return SurfaceResult(
            surface_type=surface_type,
            values=values_out,
            xs=xs_orig.astype(np.float32),
            ys=ys_orig.astype(np.float32),
            t_query=t_query,
            label=label,
            unit=unit,
            comparable=comparable,
            n_samples=n_samp,
            history_times=np.asarray(history_times),
            history_locs=np.asarray(history_locs),
        )

    def evaluate_sequence(
        self,
        spec: SurfaceEvalSpec,
    ) -> list[SurfaceResult]:
        """Evaluate a sequence of surface frames according to ``spec``.

        Handles data extraction from the runner's data module, history window
        selection, t_query resolution, and optional rolling history.

        Parameters
        ----------
        spec : SurfaceEvalSpec

        Returns
        -------
        list[SurfaceResult], one per resolved t_query.
        """
        dm = self._runner._data_module
        seq = dm.get_original_sequence(spec.split, spec.seq_idx)
        all_times = seq["times"]
        all_locs  = seq["locations"]

        hist_t, hist_s = self._select_history(all_times, all_locs, spec)
        t_queries      = self._resolve_t_queries(hist_t, all_times, spec)

        rolling = spec.history_mode == "rolling"
        surfaces = []
        for t_q in t_queries:
            if rolling:
                ht, hs = self._rolling_history(all_times, all_locs, t_q, spec.history_length)
            else:
                ht, hs = hist_t, hist_s

            _LOG.info(
                "[surface_eval] t_query=%.4f | hist_len=%d | hist_t=[%.4f…%.4f]",
                t_q,
                len(ht),
                float(ht[0]) if len(ht) > 0 else float("nan"),
                float(ht[-1]) if len(ht) > 0 else float("nan"),
            )
            result = self.evaluate_frame(
                history_times=ht,
                history_locs=hs,
                t_query=t_q,
                spatial_domain=spec.spatial_domain,
                n_grid=spec.n_grid,
                n_samples=spec.n_samples,
            )
            surfaces.append(result)

        return surfaces

    # ------------------------------------------------------------------
    # Internal helpers — normalization
    # ------------------------------------------------------------------

    @property
    def _ds(self):
        return self._runner._data_module._train_dataset

    def _normalize_history(
        self,
        history_times: np.ndarray,
        history_locs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ds = self._ds
        t = (np.asarray(history_times, dtype=np.float64) - ds.time_mean) / max(ds.time_std, 1e-8)
        s = (np.asarray(history_locs, dtype=np.float64) - ds.loc_mean) / np.maximum(ds.loc_std, 1e-8)
        return t.astype(np.float32), s.astype(np.float32)

    def _normalize_t(self, t: float) -> float:
        ds = self._ds
        return float((t - ds.time_mean) / max(ds.time_std, 1e-8))

    def _grid_bounds_norm(
        self,
        s_norm: np.ndarray,
        spatial_domain: Optional[Tuple],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute grid bounds in normalized space.

        ``spatial_domain`` is in original space as ((x_lo, x_hi), (y_lo, y_hi)).
        When None, auto-compute from history ± 0.5σ.
        """
        ds = self._ds
        d = s_norm.shape[-1] if s_norm.ndim >= 2 else 1
        has_history = s_norm.shape[0] > 0

        s_lo = np.zeros(d, dtype=np.float32)
        s_hi = np.zeros(d, dtype=np.float32)

        for i in range(d):
            if spatial_domain is not None and i < len(spatial_domain) and spatial_domain[i] is not None:
                lo_orig, hi_orig = spatial_domain[i]
                s_lo[i] = (lo_orig - ds.loc_mean[i]) / max(float(ds.loc_std[i]), 1e-8)
                s_hi[i] = (hi_orig - ds.loc_mean[i]) / max(float(ds.loc_std[i]), 1e-8)
            else:
                col = s_norm[:, i] if s_norm.ndim >= 2 else s_norm
                s_lo[i] = float(col.min() - 0.5) if has_history else -3.0
                s_hi[i] = float(col.max() + 0.5) if has_history else  3.0

        return s_lo, s_hi

    # ------------------------------------------------------------------
    # Internal helpers — history & t_query selection
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_history(
        all_times: np.ndarray,
        all_locs: np.ndarray,
        t_query: float,
        history_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (hist_t, hist_s) of all events strictly before ``t_query``,
        capped to the most recent ``history_length`` events."""
        mask = all_times < t_query
        ht = all_times[mask]
        hs = all_locs[mask]
        if history_length > 0 and len(ht) > history_length:
            ht = ht[-history_length:]
            hs = hs[-history_length:]
        return ht, hs

    @staticmethod
    def _select_history(
        all_times: np.ndarray,
        all_locs: np.ndarray,
        spec: SurfaceEvalSpec,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select the fixed history window from the full sequence."""
        L = len(all_times)
        length = L if spec.history_length <= 0 else min(spec.history_length, L)

        if spec.history_anchor == "first_n":
            idx = slice(0, length)
        elif spec.history_anchor == "from_event":
            anchor = spec.history_anchor_event_idx
            if anchor is None:
                raise ValueError(
                    "history_anchor_event_idx must be set for history_anchor='from_event'."
                )
            start = max(0, anchor - length + 1)
            idx = slice(start, anchor + 1)
        else:  # "last_n"
            idx = slice(max(0, L - length), L)

        return all_times[idx], all_locs[idx]

    @staticmethod
    def _resolve_t_queries(
        hist_t: np.ndarray,
        all_times: np.ndarray,
        spec: SurfaceEvalSpec,
    ) -> list[float]:
        """Resolve t_query list from spec."""
        import warnings as _warnings

        n = spec.n_time_steps

        if spec.t_query_mode == "explicit":
            if not spec.t_queries:
                raise ValueError(
                    "t_queries must be a non-empty list when t_query_mode='explicit'."
                )
            return [float(t) for t in spec.t_queries]

        if spec.t_query_mode == "after_history":
            t_last = float(hist_t[-1]) if len(hist_t) > 0 else float(all_times[-1])
            return list(np.linspace(t_last, t_last + spec.horizon, n + 1, endpoint=True)[1:])

        # "uniform" — n equally-spaced points spanning [t_lo, t_hi]
        t_lo, t_hi = float(all_times.min()), float(all_times.max())
        return list(np.linspace(t_lo, t_hi, n))


# ---------------------------------------------------------------------------
# SurfaceQuery — backward-compatibility shim
# ---------------------------------------------------------------------------

class SurfaceQuery:
    """Backward-compatibility shim.  New code should use ``SurfaceEvaluator``.

    Wraps ``SurfaceEvaluator.evaluate_frame()`` behind the old ``query()`` API.
    The runner must have been fitted before constructing a ``SurfaceQuery``.
    """

    def __init__(self, runner: "STPPRunner"):
        self._evaluator = SurfaceEvaluator(runner)

    def query(
        self,
        history_times: np.ndarray,
        history_locs: np.ndarray,
        t_query: float,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        n_grid: int = 50,
        n_samples: int = 500,
    ) -> SurfaceResult:
        """Query a spatial surface at fixed time ``t_query`` given history.

        Parameters
        ----------
        history_times : (L,) array, original space
        history_locs  : (L, d) array, original space
        t_query       : query time in original space
        x_range       : (lo, hi) for x-axis; None → auto
        y_range       : (lo, hi) for y-axis; None → auto
        n_grid        : grid resolution per axis
        n_samples     : samples for proxy_kde path

        Returns
        -------
        SurfaceResult
        """
        spatial_domain = None
        if x_range is not None or y_range is not None:
            spatial_domain = (x_range, y_range)
        return self._evaluator.evaluate_frame(
            history_times=history_times,
            history_locs=history_locs,
            t_query=t_query,
            spatial_domain=spatial_domain,
            n_grid=n_grid,
            n_samples=n_samples,
        )
