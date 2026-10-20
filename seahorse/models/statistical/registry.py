"""Preset resolution for the NumPy/SciPy statistical models."""

from __future__ import annotations

from typing import Mapping

from .core import as_support
from .hawkes import HawkesSTPP
from .poisson import PoissonSTPP


STATISTICAL_PRESETS = {
    "homogeneous_poisson": {"kind": "poisson", "grid": (1, 1)},
    "spatial_poisson": {"kind": "poisson", "grid": (8, 8)},
    # Explicit product: spatial probabilities are invariant across phases.
    "seasonal_poisson": {"kind": "poisson", "grid": (8, 8), "period": 7.0, "n_phases": 7},
    "hawkes_exp_gaussian": {"kind": "hawkes", "temporal": "exponential", "spatial": "gaussian"},
    "hawkes_omori_pareto": {"kind": "hawkes", "temporal": "omori", "spatial": "pareto"},
}


def build_statistical_model(preset: str, options: Mapping | None = None):
    """Build an unfitted model from a named preset and declared options.

    ``support`` is intentionally mandatory: it controls continuous density
    units, boundary compensators and sampling, so discovering it from held-out
    events would leak evaluation information.
    """
    if preset not in STATISTICAL_PRESETS:
        raise KeyError(f"Unknown statistical preset {preset!r}; available={sorted(STATISTICAL_PRESETS)}")
    merged = dict(STATISTICAL_PRESETS[preset])
    merged.update(dict(options or {}))
    if "support" not in merged:
        raise ValueError("Statistical model options must declare support=(xmin,xmax,ymin,ymax).")
    support = as_support(merged.pop("support"))
    kind = merged.pop("kind")
    if kind == "poisson":
        raw_grid = tuple(merged.pop("grid", (1, 1)))
        if len(raw_grid) != 2 or any(int(v) != v for v in raw_grid):
            raise ValueError("grid dimensions must be two positive integers.")
        grid = tuple(int(v) for v in raw_grid)
        if min(grid) < 1:
            raise ValueError("grid dimensions must be positive.")
        period = merged.pop("period", None)
        n_phases = int(merged.pop("n_phases", 1))
        if period is not None and (float(period) <= 0 or n_phases < 1):
            raise ValueError("seasonal period and n_phases must be positive.")
        model = PoissonSTPP(support=support, grid=grid, period=None if period is None else float(period),
                            n_phases=n_phases, phase_origin=float(merged.pop("phase_origin", 0.0)),
                            alpha=float(merged.pop("alpha", 0.5)))
    elif kind == "hawkes":
        model = HawkesSTPP(support=support, temporal=str(merged.pop("temporal")), spatial=str(merged.pop("spatial")),
                           quadrature_order=int(merged.pop("quadrature_order", 32)),
                           edge_correction=str(merged.pop("edge_correction", "auto")),
                           sampler_iterations=int(merged.pop("sampler_iterations", 2000)))
    else:
        raise AssertionError(f"Unsupported statistical kind {kind!r}")
    if merged:
        raise ValueError(f"Unknown options for {preset}: {sorted(merged)}")
    model.preset = preset
    quality = "exact" if kind == "poisson" else model.nll_kind
    method = "analytic" if kind == "poisson" else model.nll_method
    model.metadata = {
        "preset": preset, "nll_kind": quality, "nll_target": "continuous_next_event_density",
        "nll_method": method,
        "nll_description": "Joint time-location conditional next-event density, including survival to target time.",
        "sampling_method": "analytic" if kind == "poisson" else model.sampling_method,
    }
    if kind == "poisson":
        model.nll_kind = quality
        model.nll_method = method
    model.nll_target = model.metadata["nll_target"]
    model.nll_description = model.metadata["nll_description"]
    return model


def model_from_state(state: Mapping):
    kind = state.get("kind")
    if kind == "poisson":
        model = PoissonSTPP.from_state(state)
    elif kind == "hawkes":
        model = HawkesSTPP.from_state(state)
    else:
        raise ValueError(f"Unknown statistical model state kind {kind!r}.")
    metadata = dict(state.get("metadata", {}))
    if not metadata:
        metadata = {"nll_kind": "exact" if kind == "poisson" else "approx",
                    "nll_target": "continuous_next_event_density"}
    model.metadata = metadata
    for key in ("preset", "nll_kind", "nll_target", "nll_method", "nll_description"):
        if key in metadata and not (kind == "hawkes" and key in {"nll_kind", "nll_method"}):
            setattr(model, key, metadata[key])
    if kind == "hawkes":
        model.metadata.update(nll_kind=model.nll_kind, nll_method=model.nll_method, sampling_method=model.sampling_method)
    return model
