"""Checkpoint-preserving numerical NLL refinement gates.

Only DNSK and NSMPP have bounded automatic refinements here.  Other numerical
paths are reported as unavailable rather than being called exact by accident.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class NumericalCheckConfig:
    absolute_tolerance: float = 1e-3
    relative_tolerance: float = 1e-2
    dnsk_query_time_nodes: int = 8
    dnsk_query_space_nodes: int = 8


def _config(value: NumericalCheckConfig | Mapping[str, Any] | None) -> NumericalCheckConfig:
    result = (
        NumericalCheckConfig()
        if value is None
        else (value if isinstance(value, NumericalCheckConfig) else NumericalCheckConfig(**value))
    )
    if result.absolute_tolerance < 0 or result.relative_tolerance < 0:
        raise ValueError("numerical tolerances must be nonnegative")
    return result


def refinement_decision(
    default: np.ndarray,
    finest: np.ndarray,
    config: NumericalCheckConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare the frozen default directly with finest, never just adjacent levels."""
    cfg = _config(config)
    default, finest = np.asarray(default, float), np.asarray(finest, float)
    if (
        default.shape != finest.shape
        or not default.size
        or not np.isfinite(default).all()
        or not np.isfinite(finest).all()
    ):
        return {"passed": False, "reason": "nonfinite_or_shape_mismatch"}
    difference = np.abs(default - finest)
    tolerance = cfg.absolute_tolerance + cfg.relative_tolerance * np.abs(finest)
    return {
        "passed": bool(np.all(difference <= tolerance)),
        "max_absolute_change": float(difference.max()),
        "max_tolerance": float(tolerance.max()),
        "violating_targets": int((difference > tolerance).sum()),
    }


def _last_nll(
    model: torch.nn.Module, batch: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, float]]:
    device = next(model.parameters()).device
    keys = ("times", "locations", "lengths", "marks", "x_event", "x_field_at_events")
    kwargs = {
        key: batch[key].to(device) for key in keys if isinstance(batch.get(key), torch.Tensor)
    }
    with torch.no_grad():
        output = model.eval_forward(**kwargs, target_scope="last")
    values = output.get("nll_per_event")
    if not isinstance(values, torch.Tensor):
        raise ValueError("model does not expose eventwise target NLL")
    values = values.detach().cpu().numpy().reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite per-target NLL")
    extra = output.get("extra_metrics", {})
    # Evidence must be JSON finite, so retain only finite scalar diagnostics.
    metrics = {
        str(key): float(value)
        for key, value in extra.items()
        if isinstance(value, (float, int, np.floating, np.integer)) and np.isfinite(value)
    }
    return values, metrics


def _configure_dnsk(
    model: torch.nn.Module, factor: int, cfg: NumericalCheckConfig
) -> dict[str, int]:
    event = model.event_model
    nodes = int(len(event.temporal_grid))
    spatial_nodes = int(round(len(event.spatial_offsets) ** 0.5))
    device, dtype = event.temporal_grid.device, event.temporal_grid.dtype
    temporal_nodes, spatial_nodes = 1 + (nodes - 1) * factor, spatial_nodes * factor
    event.temporal_grid = torch.linspace(
        0, event.tau_max, temporal_nodes, device=device, dtype=dtype
    )
    axis = (torch.arange(spatial_nodes, device=device, dtype=dtype) + 0.5) * (
        2 * event.radius / spatial_nodes
    ) - event.radius
    offsets = torch.cartesian_prod(axis, axis)
    event.spatial_offsets = offsets
    event.spatial_offset_weights = (offsets.square().sum(-1) <= event.radius**2).to(dtype) * (
        2 * event.radius / spatial_nodes
    ) ** 2
    # This is an explicit evaluation-only positivity probe on the copied model.
    event.barrier_time_nodes = int(cfg.dnsk_query_time_nodes)
    event.barrier_space_nodes = int(cfg.dnsk_query_space_nodes)
    return {
        "temporal_nodes": temporal_nodes,
        "spatial_nodes": spatial_nodes,
        "positivity_time_nodes": event.barrier_time_nodes,
        "positivity_space_nodes": event.barrier_space_nodes,
    }


def _configure_nsmpp(model: torch.nn.Module, factor: int) -> dict[str, int]:
    event = model.event_model
    event.interval_int_res = int(event.interval_int_res) * factor
    return {"interval_midpoint_nodes": int(event.interval_int_res)}


def _configure_thp(model: torch.nn.Module, factor: int) -> dict[str, int]:
    """Refine THP's deterministic *evaluation* composite-midpoint rule.

    THP calls its training compensator Monte Carlo, but its implementation uses
    fixed midpoint nodes in eval mode.  Increasing this instance attribute on
    a copied temporal process is consequently a deterministic quadrature
    oracle, not a repeated stochastic estimate.
    """
    temporal = model.event_model.temporal_model
    samples = int(temporal.MC_SAMPLES) * factor
    temporal.MC_SAMPLES = samples
    return {"evaluation_quadrature": "composite_midpoint", "midpoint_nodes": samples}


def _configure_neural_ode(model: torch.nn.Module, factor: int) -> dict[str, float | str]:
    solver = model.state_model.temporal_core.ode_solver
    solver.atol = float(solver.atol) / factor
    solver.rtol = float(solver.rtol) / factor
    return {"solver": str(solver.method), "atol": float(solver.atol), "rtol": float(solver.rtol)}


def _family(runner: Any) -> str:
    preset = getattr(getattr(getattr(runner, "config", None), "model", None), "preset", None)
    event = runner.model.event_model
    if preset == "dnsk" or event.__class__.__name__ == "DNSKEventModel":
        return "dnsk"
    if preset == "nsmpp" or event.__class__.__name__ == "NSMPPDeepBasisEventModel":
        return "nsmpp"
    if (
        preset == "thp_gmm"
        or event.__class__.__name__ == "FactorizedEventModel"
        and event.temporal_model.__class__.__name__ == "THPTemporalProcess"
    ):
        return "thp"
    if str(preset).startswith("neural_stpp") or hasattr(
        getattr(runner.model, "state_model", None), "temporal_core"
    ):
        return "neural_ode"
    return str(preset or "unknown")


def run_numerical_nll_checks(
    runner: Any,
    batch: Mapping[str, Any],
    config: NumericalCheckConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run default/x2/x4 quadrature checks on deep copies of a loaded model.

    The original checkpoint object, its buffers, parameters and training state
    are not changed.  A nonfinite result or positivity exception is evidence
    of failure, never repaired by clipping.
    """
    cfg, family = _config(config), _family(runner)
    evidence: dict[str, Any] = {
        "family": family,
        "config": asdict(cfg),
        "status": "unavailable",
        "checks": [],
    }
    if family not in {"dnsk", "nsmpp", "thp", "neural_ode"}:
        evidence["reason"] = "automatic refinement is pending/manual for this NLL path"
        return evidence
    try:
        variants = []
        for factor, label in ((1, "default"), (2, "refined"), (4, "finest")):
            clone = copy.deepcopy(runner.model).eval()
            if family == "dnsk":
                setting = _configure_dnsk(clone, factor, cfg)
            elif family == "nsmpp":
                setting = _configure_nsmpp(clone, factor)
            elif family == "thp":
                setting = _configure_thp(clone, factor)
            else:
                setting = _configure_neural_ode(clone, factor)
            values, metrics = _last_nll(clone, batch)
            variants.append(values)
            record = {
                "label": label,
                "factor": factor,
                "setting": setting,
                "targets": int(len(values)),
                "mean_per_target_nll": float(values.mean()),
                "finite": True,
            }
            if family == "dnsk":
                grid_min = metrics.get("dnsk_grid_min_intensity")
                if grid_min is None or grid_min <= 0:
                    raise ValueError(
                        "DNSK positivity query grid did not return a positive finite minimum"
                    )
                record["scored_event_positivity"] = "passed by DNSK evaluation"
                record["query_grid_min_intensity"] = grid_min
            evidence["checks"].append(record)
        decision = refinement_decision(variants[0], variants[-1], cfg)
        evidence["default_vs_finest"] = decision
        if family == "dnsk":
            evidence["positivity"] = {
                "scored_events": "passed by DNSK evaluation",
                "query_grid": "passed: positive finite minima recorded per refinement",
            }
        evidence["status"] = "pass" if decision["passed"] else "fail"
    except Exception as exc:
        evidence.update(status="fail", failure_type=type(exc).__name__, failure=str(exc))
    return evidence


__all__ = ["NumericalCheckConfig", "refinement_decision", "run_numerical_nll_checks"]
