#!/usr/bin/env python3
"""Bounded executable smoke test for the frozen model roster.

This is deliberately an integration preflight, not a fit, HPO sweep, or paper
result. It exercises two fixed 20-history/one-target contexts per preset.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import numpy as np


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Strictly increasing, two deterministic 21-event sequences in [0,1]^2.
    base = torch.arange(21, dtype=torch.float32, device=device) / 24.0
    times = torch.stack((base, base + .01))
    i = torch.arange(21, dtype=torch.float32, device=device)
    locations = torch.stack((torch.stack(((i + 1) / 24, (2 * i + 3).remainder(23) / 24), -1),
                             torch.stack(((3 * i + 2).remainder(23) / 24, (i + 4) / 25), -1)))
    return times, locations, torch.tensor([21, 21], dtype=torch.long, device=device)


def _scalar_finite(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.numel() == 1 and bool(torch.isfinite(value).item())


def _native_sample(model, times, locations, lengths) -> dict[str, Any]:
    caps = model.event_model.capabilities
    if not caps.has_native_sampler:
        return {"attempted": False}
    state = model.state_model.encode_sampling_history(times=times, locations=locations, lengths=lengths)
    sample = model.event_model.sample_native(state=state)
    if isinstance(sample, dict):
        sample = sample.get("samples")
    if not isinstance(sample, torch.Tensor):
        return {"attempted": True, "finite": False, "reason": "native sampler returned no tensor or samples key"}
    return {"attempted": True, "shape": list(sample.shape), "finite": bool(torch.isfinite(sample).all().item())}


def _dnsk_checks(model, times, locations, lengths, *, tolerance: float = 1e-2) -> dict[str, Any]:
    event = model.event_model
    if type(event).__name__ != "DNSKEventModel":
        return {"applicable": False}
    state = model.state_model.encode_history(times=times, locations=locations, lengths=lengths)
    grid_t = torch.linspace(.05, .8, 9, device=times.device).repeat(2, 1)
    grid_x = torch.stack(torch.meshgrid(torch.linspace(.1,.9,3,device=times.device), torch.linspace(.1,.9,3,device=times.device), indexing="ij"), -1).reshape(-1,2)
    grid_x = grid_x.unsqueeze(0).expand(2,-1,-1)
    values = event.intensity(state=state, query_times=grid_t[:, :grid_x.shape[1]], query_locations=grid_x)
    # Compare the same full-history DNSK criterion before the optimizer update.
    # Target selection is irrelevant to quadrature convergence and would mask
    # compensator error at the other observed events.
    default = event.training_loss(times=times, locations=locations, lengths=lengths, state=state)["nll"]
    values_by_nodes = [(len(event.temporal_grid), int(round(len(event.spatial_offsets) ** .5)), float(default.detach()))]
    current = event
    for _ in range(2):
        refined = copy.deepcopy(current)
        temporal_nodes = 2 * len(current.temporal_grid) - 1
        spatial_nodes = 2 * int(round(len(current.spatial_offsets) ** .5))
        refined.temporal_grid = torch.linspace(0, event.tau_max, temporal_nodes, device=times.device)
        axis = (torch.arange(spatial_nodes, device=times.device, dtype=times.dtype) + .5) * (2*event.radius/spatial_nodes) - event.radius
        offsets = torch.cartesian_prod(axis,axis)
        refined.spatial_offsets = offsets
        refined.spatial_offset_weights = (offsets.square().sum(-1) <= event.radius**2).to(times.dtype) * (2*event.radius/spatial_nodes)**2
        nll = refined.training_loss(times=times, locations=locations, lengths=lengths, state=state)["nll"]
        values_by_nodes.append((temporal_nodes, spatial_nodes, float(nll.detach())))
        current = refined
    previous, final = values_by_nodes[-2][-1], values_by_nodes[-1][-1]
    relative_change = abs(final - previous) / max(abs(previous), 1e-12)
    return {"applicable": True, "pre_optimizer": True,
            "positive_query_grid": bool(torch.isfinite(values).all() and (values > 0).all()),
            "quadrature_levels": [{"temporal_nodes":t,"spatial_nodes":s,"nll":n} for t,s,n in values_by_nodes],
            "last_refinement_relative_change": relative_change, "tolerance": tolerance,
            "converged": bool(np.isfinite(relative_change) and relative_change <= tolerance)}


def preflight_one(entry: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from seahorse.config.schema import STPPConfig
    preset = entry["preset"]
    result: dict[str, Any] = {"id": entry["id"], "preset": preset, "label": entry["label"], "nll_path": entry["nll_path"]}
    try:
        torch.manual_seed(20260910)
        config = STPPConfig.from_preset(preset)
        effective = config.model_dump(mode="json")
        model = config.model.build_model().to(device)
        times, locations, lengths = _batch(device)
        model.train()
        out = model(times, locations, lengths, target_scope="last")
        loss_result = model.compute_loss(out)
        matrix, mask = out.get("nll_matrix"), out.get("mask")
        per_target_finite = None
        if isinstance(matrix, torch.Tensor) and isinstance(mask, torch.Tensor):
            selected = matrix[mask.bool()]
            per_target_finite = bool(selected.numel() == 2 and torch.isfinite(selected).all().item())
        result.update(total_events=float(loss_result.total_events.detach()), loss_finite=_scalar_finite(loss_result.loss), nll_finite=_scalar_finite(loss_result.nll), per_target_finite=per_target_finite)
        if float(loss_result.total_events.detach()) != 2.:
            raise AssertionError(f"target_scope=last emitted {float(loss_result.total_events)} rather than exactly two targets")
        # DNSK's compensator is a quadrature approximation. Check convergence at
        # the initial parameters: an optimizer update must not turn this into a
        # one-step stability test or conceal a discretization failure.
        dnsk = _dnsk_checks(model, times, locations, lengths)
        if dnsk["applicable"] and not dnsk["converged"]:
            raise AssertionError(
                "DNSK quadrature did not converge under prescribed refinements: "
                f"relative change={dnsk['last_refinement_relative_change']:.6g}, "
                f"tolerance={dnsk['tolerance']:.6g}"
            )
        if dnsk["applicable"] and not dnsk["positive_query_grid"]:
            raise AssertionError("DNSK intensity failed the finite-positive query grid check")
        loss_result.loss.backward()
        grad_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        if not grad_finite: raise AssertionError("non-finite gradient")
        optimizer = config.training.build_optimizer(model.parameters()); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        updater = getattr(model.event_model, "update_ema", None)
        if callable(updater): updater()
        model.eval()
        with torch.no_grad():
            eval_out = model.eval_forward(times, locations, lengths, target_scope="last")
        caps = model.event_model.capabilities
        result.update(capabilities={"nll_kind": caps.nll_kind, "native_sampler": caps.has_native_sampler,
                                    "objective": caps.training_objective},
                      eval_metric_finite=_scalar_finite(eval_out.get("nll")), native_sample=_native_sample(model,times,locations,lengths),
                      dnsk=dnsk)
        if caps.nll_kind == "exact" and not result["eval_metric_finite"]:
            raise AssertionError("exact-NLL preset returned a non-finite validation quantity")
        if caps.nll_kind == "exact" and result["per_target_finite"] is not True:
            raise AssertionError("exact-NLL preset did not expose two finite selected target terms")
        if caps.nll_kind == "none" and not result["native_sample"].get("finite", False):
            raise AssertionError("non-likelihood preset has no finite native sample")
        checkpoint = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        reload_model = config.model.build_model().to(device)
        incompatible = reload_model.load_state_dict(checkpoint, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys: raise AssertionError("checkpoint state mismatch")
        reload_config = STPPConfig.model_validate(effective).model_dump(mode="json")
        if reload_config != effective: raise AssertionError("effective config failed JSON round-trip")
        result.update(status="pass", checkpoint_reload="strict", effective_config_sha256=hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest())
    except Exception as exc:
        result.update(status="fail", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(limit=8))
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path("experiments/campaign_2026/spec_v1.json"))
    parser.add_argument("--models", nargs="*", help="model IDs or presets; default all configured entries")
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    spec=json.loads(args.spec.read_text()); entries=spec["models"]
    if args.models:
        wanted=set(args.models); entries=[e for e in entries if e["id"] in wanted or e["preset"] in wanted]
        unknown=wanted-{x for e in entries for x in (e["id"],e["preset"])}
        if unknown: raise SystemExit(f"Unknown configured model selector(s): {sorted(unknown)}")
    device=torch.device(args.device)
    rows=[preflight_one(entry,device) for entry in entries]
    report={"purpose":"bounded executable roster smoke only; not fit/HPO/scientific evidence", "created_utc":datetime.now(timezone.utc).isoformat(),
            "spec":str(args.spec), "spec_sha256":_sha(args.spec), "device":str(device), "contexts":2, "history_events":20,
            "results":rows, "passed":sum(r["status"]=="pass" for r in rows), "failed":sum(r["status"]!="pass" for r in rows)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"passed":report["passed"],"failed":report["failed"]}))
    return 1 if report["failed"] else 0


if __name__ == "__main__": raise SystemExit(main())
