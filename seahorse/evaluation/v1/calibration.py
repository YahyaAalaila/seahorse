"""Calibration metrics computed from the NLL compensator (no sampler).

For every frozen fixed-history test target of a completed campaign cell this
computes, from the same likelihood machinery the campaign NLL pass uses:

* temporal: ``Lambda_i``, the NLL's own compensator increment over
  ``(t_{H-1}, t_H]``, gives the time-rescaling PIT ``u_i = 1 - exp(-Lambda_i)``;
  the cumulative hazard ``Lambda_i(tau)`` on a waiting-time grid in raw time
  units gives the next-event CDF ``F_i(tau) = 1 - exp(-Lambda_i(tau))``, the
  threshold-weighted ``CRPS_c`` (exact integral of the linearly interpolated
  CDF), predictive quantiles and their absolute error.
* spatial: the HPD-PIT at the true event time, ``u_i = P[p(S) >= p(s_i)]``,
  from the model's conditional spatial density on an ``N x N`` cell grid of its
  spatial domain (piecewise-constant discretisation, randomised within the
  observation's cell, hence exactly uniform for the discretised model). CNF
  decoders are not covered by this route.

Scoring primitives come from :mod:`seahorse.evaluation.v1.scoring`; the
per-family model access lives in :mod:`seahorse.evaluation.v1.calibration_families`.
Outputs are written only under ``out``; campaign directories, checkpoints and
data are opened read-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .scoring import (
    crps_exponential_truncated,
    crps_from_cdf_grid,
    ks_uniform,
    pit_from_compensator,
    pit_histogram,
    pit_tail_fractions,
    tau_grid_quadratic,
)

CALIBRATION_VERSION = "1.0.0"

DEFINITIONS = {
    "temporal_pit": "u = 1 - exp(-Lambda_i), Lambda_i = the NLL path's compensator over (t_{H-1}, t_H]",
    "crps": (
        "CRPS_c = int_0^c (F(tau) - 1{tau>=y})^2 dtau, F = 1 - exp(-Lambda(tau)), "
        "tau grid tau_j = tau_max (j/(n-1))^2 in raw units, F linear between nodes"
    ),
    "quantiles": (
        "tau_p = Lambda^{-1}(-log(1-p)): from the tau grid (linear interpolation) where the grid "
        "is computed, else by bisection of the family's closed-form / quadrature cumulative hazard"
    ),
    "point_forecast": "predictive median tau_0.5; abs error |tau_0.5 - y| in hours; q10/q90 coverage",
    "spatial_hpd_pit": (
        "u = P(p > p_obs) + V P(p == p_obs) on an N x N cell grid of the model's spatial domain "
        "(piecewise-constant discretisation, V ~ U(0,1) seeded)"
    ),
}


# ---------------------------------------------------------------------------
# Primitives not in scoring.py
# ---------------------------------------------------------------------------


def ks_uniform_with_critical(u: np.ndarray) -> dict:
    """:func:`ks_uniform` plus the asymptotic 95% critical value ``1.358 / sqrt(n)``."""
    out = ks_uniform(u)
    n = out["n"]
    out["ks_crit_95"] = float(1.358 / math.sqrt(n)) if n else float("nan")
    return out


def quantiles_from_cumhazard(
    cumhazard: np.ndarray, tau_grid: np.ndarray, probs=(0.1, 0.5, 0.9)
) -> dict[float, np.ndarray]:
    """``tau_p = Lambda^{-1}(-log(1-p))`` by linear interpolation; ``+inf`` if censored."""
    lg = np.maximum.accumulate(np.asarray(cumhazard, dtype=np.float64), axis=1)
    out = {}
    for p in probs:
        target = -math.log1p(-p)
        q = np.full(lg.shape[0], np.inf)
        ok = lg[:, -1] >= target
        for i in np.flatnonzero(ok):
            q[i] = np.interp(target, lg[i], tau_grid)
        out[p] = q
    return out


def to64(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu", torch.float64).numpy()


def bisect_quantiles(
    cumhaz_rows: Callable,
    n_rows: int,
    tau_max: float,
    probs=(0.1, 0.5, 0.9),
    iters: int = 24,
    xp=np,
    ref: torch.Tensor | None = None,
) -> dict[float, np.ndarray]:
    """``tau_p = Lambda^{-1}(-log(1-p))`` per row by joint bisection on ``[0, tau_max]``.

    ``cumhaz_rows(tau (B,K)) -> Lambda (B,K)`` is called once per iteration for
    all ``K`` probes (NumPy or torch; ``ref`` supplies device/dtype for torch).
    Rows with ``Lambda(tau_max) < target`` get ``+inf``.
    """
    k = len(probs)
    targets_np = np.array([-math.log1p(-p) for p in probs])
    if xp is np:
        def full(v):
            return np.full((n_rows, k), float(v))
        where = np.where
        asnp = np.asarray
        targets = targets_np[None, :]
    else:
        def full(v):
            return torch.full((n_rows, k), float(v), device=ref.device, dtype=ref.dtype)
        where = torch.where
        asnp = to64
        targets = torch.as_tensor(targets_np, device=ref.device, dtype=ref.dtype)[None, :]
    top = cumhaz_rows(full(tau_max))
    lo, hi = full(0.0), full(tau_max)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ge = cumhaz_rows(mid) >= targets
        hi = where(ge, mid, hi)
        lo = where(ge, lo, mid)
    q = np.asarray(asnp(0.5 * (lo + hi)), dtype=np.float64)
    q[np.asarray(asnp(top), dtype=np.float64) < targets_np[None, :]] = np.inf
    return {p: q[:, i] for i, p in enumerate(probs)}


def hpd_pit_from_cells(probs: np.ndarray, cell_obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomised HPD-PIT of a piecewise-constant cell model.

    ``u = P(p > p_obs) + V * P(p == p_obs)`` with ``V ~ U(0, 1)``; ``nan`` where the
    observation lies outside the grid (``cell_obs < 0``).
    """
    probs = np.asarray(probs, dtype=np.float64)
    u = np.full(probs.shape[0], np.nan)
    ok = np.asarray(cell_obs) >= 0
    if ok.any():
        p = probs[ok]
        po = p[np.arange(p.shape[0]), np.asarray(cell_obs)[ok]]
        above = (p * (p > po[:, None])).sum(1)
        tie = (p * (p == po[:, None])).sum(1)
        u[ok] = above + rng.random(p.shape[0]) * tie
    return u


def cell_index(xy: np.ndarray, box, n: int) -> np.ndarray:
    """Cell index on an ``n x n`` grid of ``box=(x0, x1, y0, y1)``; ``-1`` outside the closed box."""
    xy = np.asarray(xy, dtype=np.float64)
    x0, x1, y0, y1 = box
    fx = (xy[:, 0] - x0) / (x1 - x0)
    fy = (xy[:, 1] - y0) / (y1 - y0)
    inside = (fx >= 0) & (fx <= 1) & (fy >= 0) & (fy <= 1)
    ix = np.clip(np.floor(fx * n), 0, n - 1).astype(int)
    iy = np.clip(np.floor(fy * n), 0, n - 1).astype(int)
    return np.where(inside, ix * n + iy, -1)


def grid_centres(box, n: int) -> np.ndarray:
    """Cell centres of an ``n x n`` grid of ``box``, shape ``(n*n, 2)``, index ``ix*n + iy``."""
    x0, x1, y0, y1 = box
    cx = x0 + (np.arange(n) + 0.5) / n * (x1 - x0)
    cy = y0 + (np.arange(n) + 0.5) / n * (y1 - y0)
    xx, yy = np.meshgrid(cx, cy, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], -1)


def round_up(x: float, step: float = 0.1) -> float:
    return math.ceil(x / step - 1e-9) * step


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def metrics_block(arrays: dict, sel: np.ndarray, c_raw: float | None, hours: float, crps_sel=None) -> dict:
    """Summary metrics on the boolean target selection ``sel``.

    CRPS summaries use ``crps_sel & sel`` when ``crps_sel`` is given. Times are
    reported in raw units and in hours (``hours`` = hours per raw unit).
    """
    out: dict[str, Any] = {"n_selected": int(sel.sum())}
    pit = arrays["pit"][sel]
    fin = np.isfinite(pit)
    out["pit"] = {
        "ks": ks_uniform_with_critical(pit[fin]),
        "hist20": pit_histogram(pit[fin]),
        "tails": pit_tail_fractions(pit[fin]),
        "mean": float(np.mean(pit[fin])) if fin.any() else None,
        "n_finite": int(fin.sum()),
        "n_nonfinite": int((~fin).sum()),
    }
    if c_raw is not None:
        cs = sel if crps_sel is None else (sel & crps_sel)
        fc = np.isfinite(arrays["crps_raw"]) & cs
        if fc.any():
            out["crps"] = {
                "crps_c_mean_raw": float(np.mean(arrays["crps_raw"][fc])),
                "crps_c_mean_hours": float(np.mean(arrays["crps_raw"][fc]) * hours),
                "crps_c_median_hours": float(np.median(arrays["crps_raw"][fc]) * hours),
                "n": int(fc.sum()),
                "surv_at_c_mean": float(np.mean(arrays["surv_at_c"][fc])),
                "frac_y_above_c": float(np.mean(arrays["y_raw"][fc] > c_raw)),
            }
        q50 = arrays["q50_raw"]
        err = np.abs(q50 - arrays["y_raw"])
        fe = np.isfinite(err) & sel
        fq = np.isfinite(arrays["q10_raw"]) & np.isfinite(arrays["q90_raw"]) & sel
        if fe.any():
            y = arrays["y_raw"]
            out["point_forecast"] = {
                "median_abs_error_hours_mean": float(np.mean(err[fe]) * hours),
                "median_abs_error_hours_median": float(np.median(err[fe]) * hours),
                "median_abs_error_raw_mean": float(np.mean(err[fe])),
                "n": int(fe.sum()),
                "frac_median_censored_beyond_grid": float(np.mean(~np.isfinite(q50[sel]))),
                "observed_wait_hours_mean": float(np.mean(y[sel]) * hours),
                "observed_wait_hours_median": float(np.median(y[sel]) * hours),
                "coverage_q10": float(np.mean(y[fq] <= arrays["q10_raw"][fq])) if fq.any() else None,
                "coverage_q90": float(np.mean(y[fq] <= arrays["q90_raw"][fq])) if fq.any() else None,
                "central80_coverage": (
                    float(np.mean((y[fq] >= arrays["q10_raw"][fq]) & (y[fq] <= arrays["q90_raw"][fq])))
                    if fq.any() else None
                ),
            }
    fs = np.isfinite(arrays["hpd_pit"]) & sel
    if fs.any():
        mass = arrays["spatial_mass_in_box"][sel]
        out["spatial"] = {
            "hpd_pit_ks": ks_uniform_with_critical(arrays["hpd_pit"][fs]),
            "hist20": pit_histogram(arrays["hpd_pit"][fs]),
            "n_finite": int(fs.sum()),
            "n_outside_box_or_missing": int((sel & ~np.isfinite(arrays["hpd_pit"])).sum()),
            "mass_in_box_mean": float(np.nanmean(mass)) if np.isfinite(mass).any() else None,
        }
    return out


# ---------------------------------------------------------------------------
# Campaign-cell evaluation
# ---------------------------------------------------------------------------


def _max_train_inter_event_time(train_dataset, scale: float) -> float:
    if hasattr(train_dataset, "_segments"):
        return max(float(np.max(np.diff(seg.times))) for seg in train_dataset._segments if len(seg.times) > 1) * scale
    return max(
        float(np.max(np.diff(np.asarray(r["times"], float))))
        for r in train_dataset.raw_sequences
        if len(r["times"]) > 1
    ) * scale


def evaluate_calibration(
    manifest_path,
    job_id: str,
    out,
    *,
    model_dir=None,
    c_hours: float | None = None,
    tau_grid_n: int = 1024,
    tau_factor: float = 4.0,
    crps_scope: str = "auto",
    crps_subset_size: int = 20000,
    crps_subset_seed: int = 20261020,
    spatial_grid: int = 128,
    spatial: bool = True,
    crps: bool = True,
    mask_npz=None,
    solver_factor: int = 1,
    skip_eval_forward: bool = False,
    limit: int | None = None,
    batch_size: int | None = None,
    device: str = "cuda",
    threads: int = 4,
    commit_every: int = 64,
    allow_binding_mismatch: bool = False,
    closed_form_check: bool = False,
) -> dict:
    """Score one completed refit cell of a frozen campaign; returns the summary.

    Writes ``per_target.npz`` and ``summary.json`` under ``out`` and resumes from
    ``progress.npz`` when the evaluation binding matches.
    """
    from torch.utils.data import DataLoader, Subset

    import seahorse
    from seahorse.experiments import campaign_run as cr
    from seahorse.experiments.campaign_progress import checkpoint_fingerprint
    from seahorse.experiments.campaign_run import (
        _coordinate_entry, _model_entry, clean_json, context_dataset, read_records, select_indices,
    )
    from seahorse.models.configs import ConfigRegistry
    from seahorse.runner.runner import STPPRunner

    from .calibration_families import FAMILIES, StatisticalCalib

    torch.set_num_threads(int(threads))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    args = {
        "model_dir": None if model_dir is None else str(model_dir), "c_hours": c_hours, "tau_grid_n": tau_grid_n,
        "tau_factor": tau_factor, "crps_scope": crps_scope, "crps_subset_size": crps_subset_size,
        "crps_subset_seed": crps_subset_seed, "spatial_grid": spatial_grid, "spatial": spatial, "crps": crps,
        "mask_npz": None if mask_npz is None else str(mask_npz), "solver_factor": solver_factor,
        "skip_eval_forward": skip_eval_forward, "limit": limit, "batch_size": batch_size, "device": device,
        "threads": threads, "closed_form_check": closed_form_check,
    }

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    job = next(j for j in manifest["jobs"] if j["id"] == job_id)
    if job["action"] != "refit":
        raise ValueError("Only final-seed refit cells are evaluated.")
    job_dir = manifest_path.parent / "jobs" / job_id
    status = json.loads((job_dir / "status.json").read_text())
    if status.get("state") != "completed":
        raise ValueError(f"cell not completed: {status.get('state')}")
    if model_dir:
        model_dir = Path(model_dir)
    elif (job_dir / "model").exists():
        model_dir = job_dir / "model"
    else:
        model_dir = Path(status["checkpoint"])
    model_entry = _model_entry(manifest, job)
    preset = model_entry["preset"]
    route = manifest["datasets"][job["dataset"]]
    protocol = manifest["spec"]["protocol"]
    coord = _coordinate_entry(manifest, job)
    statistical = ConfigRegistry.runner_kind(preset) == "statstpp"
    device = "cpu" if statistical else device
    if model_entry["nll_path"] in {"none", "bound", "surrogate"}:
        raise ValueError(f"{preset}: no evaluable likelihood ({model_entry['nll_path']})")
    neural = preset in ("njsde", "neural_attncnf", "neural_jumpcnf")

    transform = manifest["coordinates"]["routes"][route["dataset_id"]]["transform"]
    time_unit = transform.get("input_time_unit", "unknown")
    hours_per_unit = 24.0 if time_unit == "day" else float("nan")
    campaign_nll = (
        json.loads((job_dir / "nll_status.json").read_text()) if (job_dir / "nll_status.json").exists() else None
    )
    stored = None
    if (job_dir / "target_nll.npz").exists():
        z = np.load(job_dir / "target_nll.npz")
        stored = {k: z[k] for k in z.files}

    summary: dict[str, Any] = {
        "calibration_version": CALIBRATION_VERSION, "seahorse_version": getattr(seahorse, "__version__", None),
        "campaign_run_py_sha256": sha256_file(cr.__file__),
        "started_utc": datetime.now(timezone.utc).isoformat(), "device": device, "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None,
        "job_id": job["id"], "preset": preset, "dataset": job["dataset"], "nll_path": model_entry["nll_path"],
        "manifest_path": str(manifest_path), "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256_field": manifest["sha256"],
        "model_dir": str(model_dir), "checkpoint_path_in_status": status.get("checkpoint"),
        "model_files_sha256": {p.name: sha256_file(p) for p in sorted(model_dir.rglob("*")) if p.is_file()},
        "coordinate_transform": coord.to_dict(), "time_unit_raw": time_unit, "hours_per_raw_unit": hours_per_unit,
        "args": args, "definitions": DEFINITIONS,
    }

    # ---------------------------------------------------------------- data + binding
    if statistical:
        from seahorse.experiments.campaign_statistical import _data

        runner = STPPRunner.load(model_dir)
        _records, dataset = _data(manifest, job, "test")
        time_scale = 1.0  # statistical contexts are in raw source units
        binding = {
            "manifest_sha256": manifest["sha256"], "job_id": job["id"],
            "checkpoint_sha256": checkpoint_fingerprint(model_dir), "test_split_sha256": route["split_sha256"]["test"],
        }
        batch_size = batch_size or 256
        seed = protocol.get("evaluation_seed", 20260916)
        collate = None
    else:
        from seahorse.data.prepared_protocol import _copy_unmarked
        from seahorse.data.registry import DataRegistry
        from seahorse.experiments.coordinates import transform_sequences

        runner = STPPRunner.load(model_dir)
        runner.model.to(device).eval()
        if neural and solver_factor != 1:
            from seahorse.experiments import numerical_checks as nc

            summary["amended_default_solver"] = nc._configure_neural_ode(runner.model, solver_factor)
            summary["solver_amendment"] = (
                f"temporal ODE solver tightened by factor {solver_factor} (atol/=F, rtol/=F); "
                "spatial CNF tolerances unchanged"
            )
        records = _copy_unmarked(list(transform_sequences(
            read_records(route["splits"]["test"], sha256=route["split_sha256"]["test"]), coord)))
        dataset = context_dataset(records, manifest, job, coord, split="test")
        time_scale = float(coord.time_scale)  # raw units per model-input unit
        batch_size = batch_size or protocol.get("evaluation_batch_size", manifest["spec"]["fit"]["batch_size"])
        seed = protocol.get("evaluation_seed", 20260916)
        binding = {
            "manifest_sha256": manifest["sha256"], "job_id": job["id"],
            "checkpoint_sha256": checkpoint_fingerprint(model_dir), "test_split_sha256": route["split_sha256"]["test"],
            "batch_size": batch_size, "seed": seed,
        }
        collate = DataRegistry.get_collate(ConfigRegistry.get_collate_key(runner.config.model.preset))
    if campaign_nll is not None:
        cb = campaign_nll.get("evaluation_binding", {})
        same = (
            all(cb.get(k) == v for k, v in binding.items() if k in cb)
            and cb.get("checkpoint_sha256") == binding["checkpoint_sha256"]
        )
        summary["campaign_binding"] = cb
        summary["binding_equals_campaign_binding"] = bool(same)
        summary["campaign_nll_status"] = campaign_nll.get("nll_status")
        summary["campaign_raw_next_event_nll"] = campaign_nll.get("raw_next_event_nll")
        if not same and not allow_binding_mismatch:
            raise ValueError(f"Evaluation binding differs from the campaign's: {cb} vs {binding}")
    summary["evaluation_binding"] = binding
    summary["log_raw_volume_per_normalized_volume"] = float(coord.log_raw_volume_per_normalized_volume)
    summary["time_scale_raw_per_model_unit"] = time_scale

    # CRPS threshold c (raw units): max training inter-event time, rounded up to 0.1
    do_crps = bool(crps)
    c_raw = None
    tau_raw = None
    if do_crps:
        if c_hours is not None:
            c_raw = c_hours / hours_per_unit
            summary["c_source"] = f"c_hours={c_hours}"
        else:
            if statistical:
                _tr, tds = _data(manifest, job, "train")
                ts_scale = 1.0
            else:
                train = _copy_unmarked(list(transform_sequences(
                    read_records(route["splits"]["train"], sha256=route["split_sha256"]["train"]), coord)))
                tds = context_dataset(train, manifest, job, coord, split="train")
                ts_scale = time_scale
            max_iet = _max_train_inter_event_time(tds, ts_scale)
            c_raw = round_up(max_iet, 0.1)
            summary["c_source"] = f"max train inter-event time {max_iet:.6g} raw units ({time_unit}) rounded up to 0.1"
            del tds
        tau_raw = tau_grid_quadratic(tau_factor * c_raw, tau_grid_n)
        summary.update({
            "c_raw": c_raw, "c_hours": c_raw * hours_per_unit, "tau_max_raw": float(tau_raw[-1]),
            "tau_grid_n": tau_grid_n, "tau_factor": tau_factor,
        })

    # ---------------------------------------------------------------- adapter
    box_t = (0.0, 1.0, 0.0, 1.0)  # campaign preprocessing rectangle in transformed coordinates
    if statistical:
        calib = StatisticalCalib(runner, dataset)
    else:
        if preset not in FAMILIES:
            raise NotImplementedError(f"preset {preset!r} not covered (families: {sorted(FAMILIES)})")
        calib = FAMILIES[preset](runner, device, box_t)
    tau_model = (
        torch.tensor(tau_raw / time_scale, dtype=torch.float32, device=device)
        if (do_crps and not statistical) else None
    )
    tau_max_model = float(tau_raw[-1] / time_scale) if do_crps else None
    do_spatial = bool(spatial) and calib.spatial_kind not in ("none", "cnf_deferred")
    n_sp = int(spatial_grid)
    summary.update({
        "family": calib.family, "spatial_kind": calib.spatial_kind, "spatial_grid_n": n_sp if do_spatial else None,
        "spatial_box_model_input_coords": list(calib.box),
        "tau_internal_scale": getattr(calib, "tau_internal_scale", 1.0),
        "spatial_status": "computed" if do_spatial else (
            "deferred: CNF decoder (density needs one flow ODE solve over the full window per query point)"
            if calib.spatial_kind == "cnf_deferred" else "skipped"
        ),
    })
    if hasattr(calib, "solver_info"):
        summary["temporal_solver"] = calib.solver_info
        if neural and solver_factor != 1:
            s_ = calib.tc.ode_solver
            summary["temporal_solver"] = {"method": str(s_.method), "atol": float(s_.atol), "rtol": float(s_.rtol)}

    # ---------------------------------------------------------------- scope policy, subsets, mask
    n_total = len(dataset)
    n_targets = min(n_total, limit) if limit else n_total
    predictive_subset = select_indices(len(dataset), protocol["predictive_targets"], protocol["predictive_target_seed"])
    pst = (
        json.loads((job_dir / "predictive_status.json").read_text())
        if (job_dir / "predictive_status.json").exists() else None
    )
    summary["predictive_subset_matches_campaign_file"] = (
        pst is not None
        and [int(i) for i in pst.get("binding", {}).get("selected_ids", [])] == [int(i) for i in predictive_subset]
    )
    if do_crps:
        scope = crps_scope
        if scope == "auto":
            scope = "all" if (n_total <= 50000 and preset != "auto_stpp") else "subset"
        crps_mask = np.ones(n_targets, bool)
        if scope == "subset":
            rng_sub = np.random.default_rng(crps_subset_seed)
            rand = rng_sub.choice(n_targets, min(crps_subset_size, n_targets), replace=False)
            crps_mask = np.zeros(n_targets, bool)
            crps_mask[rand] = True
            crps_mask[[i for i in predictive_subset if i < n_targets]] = True
        grid_all = (scope == "all") or neural  # the neural grid is one ODE solve per batch
        quantile_method = "grid" if grid_all else "bisect"
        summary["crps_scope"] = {
            "policy": scope, "n_crps_targets": int(crps_mask.sum()),
            "subset_seed": crps_subset_seed if scope == "subset" else None,
            "subset_size_requested": crps_subset_size if scope == "subset" else None,
            "includes_predictive_targets": scope == "subset",
            "grid_computed_for_all_targets": bool(grid_all), "quantile_method": quantile_method,
            "rule": "all targets when N <= 50k and preset != auto_stpp; otherwise the campaign's predictive targets plus a fixed-seed random subset",
        }
    else:
        crps_mask = np.zeros(n_targets, bool)
        grid_all = False
        quantile_method = None
    mask_excluded = None
    if mask_npz is not None and Path(mask_npz).exists():
        mz = np.load(mask_npz)
        mask_excluded = {
            "seq": mz["sequence_index"], "tgt": mz["target_event_index"], "excluded": mz["excluded"].astype(bool),
            "path": str(mask_npz), "sha256": sha256_file(mask_npz),
        }

    # ---------------------------------------------------------------- arrays + resume
    keys = (
        "nll_campaign_path", "d_lambda", "log_lam_T_model", "y_raw", "crps_raw", "surv_at_c", "lambda_grid_at_y",
        "q10_raw", "q50_raw", "q90_raw", "hpd_pit", "spatial_mass_in_box",
    )
    arrays: dict[str, np.ndarray] = {k: np.full(n_targets, np.nan) for k in keys}
    arrays["sequence_index"] = np.full(n_targets, -1, np.int64)
    arrays["target_event_index"] = np.full(n_targets, -1, np.int64)
    binding_sha = hashlib.sha256(json.dumps({
        **binding, "c_raw": c_raw, "tau_grid_n": tau_grid_n, "tau_factor": tau_factor,
        "spatial_grid": n_sp if do_spatial else None, "limit": n_targets, "calibration_version": CALIBRATION_VERSION,
        "crps_scope": summary.get("crps_scope"), "solver_factor": solver_factor, "skip_eval_forward": skip_eval_forward,
    }, sort_keys=True, default=str).encode()).hexdigest()
    prog_path = out / "progress.npz"
    position = 0
    timing = {"eval_forward": 0.0, "terms_pit": 0.0, "grid": 0.0, "quantiles": 0.0, "crps_numpy": 0.0, "spatial": 0.0}
    checks: list[dict] = []
    errors: list[dict] = []
    if prog_path.exists():
        pz = np.load(prog_path, allow_pickle=True)
        if str(pz["binding_sha"]) == binding_sha:
            position = int(pz["position"])
            for k in arrays:
                arrays[k][:position] = pz[k][:position]
            timing = json.loads(str(pz["timing"]))
            checks = json.loads(str(pz["checks"]))
            print(f"[resume] {position}/{n_targets} targets from {prog_path}", flush=True)
        else:
            print("[resume] progress file bound to a different run; starting over", flush=True)
    summary["resumed_from_target"] = position

    def commit(pos: int) -> None:
        tmp = prog_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, position=pos, binding_sha=binding_sha, timing=json.dumps(timing), checks=json.dumps(checks[-200:]),
            **arrays,
        )
        tmp.replace(prog_path)

    def batches():
        if statistical:
            for pos in range(position, n_targets, batch_size):
                yield pos, min(batch_size, n_targets - pos), list(range(pos, min(n_targets, pos + batch_size)))
        else:
            loader = DataLoader(Subset(dataset, range(position, n_targets)), batch_size=batch_size, shuffle=False,
                                collate_fn=collate)
            pos = position
            for batch in loader:
                size = len(batch["lengths"])
                yield pos, size, batch
                pos += size

    from seahorse.evaluation.likelihood import _reporting_correction
    from seahorse.runner.results import resolve_loss_result_reporting

    log_raw_vol = float(coord.log_raw_volume_per_normalized_volume)
    n_batches, t_loop, t_commit = 0, time.monotonic(), time.monotonic()
    oom_retries = 0
    for pos, size, batch in batches():
        for attempt in range(61):
            sl = slice(pos, pos + size)
            rng = np.random.default_rng(seed + pos)
            rows_crps = np.flatnonzero(crps_mask[sl]) if do_crps else np.zeros(0, int)
            lam_grid = None
            rows_grid = np.zeros(0, int)
            try:
                if statistical:
                    ctx = calib.contexts(batch)
                    arrays["sequence_index"][sl] = [m["target_source_sequence_index"] for m in ctx["meta"]]
                    arrays["target_event_index"][sl] = [m["target_global_event_index"] for m in ctx["meta"]]
                    t0 = time.monotonic()
                    tm = calib.terms(ctx, None)
                    timing["terms_pit"] += time.monotonic() - t0
                    arrays["nll_campaign_path"][sl] = tm["nll"]
                    arrays["d_lambda"][sl] = tm["d_lambda"]
                    arrays["log_lam_T_model"][sl] = tm.get("log_lam_T", np.full(size, np.nan))
                    arrays["y_raw"][sl] = ctx["next_time"] - ctx["origin"]
                    checks.append({k: v for k, v in tm.items() if k.startswith("check_")})
                    if do_spatial:
                        t0 = time.monotonic()
                        sp = calib.spatial_cells(ctx, n_sp)
                        if sp is not None:
                            probs, cell, mass, _desc = sp
                            arrays["hpd_pit"][sl] = hpd_pit_from_cells(probs, cell, rng)
                            arrays["spatial_mass_in_box"][sl] = mass
                        else:
                            summary["spatial_status"] = "not informative: single spatial cell (HPD-PIT = V by construction)"
                        timing["spatial"] += time.monotonic() - t0
                    if do_crps:
                        t0 = time.monotonic()
                        if grid_all:
                            lam_grid = calib.terms(ctx, tau_raw)["lam_grid"]
                            rows_grid = np.arange(size)
                        else:
                            q = bisect_quantiles(lambda t: calib.cumhaz_rows_np(ctx, t), size, float(tau_raw[-1]))
                            arrays["q10_raw"][sl], arrays["q50_raw"][sl], arrays["q90_raw"][sl] = q[0.1], q[0.5], q[0.9]
                            timing["quantiles"] += time.monotonic() - t0
                            t0 = time.monotonic()
                            if len(rows_crps):
                                sub = {k: (v[rows_crps] if isinstance(v, np.ndarray) else v)
                                       for k, v in ctx.items() if k != "meta"}
                                lam_grid = calib.terms(sub, tau_raw)["lam_grid"]
                                rows_grid = rows_crps
                        timing["grid"] += time.monotonic() - t0
                else:
                    md = batch["observation_metadata"]
                    arrays["sequence_index"][sl] = [m["target_source_sequence_index"] for m in md]
                    arrays["target_event_index"][sl] = [m["target_global_event_index"] for m in md]
                    kw = {k: batch[k].to(device) for k in ("times", "locations", "lengths")}
                    output = None
                    if not (neural and skip_eval_forward):
                        _sync(device)
                        t0 = time.monotonic()
                        cuda_devices = [torch.cuda.current_device()] if str(device).startswith("cuda") else []
                        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                            torch.manual_seed(seed + pos)
                            output = runner.model.eval_forward(**kw, target_scope="last")
                            result = runner.model.compute_loss(output)
                        reported, _, _, _, space = resolve_loss_result_reporting(result, requested_space="raw")
                        correction = _reporting_correction(result, reported, space)
                        state_transform = getattr(runner.model.state_model, "_input_transform_spec", {})
                        if space != "raw" and state_transform and state_transform.get("type") != "identity":
                            raise ValueError("Missing family coordinate correction for raw-space comparison.")
                        arrays["nll_campaign_path"][sl] = (
                            output["nll_per_event"].detach().cpu().numpy().reshape(-1) + correction + log_raw_vol
                        )
                        _sync(device)
                        timing["eval_forward"] += time.monotonic() - t0
                    t0 = time.monotonic()
                    tm = calib.terms(kw["times"], kw["locations"], kw["lengths"],
                                     tau_model if (do_crps and grid_all) else None, output=output)
                    _sync(device)
                    timing["terms_pit"] += time.monotonic() - t0
                    arrays["d_lambda"][sl] = to64(tm["d_lambda"])
                    arrays["log_lam_T_model"][sl] = to64(tm["log_lam_T"])
                    checks.append({k: v for k, v in tm.items() if k.startswith("check_")})
                    idx = torch.arange(size, device=device)
                    arrays["y_raw"][sl] = to64(
                        kw["times"][idx, kw["lengths"] - 1] - kw["times"][idx, kw["lengths"] - 2]
                    ) * time_scale
                    if do_spatial:
                        t0 = time.monotonic()
                        sp = calib.spatial_cells(kw["times"], kw["locations"], kw["lengths"], n_sp)
                        if sp is not None:
                            probs, cell, mass = sp
                            arrays["hpd_pit"][sl] = hpd_pit_from_cells(probs, cell, rng)
                            arrays["spatial_mass_in_box"][sl] = mass
                        _sync(device)
                        timing["spatial"] += time.monotonic() - t0
                    if do_crps:
                        if grid_all:
                            lam_grid = to64(tm["lam_grid"])
                            rows_grid = np.arange(size)
                        else:
                            t0 = time.monotonic()
                            q = bisect_quantiles(calib.cumhaz_rows, size, tau_max_model, xp=torch,
                                                 ref=tm["d_lambda"].float())
                            for p_ in q:
                                q[p_] = q[p_] * time_scale  # model-input units -> raw
                            arrays["q10_raw"][sl], arrays["q50_raw"][sl], arrays["q90_raw"][sl] = q[0.1], q[0.5], q[0.9]
                            _sync(device)
                            timing["quantiles"] += time.monotonic() - t0
                            if len(rows_crps):
                                t0 = time.monotonic()
                                r = torch.as_tensor(rows_crps, device=device)
                                lam_grid = to64(calib.terms(kw["times"][r], kw["locations"][r], kw["lengths"][r],
                                                            tau_model)["lam_grid"])
                                rows_grid = rows_crps
                                _sync(device)
                                timing["grid"] += time.monotonic() - t0
                if do_crps and lam_grid is not None:
                    t0 = time.monotonic()
                    lg = np.maximum.accumulate(np.asarray(lam_grid, dtype=np.float64), axis=1)
                    cdf = -np.expm1(-lg)
                    g_abs = pos + rows_grid
                    y = arrays["y_raw"][g_abs]
                    arrays["crps_raw"][g_abs] = crps_from_cdf_grid(cdf, tau_raw, y, c_raw)
                    arrays["lambda_grid_at_y"][g_abs] = [np.interp(y[i], tau_raw, lg[i]) for i in range(len(g_abs))]
                    arrays["surv_at_c"][g_abs] = np.exp(
                        -np.array([np.interp(c_raw, tau_raw, lg[i]) for i in range(len(g_abs))])
                    )
                    if quantile_method == "grid":
                        q = quantiles_from_cumhazard(lg, tau_raw)
                        arrays["q10_raw"][g_abs], arrays["q50_raw"][g_abs], arrays["q90_raw"][g_abs] = q[0.1], q[0.5], q[0.9]
                    timing["crps_numpy"] += time.monotonic() - t0
            except torch.cuda.OutOfMemoryError:
                oom_retries += 1
                if attempt >= 60:
                    commit(pos)
                    print(f"[oom] giving up after {attempt} waits at target {pos}; progress committed", flush=True)
                    sys.exit(3)
                torch.cuda.empty_cache()
                print(f"[oom] batch at {pos}: waiting 60 s (attempt {attempt + 1})", flush=True)
                time.sleep(60)
                continue
            except MemoryError:
                commit(pos)
                raise
            except Exception as exc:
                errors.append({"start": pos, "count": size, "error": str(exc), "type": type(exc).__name__,
                               "tb": traceback.format_exc()[-3000:]})
                if len(errors) > 5:
                    commit(pos)
                    raise
            break
        n_batches += 1
        if n_batches % commit_every == 0 or time.monotonic() - t_commit > 300:
            commit(pos + size)
            t_commit = time.monotonic()
        if n_batches % 50 == 0 or pos + size >= n_targets:
            print(f"[{pos + size}/{n_targets}] t={time.monotonic() - t_loop:.0f}s "
                  + " ".join(f"{k}={v:.1f}s" for k, v in timing.items()), flush=True)
    commit(n_targets)
    summary["oom_retries"] = oom_retries

    # ---------------------------------------------------------------- scores
    arrays["pit"] = pit_from_compensator(arrays["d_lambda"])
    if stored is not None and np.isfinite(arrays["nll_campaign_path"]).any():
        n_cmp = min(n_targets, stored["nll"].shape[0])
        diff = np.abs(arrays["nll_campaign_path"][:n_cmp] - stored["nll"][:n_cmp])
        summary["campaign_nll_reproduction"] = {
            "n": int(n_cmp),
            "targets_aligned": bool(
                np.array_equal(stored["sequence_index"][:n_cmp], arrays["sequence_index"][:n_cmp])
                and np.array_equal(stored["target_event_index"][:n_cmp], arrays["target_event_index"][:n_cmp])
            ),
            "max_abs_diff": float(np.nanmax(diff)), "mean_abs_diff": float(np.nanmean(diff)),
            "n_nonfinite_here": int((~np.isfinite(arrays["nll_campaign_path"][:n_cmp])).sum()),
            "mean_nll_here": float(np.nanmean(arrays["nll_campaign_path"][:n_cmp])),
            "mean_nll_stored": float(np.nanmean(stored["nll"][:n_cmp])),
        }
    elif neural and skip_eval_forward:
        summary["campaign_nll_reproduction"] = {
            "skipped": "skip_eval_forward (ODE cost); PIT uses the same integrate_lambda the state model runs"
        }
    hours = hours_per_unit
    sel_all = np.ones(n_targets, bool)
    summary["all_targets"] = metrics_block(arrays, sel_all, c_raw, hours, crps_sel=crps_mask if do_crps else None)
    for k in ("pit", "crps", "point_forecast", "spatial"):
        if k in summary["all_targets"]:
            summary[k] = summary["all_targets"][k]
    if do_crps and grid_all and (summary.get("crps_scope", {}).get("policy") == "subset"):
        summary["crps_all_targets_extra"] = metrics_block(arrays, sel_all, c_raw, hours, crps_sel=None).get("crps")
    if do_crps:
        rel = np.abs(arrays["lambda_grid_at_y"] - arrays["d_lambda"]) / np.maximum(np.abs(arrays["d_lambda"]), 1e-6)
        fg = np.isfinite(arrays["lambda_grid_at_y"])
        summary["check_grid_vs_nll_compensator"] = {
            "n": int(fg.sum()),
            "max_abs": float(np.nanmax(np.abs(arrays["lambda_grid_at_y"] - arrays["d_lambda"])[fg])) if fg.any() else None,
            "median_rel": float(np.nanmedian(rel[fg])) if fg.any() else None,
            "max_rel": float(np.nanmax(rel[fg])) if fg.any() else None,
        }
    if mask_excluded is not None:
        key = {(int(s_), int(t_)): bool(e_)
               for s_, t_, e_ in zip(mask_excluded["seq"], mask_excluded["tgt"], mask_excluded["excluded"])}
        exc = np.array([key.get((int(s_), int(t_)), False)
                        for s_, t_ in zip(arrays["sequence_index"], arrays["target_event_index"])])
        matched = np.array([(int(s_), int(t_)) in key
                            for s_, t_ in zip(arrays["sequence_index"], arrays["target_event_index"])])
        retained = matched & ~exc
        summary["masked"] = {
            "mask_path": mask_excluded["path"], "mask_sha256": mask_excluded["sha256"],
            "n_excluded_in_scored": int(exc.sum()), "n_unmatched_targets": int((~matched).sum()),
            "n_retained": int(retained.sum()),
            "retained_targets": metrics_block(arrays, retained, c_raw, hours, crps_sel=crps_mask if do_crps else None),
        }
        arrays["mask_excluded"] = exc
    summary["checks_max"] = {
        k: float(np.nanmax([c_[k] for c_ in checks if k in c_])) for k in {k for c_ in checks for k in c_}
    }
    summary["timing"] = {
        "wall_seconds_loop": time.monotonic() - t_loop, "n_targets": n_targets, "stages_seconds": timing,
        "seconds_per_1000_targets": {k: 1000.0 * v / max(n_targets, 1) for k, v in timing.items()},
    }
    summary["n_targets_total"] = n_total
    summary["n_targets_scored"] = n_targets
    summary["errors"] = errors
    summary["predictive_subset_ids_sha256"] = hashlib.sha256(json.dumps(predictive_subset).encode()).hexdigest()

    # ---------------------------------------------------------------- closed-form checks (statistical)
    if statistical and closed_form_check and do_crps:
        summary["closed_form_check"] = _statistical_closed_form_check(calib, arrays, c_raw, tau_raw, tau_factor, tau_grid_n)

    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary["total_seconds"] = time.monotonic() - t_start
    np.savez_compressed(
        out / "per_target.npz", **arrays, crps_subset_mask=crps_mask,
        tau_grid_raw=tau_raw if do_crps else np.zeros(0), predictive_subset_ids=np.asarray(predictive_subset),
    )
    (out / "summary.json").write_text(json.dumps(clean_json(summary), indent=1))
    if prog_path.exists():
        prog_path.unlink()
    return summary


def _statistical_closed_form_check(calib, arrays, c_raw, tau_raw, tau_factor, tau_grid_n) -> dict:
    """Compare grid CRPS/PIT of the statistical presets with analytic or quadrature references."""
    cf: dict[str, Any] = {}
    rows_cf = np.flatnonzero(np.isfinite(arrays["crps_raw"]))
    if calib.kind.startswith("Poisson") and calib.model.period is None and len(rows_cf):
        r = float(calib.model.rates[0])
        ref = np.array([crps_exponential_truncated(r, arrays["y_raw"][i], c_raw) for i in rows_cf])
        d = np.abs(ref - arrays["crps_raw"][rows_cf])
        fine = tau_grid_quadratic(tau_factor * c_raw, 16 * tau_grid_n)
        sub = rows_cf[:: max(1, len(rows_cf) // 256)]
        cdf_fine = -np.expm1(-r * fine[None, :]).repeat(len(sub), 0)
        dfine = np.abs(
            crps_from_cdf_grid(cdf_fine, fine, arrays["y_raw"][sub], c_raw)
            - np.array([crps_exponential_truncated(r, arrays["y_raw"][i], c_raw) for i in sub])
        )
        cf["exponential_closed_form"] = {
            "rate": r, "n": int(len(rows_cf)), "max_abs_diff_grid": float(d.max()), "mean_abs_diff_grid": float(d.mean()),
            "max_rel_diff_grid": float(np.max(d / ref)), f"max_abs_diff_grid{16 * tau_grid_n}": float(dfine.max()),
            "pit_closed_form_max_abs_diff": float(np.max(np.abs(arrays["pit"] - (-np.expm1(-r * arrays["y_raw"]))))),
            "note": "grid-vs-closed-form differences are the piecewise-linear CDF interpolation error; they shrink ~1/n^2 with the grid",
        }
    if calib.kind.startswith("Hawkes") and len(rows_cf):
        from scipy.integrate import quad

        sub = rows_cf[:: max(1, len(rows_cf) // 64)]
        ctx = calib.contexts(sub)
        mu, alpha, beta, _sigma, _masses, cm = calib.hawkes_pieces(ctx)
        area = calib.core["area"](calib.support)
        ref = []
        for i in range(len(sub)):
            a = ctx["origin"][i] - ctx["times"][i]
            w = cm[i] * alpha / beta

            def cumhaz(t, a=a, w=w):
                return mu * area * t + np.sum(w * (np.exp(-beta * a) - np.exp(-beta * (a + t))))

            yi = min(arrays["y_raw"][sub[i]], c_raw)
            ref.append(
                quad(lambda t: (-np.expm1(-cumhaz(t))) ** 2, 0, yi, limit=200)[0]
                + quad(lambda t: np.exp(-2 * cumhaz(t)), yi, c_raw, limit=200)[0]
            )
        ref = np.array(ref)
        d = np.abs(ref - arrays["crps_raw"][sub])
        cf["hawkes_quadrature_reference"] = {
            "n": int(len(sub)), "max_abs_diff": float(d.max()), "max_rel_diff": float(np.max(d / ref)),
            "note": "scipy.integrate.quad of the closed-form CDF vs the grid integral",
        }
    return cf


__all__ = [
    "CALIBRATION_VERSION",
    "DEFINITIONS",
    "bisect_quantiles",
    "cell_index",
    "evaluate_calibration",
    "grid_centres",
    "hpd_pit_from_cells",
    "ks_uniform_with_critical",
    "metrics_block",
    "quantiles_from_cumhazard",
    "round_up",
    "sha256_file",
    "to64",
]
