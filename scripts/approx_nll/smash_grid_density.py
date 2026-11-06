#!/usr/bin/env python3
"""SMASH: temporal NLL and a grid estimate of the spatial density of its own score model.

SMASH defines no likelihood. This script derives an approximate, framework-derived density
from a finished ``smash`` campaign job, per test target:

* Temporal: ``ScoreMatchingProcess.nll_terms`` (the network queried in its training
  coordinates y = 2u - 1, compensator from y = -1), per unit of the [0, 1] time token u_t,
  converted to raw units with

      NLL_t,raw = NLL_t,u + log R_t + log max(tau_c, 1e-4) [log-normalised time only] + log time_scale

  where tau_c is the inter-event time in campaign units. It is the NLL of a sub-density
  (an upper bound on the NLL of its normalised version).
* Spatial: the location score s(y_s | y_t, H) is evaluated on a G x G node grid over the token
  box [-1, 1]^2 with y_t fixed at the target's value; phi = argmin ||D phi / h - s_mid||^2
  (forward differences, scores at edge midpoints) solves the Neumann Poisson problem, exactly
  by DCT-II; log p = phi - log trapz(exp phi); bilinear interpolation at the true location;

      NLL_s,raw = NLL_s,y + sum_d log(R_d / 2) + log(scale_lon * scale_lat).

  ``resid_ratio`` = ||grad phi - s|| / ||s|| measures the non-conservative part of the score
  field. The density is restricted to the token box.

Joint = temporal + grid spatial. ``--n-sub N`` scores one uniform draw from each of N equal
strata of the target list (fixed ``--seed``); ``--n-sub 0`` scores every target.
``--no-grid`` skips the spatial step (temporal term on all selected targets only).

Writes ``<out>/target_nll.npz`` (``target_index``, ``sequence_index``, ``target_event_index``,
``nll_temporal_raw`` and, per grid size G, ``nll_spatial_grid_G<G>_raw``,
``nll_joint_grid_G<G>_raw``, ``resid_ratio_G<G>``) and ``<out>/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dctn, idctn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _campaign_io as io  # noqa: E402


def stratified(n_total: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, n_total, n + 1).astype(np.int64)
    return np.array([rng.integers(edges[i], max(edges[i + 1], edges[i] + 1)) for i in range(n)], np.int64)


def poisson_ls(gx: np.ndarray, gy: np.ndarray, h: float) -> tuple[np.ndarray, float]:
    """Least-squares potential of a 2-D vector field given at the nodes of a square grid.

    ``gx``, ``gy``: (G, G) field components, axis 0 = first coordinate. Returns phi (up to a
    constant) and the residual ratio ||D phi / h - g_mid|| / ||g_mid||.
    """
    G = gx.shape[0]
    gxm = 0.5 * (gx[1:, :] + gx[:-1, :])
    gym = 0.5 * (gy[:, 1:] + gy[:, :-1])
    b = np.zeros((G, G))
    b[:-1, :] -= gxm
    b[1:, :] += gxm
    b[:, :-1] -= gym
    b[:, 1:] += gym
    b /= h
    lam1 = (2 - 2 * np.cos(np.pi * np.arange(G) / G)) / h ** 2
    lam = lam1[:, None] + lam1[None, :]
    lam[0, 0] = 1.0
    ph = dctn(b, type=2, norm="ortho") / lam
    ph[0, 0] = 0.0
    phi = idctn(ph, type=2, norm="ortho")
    rx = (phi[1:, :] - phi[:-1, :]) / h - gxm
    ry = (phi[:, 1:] - phi[:, :-1]) / h - gym
    norm = max(math.sqrt((gxm ** 2).sum() + (gym ** 2).sum()), 1e-300)
    return phi, math.sqrt((rx ** 2).sum() + (ry ** 2).sum()) / norm


def trapz_log_normaliser(phi: np.ndarray, h: float) -> float:
    w = np.ones(phi.shape[0])
    w[0] = w[-1] = 0.5
    W = (w[:, None] * w[None, :]) * h * h
    m = phi.max()
    return m + math.log((W * np.exp(phi - m)).sum())


def bilinear(phi: np.ndarray, nodes: np.ndarray, y: np.ndarray) -> float:
    G, h = len(nodes), nodes[1] - nodes[0]
    fx = np.clip((y[0] - nodes[0]) / h, 0, G - 1 - 1e-9)
    fy = np.clip((y[1] - nodes[0]) / h, 0, G - 1 - 1e-9)
    i, j = int(fx), int(fy)
    a, b = fx - i, fy - j
    return ((1 - a) * (1 - b) * phi[i, j] + a * (1 - b) * phi[i + 1, j]
            + (1 - a) * b * phi[i, j + 1] + a * b * phi[i + 1, j + 1])


def grid_spatial_nll_y(sp, y_target: torch.Tensor, cond: torch.Tensor, G: int, chunk: int = 65536):
    """Grid NLL of the location in the y coordinates for one target (y_target: (3,))."""
    nodes = np.linspace(-1.0, 1.0, G)
    h = nodes[1] - nodes[0]
    X, Y = np.meshgrid(nodes, nodes, indexing="ij")
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), device=y_target.device, dtype=y_target.dtype)
    q = torch.cat([y_target[0].expand(len(pts), 1), pts], -1)[:, None, :]
    scores = []
    for c0 in range(0, len(pts), chunk):
        qc = q[c0:c0 + chunk]
        scores.append(sp.model.get_score_loc(qc, cond.expand(len(qc), -1, -1)).squeeze(1))
    s = torch.cat(scores).double().cpu().numpy()
    phi, resid = poisson_ls(s[:, 0].reshape(G, G), s[:, 1].reshape(G, G), h)
    ys = y_target[1:].double().cpu().numpy()
    nll = -(bilinear(phi, nodes, ys) - trapz_log_normaliser(phi, h))
    return nll, resid, bool((np.abs(ys) > 1).any())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io.add_common_args(ap)
    ap.add_argument("--n-sub", type=int, default=2000, help="stratified subsample size (0 = all targets)")
    ap.add_argument("--seed", type=int, default=20261105, help="seed of the stratified subsample")
    ap.add_argument("--grids", default="128", help="comma-separated grid sizes G")
    ap.add_argument("--no-grid", action="store_true", help="temporal term only")
    ap.add_argument("--n-quad", type=int, default=20, help="quadrature nodes of the compensator")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    job = io.load_job(args, model_kind="smash")
    sm, sp = job.model.state_model, job.model.event_model.score_matching
    if sm.num_types != 1:
        raise SystemExit("unmarked SMASH only")
    N = len(job.dataset)
    idx = np.arange(N) if args.n_sub <= 0 or args.n_sub >= N else stratified(N, args.n_sub, args.seed)
    grids = [] if args.no_grid else [int(g) for g in args.grids.split(",")]

    R_t = float(sm.token_time_range_log if sm.log_normalization else sm.token_time_range_raw) if sm.minmax_normalize_time else 1.0
    R_s = sm.token_loc_range.clamp(min=1e-8).tolist() if sm.minmax_normalize_loc else [1.0, 1.0]
    log_time_scale = math.log(job.coord.time_scale)
    sp_jac = sum(math.log(r / 2.0) for r in R_s) + math.log(job.coord.space_scale[0] * job.coord.space_scale[1])

    n = len(idx)
    temporal = np.empty(n)
    seq_ids, ev_ids = np.empty(n, np.int64), np.empty(n, np.int64)
    spatial = {G: np.empty(n) for G in grids}
    resid = {G: np.empty(n) for G in grids}
    outside = {G: 0 for G in grids}
    n_clamped, pos = 0, 0
    loader = DataLoader(Subset(job.dataset, idx.tolist()), batch_size=args.batch_size, shuffle=False,
                        collate_fn=job.collate)
    for batch in loader:
        size = len(batch["lengths"])
        s_ids, e_ids = io.target_ids(batch)
        seq_ids[pos:pos + size], ev_ids[pos:pos + size] = s_ids, e_ids
        with torch.no_grad():
            state, kw = io.last_event_state(job.model, batch, args.device)
            u, cond = state.payload["smash_img"], state.payload["smash_cond"]
            terms = sp.nll_terms(u, cond, n_quad=args.n_quad)
            L, rows = kw["lengths"], torch.arange(size, device=u.device)
            tau = (kw["times"][rows, L - 1] - kw["times"][rows, L - 2]).double().cpu().numpy()
            if sm.log_normalization:
                minus_log_g_prime = np.log(np.maximum(tau, 1e-4))
                n_clamped += int((tau < 1e-4).sum())
            else:
                minus_log_g_prime = np.zeros_like(tau)
            temporal[pos:pos + size] = (terms["temporal"].double().cpu().numpy() + math.log(R_t)
                                        + minus_log_g_prime + log_time_scale)
            y = u * 2 - 1
            for G in grids:
                for b in range(size):
                    nll_y, r, off = grid_spatial_nll_y(sp, y[b, 0], cond[b:b + 1], G)
                    spatial[G][pos + b] = nll_y + sp_jac
                    resid[G][pos + b] = r
                    outside[G] += int(off)
        pos += size
    assert pos == n

    def se(v):
        return float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")

    arrays = {"target_index": idx, "sequence_index": seq_ids, "target_event_index": ev_ids,
              "nll_temporal_raw": temporal}
    summary = {**io.provenance(job), "population": N, "n": n, "seed": args.seed, "n_quad": args.n_quad,
               "R_t": R_t, "R_s": R_s, "log_normalization": bool(sm.log_normalization),
               "spatial_jacobian_raw": sp_jac, "n_clamped_targets": n_clamped,
               "temporal_raw_mean": float(temporal.mean()), "temporal_raw_se": se(temporal),
               "grids": {}, "nll_kind": "approximate, framework-derived (SMASH defines no likelihood)",
               "alignment_with_ref": io.check_alignment(args.ref, seq_ids, ev_ids, positions=idx)}
    for G in grids:
        joint = spatial[G] + temporal
        arrays[f"nll_spatial_grid_G{G}_raw"] = spatial[G]
        arrays[f"nll_joint_grid_G{G}_raw"] = joint
        arrays[f"resid_ratio_G{G}"] = resid[G]
        summary["grids"][str(G)] = {"spatial_grid_raw_mean": float(spatial[G].mean()),
                                    "spatial_grid_raw_se": se(spatial[G]),
                                    "joint_raw_mean": float(joint.mean()), "joint_raw_se": se(joint),
                                    "resid_ratio_median": float(np.median(resid[G])),
                                    "n_true_loc_outside_grid": outside[G]}
    np.savez_compressed(out / "target_nll.npz", **arrays)
    summary["torch"], summary["elapsed_s"] = torch.__version__, time.time() - t0
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n", "temporal_raw_mean", "grids")}))


if __name__ == "__main__":
    main()
