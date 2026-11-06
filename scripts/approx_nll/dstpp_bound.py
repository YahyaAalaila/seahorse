#!/usr/bin/env python3
"""DSTPP full variational bound per test target, in raw units.

For every target of a finished ``diffusion_stpp`` campaign job, evaluates
``GaussianDiffusionST.NLL_cal`` (all T steps: Gaussian decoder at t = 0, KL terms, prior KL),
sums the per-dimension means over the three token dimensions and adds the Jacobians of the
token normalisation and of the campaign coordinate transform:

    NLL_raw = D * NLL_y,mean - sum_d log(2 / R_d) + log(time_scale * scale_lon * scale_lat)

with y = 2u - 1, u = (x - min_d) / R_d the train-split min/max token (D = 3). The bound is an
upper bound on the model's NLL, not a density value. The target token is rebuilt without the
state model's clamp to [0, 1] (the decoder density lives on R^3); the number of targets the
clamp would have moved is reported as ``n_clamped_targets``.

Writes ``<out>/target_nll.npz`` (``nll``, ``nll_temporal_raw``, ``nll_spatial_raw``,
``nll_token_y``, ``sequence_index``, ``target_event_index``) and ``<out>/summary.json``.
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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _campaign_io as io  # noqa: E402


def target_tokens(sm, times, locations, lengths):
    """Unclamped [0, 1] token of the last event of each window."""
    idx = torch.arange(len(lengths), device=times.device)
    dt = times[idx, lengths - 1] - times[idx, lengths - 2]
    loc = locations[idx, lengths - 1]
    u_t = (dt - sm.token_delta_t_min) / sm.token_delta_t_range.clamp(min=1e-8) if sm.minmax_normalize_time else dt
    u_s = (loc - sm.token_loc_min) / sm.token_loc_range.clamp(min=1e-8) if sm.minmax_normalize_loc else loc
    return torch.cat([u_t[:, None], u_s], -1)[:, None, :]


def log_dy_dx(sm) -> tuple[float, float]:
    """(temporal, total) log|dy/dx| of y = 2u - 1 with respect to the campaign coordinate."""
    lt = math.log(2.0) - (math.log(float(sm.token_delta_t_range.clamp(min=1e-8))) if sm.minmax_normalize_time else 0.0)
    total = lt
    for r in sm.token_loc_range.clamp(min=1e-8).tolist():
        total += math.log(2.0) - (math.log(r) if sm.minmax_normalize_loc else 0.0)
    return lt, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io.add_common_args(ap)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--noise-seed", type=int, default=20261105,
                    help="seed of the forward-noise draws (one draw per step and target)")
    ap.add_argument("--max-targets", type=int, default=0, help="score only the first N targets (0 = all)")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    job = io.load_job(args, model_kind="diffusion_stpp")
    sm, diffusion = job.model.state_model, job.model.event_model.diffusion
    N = len(job.dataset) if not args.max_targets else min(args.max_targets, len(job.dataset))
    loader = DataLoader(Subset(job.dataset, range(N)), batch_size=args.batch_size, shuffle=False,
                        collate_fn=job.collate)
    lj_t, lj = log_dy_dx(sm)
    lrv = job.coord.log_raw_volume_per_normalized_volume
    log_time_scale = math.log(job.coord.time_scale)
    log_space_scale = math.log(job.coord.space_scale[0] * job.coord.space_scale[1])
    D = diffusion.seq_length
    cols = {k: np.empty(N) for k in ("nll", "nll_temporal_raw", "nll_spatial_raw", "nll_token_y")}
    seq_ids, ev_ids = np.empty(N, np.int64), np.empty(N, np.int64)
    n_clamped, pos = 0, 0
    for batch in loader:
        size = len(batch["lengths"])
        s_ids, e_ids = io.target_ids(batch)
        seq_ids[pos:pos + size], ev_ids[pos:pos + size] = s_ids, e_ids
        torch.manual_seed(args.noise_seed * 1000003 + pos)
        with torch.no_grad():
            state, kw = io.last_event_state(job.model, batch, args.device)
            img = target_tokens(sm, **kw)
            n_clamped += int((state.payload["diff_img"] != img).any(-1).sum())
            tot, tem, spa = diffusion.NLL_cal(img, state.payload["diff_cond"], clip_denoised=True)
        tot, tem, spa = (v.double().cpu().numpy() for v in (tot * D, tem, spa * (D - 1)))
        sl = slice(pos, pos + size)
        cols["nll_token_y"][sl] = tot
        cols["nll"][sl] = tot - lj + lrv
        cols["nll_temporal_raw"][sl] = tem - lj_t + log_time_scale
        cols["nll_spatial_raw"][sl] = spa - (lj - lj_t) + log_space_scale
        pos += size
    np.savez_compressed(out / "target_nll.npz", sequence_index=seq_ids, target_event_index=ev_ids, **cols)
    summary = {**io.provenance(job), "n": N, "population": len(job.dataset),
               "means": {k: float(v.mean()) for k, v in cols.items()},
               "median_nll": float(np.median(cols["nll"])), "n_clamped_targets": n_clamped,
               "log_dy_dx_sum": lj, "noise_seed": args.noise_seed, "batch_size": args.batch_size,
               "alignment_with_ref": io.check_alignment(args.ref, seq_ids, ev_ids),
               "nll_kind": "upper bound (DSTPP variational bound), raw units",
               "torch": torch.__version__, "elapsed_s": time.time() - t0}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["means"]))


if __name__ == "__main__":
    main()
