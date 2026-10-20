"""Sample-only presets (SMASH, DSTPP): native next-event samplers.

These presets have no tractable intensity; every predictive metric is computed
from K i.i.d. draws of the next event from the preset's own sampler
(``method = native_samples``). Differences to the paper-era path:

* waiting times are **not** clamped: a draw with ``dt <= 0`` (possible for
  DSTPP, whose diffusion decoder outputs an unconstrained min-max-scaled
  ``dt``) is counted as an *invalid sample* and excluded, and the invalid
  fraction is reported (paper-era code silently replaced it with 0);
* K defaults to 200 (paper-era: 32).

SMASH is sampled with its own annealed-Langevin sampler as implemented in
Seahorse (``sampling_timesteps`` steps, step size ``langevin_step``, joint 3-D
chain, Tweedie denoising at the last step). This differs from the SMASH
paper's procedure (1000-2000 steps, time then location); see METRICS_SPEC.md.
"""

from __future__ import annotations

import numpy as np
import torch

from seahorse.models.abstractions import StateContext
from seahorse.evaluation.predictive.rollout import (
    _denormalize_spatial_samples,
    _denormalize_time_samples,
    _flatten_sample_tensor,
    normalize_history_for_runner,
)

from .base import FamilyAdapter, Prepared, chunked


class NativeAdapter(FamilyAdapter):
    family = "native_sampler"
    nll_kind = "framework_approx"
    spatial_domain = "n/a"
    spatial_sampler = "native"
    has_intensity = False
    min_history = 1

    def __init__(self, runner, device):
        super().__init__(runner, device)
        if self.preset not in ("smash", "diffusion_stpp"):
            raise NotImplementedError(self.preset)

    def prepare(self, times, locs) -> Prepared:
        times = np.asarray(times, float)
        locs = np.asarray(locs, float)
        if times.ndim == 1:
            times, locs = times[None], locs[None]
        return Prepared(times=times, locs=locs)

    @torch.no_grad()
    def sample_next(self, prep, b_idx, k_idx, K: int, *, chunk_samples: int = 40000):
        """K draws of (dt_raw, loc_raw) for each context; returns (R,K), (R,K,2)."""
        R = len(b_idx)
        dt = np.empty((R, K))
        loc = np.empty((R, K, 2))
        per = max(1, chunk_samples // K)
        for sl in chunked(R, per):
            bb, kk = b_idx[sl], k_idx[sl]
            r = len(bb)
            L = int(kk.max()) + 1
            pt = np.zeros((r, L), np.float32)
            pl = np.zeros((r, L, 2), np.float32)
            for i in range(r):
                tn, ln = normalize_history_for_runner(
                    {"times": prep.times[bb[i], : kk[i] + 1], "locations": prep.locs[bb[i], : kk[i] + 1]},
                    self.runner.norm_stats,
                )
                pt[i, : kk[i] + 1] = tn
                pl[i, : kk[i] + 1] = ln
            st = self.model.state_model.encode_sampling_history(
                times=torch.as_tensor(pt, device=self.device),
                locations=torch.as_tensor(pl, device=self.device),
                lengths=torch.as_tensor(kk + 1, device=self.device),
            )
            em = self.model.event_model
            if self.preset == "smash":
                cond = st.payload["smash_cond_last"].repeat_interleave(K, dim=0)
                with torch.enable_grad():
                    out = em.sample_native(
                        state=StateContext(payload={"smash_cond_last": cond}),
                        step=int(em.score_matching.sampling_timesteps),
                        is_last=True,
                        n_samples=1,
                        batch_size=r * K,
                        device=self.device,
                    )
            else:
                cond = st.payload["diff_cond_last"].repeat_interleave(K, dim=0)
                out = em.sample_native(
                    state=StateContext(payload={"diff_cond_last": cond}),
                    batch_size=r * K,
                    device=self.device,
                )
            flat = _flatten_sample_tensor(out["samples"])[: r * K].detach().cpu().numpy()
            dt[sl] = _denormalize_time_samples(self.runner, flat).reshape(r, K)
            loc[sl] = _denormalize_spatial_samples(self.runner, flat).reshape(r, K, 2)
        return dt, loc
