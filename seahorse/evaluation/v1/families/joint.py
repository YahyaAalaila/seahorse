"""NSMPP (DeepBasis) adapter: direct joint intensity on a bounded support box.

lambda(t, s | H) = softplus(mu_raw + sum_{t_j < t} K((t, s), (t_j, s_j))) + eps,
with a finite-rank neural kernel. The NLL integrates lambda over the preset's
``support box`` S (the training bounding box) by quadrature; after the fix in
``29458c6`` the per-event compensator is ``_interval_compensators`` (composite
midpoint in time, ``int_res x int_res`` node grid in space). We use exactly that
for the time-rescaling PIT, and a fine grid for spatial densities.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import EventTerms, FamilyAdapter, Prepared, chunked, to_np
from .windowed import _grid_logpdf, _sample_grid


class NSMPPAdapter(FamilyAdapter):
    family = "nsmpp"
    nll_kind = "numerical_quadrature"
    spatial_domain = "support_box"
    spatial_sampler = "fine_grid"
    min_history = 1

    def __init__(self, runner, device, grid_n: int = 128):
        super().__init__(runner, device)
        self.em = self.model.event_model
        tr = self.model.state_model._input_transform
        if tr is not None and type(tr).__name__ != "IdentityTransformArtifact":
            raise NotImplementedError(f"NSMPP adapter assumes identity transform, got {type(tr).__name__}")
        self.require_identity_runner_norm()
        self.s_min = self.em.support_space_min.detach().cpu().double().numpy()
        self.s_max = self.em.support_space_max.detach().cpu().double().numpy()
        self.s_range = self.s_max - self.s_min
        self.log_area = float(np.log(self.s_range).sum())
        self.grid_n = int(grid_n)

    def _hat(self, s):
        return (np.asarray(s, float) - self.s_min) / self.s_range

    def _unhat(self, s):
        return np.asarray(s, float) * self.s_range + self.s_min

    @torch.no_grad()
    def prepare(self, times, locs) -> Prepared:
        times = np.asarray(times, float)
        locs = np.asarray(locs, float)
        if times.ndim == 1:
            times, locs = times[None], locs[None]
        ev = torch.cat([self.t(times)[..., None], self.t(locs)], -1)
        prep = Prepared(times=times, locs=locs)
        prep.cache["ev"] = ev
        prep.cache["valid"] = torch.ones(ev.shape[:2], dtype=torch.bool, device=self.device)
        return prep

    def _lam(self, prep, b_idx, q_t, q_s, counts):
        """lambda at queries; q_t (R,M), q_s (R,M,2), history = first counts[r] events."""
        ev = prep.cache["ev"][torch.as_tensor(b_idx, device=self.device)]
        valid = prep.cache["valid"][torch.as_tensor(b_idx, device=self.device)]
        q = torch.cat([q_t[..., None], q_s], -1)
        pc = counts[:, None].expand(q.shape[0], q.shape[1])
        ks = self.em.kernel.sum_over_prefix_history(q, pc, ev, valid)
        return F.softplus(ks + self.em.raw_mu) + self.em.intensity_eps

    @torch.no_grad()
    def event_terms(self, prep) -> EventTerms:
        ev, valid = prep.cache["ev"], prep.cache["valid"]
        T = prep.T
        lam = self.em._event_intensities(ev, valid)[0]
        dlam = self.em._interval_compensators(ev, valid)[0]
        # spatial conditional: lambda(t_i, s_i) / int_S lambda(t_i, s) ds on a fine grid
        k = np.arange(0, T - 1)
        lg = self.ground_intensity_fine(prep, np.zeros_like(k), k, prep.times[0, 1:])
        log_ps = np.full(T, np.nan)
        log_ps[1:] = to_np(torch.log(lam[1:])) - np.log(lg)
        defined = np.ones(T, bool)
        defined[0] = False
        return EventTerms(
            log_lambda=to_np(torch.log(lam)), d_lambda=to_np(dlam), log_ps=log_ps, defined=defined
        )

    def _cell_centres(self, n):
        c = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(c, c, indexing="ij")
        return self._unhat(np.stack([X.ravel(), Y.ravel()], -1))

    @torch.no_grad()
    def _ground_on_nodes(self, prep, b_idx, k_idx, t_query, nodes, weight):
        """sum_g lambda(t, g) * weight for given spatial nodes; t_query (R, Gt)."""
        R, Gt = t_query.shape
        Mn = nodes.shape[0]
        out = np.empty((R, Gt))
        nod = self.t(nodes)
        per = max(1, (1 << 22) // (Gt * Mn))
        for sl in chunked(R, per):
            r = sl.stop - sl.start
            qt = self.t(t_query[sl])[:, :, None].expand(r, Gt, Mn).reshape(r, Gt * Mn)
            qs = nod[None, None].expand(r, Gt, Mn, 2).reshape(r, Gt * Mn, 2)
            counts = torch.as_tensor(k_idx[sl] + 1, device=self.device)
            lam = self._lam(prep, b_idx[sl], qt, qs, counts).view(r, Gt, Mn)
            out[sl] = to_np(lam.sum(-1)) * weight
        return out

    def ground_intensity(self, prep, b_idx, k_idx, t_query):
        """NLL-consistent ground intensity (the model's int_res x int_res node rule)."""
        nodes = self.em.support_spatial_grid.detach().cpu().double().numpy()
        w = float(self.em.spatial_cell_area.item())
        return self._ground_on_nodes(prep, b_idx, k_idx, np.asarray(t_query, float)[:, None], nodes, w)[:, 0]

    def ground_intensity_fine(self, prep, b_idx, k_idx, t_query, n=64):
        nodes = self._cell_centres(n)
        w = float(np.exp(self.log_area)) / (n * n)
        return self._ground_on_nodes(prep, b_idx, k_idx, np.asarray(t_query, float)[:, None], nodes, w)[:, 0]

    @torch.no_grad()
    def cum_hazard(self, prep, b_idx, k_idx, tau_grid) -> np.ndarray:
        """Cumulative trapezoid in time of the NLL's spatial node rule."""
        tg = np.asarray(tau_grid, float)
        tk = prep.times[b_idx, k_idx]
        tq = tk[:, None] + tg[None, :]
        nodes = self.em.support_spatial_grid.detach().cpu().double().numpy()
        w = float(self.em.spatial_cell_area.item())
        lg = self._ground_on_nodes(prep, b_idx, k_idx, tq, nodes, w)
        dt = np.diff(tg)
        cum = np.concatenate([np.zeros((lg.shape[0], 1)), np.cumsum(0.5 * (lg[:, 1:] + lg[:, :-1]) * dt, 1)], 1)
        return cum

    @torch.no_grad()
    def intensity(self, prep, b_idx, k_idx, t_query, points):
        pts = np.asarray(points, float)
        R, M, _ = pts.shape
        out = np.empty((R, M))
        for sl in chunked(R, max(1, (1 << 20) // M)):
            r = sl.stop - sl.start
            qt = self.t(np.asarray(t_query, float)[sl])[:, None].expand(r, M)
            lam = self._lam(prep, b_idx[sl], qt, self.t(pts[sl]), torch.as_tensor(k_idx[sl] + 1, device=self.device))
            v = to_np(lam)
            inside = np.all((self._hat(pts[sl]) >= 0) & (self._hat(pts[sl]) <= 1), -1)
            v[~inside] = 0.0
            out[sl] = v
        return out

    def spatial_logpdf(self, prep, b_idx, k_idx, t_query, points):
        lam = self.intensity(prep, b_idx, k_idx, t_query, points)
        lg = self.ground_intensity_fine(prep, b_idx, k_idx, t_query)
        return np.log(np.maximum(lam, 1e-300)) - np.log(lg)[:, None]

    def _grid_probs(self, prep, b_idx, k_idx, t_query, n):
        cc = self._cell_centres(n)
        R = len(b_idx)
        lam = self.intensity(prep, b_idx, k_idx, t_query, np.broadcast_to(cc, (R,) + cc.shape))
        return lam / lam.sum(-1, keepdims=True)

    def sample_space(self, prep, b_idx, k_idx, t_query, rng, grid_n=None, **kw):
        n = int(grid_n or self.grid_n)
        probs = self._grid_probs(prep, b_idx, k_idx, t_query, n)
        return _sample_grid(probs, n, rng, self._unhat, self.log_area)

    def sampling_logpdf(self, prep, b_idx, k_idx, t_query, points, grid_n=None):
        n = int(grid_n or self.grid_n)
        probs = self._grid_probs(prep, b_idx, k_idx, t_query, n)
        return _grid_logpdf(probs, n, self._hat(points), self.log_area)

    def sample_space_k(self, prep, b_idx, k_idx, t_query, K, rng, grid_n=None):
        n = int(grid_n or self.grid_n)
        probs = self._grid_probs(prep, b_idx, k_idx, t_query, n)
        R = probs.shape[0]
        pr = np.repeat(probs, K, axis=0)
        x, lp = _sample_grid(pr, n, rng, self._unhat, self.log_area)
        return x.reshape(R, K, 2), lp.reshape(R, K)
