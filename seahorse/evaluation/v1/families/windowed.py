"""AutoSTPP and DeepSTPP adapters (fixed 20-event 'paper window' presets).

Both presets define the conditional law of event ``i`` from the window of the
``lookback`` (= 20) events before it, in 'paper-scaled' coordinates

    s_hat = (s - loc_min) / loc_range,     dt_hat = (dt - dt_min) / dt_range.

Targets ``i < lookback`` are *undefined* under these models (no window); the
0.1.0 benchmark scored them as NLL = 0 (fixed in 1.0.0).

AutoSTPP (Zhou & Yu 2023): lambda(t, s) = mu + sum_{j in W} f(s - s_j, t - t_j)
with ``f`` the mixed derivative of a monotone ProdNet, so both the temporal
intensity ``int_[0,1]^2 lambda ds`` and the compensator are closed-form
(spatial domain S = unit square in scaled space = the training bounding box).

DeepSTPP (Zhou et al. 2022): lambda(t) = sum_i w_i exp(-b_i (t - t_i)) over
window events + latent background points, p(s | t) a Gaussian mixture with
weights ~ w_i exp(-b_i (t - t_i)); the spatial domain is R^2.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .base import EventTerms, FamilyAdapter, Prepared, chunked, to_np


class _WindowBase(FamilyAdapter):
    def __init__(self, runner, device):
        super().__init__(runner, device)
        sm = self.model.state_model
        self.sm = sm
        ns = runner.norm_stats or {}
        if bool(ns.get("normalize", False)):
            tm = float(ns.get("time_mean", 0.0)); ts = float(ns.get("time_std", 1.0))
            lm = np.asarray(ns.get("loc_mean", [0, 0]), float); ls = np.asarray(ns.get("loc_std", [1, 1]), float)
            if abs(tm) > 0 or abs(ts - 1) > 0 or np.any(lm != 0) or np.any(ls != 1):
                raise NotImplementedError("windowed adapters assume identity runner norm_stats")
        if bool(sm.input_normalized_flag.item() > 0.5):
            raise NotImplementedError("windowed adapters assume input_normalized=False")
        self.dt_min = float(sm.paper_dt_min.item())
        self.dt_range = float(sm.paper_dt_range.item())
        self.loc_min = sm.paper_loc_min.detach().cpu().double().numpy()
        self.loc_range = sm.paper_loc_range.detach().cpu().double().numpy()
        self.log_jac_s = float(np.log(self.loc_range).sum())
        self.log_jac_t = float(np.log(self.dt_range))

    def _scale_locs(self, s):
        return (np.asarray(s, float) - self.loc_min) / self.loc_range

    def _unscale_locs(self, s):
        return np.asarray(s, float) * self.loc_range + self.loc_min

    def _tau_scaled(self, tau_raw):
        return (np.asarray(tau_raw, float) - self.dt_min) / self.dt_range

    def _window_arrays(self, prep, b_idx, k_idx):
        """Scaled window locations (R, L, 2) and lags a_j = c_k - c_j (R, L).

        With ``self.full_history_diagnostic`` (AutoSTPP only, diagnostics), the
        window is replaced by *all* events 0..k (padded with zero-weight
        entries), which is what the legacy thinning sampler conditioned on via
        ``AutoSTPPEventModel.intensity`` (``trunc=False``).
        """
        if getattr(self, "full_history_diagnostic", False):
            return self._full_arrays(prep, b_idx, k_idx)
        L = self.lookback
        c = prep.cache
        hs = c["hist_scaled"]  # (B, T, 3) [x, y, dt_hat]
        cum = c["cum_scaled"]  # (B, T)
        b = torch.as_tensor(b_idx, device=self.device)
        k = torch.as_tensor(k_idx, device=self.device)
        offs = torch.arange(L - 1, -1, -1, device=self.device)  # L-1 .. 0
        j = k[:, None] - offs[None, :]  # (R, L) indices k-L+1 .. k
        if bool((j < 0).any()):
            raise ValueError("context has fewer than lookback events")
        s = hs[b[:, None], j, :2]
        a = cum[b, k][:, None] - cum[b[:, None], j]
        return s, a, j

    def _full_arrays(self, prep, b_idx, k_idx):
        c = prep.cache
        hs, cum = c["hist_scaled"], c["cum_scaled"]
        b = torch.as_tensor(b_idx, device=self.device)
        k = torch.as_tensor(k_idx, device=self.device)
        Lmax = int(k.max().item()) + 1
        j = torch.arange(Lmax, device=self.device)[None, :].expand(len(b_idx), -1)
        valid = j <= k[:, None]
        jj = torch.where(valid, j, k[:, None].expand_as(j))
        s = hs[b[:, None], jj, :2]
        a = cum[b, k][:, None] - cum[b[:, None], jj]
        # padded entries: put them far in the past so their contribution vanishes
        a = torch.where(valid, a, torch.full_like(a, 1e6))
        self.lookback_eff = Lmax
        return s, a, jj

    def prepare(self, times, locs) -> Prepared:
        times = np.asarray(times, float)
        locs = np.asarray(locs, float)
        if times.ndim == 1:
            times, locs = times[None], locs[None]
        B, T = times.shape
        dt = np.diff(np.concatenate([np.zeros((B, 1)), times], axis=1), axis=1)
        dt_hat = (dt - self.dt_min) / self.dt_range
        hs = np.concatenate([self._scale_locs(locs), dt_hat[..., None]], axis=-1)
        prep = Prepared(times=times, locs=locs)
        prep.cache["hist_scaled"] = self.t(hs)
        prep.cache["cum_scaled"] = self.t(np.cumsum(dt_hat, axis=1), dtype=torch.float32)
        prep.cache["times_t"] = self.t(times)
        prep.cache["locs_t"] = self.t(locs)
        return prep


class AutoSTPPAdapter(_WindowBase):
    family = "auto_stpp"
    nll_kind = "exact"
    spatial_domain = "support_box"
    spatial_sampler = "fine_grid"

    def __init__(self, runner, device, grid_n: int = 128):
        super().__init__(runner, device)
        em = self.model.event_model
        self.em = em
        self.kernel = em.kernel
        self.lookback = int(em.lookback)
        self.min_history = self.lookback
        if em.trunc:
            raise NotImplementedError("AutoSTPP trunc=True not supported")
        self.grid_n = int(grid_n)

    def _bg(self):
        return float(torch.exp(self.em.background).item())

    @torch.no_grad()
    def event_terms(self, prep) -> EventTerms:
        if prep.B != 1:
            raise ValueError
        T = prep.T
        L = self.lookback
        out = dict(
            log_lambda=np.full(T, np.nan), d_lambda=np.full(T, np.nan), log_ps=np.full(T, np.nan)
        )
        k = np.arange(L - 1, T - 1)
        if len(k):
            b = np.zeros_like(k)
            s, a, _ = self._window_arrays(prep, b, k)
            tgt = torch.as_tensor(k + 1, device=self.device)
            s_y = prep.cache["hist_scaled"][0, tgt, :2]
            dt_y = prep.cache["hist_scaled"][0, tgt, 2]
            R = len(k)
            t_diff = a + dt_y[:, None]
            st = torch.cat([s_y[:, None, :] - s, t_diff[..., None]], dim=-1)
            lam = self.kernel.forward(st.reshape(-1, 3)).view(R, L).sum(-1) + self._bg()
            lam_t = self.kernel.lamb_t_stpp(s.reshape(-1, 2), t_diff.reshape(-1, 1)).view(R, L).sum(-1) + self._bg()
            ints = self.kernel.int_lamb_stpp(s.reshape(-1, 2), a.reshape(-1, 1), t_diff.reshape(-1, 1)).view(R, L).sum(-1)
            ints = ints + dt_y * self._bg()
            out["log_lambda"][k + 1] = to_np(torch.log(lam)) - self.log_jac_t - self.log_jac_s
            out["d_lambda"][k + 1] = to_np(ints)
            out["log_ps"][k + 1] = to_np(torch.log(lam) - torch.log(lam_t)) - self.log_jac_s
        defined = np.isfinite(out["d_lambda"])
        return EventTerms(defined=defined, **out)

    @torch.no_grad()
    def cum_hazard(self, prep, b_idx, k_idx, tau_grid, chunk: int = 64) -> np.ndarray:
        L = self.lookback
        tg = self.t(np.maximum(self._tau_scaled(tau_grid), 0.0))
        G = tg.shape[0]
        out = np.empty((len(b_idx), G))
        for sl in chunked(len(b_idx), chunk):
            s, a, _ = self._window_arrays(prep, b_idx[sl], k_idx[sl])
            R, L = s.shape[0], s.shape[1]
            ss = s[:, :, None, :].expand(R, L, G, 2).reshape(-1, 2)
            aa = a[:, :, None].expand(R, L, G).reshape(-1, 1)
            bb = (a[:, :, None] + tg[None, None, :]).reshape(-1, 1)
            v = self.kernel.int_lamb_stpp(ss, aa, bb).view(R, L, G).sum(1)
            out[sl] = to_np(v + self._bg() * tg[None, :])
        return out

    @torch.no_grad()
    def cum_hazard_at(self, prep, b_idx, k_idx, tau_raw, chunk: int = 32) -> np.ndarray:
        """Lambda_k(tau_r) at one waiting time per row (memory O(R x L))."""
        out = np.empty(len(b_idx))
        tq = np.asarray(tau_raw, float)
        for sl in chunked(len(b_idx), chunk):
            s, a, _ = self._window_arrays(prep, b_idx[sl], k_idx[sl])
            R, L = s.shape[0], s.shape[1]
            ts = self.t(np.maximum(self._tau_scaled(tq[sl]), 0.0))
            v = self.kernel.int_lamb_stpp(s.reshape(-1, 2), a.reshape(-1, 1),
                                          (a + ts[:, None]).reshape(-1, 1)).view(R, L).sum(1)
            out[sl] = to_np(v + self._bg() * ts)
        return out

    @torch.no_grad()
    def ground_intensity(self, prep, b_idx, k_idx, t_query) -> np.ndarray:
        L = self.lookback
        out = []
        tq = np.asarray(t_query, float)
        for sl in chunked(len(b_idx), 4096):
            s, a, _ = self._window_arrays(prep, b_idx[sl], k_idx[sl])
            tk = prep.times[b_idx[sl], k_idx[sl]]
            tau = self.t(np.maximum(self._tau_scaled(tq[sl] - tk), 0.0))
            R, L = s.shape[0], s.shape[1]
            v = self.kernel.lamb_t_stpp(s.reshape(-1, 2), (a + tau[:, None]).reshape(-1, 1)).view(R, L).sum(-1)
            out.append(to_np(v + self._bg()) / self.dt_range)
        return np.concatenate(out)

    # --- spatial ----------------------------------------------------------
    def _axis_grads(self, net, x):
        _, g = net.forward_with_grad(x.reshape(-1, 1))
        return g.reshape(x.shape)

    def _lambda_scaled_at(self, s_win, a, tau_hat, pts_hat):
        """lambda_hat at points; s_win (R,L,2), a (R,L), tau_hat (R,), pts_hat (R,M,2)."""
        R, L, _ = s_win.shape
        M = pts_hat.shape[1]
        dx = pts_hat[:, None, :, 0] - s_win[:, :, None, 0]  # (R,L,M)
        dy = pts_hat[:, None, :, 1] - s_win[:, :, None, 1]
        tt = (a + tau_hat[:, None])  # (R,L)
        total = torch.zeros(R, M, device=self.device)
        for sign, sp in ((3.0, self.kernel.M), (-3.0, self.kernel.L)):
            for net in sp.nets:
                gx = self._axis_grads(net.x_net, dx)
                gy = self._axis_grads(net.y_net, dy)
                gt = self._axis_grads(net.t_net, tt)
                total = total + sign * (gx * gy * gt[:, :, None]).sum(1)
        return total + self._bg()

    def _grid_hat(self, n):
        c = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(c, c, indexing="ij")
        return np.stack([X.ravel(), Y.ravel()], -1)

    @torch.no_grad()
    def spatial_logpdf(self, prep, b_idx, k_idx, t_query, points) -> np.ndarray:
        """Exact model density p(s|t) = lambda(t,s) / lambda_t(t) (0 outside the box)."""
        pts = np.asarray(points, float)
        R, M, _ = pts.shape
        out = np.empty((R, M))
        tq = np.asarray(t_query, float)
        for sl in chunked(R, 256):
            s, a, _ = self._window_arrays(prep, b_idx[sl], k_idx[sl])
            tk = prep.times[b_idx[sl], k_idx[sl]]
            tau = self.t(np.maximum(self._tau_scaled(tq[sl] - tk), 0.0))
            ph = self._scale_locs(pts[sl])
            inside = np.all((ph >= 0) & (ph <= 1), axis=-1)
            lam = to_np(self._lambda_scaled_at(s, a, tau, self.t(ph)))
            r = s.shape[0]
            lam_t = to_np(self.kernel.lamb_t_stpp(s.reshape(-1, 2), (a + tau[:, None]).reshape(-1, 1)).view(r, -1).sum(-1) + self._bg())
            lp = np.log(np.maximum(lam, 1e-300)) - np.log(lam_t)[:, None] - self.log_jac_s
            lp[~inside] = -np.inf
            out[sl] = lp
        return out

    @torch.no_grad()
    def intensity(self, prep, b_idx, k_idx, t_query, points) -> np.ndarray:
        pts = np.asarray(points, float)
        R = pts.shape[0]
        out = np.empty(pts.shape[:2])
        tq = np.asarray(t_query, float)
        for sl in chunked(R, 256):
            s, a, _ = self._window_arrays(prep, b_idx[sl], k_idx[sl])
            tk = prep.times[b_idx[sl], k_idx[sl]]
            tau = self.t(np.maximum(self._tau_scaled(tq[sl] - tk), 0.0))
            ph = self._scale_locs(pts[sl])
            lam = to_np(self._lambda_scaled_at(s, a, tau, self.t(ph)))
            inside = np.all((ph >= 0) & (ph <= 1), axis=-1)
            lam[~inside] = 0.0
            out[sl] = lam / self.dt_range / np.exp(self.log_jac_s)
        return out

    @torch.no_grad()
    @torch.no_grad()
    def _grid_probs(self, prep, b_idx, k_idx, t_query, n):
        """Cell probabilities on an n x n grid of the unit square (scaled space).

        Uses the ProdNet separability: lambda = mu + sum_{p,j} c_p g^t_p(a_j+tau)
        g^x_p(x - x_j) g^y_p(y - y_j), so the grid is an outer-product sum
        evaluated with one batched matmul per chunk (exact, not approximate).
        """
        c1 = (torch.arange(n, device=self.device, dtype=torch.float32) + 0.5) / n
        R = len(b_idx)
        probs = np.empty((R, n * n))
        tq = np.asarray(t_query, float)
        for sl in chunked(R, 512):
            s, a, _ = self._window_arrays(prep, b_idx[sl], k_idx[sl])
            tk = prep.times[b_idx[sl], k_idx[sl]]
            tau = self.t(np.maximum(self._tau_scaled(tq[sl] - tk), 0.0))
            r, L, _ = s.shape
            tt = a + tau[:, None]  # (r, L)
            dx = c1[None, None, :] - s[:, :, None, 0]  # (r, L, n)
            dy = c1[None, None, :] - s[:, :, None, 1]
            lam = torch.zeros(r, n, n, device=self.device)
            for sign, sp in ((3.0, self.kernel.M), (-3.0, self.kernel.L)):
                for net in sp.nets:
                    gx = self._axis_grads(net.x_net, dx)
                    gy = self._axis_grads(net.y_net, dy)
                    gt = self._axis_grads(net.t_net, tt) * sign  # (r, L)
                    lam = lam + torch.einsum("rl,rlx,rly->rxy", gt, gx, gy)
            lam = (lam + self._bg()).clamp(min=0).reshape(r, n * n)
            probs[sl] = to_np(lam / lam.sum(-1, keepdim=True))
        return probs

    def sample_space(self, prep, b_idx, k_idx, t_query, rng, grid_n=None, **kw):
        n = int(grid_n or self.grid_n)
        probs = self._grid_probs(prep, b_idx, k_idx, t_query, n)
        return _sample_grid(probs, n, rng, self._unscale_locs, self.log_jac_s)

    def sampling_logpdf(self, prep, b_idx, k_idx, t_query, points, grid_n=None):
        n = int(grid_n or self.grid_n)
        probs = self._grid_probs(prep, b_idx, k_idx, t_query, n)
        return _grid_logpdf(probs, n, self._scale_locs(points), self.log_jac_s)

    def sample_space_k(self, prep, b_idx, k_idx, t_query, K, rng, grid_n=None):
        n = int(grid_n or self.grid_n)
        probs = self._grid_probs(prep, b_idx, k_idx, t_query, n)
        R = probs.shape[0]
        pr = np.repeat(probs, K, axis=0)
        x, lp = _sample_grid(pr, n, rng, self._unscale_locs, self.log_jac_s)
        return x.reshape(R, K, 2), lp.reshape(R, K)


class DeepSTPPAdapter(_WindowBase):
    family = "deep_stpp"
    nll_kind = "exact"
    spatial_domain = "R2"
    spatial_sampler = "exact_gmm"

    def __init__(self, runner, device):
        super().__init__(runner, device)
        em = self.model.event_model
        self.em = em
        self.dec = em.hawkes_decoder
        self.lookback = int(em.seq_len)
        self.min_history = self.lookback
        self.num_points = int(em.num_points)

    def _z(self, prep, b_idx, k_idx):
        """Window latents z (= encoder mean in eval) for contexts (b, k), cached."""
        L = self.lookback
        cache = prep.cache.setdefault("z_cache", {})
        keys = list(zip(np.asarray(b_idx).tolist(), np.asarray(k_idx).tolist()))
        missing = sorted({kk for kk in keys if kk not in cache})
        if missing:
            hs = prep.cache["hist_scaled"]
            for sl in chunked(len(missing), 4096):
                part = missing[sl]
                win = torch.stack([hs[b, k - L + 1 : k + 1] for b, k in part], 0)
                z, _, _, _ = self.sm._encode_windows(win, sample_latent=bool(getattr(self.sm, "vae", False)))
                for (b, k), zz in zip(part, z):
                    cache[(b, k)] = zz
        return torch.stack([cache[kk] for kk in keys], 0)

    def _params(self, prep, b_idx, k_idx):
        z = self._z(prep, b_idx, k_idx)
        w, bdec, sigma, inv_var = self.dec.decode(z)
        s, a, _ = self._window_arrays(prep, b_idx, k_idx)
        R = s.shape[0]
        tn = torch.cat([a, torch.zeros(R, self.num_points, device=self.device)], dim=1)
        bgp = self.dec.background
        if bgp is not None:
            centers = torch.cat([s, bgp.to(s).unsqueeze(0).expand(R, -1, -1)], dim=1)
        else:
            centers = s
        bsafe = self.dec._safe_decay(bdec)
        return w, bsafe, sigma, tn, centers

    @torch.no_grad()
    def event_terms(self, prep) -> EventTerms:
        T = prep.T
        L = self.lookback
        out = dict(log_lambda=np.full(T, np.nan), d_lambda=np.full(T, np.nan), log_ps=np.full(T, np.nan))
        k = np.arange(L - 1, T - 1)
        if len(k):
            b = np.zeros_like(k)
            w, bs, sigma, tn, centers = self._params(prep, b, k)
            tgt = torch.as_tensor(k + 1, device=self.device)
            s_y = prep.cache["hist_scaled"][0, tgt, :2]
            dt_y = prep.cache["hist_scaled"][0, tgt, 2]
            t_ti = (tn + dt_y[:, None]).clamp(min=1e-6)
            log_v = torch.log(w) - bs * t_ti
            log_lam_t = torch.logsumexp(log_v, -1)
            dlam = (w / bs * (torch.exp(-bs * tn.clamp(min=0)) - torch.exp(-bs * t_ti))).sum(-1)
            lps = self._mix_logpdf(log_v, centers, sigma, s_y[:, None, :])[:, 0]
            out["log_lambda"][k + 1] = to_np(log_lam_t + lps) - self.log_jac_t - self.log_jac_s
            out["d_lambda"][k + 1] = to_np(dlam)
            out["log_ps"][k + 1] = to_np(lps) - self.log_jac_s
        return EventTerms(defined=np.isfinite(out["d_lambda"]), **out)

    @staticmethod
    def _mix_logpdf(log_v, centers, sigma, pts):
        """log sum_i softmax(log_v)_i N(pts | centers_i, diag(sigma_i)); pts (R,M,2)."""
        log_vn = log_v - torch.logsumexp(log_v, -1, keepdim=True)
        d = pts[:, :, None, :] - centers[:, None, :, :]  # (R,M,m,2)
        lg = (-0.5 * torch.log(sigma).sum(-1)[:, None, :] - math.log(2 * math.pi)
              - 0.5 * (d * d / sigma[:, None, :, :]).sum(-1))
        return torch.logsumexp(log_vn[:, None, :] + lg, -1)

    @torch.no_grad()
    def cum_hazard(self, prep, b_idx, k_idx, tau_grid, chunk: int = 256) -> np.ndarray:
        tg = self.t(np.maximum(self._tau_scaled(tau_grid), 0.0))
        out = np.empty((len(b_idx), tg.shape[0]))
        for sl in chunked(len(b_idx), chunk):
            w, bs, _, tn, _ = self._params(prep, b_idx[sl], k_idx[sl])
            e0 = torch.exp(-bs * tn.clamp(min=0))  # (R,m)
            e1 = torch.exp(-bs[:, :, None] * (tn[:, :, None] + tg[None, None, :]))
            out[sl] = to_np(((w / bs)[:, :, None] * (e0[:, :, None] - e1)).sum(1))
        return out

    @torch.no_grad()
    def ground_intensity(self, prep, b_idx, k_idx, t_query):
        out = []
        tq = np.asarray(t_query, float)
        for sl in chunked(len(b_idx), 4096):
            w, bs, _, tn, _ = self._params(prep, b_idx[sl], k_idx[sl])
            tk = prep.times[b_idx[sl], k_idx[sl]]
            tau = self.t(np.maximum(self._tau_scaled(tq[sl] - tk), 0.0))
            t_ti = (tn + tau[:, None]).clamp(min=1e-6)
            out.append(to_np((w * torch.exp(-bs * t_ti)).sum(-1)) / self.dt_range)
        return np.concatenate(out)

    @torch.no_grad()
    def spatial_logpdf(self, prep, b_idx, k_idx, t_query, points):
        pts = np.asarray(points, float)
        out = np.empty(pts.shape[:2])
        tq = np.asarray(t_query, float)
        for sl in chunked(len(b_idx), 512):
            w, bs, sigma, tn, centers = self._params(prep, b_idx[sl], k_idx[sl])
            tk = prep.times[b_idx[sl], k_idx[sl]]
            tau = self.t(np.maximum(self._tau_scaled(tq[sl] - tk), 0.0))
            t_ti = (tn + tau[:, None]).clamp(min=1e-6)
            log_v = torch.log(w) - bs * t_ti
            out[sl] = to_np(self._mix_logpdf(log_v, centers, sigma, self.t(self._scale_locs(pts[sl])))) - self.log_jac_s
        return out

    @torch.no_grad()
    def sample_space(self, prep, b_idx, k_idx, t_query, rng, **kw):
        R = len(b_idx)
        locs = np.empty((R, 2))
        tq = np.asarray(t_query, float)
        for sl in chunked(R, 4096):
            w, bs, sigma, tn, centers = self._params(prep, b_idx[sl], k_idx[sl])
            tk = prep.times[b_idx[sl], k_idx[sl]]
            tau = self.t(np.maximum(self._tau_scaled(tq[sl] - tk), 0.0))
            t_ti = (tn + tau[:, None]).clamp(min=1e-6)
            log_v = to_np(torch.log(w) - bs * t_ti)
            pv = np.exp(log_v - log_v.max(-1, keepdims=True))
            pv /= pv.sum(-1, keepdims=True)
            u = rng.random((pv.shape[0], 1))
            comp = (np.cumsum(pv, -1) < u).sum(-1).clip(max=pv.shape[1] - 1)
            c = to_np(centers)[np.arange(pv.shape[0]), comp]
            sd = np.sqrt(to_np(sigma)[np.arange(pv.shape[0]), comp])
            locs[sl] = self._unscale_locs(c + sd * rng.standard_normal(c.shape))
        logp = self.spatial_logpdf(prep, b_idx, k_idx, tq, locs[:, None, :])[:, 0]
        return locs, logp


def _sample_grid(probs, n, rng, unscale, log_jac_s):
    """Piecewise-constant sampling on an n x n cell grid of the unit square."""
    R = probs.shape[0]
    cdf = np.cumsum(probs, -1)
    u = rng.random((R, 1)) * cdf[:, -1:]
    cell = (cdf < u).sum(-1).clip(max=n * n - 1)
    ix, iy = cell // n, cell % n
    xh = (ix + rng.random(R)) / n
    yh = (iy + rng.random(R)) / n
    logp = np.log(probs[np.arange(R), cell] * n * n) - log_jac_s
    return unscale(np.stack([xh, yh], -1)), logp


def _grid_logpdf(probs, n, pts_hat, log_jac_s):
    pts_hat = np.asarray(pts_hat, float)
    R, M, _ = pts_hat.shape
    ix = np.clip(np.floor(pts_hat[..., 0] * n).astype(int), 0, n - 1)
    iy = np.clip(np.floor(pts_hat[..., 1] * n).astype(int), 0, n - 1)
    cell = ix * n + iy
    p = np.take_along_axis(probs, cell, axis=1) * n * n
    out = np.log(np.maximum(p, 1e-300)) - log_jac_s
    inside = np.all((pts_hat >= 0) & (pts_hat <= 1), axis=-1)
    out[~inside] = -np.inf
    return out
