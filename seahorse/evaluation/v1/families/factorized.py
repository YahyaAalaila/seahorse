"""Factorized RMTPP/THP + Gaussian-mixture adapters (``rmtpp_gmm``, ``thp_gmm``).

lambda(t, s | H) = lambda_T(t | H) p(s | H), with p a mixture of isotropic
Gaussians centred at past events (weights ~ exp(-tau (t - t_j)); normalised
over j the weights do not depend on t). Spatial domain R^2.

RMTPP: lambda_T = a exp(-b (t - t_k)), closed-form compensator (exact).
       Note ``a, b > 0`` -> the waiting-time law is *defective*: with
       probability exp(-a/b) no further event ever occurs.
THP:   lambda_T = softplus(f(h_k) + beta (t - t_k)); the model's NLL uses a
       30-node midpoint rule. We report the PIT from a 64-node Gauss-Legendre
       compensator (``nll_kind = numerical_quadrature``) and record the
       difference to the NLL-path rule.

Inputs are z-scored by the preset (time and space) and shifted so that the
first event of each sequence is at 0; raw-unit quantities include the
corresponding Jacobians.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .base import EventTerms, FamilyAdapter, Prepared, chunked, to_np


class FactorizedAdapter(FamilyAdapter):
    family = "factorized"
    spatial_domain = "R2"
    spatial_sampler = "exact_gmm"
    min_history = 1

    def __init__(self, runner, device):
        super().__init__(runner, device)
        em = self.model.event_model
        self.em = em
        self.tm = em.temporal_model
        self.sp = em.spatial_model
        self.kind = type(self.tm).__name__
        if self.kind.startswith("RMTPP"):
            self.nll_kind = "exact"
        elif self.kind.startswith("THP"):
            self.nll_kind = "numerical_quadrature"
        else:
            raise NotImplementedError(self.kind)
        if em.t1 is not None:
            raise NotImplementedError("fixed t1 not supported")
        tr = self.model.state_model._input_transform
        ns = runner.norm_stats or {}
        if bool(ns.get("normalize", False)) and (
            float(ns.get("time_std", 1.0)) != 1.0 or float(ns.get("time_mean", 0.0)) != 0.0
        ):
            raise NotImplementedError("factorized adapter assumes identity runner norm_stats")
        self.tr = tr
        if tr is not None and type(tr).__name__ == "ZScoreTransformArtifact":
            self.t_mean = float(tr.time_mean) if tr.normalize_time else 0.0
            self.t_std = float(tr.time_std) if tr.normalize_time else 1.0
            self.l_mean = np.asarray(tr.loc_mean, float) if tr.normalize_space else np.zeros(2)
            self.l_std = np.asarray(tr.loc_std, float) if tr.normalize_space else np.ones(2)
        elif tr is None or type(tr).__name__ == "IdentityTransformArtifact":
            self.t_mean, self.t_std, self.l_mean, self.l_std = 0.0, 1.0, np.zeros(2), np.ones(2)
        else:
            raise NotImplementedError(type(tr).__name__)
        self.log_jac_t = math.log(self.t_std)
        self.log_jac_s = float(np.log(self.l_std).sum())

    def _n_locs(self, s):
        return (np.asarray(s, float) - self.l_mean) / self.l_std

    def _r_locs(self, s):
        return np.asarray(s, float) * self.l_std + self.l_mean

    @torch.no_grad()
    def prepare(self, times, locs) -> Prepared:
        times = np.asarray(times, float)
        locs = np.asarray(locs, float)
        if times.ndim == 1:
            times, locs = times[None], locs[None]
        B, T = times.shape
        tn = (times - self.t_mean) / self.t_std
        ts = np.clip(tn - tn[:, :1], 0.0, None)  # sequence-relative shift as in the model
        ts_t = self.t(ts)
        # RMTPP packs sequences (lengths must live on the CPU); THP builds masks on-device
        lens = torch.full((B,), T, dtype=torch.long,
                          device="cpu" if self.kind.startswith("RMTPP") else self.device)
        hidden, delta_t = self.tm._encode(ts_t, lens)
        prep = Prepared(times=times, locs=locs)
        prep.cache.update(ts=ts_t, hidden=hidden, delta_t=delta_t, s_n=self.t(self._n_locs(locs)))
        if self.kind.startswith("RMTPP"):
            ab = self.tm.mlp(hidden)
            prep.cache["a"] = F.softplus(ab[..., 0])
            prep.cache["b"] = F.softplus(ab[..., 1]).clamp(min=1e-6)
        else:
            prep.cache["f"] = self.tm.f(hidden).squeeze(-1)  # (B, T+1)
            prep.cache["beta"] = self.tm.beta.reshape(())
        return prep

    # hidden index for context k (history events 0..k) is k+1
    def _lam_T_native(self, prep, b, k, tau_n):
        c = prep.cache
        if self.kind.startswith("RMTPP"):
            a = c["a"][b, k + 1][:, None].double()
            bb = c["b"][b, k + 1][:, None].double()
            return a * torch.exp(-bb * tau_n.double())
        f = c["f"][b, k + 1][:, None]
        return F.softplus(f + c["beta"] * tau_n)

    def _cum_native(self, prep, b, k, tau_n):
        c = prep.cache
        if self.kind.startswith("RMTPP"):
            # float64 + expm1: the float32 form 1 - exp(-b tau) is exactly 0 for
            # b tau < ~3e-8 (RMTPP expm1 fix)
            a = c["a"][b, k + 1][:, None].double()
            bb = c["b"][b, k + 1][:, None].double()
            return (a / bb) * (-torch.expm1(-bb * tau_n.double()))
        # THP: Gauss-Legendre per [0, tau] (smooth integrand)
        xg, wg = np.polynomial.legendre.leggauss(64)
        xg = self.t(0.5 * (xg + 1.0))
        wg = self.t(0.5 * wg)
        f = c["f"][b, k + 1][:, None, None]
        nodes = tau_n[..., None] * xg
        val = F.softplus(f + c["beta"] * nodes)
        return (val * wg).sum(-1) * tau_n

    @torch.no_grad()
    def event_terms(self, prep) -> EventTerms:
        T = prep.T
        k = np.arange(0, T - 1)
        b = np.zeros_like(k)
        bt = torch.as_tensor(b, device=self.device)
        kt = torch.as_tensor(k, device=self.device)
        dtn = prep.cache["delta_t"][0, 1:][:, None]  # target i = k+1 wait (native)
        lam_t = self._lam_T_native(prep, bt, kt, dtn)[:, 0]
        dlam = self._cum_native(prep, bt, kt, dtn)[:, 0]
        log_ps_n = to_np(self.sp.logprob(prep.cache["ts"], prep.cache["s_n"], torch.ones(1, T, device=self.device))[0])
        out_ll = np.full(T, np.nan)
        out_dl = np.full(T, np.nan)
        out_ps = np.full(T, np.nan)
        out_ps[1:] = log_ps_n[1:] - self.log_jac_s
        out_ll[1:] = to_np(torch.log(lam_t + 1e-20)) - self.log_jac_t + out_ps[1:]
        out_dl[1:] = to_np(dlam)
        extra = {}
        if self.kind.startswith("THP"):
            S = self.tm.MC_SAMPLES
            nodes = self.t((np.arange(S) + 0.5) / S)
            f = prep.cache["f"][0, kt + 1][:, None]
            nll_path = (F.softplus(f + prep.cache["beta"] * dtn * nodes[None, :]) * dtn).mean(-1)
            ep = np.full(T, np.nan)
            ep[1:] = to_np(nll_path)
            extra["d_lambda_nll_path"] = ep
        defined = np.ones(T, bool)
        defined[0] = False
        return EventTerms(log_lambda=out_ll, d_lambda=out_dl, log_ps=out_ps, defined=defined, extra=extra)

    @torch.no_grad()
    def cum_hazard(self, prep, b_idx, k_idx, tau_grid) -> np.ndarray:
        tn = self.t(np.asarray(tau_grid, float) / self.t_std)
        out = np.empty((len(b_idx), tn.shape[0]))
        for sl in chunked(len(b_idx), 256):
            b = torch.as_tensor(b_idx[sl], device=self.device)
            k = torch.as_tensor(k_idx[sl], device=self.device)
            out[sl] = to_np(self._cum_native(prep, b, k, tn[None, :].expand(len(b), -1)))
        return out

    @torch.no_grad()
    def ground_intensity(self, prep, b_idx, k_idx, t_query):
        tk = prep.times[b_idx, k_idx]
        tau_n = self.t((np.asarray(t_query, float) - tk) / self.t_std)[:, None]
        v = self._lam_T_native(prep, torch.as_tensor(b_idx, device=self.device), torch.as_tensor(k_idx, device=self.device), tau_n)[:, 0]
        return to_np(v) / self.t_std

    # --- spatial GMM ----------------------------------------------------
    def _gmm(self, prep, b, k):
        sig = float(F.softplus(self.sp.log_sigma_kernel).item())
        tau = float(F.softplus(self.sp.log_tau).item())
        return sig, tau

    @torch.no_grad()
    def spatial_logpdf(self, prep, b_idx, k_idx, t_query, points):
        sig, tau = self._gmm(prep, None, None)
        pts = self._n_locs(points)
        out = np.empty(pts.shape[:2])
        ts = to_np(prep.cache["ts"])
        sn = to_np(prep.cache["s_n"])
        for r in range(len(b_idx)):
            b, k = int(b_idx[r]), int(k_idx[r])
            lw = tau * ts[b, : k + 1]
            lw = lw - _lse(lw)
            d = pts[r][:, None, :] - sn[b, : k + 1][None]
            lg = -0.5 * (d * d).sum(-1) / sig**2 - math.log(2 * math.pi) - 2 * math.log(sig)
            out[r] = _lse2(lw[None] + lg) - self.log_jac_s
        return out

    @torch.no_grad()
    def sample_space(self, prep, b_idx, k_idx, t_query, rng, **kw):
        sig, tau = self._gmm(prep, None, None)
        ts = to_np(prep.cache["ts"])
        sn = to_np(prep.cache["s_n"])
        R = len(b_idx)
        x = np.empty((R, 2))
        for r in range(R):
            b, k = int(b_idx[r]), int(k_idx[r])
            lw = tau * ts[b, : k + 1]
            w = np.exp(lw - lw.max())
            w /= w.sum()
            j = min(int((np.cumsum(w) < rng.random()).sum()), k)
            x[r] = sn[b, j] + sig * rng.standard_normal(2)
        locs = self._r_locs(x)
        logp = self.spatial_logpdf(prep, b_idx, k_idx, t_query, locs[:, None, :])[:, 0]
        return locs, logp


def _lse(a):
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def _lse2(a):
    m = a.max(-1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(-1, keepdims=True)))[..., 0]
