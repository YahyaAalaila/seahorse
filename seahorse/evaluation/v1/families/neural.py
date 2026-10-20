"""Neural-STPP family adapter (NJSDE / Jump-CNF / Attn-CNF).

Model structure used here (Chen et al. 2021, as ported in ``seahorse``):

* temporal: a latent ODE ``h(t)`` with ground intensity ``lambda(t) = g(h(t))``
  and jumps ``h <- update(h, s_i)`` at events. The compensator
  ``Lambda(t) = int lambda`` is integrated *as part of the ODE state*
  (``NeuralPointProcess.integrate_lambda``), so ``dLambda_i`` and the
  cumulative hazard on a waiting-time grid are exact up to the solver tolerance.
* spatial: ``p(s | t, H)`` on ``R^2`` (standardised coordinates), a conditional
  GMM (NJSDE) or a continuous normalising flow (Jump-CNF, Attn-CNF).

Joint intensity ``lambda(t, s) = lambda(t) p(s | t, H)`` integrates to
``lambda(t)`` over ``R^2``, so the NLL's spatial domain is ``R^2``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .base import EventTerms, FamilyAdapter, Prepared, chunked, to_np


def _odeint():
    from torchdiffeq import odeint

    return odeint


class NeuralAdapter(FamilyAdapter):
    family = "neural_stpp"
    nll_kind = "exact_ode"  # exact up to adaptive-solver tolerance
    spatial_domain = "R2"
    min_history = 1

    def __init__(self, runner, device):
        super().__init__(runner, device)
        sm = self.model.state_model
        self.sm = sm
        self.tc = sm.temporal_core
        self.dec = self.model.event_model.spatial_decoder
        self.spatial_type = type(self.dec).__name__
        tr = sm._input_transform
        if tr is not None and getattr(tr, "normalize_time", False):
            raise NotImplementedError("Neural adapter assumes raw-time inputs (normalize_time=False).")
        if sm.normalize_time_inputs:
            raise NotImplementedError("Neural adapter assumes normalize_time_inputs=False.")
        self.require_identity_runner_norm()
        if tr is not None and getattr(tr, "normalize_space", False):
            self.loc_mean = np.asarray(tr.loc_mean, dtype=np.float64)
            self.loc_std = np.asarray(tr.loc_std, dtype=np.float64)
        else:
            self.loc_mean = np.zeros(2)
            self.loc_std = np.ones(2)
        self.log_jac_s = float(np.log(self.loc_std).sum())  # raw = native - log_jac
        self.uses_aux = bool(getattr(self.dec, "USES_NEURAL_AUX_STATE", False))
        if "GMM" in self.spatial_type:
            self.spatial_sampler = "exact_gmm"
        elif "JumpCNF" in self.spatial_type:
            self.spatial_sampler = "exact_inverse_flow_jumpcnf"
        elif "AttnCNF" in self.spatial_type:
            self.spatial_sampler = "exact_inverse_flow_attncnf"
        else:
            self.spatial_sampler = "unsupported"

    # ------------------------------------------------------------------
    def _norm_locs(self, locs_raw: np.ndarray) -> np.ndarray:
        return (np.asarray(locs_raw, dtype=np.float64) - self.loc_mean) / self.loc_std

    def _raw_locs(self, locs_norm: np.ndarray) -> np.ndarray:
        return np.asarray(locs_norm, dtype=np.float64) * self.loc_std + self.loc_mean

    @torch.no_grad()
    def prepare(self, times: np.ndarray, locs: np.ndarray) -> Prepared:
        times = np.asarray(times, dtype=np.float64)
        locs = np.asarray(locs, dtype=np.float64)
        if times.ndim == 1:
            times, locs = times[None], locs[None]
        B, T = times.shape
        times = np.stack([_strict_rows(r) for r in times])
        t_raw = self.t(times)
        s_norm = self.t(self._norm_locs(locs))
        mask = torch.ones(B, T, device=self.device)
        tc = self.tc
        intens, _lam_final, prejump, details = tc.integrate_lambda(
            t_raw, s_norm if tc.cond else None, mask, t0=0.0, t1=None, nlinspace=1, return_details=True
        )
        cum = details["cumulative_lambdas"]  # (B, T)
        h_pre = prejump[:, 1 : T + 1, :]  # event-aligned pre-jump hidden h(t_i^-)
        H = h_pre.shape[-1]
        flat_pre = h_pre.reshape(B * T, H)
        cond = s_norm.reshape(B * T, -1) if tc.cond else None
        h_post = tc.hidden_state_dynamics.update_state(
            t_raw.reshape(-1), flat_pre, cond=cond
        ).reshape(B, T, H)
        prep = Prepared(times=times, locs=locs)
        prep.cache.update(
            t_raw=t_raw, s_norm=s_norm, intens=intens, cum=cum, h_pre=h_pre, h_post=h_post
        )
        return prep

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout_init(self, seq_prep, k0, M):
        c = seq_prep.cache
        prep = Prepared(times=np.repeat(seq_prep.times[:, : k0 + 1], M, 0),
                        locs=np.repeat(seq_prep.locs[:, : k0 + 1], M, 0))
        for key in ("t_raw", "s_norm", "h_pre", "h_post"):
            prep.cache[key] = c[key][:, : k0 + 1].repeat_interleave(M, dim=0).contiguous()
        return prep

    @torch.no_grad()
    def extend(self, prep, t_new, s_new):
        """Incremental update: integrate the hidden ODE to t_new, jump with s_new."""
        c = prep.cache
        t_new = np.asarray(t_new, float)
        last = c["t_raw"][:, -1]
        tn = self.t(t_new)
        tn = torch.where(tn > last, tn, torch.nextafter(last, torch.full_like(last, float("inf"))))
        h_pre, _, _ = self.tc.integrate_hidden(c["h_post"][:, -1], last, tn)
        sn = self.t(self._norm_locs(np.asarray(s_new, float)))
        h_post = self.tc.hidden_state_dynamics.update_state(tn, h_pre, cond=sn if self.tc.cond else None)
        out = Prepared(times=np.concatenate([prep.times, to_np(tn)[:, None]], 1),
                       locs=np.concatenate([prep.locs, np.asarray(s_new, float)[:, None, :]], 1))
        out.cache["t_raw"] = torch.cat([c["t_raw"], tn[:, None]], 1)
        out.cache["s_norm"] = torch.cat([c["s_norm"], sn[:, None, :]], 1)
        out.cache["h_pre"] = torch.cat([c["h_pre"], h_pre[:, None, :]], 1)
        out.cache["h_post"] = torch.cat([c["h_post"], h_post[:, None, :]], 1)
        # attn end states: the new row's end state requires a joint solve; recompute lazily
        return out

    # ------------------------------------------------------------------
    def _spatial_z_seq(self, h_seq: torch.Tensor) -> torch.Tensor:
        if self.uses_aux:
            aux_dim = int(self.sm.spatial_aux_dim)
            return h_seq[..., -aux_dim:] if aux_dim > 0 else h_seq[..., :0]
        return h_seq

    @torch.no_grad()
    def event_terms(self, prep: Prepared) -> EventTerms:
        if prep.B != 1:
            raise ValueError("event_terms expects one sequence")
        c = prep.cache
        T = prep.T
        intens = c["intens"][0]
        cum = c["cum"][0]
        d_lam = torch.cat([cum[:1], cum[1:] - cum[:-1]])
        log_lam_t = torch.log(intens + 1e-8)  # identical to the model's temporal_nll
        mask = torch.ones(1, T, device=self.device)
        t_prev = torch.cat([torch.zeros(1, 1, device=self.device), c["t_raw"][:, :-1]], dim=1)
        with torch.enable_grad():
            sp_nll = self.dec.sequence_nll(
                z_seq=self._spatial_z_seq(c["h_pre"]),
                t_seq=c["t_raw"].unsqueeze(-1),
                s_seq=c["s_norm"],
                t_prev_seq=t_prev.unsqueeze(-1),
                lengths=torch.tensor([T], device=self.device),
                mask=mask,
            )
        log_ps_raw = -to_np(sp_nll[0]) - self.log_jac_s
        log_lam_raw = to_np(log_lam_t) + log_ps_raw
        defined = np.ones(T, dtype=bool)
        defined[0] = False  # targets start at index 1 (benchmark convention)
        return EventTerms(
            log_lambda=log_lam_raw,
            d_lambda=to_np(d_lam),
            log_ps=log_ps_raw,
            defined=defined,
            extra={"log_lambda_ground": to_np(log_lam_t)},
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def cum_hazard(self, prep, b_idx, k_idx, tau_grid, chunk: int = 512) -> np.ndarray:
        c = prep.cache
        odeint = _odeint()
        solver = self.tc.ode_solver
        tau = np.asarray(tau_grid, dtype=np.float64)
        tau_max = float(tau[-1])
        s_grid = torch.as_tensor(tau / tau_max, dtype=torch.float32, device=self.device)
        out = np.empty((len(b_idx), len(tau)), dtype=np.float64)
        for sl in chunked(len(b_idx), chunk):
            bb = torch.as_tensor(b_idx[sl], device=self.device)
            kk = torch.as_tensor(k_idx[sl], device=self.device)
            h0 = c["h_post"][bb, kk]
            t0 = c["t_raw"][bb, kk]
            t1 = t0 + tau_max
            R = h0.shape[0]
            state = (
                t0,
                t1,
                torch.zeros(1, device=self.device),
                torch.zeros(R, device=self.device),
                h0,
            )
            sol = odeint(
                solver,
                state,
                s_grid,
                rtol=solver.rtol,
                atol=solver.atol,
                method="dopri5",
                options={"dtype": torch.float64},
            )
            out[sl] = to_np(sol[3]).T  # (R, G)
        return out

    @torch.no_grad()
    def hidden_at(self, prep, b_idx, k_idx, t_query: np.ndarray) -> torch.Tensor:
        c = prep.cache
        bb = torch.as_tensor(b_idx, device=self.device)
        kk = torch.as_tensor(k_idx, device=self.device)
        h0 = c["h_post"][bb, kk]
        t0 = c["t_raw"][bb, kk]
        tq = self.t(t_query)
        h, _lam, _ = self.tc.integrate_hidden(h0, t0, tq)
        return h

    @torch.no_grad()
    def ground_intensity(self, prep, b_idx, k_idx, t_query) -> np.ndarray:
        out = []
        for sl in chunked(len(b_idx), 2048):
            h = self.hidden_at(prep, b_idx[sl], k_idx[sl], np.asarray(t_query)[sl])
            out.append(to_np(self.tc.get_intensity(h).reshape(-1)))
        return np.concatenate(out)

    # ------------------------------------------------------------------
    # Spatial: conditional GMM (NJSDE)
    def _gmm_params(self, h: torch.Tensor) -> torch.Tensor:
        aux = self.dec._select_aux(h)
        return self.dec._params_from_aux(aux)

    @torch.no_grad()
    def sample_space(self, prep, b_idx, k_idx, t_query, rng, **kw):
        if self.spatial_sampler == "exact_gmm":
            return self._sample_gmm(prep, b_idx, k_idx, t_query, rng)
        if self.spatial_sampler.startswith("exact_inverse_flow"):
            from .neural_flows import sample_cnf

            return sample_cnf(self, prep, b_idx, k_idx, np.asarray(t_query), rng, **kw)
        raise NotImplementedError(self.spatial_sampler)

    def _sample_gmm(self, prep, b_idx, k_idx, t_query, rng):
        R = len(b_idx)
        locs = np.empty((R, 2))
        logp = np.empty(R)
        nmix = self.dec.n_mixtures
        for sl in chunked(R, 4096):
            h = self.hidden_at(prep, b_idx[sl], k_idx[sl], np.asarray(t_query)[sl])
            params = self._gmm_params(h).reshape(-1, 2, 3, nmix)
            p = params.to(torch.float64).cpu().numpy()
            logits, means, log_std = p[:, :, 0, :], p[:, :, 1, :], p[:, :, 2, :]
            w = np.exp(logits - logits.max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            n = p.shape[0]
            u = rng.random((n, 2, 1))
            comp = (np.cumsum(w, -1) < u).sum(-1).clip(max=nmix - 1)  # (n, 2)
            mu = np.take_along_axis(means, comp[..., None], -1)[..., 0]
            sd = np.exp(np.take_along_axis(log_std, comp[..., None], -1)[..., 0])
            x = mu + sd * rng.standard_normal((n, 2))
            locs[sl] = self._raw_locs(x)
            logp[sl] = self._gmm_logpdf_np(p, x) - self.log_jac_s
        return locs, logp

    @staticmethod
    def _gmm_logpdf_np(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Per-dimension GMM log-density summed over dims. p: (n,2,3,nmix), x: (n,2)."""
        logits, means, log_std = p[:, :, 0, :], p[:, :, 1, :], p[:, :, 2, :]
        logw = logits - _logsumexp(logits, -1)[..., None]
        z = (x[..., None] - means) / np.exp(log_std)
        lg = -0.5 * z * z - log_std - 0.5 * math.log(2 * math.pi)
        return _logsumexp(logw + lg, -1).sum(-1)

    @torch.no_grad()
    def spatial_logpdf(self, prep, b_idx, k_idx, t_query, points) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        R, M, _ = pts.shape
        if self.spatial_sampler == "exact_gmm":
            out = np.empty((R, M))
            nmix = self.dec.n_mixtures
            for sl in chunked(R, 2048):
                h = self.hidden_at(prep, b_idx[sl], k_idx[sl], np.asarray(t_query)[sl])
                p = self._gmm_params(h).reshape(-1, 2, 3, nmix).to(torch.float64).cpu().numpy()
                x = self._norm_locs(pts[sl])  # (r, M, 2)
                r = x.shape[0]
                pp = np.repeat(p, M, axis=0)
                out[sl] = self._gmm_logpdf_np(pp, x.reshape(r * M, 2)).reshape(r, M) - self.log_jac_s
            return out
        from .neural_flows import cnf_logpdf_points

        return cnf_logpdf_points(self, prep, b_idx, k_idx, np.asarray(t_query), pts)


def _strict_rows(row: np.ndarray) -> np.ndarray:
    """float32 strictly-increasing repair (as in the NLL path); generated
    rollout events can tie their predecessor after the float32 cast."""
    t32 = np.asarray(row, dtype=np.float32).copy()
    for i in range(1, t32.size):
        if not (t32[i] > t32[i - 1]):
            t32[i] = np.nextafter(t32[i - 1], np.float32(np.inf))
    return t32.astype(np.float64)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)
