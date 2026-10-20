"""Per-family model access for :mod:`seahorse.evaluation.v1.calibration`.

Common contract of the torch adapters (batch = one ``H+1``-event window per row,
from the campaign collate):

* ``terms(times, locations, lengths, tau_model, output) -> dict`` with
  ``d_lambda (B,)`` the compensator over ``(t_{H-1}, t_H]`` (the NLL path's own
  increment), ``log_lam_T (B,)`` the log ground intensity at ``t_H`` in model
  time units (diagnostic), ``lam_grid (B, G)`` ``Lambda(tau_model)`` from the
  post-history state (only when ``tau_model`` is given) and ``check_*``
  identity checks against the ``eval_forward`` output (0.0 expected).
* ``cumhaz_rows(tau (B, K)) -> Lambda (B, K)`` for bisection quantiles.
* ``spatial_cells(times, locations, lengths, n)`` returns
  ``(probs (B, n*n), cell_obs (B,), mass_in_box (B,))`` or ``None``.
* ``box``: the spatial domain of the family in the model's input coordinates.
* ``tau_internal_scale``: model-internal time units per model-input time unit
  (1 unless the family z-scores time internally).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .calibration import cell_index, grid_centres, to64


class _Base:
    family = "base"
    spatial_kind = "none"
    tau_internal_scale = 1.0
    box = (0.0, 1.0, 0.0, 1.0)

    def __init__(self, runner, device, box):
        self.runner = runner
        self.model = runner.model
        self.device = device
        self.box = tuple(float(v) for v in box)

    def spatial_cells(self, times, locations, lengths, n):
        return None


class DeepSTPPCalib(_Base):
    family = "deep_stpp"
    spatial_kind = "gmm_window_closed_form"

    def __init__(self, runner, device, box):
        super().__init__(runner, device, box)
        self.sm, self.em = self.model.state_model, self.model.event_model
        self.dec = self.em.hawkes_decoder
        if bool(self.sm.input_normalized_flag.item() > 0.5):
            raise NotImplementedError("deep_stpp: input_normalized=True not supported")
        self.dt_min, self.dt_range = float(self.sm.paper_dt_min.item()), float(self.sm.paper_dt_range.item())
        self.loc_min = to64(self.sm.paper_loc_min)
        self.loc_range = to64(self.sm.paper_loc_range)

    @torch.no_grad()
    def _state(self, times, locations, lengths):
        p = self.sm.encode_history(times=times, locations=locations, lengths=lengths).payload
        B = int(lengths.shape[0])
        x, y = p["paper_st_x"], p["paper_st_y"]
        if x.shape[0] != B or not bool((p["paper_window_counts"] == 1).all()):
            raise ValueError("expects exactly one DeepSTPP window (H+1 events) per batch row")
        w, b, _sigma, inv_var = self.dec.decode(p["z"])
        t_hist = torch.cumsum(x[:, :, 2], dim=-1)
        tn = torch.cat([t_hist[:, -1:] - t_hist, torch.zeros(B, self.em.num_points, device=x.device, dtype=x.dtype)], -1)
        dt_y = y[:, 0, 2]
        bg = self.dec.background
        centers = torch.cat([x[:, :, :2], bg.to(x).unsqueeze(0).expand(B, -1, -1)], 1) if bg is not None else x[:, :, :2]
        return dict(w=w, b=b, inv_var=inv_var, tn=tn, dt_y=dt_y, centers=centers)

    @torch.no_grad()
    def terms(self, times, locations, lengths, tau_model, output=None):
        s = self._state(times, locations, lengths)
        w, b, tn, dt_y = s["w"], s["b"], s["tn"], s["dt_y"]
        bs = self.dec._safe_decay(b)  # same decay as the NLL path (log_ft)
        e0 = torch.exp(-bs * tn.clamp(min=0.0))
        t_ti = (tn + dt_y[:, None]).clamp(min=1e-6)
        d_lambda = (w / bs * (e0 - torch.exp(-bs * t_ti))).sum(-1)
        log_lam_T_paper = torch.logsumexp(torch.log(w) - bs * t_ti, dim=-1)
        out = {"d_lambda": d_lambda, "log_lam_T": log_lam_T_paper - math.log(self.dt_range)}
        if output is not None:
            out["check_tll"] = (output["tll_matrix"][:, 0] - (log_lam_T_paper - d_lambda)).abs().max().item()
        self._cache = dict(w=w, bs=bs, tn=tn, e0=e0)
        if tau_model is not None:
            tau_hat = ((tau_model - self.dt_min) / self.dt_range).clamp(min=0.0)
            tg = (tn[:, :, None] + tau_hat[None, None, :]).clamp(min=1e-6)
            out["lam_grid"] = ((w / bs)[:, :, None] * (e0[:, :, None] - torch.exp(-bs[:, :, None] * tg))).sum(1)
        return out

    @torch.no_grad()
    def cumhaz_rows(self, tau):
        c = self._cache
        tau_hat = ((tau - self.dt_min) / self.dt_range).clamp(min=0.0)
        t_ti = (c["tn"][:, None, :] + tau_hat[:, :, None]).clamp(min=1e-6)
        return ((c["w"] / c["bs"])[:, None, :] * (c["e0"][:, None, :] - torch.exp(-c["bs"][:, None, :] * t_ti))).sum(-1)

    @torch.no_grad()
    def spatial_cells(self, times, locations, lengths, n):
        s = self._state(times, locations, lengths)
        w, b, inv_var, tn, dt_y, centers = s["w"], s["b"], s["inv_var"], s["tn"], s["dt_y"], s["centers"]
        t_ti = (tn + dt_y[:, None]).clamp(min=1e-6)
        log_v = torch.log(w) - b * t_ti  # exactly log_s_intensity's weights (b, not b_safe)
        log_vn = log_v - torch.logsumexp(log_v, -1, keepdim=True)
        B = w.shape[0]
        pts_T = grid_centres(self.box, n)
        pts = torch.as_tensor((pts_T - self.loc_min) / self.loc_range, dtype=w.dtype, device=w.device)
        log_norm = 0.5 * torch.log(inv_var.prod(-1).clamp(min=1e-12)) - math.log(2 * math.pi)
        probs = np.empty((B, n * n))
        mass = np.empty(B)
        cell_area_T = (self.box[1] - self.box[0]) * (self.box[3] - self.box[2]) / (n * n)
        log_jac = float(np.log(self.loc_range).sum())
        step = max(1, (1 << 22) // (n * n * w.shape[1]))
        for r0 in range(0, B, step):
            sl = slice(r0, min(B, r0 + step))
            d = pts[None, :, None, :] - centers[sl][:, None, :, :]
            lg = log_norm[sl][:, None, :] - 0.5 * (d * d * inv_var[sl][:, None, :, :]).sum(-1)
            logp = torch.logsumexp(log_vn[sl][:, None, :] + lg, -1)
            m = to64(torch.exp(logp)) * math.exp(-log_jac) * cell_area_T
            mass[sl] = m.sum(1)
            probs[sl] = m / m.sum(1, keepdims=True)
        idx = torch.arange(B, device=times.device)
        obs = to64(locations[idx, lengths - 1])
        return probs, cell_index(obs, self.box, n), mass


class AutoSTPPCalib(_Base):
    family = "auto_stpp"
    spatial_kind = "prodnet_separable_closed_form"

    def __init__(self, runner, device, box):
        super().__init__(runner, device, box)
        self.sm, self.em = self.model.state_model, self.model.event_model
        self.kernel = self.em.kernel
        if bool(self.sm.input_normalized_flag.item() > 0.5):
            raise NotImplementedError("auto_stpp: input_normalized=True not supported")
        self.dt_min, self.dt_range = float(self.sm.paper_dt_min.item()), float(self.sm.paper_dt_range.item())
        self.loc_min = to64(self.sm.paper_loc_min)
        self.loc_range = to64(self.sm.paper_loc_range)
        self.L = int(self.em.lookback)
        # AutoSTPP's likelihood integrates over the unit paper square = the training bounding box
        self.box = (float(self.loc_min[0]), float(self.loc_min[0] + self.loc_range[0]),
                    float(self.loc_min[1]), float(self.loc_min[1] + self.loc_range[1]))

    @torch.no_grad()
    def _state(self, times, locations, lengths):
        p = self.sm.encode_history(times=times, locations=locations, lengths=lengths).payload
        B = int(lengths.shape[0])
        x, y = p["paper_st_x"], p["paper_st_y"]
        if x.shape[0] != B or not bool((p["paper_window_counts"] == 1).all()):
            raise ValueError("expects exactly one AutoSTPP window per batch row")
        t_x_cum = torch.cumsum(x[..., -1], dim=-1)
        return dict(s_x=x[..., :2], a=t_x_cum[:, -1:] - t_x_cum, dt_y=y[:, 0, -1],
                    bg=self.em._background_rate(device=x.device, dtype=x.dtype))

    @torch.no_grad()
    def terms(self, times, locations, lengths, tau_model, output=None):
        s = self._state(times, locations, lengths)
        s_x, a, dt_y, bg = s["s_x"], s["a"], s["dt_y"], s["bg"]
        B, L = a.shape
        d_lambda = self.kernel.int_lamb_stpp(
            s_x.reshape(-1, 2), a.reshape(-1, 1), (a + dt_y[:, None]).reshape(-1, 1)
        ).view(B, L).sum(-1) + dt_y * bg
        if output is not None:
            lam_t = output["lamb_t"][:, 0]
            check = (output["lamb_ints"][:, 0] - d_lambda).abs().max().item()
        else:
            lam_t = self.em._temporal_intensity_exact(s_x, a + dt_y[:, None], device=a.device, dtype=a.dtype)
            check = float("nan")
        out = {"d_lambda": d_lambda, "log_lam_T": torch.log(lam_t) - math.log(self.dt_range), "check_lamb_ints": check}
        self._cache = dict(s_x=s_x, a=a, bg=bg)
        if tau_model is not None:
            tau_hat = ((tau_model - self.dt_min) / self.dt_range).clamp(min=0.0)
            G = tau_hat.shape[0]
            grid = torch.empty(B, G, device=a.device, dtype=a.dtype)
            step = max(1, (1 << 16) // (B * L))
            for g0 in range(0, G, step):
                th = tau_hat[g0:g0 + step]
                g = th.shape[0]
                ss = s_x[:, :, None, :].expand(B, L, g, 2).reshape(-1, 2)
                aa = a[:, :, None].expand(B, L, g).reshape(-1, 1)
                bb = (a[:, :, None] + th[None, None, :]).reshape(-1, 1)
                grid[:, g0:g0 + g] = self.kernel.int_lamb_stpp(ss, aa, bb).view(B, L, g).sum(1) + bg * th[None, :]
            out["lam_grid"] = grid
        return out

    @torch.no_grad()
    def cumhaz_rows(self, tau):
        c = self._cache
        s_x, a, bg = c["s_x"], c["a"], c["bg"]
        B, L = a.shape
        K = tau.shape[1]
        th = ((tau - self.dt_min) / self.dt_range).clamp(min=0.0)
        ss = s_x[:, :, None, :].expand(B, L, K, 2).reshape(-1, 2)
        aa = a[:, :, None].expand(B, L, K).reshape(-1, 1)
        bb = (a[:, :, None] + th[:, None, :]).reshape(-1, 1)
        return self.kernel.int_lamb_stpp(ss, aa, bb).view(B, L, K).sum(1) + bg * th

    @staticmethod
    def _axis_grad(net, x):
        _, g = net.forward_with_grad(x.reshape(-1, 1))
        return g.reshape(x.shape)

    @torch.no_grad()
    def spatial_cells(self, times, locations, lengths, n):
        s = self._state(times, locations, lengths)
        s_x, a, dt_y, bg = s["s_x"], s["a"], s["dt_y"], s["bg"]
        B, L = a.shape
        c1 = (torch.arange(n, device=a.device, dtype=a.dtype) + 0.5) / n  # unit paper square
        tt = a + dt_y[:, None]
        probs = np.empty((B, n * n))
        step = max(1, (1 << 18) // (L * n))
        for r0 in range(0, B, step):
            sl = slice(r0, min(B, r0 + step))
            b = sl.stop - sl.start
            dx = c1[None, None, :] - s_x[sl][:, :, None, 0]
            dy = c1[None, None, :] - s_x[sl][:, :, None, 1]
            lam = torch.zeros(b, n, n, device=a.device, dtype=a.dtype)
            for sign, sp in ((3.0, self.kernel.M), (-3.0, self.kernel.L)):
                for net in sp.nets:
                    gx = self._axis_grad(net.x_net, dx)
                    gy = self._axis_grad(net.y_net, dy)
                    gt = self._axis_grad(net.t_net, tt[sl]) * sign
                    lam = lam + torch.einsum("rl,rlx,rly->rxy", gt, gx, gy)
            lam = (lam + bg).clamp(min=0).reshape(b, n * n)
            probs[sl] = to64(lam / lam.sum(-1, keepdim=True))
        idx = torch.arange(B, device=times.device)
        obs = to64(locations[idx, lengths - 1])
        return probs, cell_index(obs, self.box, n), np.full(B, np.nan)


class NeuralCalib(_Base):
    family = "neural_stpp"

    def __init__(self, runner, device, box):
        super().__init__(runner, device, box)
        self.sm = self.model.state_model
        self.tc = self.sm.temporal_core
        self.dec = self.model.event_model.spatial_decoder
        tr = self.sm._input_transform
        if (tr is not None and getattr(tr, "normalize_time", False)) or self.sm.normalize_time_inputs:
            raise NotImplementedError("neural family: time normalisation not supported")
        if tr is not None and getattr(tr, "normalize_space", False):
            self.loc_mean, self.loc_std = np.asarray(tr.loc_mean, float), np.asarray(tr.loc_std, float)
        else:
            self.loc_mean, self.loc_std = np.zeros(2), np.ones(2)
        self.spatial_type = type(self.dec).__name__
        self.spatial_kind = "gmm_per_dim_closed_form" if "GMM" in self.spatial_type else "cnf_deferred"
        s = self.tc.ode_solver
        self.solver_info = {"method": str(s.method), "atol": float(s.atol), "rtol": float(s.rtol)}
        self._cache = None

    @torch.no_grad()
    def terms(self, times, locations, lengths, tau_model, output=None):
        from seahorse.models.state_models.neural import _repair_strictly_increasing_times

        B = int(lengths.shape[0])
        max_len = int(lengths.max().item())
        tr = self.sm._input_transform
        if tr is not None:
            times_raw = tr.inverse_times(times.to(torch.float64), lengths).to(times.dtype)
            locs = tr.forward_locations(locations, lengths)
        else:
            times_raw, locs = times, locations
        times_raw = _repair_strictly_increasing_times(times_raw[:, :max_len], lengths)
        mask = torch.arange(max_len, device=times.device).unsqueeze(0) < lengths.unsqueeze(1)
        intens, _lf, prejump, det = self.tc.integrate_lambda(
            times_raw, locs[:, :max_len, :], mask, t0=torch.zeros(B, device=times.device, dtype=times.dtype),
            t1=None, nlinspace=1, return_details=True)
        cum = det["cumulative_lambdas"]
        idx = torch.arange(B, device=times.device)
        k = lengths - 1
        d_lambda = cum[idx, k] - cum[idx, k - 1]
        lam = intens[idx, k]
        out = {"d_lambda": d_lambda, "log_lam_T": torch.log(lam + 1e-8)}
        if output is not None:
            out["check_temporal_nll"] = (
                output["temporal_nll_matrix"][idx, k] - (-torch.log(lam + 1e-8) + d_lambda)
            ).abs().max().item()
        h_pre = prejump[idx, k]  # h(t_{k-1}^-): prejump[:, j] = h(t_{j-1}^-)
        t_prev = times_raw[idx, k - 1]
        cond = locs[idx, k - 1] if self.tc.cond else None
        h_post = self.tc.hidden_state_dynamics.update_state(t_prev, h_pre, cond=cond)
        self._cache = dict(prejump=prejump, locs=locs, times_raw=times_raw, idx=idx, k=k, h_post=h_post, t_prev=t_prev)
        if tau_model is not None:
            out["lam_grid"] = self._grid(h_post, t_prev, tau_model)
        return out

    @torch.no_grad()
    def cumhaz_rows(self, tau):
        c = self._cache
        B, K = tau.shape
        cols = [
            self.tc.integrate_hidden(c["h_post"], c["t_prev"], c["t_prev"] + tau[:, k].clamp(min=1e-6))[1]
            for k in range(K)
        ]
        return torch.stack(cols, 1)

    @torch.no_grad()
    def _grid(self, h0, t0, tau):
        from torchdiffeq import odeint

        solver = self.tc.ode_solver
        tau_max = float(tau[-1].item())
        s_grid = (tau / tau_max).to(torch.float32)
        R = h0.shape[0]
        state = (t0, t0 + tau_max, torch.zeros(1, device=h0.device, dtype=h0.dtype),
                 torch.zeros(R, device=h0.device, dtype=h0.dtype), h0)
        sol = odeint(solver, state, s_grid, rtol=solver.rtol, atol=solver.atol, method="dopri5",
                     options={"dtype": torch.float64})
        return sol[3].T

    @torch.no_grad()
    def spatial_cells(self, times, locations, lengths, n):
        if self.spatial_kind != "gmm_per_dim_closed_form":
            return None
        from seahorse.models.spatial_models.njsde import _gaussian_loglik

        c = self._cache
        if c is None:
            raise RuntimeError("terms() must run before spatial_cells()")
        h = c["prejump"][c["idx"], c["k"] + 1]  # h(t_k^-), the state the NLL's spatial term uses
        nm = self.dec.n_mixtures
        params = self.dec._params_from_aux(self.dec._select_aux(h)).reshape(-1, 2, 3, nm)
        logits, means, log_std = params[:, :, 0, :], params[:, :, 1, :], params[:, :, 2, :]
        logw = logits - torch.logsumexp(logits, -1, keepdim=True)
        x0, x1, y0, y1 = self.box
        ax = [
            torch.as_tensor(((x0 + (np.arange(n) + 0.5) / n * (x1 - x0)) - self.loc_mean[0]) / self.loc_std[0],
                            dtype=h.dtype, device=h.device),
            torch.as_tensor(((y0 + (np.arange(n) + 0.5) / n * (y1 - y0)) - self.loc_mean[1]) / self.loc_std[1],
                            dtype=h.dtype, device=h.device),
        ]
        lp = []
        for d in range(2):
            z = ax[d][None, :, None]
            lp.append(torch.logsumexp(
                logw[:, d, None, :] + _gaussian_loglik(z, means[:, d, None, :], log_std[:, d, None, :]), -1))
        logp = lp[0][:, :, None] + lp[1][:, None, :]
        B = h.shape[0]
        cell_area_T = (x1 - x0) * (y1 - y0) / (n * n)
        m = to64(torch.exp(logp)).reshape(B, n * n) / float(np.prod(self.loc_std)) * cell_area_T
        mass = m.sum(1)
        probs = m / mass[:, None]
        obs = to64(locations[c["idx"], c["k"]])
        return probs, cell_index(obs, self.box, n), mass


class FactorizedCalib(_Base):
    """rmtpp_gmm / thp_gmm / poisson_gmm / hawkes_gmm / selfcorrecting_gmm."""

    family = "factorized"
    spatial_kind = "gmm_history_closed_form"

    def __init__(self, runner, device, box):
        super().__init__(runner, device, box)
        self.sm, self.em = self.model.state_model, self.model.event_model
        self.tm, self.spm = self.em.temporal_model, self.em.spatial_model
        self.kind = type(self.tm).__name__
        if self.em.t1 is not None:
            raise NotImplementedError("fixed t1 not supported")
        tr = self.sm._input_transform
        self.time_std = float(tr.time_std) if (tr is not None and getattr(tr, "normalize_time", False)) else 1.0
        self.tau_internal_scale = 1.0 / self.time_std
        if tr is not None and getattr(tr, "normalize_space", False):
            self.loc_mean, self.loc_std = np.asarray(tr.loc_mean, float), np.asarray(tr.loc_std, float)
        else:
            self.loc_mean, self.loc_std = np.zeros(2), np.ones(2)
        self._cache = None

    @torch.no_grad()
    def _state(self, times, locations, lengths):
        p = self.sm.encode_history(times=times, locations=locations, lengths=lengths).payload
        tn, ln = p["times"].to(self.device), p["locations"].to(self.device)
        ts = (tn - tn[:, :1]).clamp(min=0.0)  # sequence-relative shift as in event_models/factorized.py
        B, T = ts.shape
        mask = (torch.arange(T, device=ts.device)[None] < lengths[:, None]).float()
        idx = torch.arange(B, device=ts.device)
        t1 = ts[idx, (lengths - 1).clamp(min=0)]
        t0 = torch.full((B,), float(self.em.t0), device=ts.device, dtype=ts.dtype)
        return dict(ts=ts, locs=ln, mask=mask, t0=t0, t1=t1, idx=idx, k=lengths - 1)

    @torch.no_grad()
    def terms(self, times, locations, lengths, tau_model, output=None):
        s = self._state(times, locations, lengths)
        ts, locs, mask, idx, k = s["ts"], s["locs"], s["mask"], s["idx"], s["k"]
        B, T = ts.shape
        self._cache = s
        tll = self.tm.event_logprob_matrix(ts, locs, mask, s["t0"], s["t1"])  # log lam - Lambda (NLL path)
        dt_k = ts[idx, k] - ts[idx, k - 1]
        if self.kind.startswith("RMTPP"):
            hidden, _ = self.tm._encode(ts, (k + 1).cpu())
            ab = self.tm.mlp(hidden[idx, k])
            log_lam = torch.log(F.softplus(ab[:, 0]) + 1e-20) - F.softplus(ab[:, 1]).clamp(min=1e-6) * dt_k
        elif self.kind.startswith("THP"):
            hidden, _ = self.tm._encode(ts, k + 1)
            f = self.tm.f(hidden[idx, k]).squeeze(-1)
            log_lam = torch.log(F.softplus(f + self.tm.beta * dt_k) + 1e-20)
        else:
            hist_mask = (torch.arange(T, device=ts.device)[None] < k[:, None]).float()
            log_lam = torch.log(self.tm.intensity_at(ts[idx, k], ts, hist_mask) + 1e-20)
        d_lambda = log_lam - tll[idx, k]
        out = {"d_lambda": d_lambda, "log_lam_T": log_lam - math.log(self.time_std)}
        if output is not None:
            out["check_temporal_nll"] = (output["temporal_nll_matrix"][idx, k] + tll[idx, k]).abs().max().item()
        if tau_model is not None:
            tau = (tau_model * self.tau_internal_scale).to(ts.dtype)
            out["lam_grid"] = self._grid(s, tau)
        return out

    @torch.no_grad()
    def _grid(self, s, tau):
        """``Lambda`` on ``tau``: ``(G,)`` shared grid -> ``(B, G)``; ``(B, K)`` per-row probes -> ``(B, K)``."""
        ts, idx, k = s["ts"], s["idx"], s["k"]
        lengths = s["k"] + 1
        B, T = ts.shape
        t_prev = ts[idx, k - 1]
        tau_b = tau[None, :] if tau.ndim == 1 else tau
        kind = self.kind
        if kind.startswith("HomogeneousPoisson"):
            lamb = F.softplus(self.tm.log_lambda).squeeze()
            return (lamb * tau_b).expand(B, -1).clone()
        if kind.startswith("HawkesProcess"):
            mu = F.softplus(self.tm.log_mu).squeeze()
            alpha = F.softplus(self.tm.log_alpha).squeeze()
            beta = F.softplus(self.tm.log_beta).squeeze()
            hist = ts[:, :-1]
            a = (t_prev[:, None] - hist).clamp(min=0.0)
            e0 = torch.exp(-beta * a)
            e1 = torch.exp(-beta * (a[:, :, None] + tau_b[:, None, :]))
            return mu * tau_b + (alpha / (beta + 1e-20)) * (e0[:, :, None] - e1).sum(1)
        if kind.startswith("SelfCorrecting"):
            mu, beta = F.softplus(self.tm.log_mu).squeeze(), F.softplus(self.tm.log_beta).squeeze()
            n_before = (lengths - 1).to(ts.dtype)
            log_comp = (mu * t_prev[:, None] - beta * n_before[:, None] - torch.log(mu + 1e-20)
                        + torch.log(torch.expm1(mu * tau_b) + 1e-20))
            return torch.exp(log_comp.clamp(max=80.0))
        if kind.startswith("RMTPP"):
            if "ab" not in s:
                hidden, _ = self.tm._encode(ts, lengths.cpu())
                s["ab"] = self.tm.mlp(hidden[idx, k])
            ab = s["ab"]
            a_ = F.softplus(ab[:, 0]).double()
            b_ = F.softplus(ab[:, 1]).clamp(min=1e-6).double()
            return ((a_ / b_)[:, None] * (-torch.expm1(-b_[:, None] * tau_b.double()))).to(ts.dtype)
        if kind.startswith("THP"):
            if "f" not in s:
                hidden, _ = self.tm._encode(ts, lengths)
                s["f"] = self.tm.f(hidden[idx, k]).squeeze(-1)
            f = s["f"]
            S = int(self.tm.MC_SAMPLES)
            nodes = (torch.arange(S, device=ts.device, dtype=ts.dtype) + 0.5) / S  # eval-mode midpoint rule
            val = F.softplus(f[:, None, None] + self.tm.beta * tau_b[:, :, None] * nodes[None, None, :])
            return val.mean(-1) * tau_b
        raise NotImplementedError(kind)

    @torch.no_grad()
    def cumhaz_rows(self, tau):
        return self._grid(self._cache, (tau * self.tau_internal_scale).to(self._cache["ts"].dtype))

    @torch.no_grad()
    def spatial_cells(self, times, locations, lengths, n):
        s = self._cache
        ts, locs, idx, k = s["ts"], s["locs"], s["idx"], s["k"]
        B, T = ts.shape
        pts_T = grid_centres(self.box, n)
        pts = torch.as_tensor((pts_T - self.loc_mean) / self.loc_std, dtype=ts.dtype, device=ts.device)
        n2 = pts.shape[0]
        hist_mask = (torch.arange(T, device=ts.device)[None] < k[:, None]).float()
        probs = np.empty((B, n2))
        mass = np.empty(B)
        cell_area_T = (self.box[1] - self.box[0]) * (self.box[3] - self.box[2]) / (n * n)
        step = max(1, (1 << 22) // (n2 * T))
        for r0 in range(0, B, step):
            sl = slice(r0, min(B, r0 + step))
            b = sl.stop - sl.start
            tq = ts[idx[sl], k[sl]][:, None].expand(b, n2).reshape(-1)
            lp = self.spm.log_spatial_density_at(
                tq, pts[None].expand(b, n2, 2).reshape(-1, 2),
                ts[sl][:, None, :].expand(b, n2, T).reshape(-1, T),
                locs[sl][:, None, :, :].expand(b, n2, T, 2).reshape(-1, T, 2),
                hist_mask[sl][:, None, :].expand(b, n2, T).reshape(-1, T),
            ).reshape(b, n2)
            m = to64(torch.exp(lp)) / float(np.prod(self.loc_std)) * cell_area_T
            mass[sl] = m.sum(1)
            probs[sl] = m / m.sum(1, keepdims=True)
        obs = to64(locations[idx, k])
        return probs, cell_index(obs, self.box, n), mass


class ParametricKernelCalib(_Base):
    """static_hawkes / gaussian_poisson (parametric kernels, identity state model)."""

    family = "parametric_kernels"
    spatial_kind = "background_plus_displacement_closed_form"

    def __init__(self, runner, device, box):
        super().__init__(runner, device, box)
        self.em = self.model.event_model
        self._cache = None

    @torch.no_grad()
    def terms(self, times, locations, lengths, tau_model, output=None):
        em = self.em
        B, T = times.shape
        idx = torch.arange(B, device=times.device)
        k = lengths - 1
        t_k, t_prev = times[idx, k], times[idx, k - 1]
        hist_t = times[:, :-1]
        hist_s = locations[:, :-1]
        ages = (t_k[:, None] - hist_t).clamp_min(1.0e-7)
        prev_ages = (t_prev[:, None] - hist_t).clamp_min(0.0)
        mu, eta, kappa = em.mu, em.eta, em.killing_rate
        log_eta = torch.where(eta > 0.0, torch.log(eta), eta.new_full((), -torch.inf))
        log_trig = log_eta + torch.log(kappa.clamp_min(1e-12)) - kappa * ages
        log_lam_T = torch.logsumexp(torch.cat([torch.log(mu).expand(B, 1), log_trig], -1), -1)
        d_lambda = (
            mu * (t_k - t_prev).clamp_min(0.0)
            + (eta * (torch.exp(-kappa * prev_ages) - torch.exp(-kappa * ages))).sum(-1)
        )
        out = {"d_lambda": d_lambda, "log_lam_T": log_lam_T}
        if output is not None:
            out["check_temporal_nll"] = (
                output["temporal_nll_matrix"][idx, k] - (-log_lam_T + d_lambda)
            ).abs().max().item()
        a = (t_prev[:, None] - hist_t).clamp_min(0.0)
        self._cache = dict(idx=idx, k=k, t_k=t_k, hist_t=hist_t, hist_s=hist_s, ages=ages, log_trig=log_trig,
                           log_lam_T=log_lam_T, a=a, mu=mu, eta=eta, kappa=kappa)
        if tau_model is not None:
            out["lam_grid"] = mu * tau_model[None, :] + (
                eta * (torch.exp(-kappa * a)[:, :, None] - torch.exp(-kappa * (a[:, :, None] + tau_model[None, None, :])))
            ).sum(1)
        return out

    @torch.no_grad()
    def cumhaz_rows(self, tau):
        c = self._cache
        a = c["a"]
        return c["mu"] * tau + (
            c["eta"] * (torch.exp(-c["kappa"] * a)[:, :, None] - torch.exp(-c["kappa"] * (a[:, :, None] + tau[:, None, :])))
        ).sum(1)

    @torch.no_grad()
    def spatial_cells(self, times, locations, lengths, n):
        em, c = self.em, self._cache
        B = times.shape[0]
        pts = torch.as_tensor(grid_centres(self.box, n), dtype=times.dtype, device=times.device)
        n2 = pts.shape[0]
        log_bg = torch.log(em.mu) + em._diagonal_gaussian_log_density(
            pts[None], em.background_mean, em.background_variance
        ).expand(B, n2)
        probs = np.empty((B, n2))
        mass = np.empty(B)
        cell_area = (self.box[1] - self.box[0]) * (self.box[3] - self.box[2]) / (n * n)
        P = c["hist_s"].shape[1]
        step = max(1, (1 << 20) // (n2 * P))
        for r0 in range(0, B, step):
            sl = slice(r0, min(B, r0 + step))
            b = sl.stop - sl.start
            disp = pts[None, :, None, :] - c["hist_s"][sl][:, None, :, :]
            ages = c["ages"][sl][:, None, :].expand(b, n2, P)
            ltd = em.transition_log_density(disp, ages)
            lj = torch.logsumexp(torch.cat([log_bg[sl][:, :, None], c["log_trig"][sl][:, None, :] + ltd], -1), -1)
            m = to64(torch.exp(lj - c["log_lam_T"][sl][:, None])) * cell_area
            mass[sl] = m.sum(1)
            probs[sl] = m / m.sum(1, keepdims=True)
        obs = to64(locations[c["idx"], c["k"]])
        return probs, cell_index(obs, self.box, n), mass


class NSMPPCalib(_Base):
    family = "nsmpp"
    spatial_kind = "kernel_intensity_grid"

    def __init__(self, runner, device, box):
        super().__init__(runner, device, box)
        self.sm, self.em = self.model.state_model, self.model.event_model
        smin, smax = to64(self.em.support_space_min), to64(self.em.support_space_max)
        self.box = (float(smin[0]), float(smax[0]), float(smin[1]), float(smax[1]))
        self._cache = None

    def _lam(self, q, counts, ev, valid):
        ks = self.em.kernel.sum_over_prefix_history(q, counts, ev, valid)
        return F.softplus(ks + self.em.raw_mu) + self.em.intensity_eps

    @torch.no_grad()
    def terms(self, times, locations, lengths, tau_model, output=None):
        p = self.sm.encode_history(times=times, locations=locations, lengths=lengths).payload
        ev, valid = p["event_vectors"].to(self.device), p["event_mask"].to(self.device).bool()
        B = ev.shape[0]
        idx = torch.arange(B, device=ev.device)
        k = lengths - 1
        lam = self.em._event_intensities(ev, valid)[idx, k]
        d_lambda = self.em._interval_compensators(ev, valid)[idx, k]
        out = {"d_lambda": d_lambda, "log_lam_T": torch.log(lam) * float("nan")}  # joint intensity only
        if output is not None:
            out["check_nll_matrix"] = (output["nll_matrix"][idx, k] - (-torch.log(lam) + d_lambda)).abs().max().item()
        self._cache = dict(ev=ev, valid=valid, idx=idx, k=k)
        if tau_model is not None:
            nodes = self.em.support_spatial_grid.to(ev)
            w = float(self.em.spatial_cell_area.item())
            M = nodes.shape[0]
            G = tau_model.shape[0]
            t_prev = ev[idx, k - 1, 0]
            lg = torch.empty(B, G, device=ev.device, dtype=ev.dtype)
            step = max(1, (1 << 20) // (B * M))
            for g0 in range(0, G, step):
                tg = tau_model[g0:g0 + step]
                g = tg.shape[0]
                qt = (t_prev[:, None] + tg[None, :])[:, :, None].expand(B, g, M).reshape(B, g * M, 1)
                qs = nodes[None, None].expand(B, g, M, 2).reshape(B, g * M, 2)
                counts = k[:, None].expand(B, g * M)
                lg[:, g0:g0 + g] = self._lam(torch.cat([qt, qs], -1), counts, ev, valid).view(B, g, M).sum(-1) * w
            dt = tau_model[1:] - tau_model[:-1]
            out["lam_grid"] = torch.cat([
                torch.zeros(B, 1, device=ev.device, dtype=ev.dtype),
                torch.cumsum(0.5 * (lg[:, 1:] + lg[:, :-1]) * dt[None, :], 1),
            ], 1)
        return out

    @torch.no_grad()
    def cumhaz_rows(self, tau, Q=16):
        """``tau (B, K)``: composite midpoint rule (``Q`` time nodes per probe) of the NLL's spatial node rule."""
        c = self._cache
        ev, valid, idx, k = c["ev"], c["valid"], c["idx"], c["k"]
        B, K = tau.shape
        nodes = self.em.support_spatial_grid.to(ev)
        M = nodes.shape[0]
        w = float(self.em.spatial_cell_area.item())
        t_prev = ev[idx, k - 1, 0]
        frac = (torch.arange(Q, device=ev.device, dtype=ev.dtype) + 0.5) / Q
        qt = (t_prev[:, None, None] + tau[:, :, None] * frac[None, None, :])[:, :, :, None].expand(B, K, Q, M).reshape(B, K * Q * M, 1)
        qs = nodes[None, None, None].expand(B, K, Q, M, 2).reshape(B, K * Q * M, 2)
        lam = self._lam(torch.cat([qt, qs], -1), k[:, None].expand(B, K * Q * M), ev, valid).view(B, K, Q, M).sum(-1) * w
        return lam.mean(-1) * tau

    @torch.no_grad()
    def spatial_cells(self, times, locations, lengths, n):
        c = self._cache
        ev, valid, idx, k = c["ev"], c["valid"], c["idx"], c["k"]
        B = ev.shape[0]
        pts = torch.as_tensor(grid_centres(self.box, n), dtype=ev.dtype, device=ev.device)
        n2 = pts.shape[0]
        probs = np.empty((B, n2))
        step = max(1, (1 << 20) // n2)
        for r0 in range(0, B, step):
            sl = slice(r0, min(B, r0 + step))
            b = sl.stop - sl.start
            qt = ev[idx[sl], k[sl], 0][:, None, None].expand(b, n2, 1)
            lam = self._lam(torch.cat([qt, pts[None].expand(b, n2, 2)], -1), k[sl][:, None].expand(b, n2), ev[sl], valid[sl])
            m = to64(lam)
            probs[sl] = m / m.sum(1, keepdims=True)
        obs = to64(locations[idx, k])
        return probs, cell_index(obs, self.box, n), np.full(B, np.nan)


class StatisticalCalib:
    """NumPy adapters for homogeneous/spatial/seasonal_poisson and hawkes_exp_gaussian."""

    family = "statistical"

    def __init__(self, runner, dataset):
        from seahorse.models.statistical.core import area, calendar_offset, contains, forecast_origin, history_arrays

        self.core = dict(calendar_offset=calendar_offset, forecast_origin=forecast_origin,
                         history_arrays=history_arrays, area=area, contains=contains)
        self.model = runner.model
        self.kind = type(self.model).__name__
        self.dataset = dataset
        self.support = tuple(float(v) for v in self.model.support)
        self.box = self.support
        if self.kind.startswith("Poisson"):
            self.spatial_kind = "model_cells_piecewise_constant"
        elif self.kind.startswith("Hawkes"):
            if (self.model.temporal, self.model.spatial) != ("exponential", "gaussian"):
                raise NotImplementedError("only hawkes_exp_gaussian is supported")
            from seahorse.models.statistical.hawkes import _rectangular_gaussian_mass

            self.gmass = _rectangular_gaussian_mass
            self.renorm = bool(self.model._source_model().components.renormalises)
            self.spatial_kind = "gaussian_mixture_plus_uniform_closed_form"
        else:
            raise NotImplementedError(self.kind)

    def contexts(self, indices):
        from seahorse.experiments.campaign_statistical import raw_context

        rows = [raw_context(self.dataset, int(i)) for i in indices]
        H = len(rows[0]["history"]["times"])
        times = np.stack([np.asarray(r["history"]["times"], float) for r in rows])
        locs = np.stack([np.asarray(r["history"]["locations"], float) for r in rows])
        origin = np.array([self.core["forecast_origin"](r["history"], np.asarray(r["history"]["times"], float))
                           for r in rows])
        offset = np.array([self.core["calendar_offset"](r["history"]) for r in rows])
        nt = np.array([float(r["next_time"]) for r in rows])
        nl = np.stack([np.asarray(r["next_location"], float).reshape(2) for r in rows])
        meta = [r["observation_metadata"] for r in rows]
        return dict(times=times, locs=locs, origin=origin, offset=offset, next_time=nt, next_loc=nl, meta=meta, H=H)

    def terms(self, ctx, tau_raw):
        return self.poisson_terms(ctx, tau_raw) if self.kind.startswith("Poisson") else self.hawkes_terms(ctx, tau_raw)

    def spatial_cells(self, ctx, n):
        return self.poisson_cells(ctx) if self.kind.startswith("Poisson") else self.hawkes_cells(ctx, n)

    # --- Poisson ------------------------------------------------------------------
    def _rate_primitive(self, t):
        m = self.model
        t = np.asarray(t, dtype=np.float64)
        if m.period is None:
            return float(m.rates[0]) * t
        width = m._phase_width
        edges = np.arange(m._n_phase) * width
        cycles, rem = np.divmod(t - m.phase_origin, m.period)
        expo = cycles[..., None] * width + np.clip(rem[..., None] - edges, 0.0, width)
        return expo @ np.asarray(m.rates, float)

    def poisson_terms(self, ctx, tau_raw):
        m = self.model
        o = ctx["origin"] + ctx["offset"]
        y = ctx["next_time"] - ctx["origin"]
        lam = np.array([m._intensity(t + off, s) for t, off, s in zip(ctx["next_time"], ctx["offset"], ctx["next_loc"])])
        d_lambda = self._rate_primitive(o + y) - self._rate_primitive(o)
        ref = np.array([m._rate_integral(a, b) for a, b in zip(o, o + y)])
        out = {"d_lambda": d_lambda, "log_lam": np.log(lam), "nll": -np.log(lam) + d_lambda,
               "check_rate_integral": float(np.max(np.abs(d_lambda - ref)))}
        if tau_raw is not None:
            out["lam_grid"] = self._rate_primitive(o[:, None] + tau_raw[None, :]) - self._rate_primitive(o)[:, None]
        return out

    def poisson_cells(self, ctx):
        from seahorse.models.statistical.poisson import _cell_indices

        m = self.model
        if m.grid[0] * m.grid[1] == 1:
            return None  # single spatial cell: HPD-PIT is V ~ U(0,1) by construction
        B = ctx["next_loc"].shape[0]
        probs = np.broadcast_to(np.asarray(m.probabilities, float)[None, :], (B, m.grid[0] * m.grid[1])).copy()
        inside = self.core["contains"](self.support, ctx["next_loc"])
        ix, iy = _cell_indices(ctx["next_loc"], self.support, m.grid)
        cell = np.where(inside, ix * m.grid[1] + iy, -1)
        return probs, cell, np.ones(B), f"model cells {m.grid[0]}x{m.grid[1]}"

    # --- Hawkes exp x Gaussian ---------------------------------------------------------
    def hawkes_pieces(self, ctx):
        mu, alpha, beta, sigma = [float(v) for v in self.model.parameters]
        times, locs = ctx["times"], ctx["locs"]
        masses = self.gmass(locs.reshape(-1, 2), sigma, self.support).reshape(times.shape)
        cm = np.ones_like(masses) if self.renorm else masses
        return mu, alpha, beta, sigma, masses, cm

    def hawkes_terms(self, ctx, tau_raw):
        mu, alpha, beta, sigma, masses, cm = self.hawkes_pieces(ctx)
        area = self.core["area"](self.support)
        times, locs, origin, tt, tl = ctx["times"], ctx["locs"], ctx["origin"], ctx["next_time"], ctx["next_loc"]
        lag = tt[:, None] - times
        temporal = alpha * np.exp(-beta * lag)
        dist = np.linalg.norm(locs - tl[:, None, :], axis=2)
        spatial = np.exp(-0.5 * (dist / sigma) ** 2) / (2.0 * math.pi * sigma ** 2)
        if self.renorm:
            spatial = spatial / masses
        lam = mu + np.sum(temporal * spatial, axis=1)
        d_lambda = mu * area * (tt - origin) + np.sum(
            cm * (alpha / beta) * (np.exp(-beta * (origin[:, None] - times)) - np.exp(-beta * lag)), axis=1)
        lam_T = mu * area + np.sum(temporal * cm, axis=1)
        out = {"d_lambda": d_lambda, "log_lam": np.log(lam), "log_lam_T": np.log(lam_T), "nll": -np.log(lam) + d_lambda}
        if tau_raw is not None:
            a = origin[:, None] - times
            out["lam_grid"] = mu * area * tau_raw[None, :] + np.einsum(
                "bj,bjg->bg", cm * (alpha / beta),
                np.exp(-beta * a)[:, :, None] - np.exp(-beta * (a[:, :, None] + tau_raw[None, None, :])))
        return out

    def cumhaz_rows_np(self, ctx, tau):
        """``Lambda(tau)`` for ``tau (B, K)`` in raw units (closed forms of the statistical presets)."""
        if self.kind.startswith("Poisson"):
            o = ctx["origin"] + ctx["offset"]
            return self._rate_primitive(o[:, None] + tau) - self._rate_primitive(o)[:, None]
        mu, alpha, beta, _sigma, _masses, cm = self.hawkes_pieces(ctx)
        area = self.core["area"](self.support)
        a = ctx["origin"][:, None] - ctx["times"]
        return mu * area * tau + np.einsum(
            "bj,bjk->bk", cm * (alpha / beta), np.exp(-beta * a)[:, :, None] - np.exp(-beta * (a[:, :, None] + tau[:, None, :])))

    def hawkes_cells(self, ctx, n):
        mu, alpha, beta, sigma, masses, cm = self.hawkes_pieces(ctx)
        area = self.core["area"](self.support)
        times, locs, tt = ctx["times"], ctx["locs"], ctx["next_time"]
        pts = grid_centres(self.support, n)
        lag = tt[:, None] - times
        temporal = alpha * np.exp(-beta * lag)
        if self.renorm:
            temporal = temporal / masses
        B, P = times.shape
        probs = np.empty((B, n * n))
        dev = getattr(self, "_grid_device", None)
        if dev is None:
            dev = "cpu"
            if torch.cuda.is_available():
                try:
                    if torch.cuda.mem_get_info()[0] > (1 << 30):
                        dev = "cuda"
                except Exception:
                    pass
            self._grid_device = dev
        tp = torch.as_tensor(pts, dtype=torch.float32, device=dev)
        tl = torch.as_tensor(locs, dtype=torch.float32, device=dev)
        tw = torch.as_tensor(temporal, dtype=torch.float32, device=dev)
        step = max(1, (1 << 24) // (n * n * P))
        with torch.no_grad():
            for r0 in range(0, B, step):
                sl = slice(r0, min(B, r0 + step))
                d2 = ((tp[None, :, None, :] - tl[sl][:, None, :, :]) ** 2).sum(-1)
                f = mu + torch.einsum("bp,bnp->bn", tw[sl], torch.exp(-0.5 * d2 / sigma ** 2)) / (2 * math.pi * sigma ** 2)
                probs[sl] = (f / f.sum(1, keepdim=True)).double().cpu().numpy()
        inside = self.core["contains"](self.support, ctx["next_loc"])
        cell = np.where(inside, cell_index(ctx["next_loc"], self.support, n), -1)
        mass = mu * area + (temporal * cm).sum(1)  # lambda_T (diagnostic only)
        return probs, cell, mass / mass, f"support grid {n}x{n}"


FAMILIES = {
    "deep_stpp": DeepSTPPCalib,
    "auto_stpp": AutoSTPPCalib,
    "njsde": NeuralCalib,
    "neural_attncnf": NeuralCalib,
    "neural_jumpcnf": NeuralCalib,
    "rmtpp_gmm": FactorizedCalib,
    "thp_gmm": FactorizedCalib,
    "poisson_gmm": FactorizedCalib,
    "hawkes_gmm": FactorizedCalib,
    "selfcorrecting_gmm": FactorizedCalib,
    "static_hawkes": ParametricKernelCalib,
    "gaussian_poisson": ParametricKernelCalib,
    "nsmpp": NSMPPCalib,
}

__all__ = [
    "FAMILIES",
    "AutoSTPPCalib",
    "DeepSTPPCalib",
    "FactorizedCalib",
    "NSMPPCalib",
    "NeuralCalib",
    "ParametricKernelCalib",
    "StatisticalCalib",
]
