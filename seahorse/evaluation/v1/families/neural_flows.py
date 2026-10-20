"""Exact inverse-flow spatial sampling for the Neural-STPP CNF decoders.

Both CNF decoders define ``log p(s | t, H)`` by transporting the query point
*backwards* through a sequence of invertible maps to a Gaussian base. Sampling
inverts those maps (base draw -> reverse ODE solves -> data space). This is the
procedure the SMASH paper (arXiv 2310.16310, App. B.3) describes for the
Neural-STPP baselines ("through their CNF procedure"). The log-density of each
draw is tracked along the inverse path (change of variables), so draws come
with their exact model log-density, which the HPD-PIT needs.

Jump-CNF (``NeuralJumpCNFSpatial.sequence_logprob``), for a query at
``tau_q = t_q + 2`` with history events ``tau_j = t_j + 2`` (j = 0..n-1),
``tau_{-1} = 0`` and intervals ``I_j = [tau_{j-1}, tau_j]`` (``tau_n := tau_q``):

    forward (density):  x --CNF(I_n)--> R_{n-1} --CNF(I_{n-1})--> ... R_0 --CNF(I_0)--> z
    each CNF(I_j) integrates (x, h) with h started at the query's own hidden
    state h_q at the *later* end of I_j (the decoder re-initialises the
    auxiliary state at every interval); R_e is a radial flow conditioned on
    (tau_e, s_e, aux(h after CNF(I_{e+1}))).

    inverse (sampling): z ~ N(z_mean, z_std); for j = 0..n:
        CNF^{-1}(I_j) from tau_{j-1} to tau_j, h started at g_j = Phi_h(h_q; tau_j -> tau_{j-1})
        if j < n: R_j^{-1} with cond (tau_j, s_j, aux(g_{j+1}))   [closed form]

Attn-CNF (``NeuralAttnCNFSpatial.sequence_logprob``): all events of a sequence
are transported *jointly* by one attentive ODE (causal attention), followed by
a per-row base CNF and a conditional Gaussian. Sampling the query row inverts
the joint ODE with the history rows started from their forward end states
(they do not depend on the query row: causal mask), then the base CNF.

Invertibility is checked numerically by ``check_roundtrip``: the log-density
tracked along the inverse path must match the model's own forward
``sequence_logprob`` evaluated at the sample.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import chunked, to_np


def _odeint():
    from torchdiffeq import odeint

    return odeint


class _RowODE(nn.Module):
    """dy/ds = f(t(s), y) * (t1 - t0), with per-row [t0, t1]; s in [0, 1]."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, s, state):
        t0, t1, y = state
        ratio = t1 - t0
        t = s * ratio + t0
        dy = self.f(t, y) * ratio.reshape(-1, *([1] * (y.ndim - 1)))
        return torch.zeros_like(t0), torch.zeros_like(t1), dy


def _solve_rows(f, t0, t1, y0, tol):
    odeint = _odeint()
    s = torch.tensor([0.0, 1.0], device=y0.device, dtype=y0.dtype)
    sol = odeint(_RowODE(f), (t0, t1, y0), s, rtol=tol, atol=tol, method="dopri5",
                 options={"dtype": torch.float64})
    return sol[2][-1]


def _cnf_solve(cnf, t0, t1, x, logpx, tol):
    """Integrate a TimeVariableCNF in either direction, bypassing endpoint repair."""
    odeint = _odeint()
    e = torch.zeros_like(x)[:, : cnf.dim]
    en = torch.zeros(1, device=x.device, dtype=x.dtype)
    jn = torch.zeros(1, device=x.device, dtype=x.dtype)
    s = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)
    sol = odeint(cnf, (t0, t1, e, x, logpx, en, jn), s, rtol=tol, atol=tol,
                 method="dopri5", options={"dtype": torch.float64})
    return sol[3][-1].detach(), sol[4][-1].detach()


def _radial_params(hyper_out: torch.Tensor, nd: int, i: int):
    start = (nd + 2) * i
    z0 = hyper_out[:, start : start + nd]
    log_alpha = hyper_out[:, start + nd : start + nd + 1] - 6.0
    beta_raw = hyper_out[:, start + nd + 1 : start + nd + 2] - 6.0
    alpha = torch.exp(log_alpha)
    beta = -alpha + F.softplus(beta_raw)
    return z0, alpha, beta, log_alpha


def _radial_inverse(flow, y: torch.Tensor, cond: torch.Tensor):
    """Inverse of ``HypernetworkRadialFlow`` (closed form per radial layer).

    Returns ``(x, sum_logdet)`` where ``sum_logdet`` is the forward log-det of
    the whole stack evaluated along the recovered pre-images (what the forward
    density pass subtracts).
    """
    nd = flow.nd
    hyper_out = flow.hypernet(cond)
    x = y
    total = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
    for i in reversed(range(flow.nflows)):
        z0, alpha, beta, _ = _radial_params(hyper_out, nd, i)
        d = x - z0
        rho = torch.norm(d, dim=-1, keepdim=True)
        bq = alpha + beta - rho
        disc = torch.sqrt(bq * bq + 4.0 * rho * alpha)
        # numerically stable positive root of r^2 + bq r - rho alpha = 0
        r = torch.where(bq >= 0, 2.0 * rho * alpha / (bq + disc).clamp_min(1e-30), 0.5 * (disc - bq))
        scale = torch.where(rho > 0, r / rho.clamp_min(1e-30), torch.ones_like(rho))
        x_prev = z0 + d * scale
        hh = 1.0 / (alpha + r)
        logdet = (nd - 1) * torch.log(1.0 + beta * hh) + torch.log(
            1.0 + beta * hh - beta * r / (alpha + r) ** 2
        )
        total = total + logdet.reshape(-1)
        x = x_prev
    return x, total


# ---------------------------------------------------------------------------
# Jump-CNF
# ---------------------------------------------------------------------------


def _jumpcnf_sample_rows(ad, hist_tau, hist_s, n_hist, tau_q, h_q, rng, tol):
    """Sample one location per row.

    hist_tau: (R, Nmax) shifted history times (t + 2), hist_s: (R, Nmax, 2) native locs,
    n_hist: (R,) history lengths, tau_q: (R,), h_q: (R, H) query pre-jump hidden.
    Returns native samples (R, 2) and native log-densities (R,).
    """
    dec = ad.dec
    dev = ad.device
    R = h_q.shape[0]
    dim = dec.spatial_dim
    aux_f = dec._aux_odefunc_ref
    hdyn = lambda t, h: aux_f(t - dec.time_offset, h)  # noqa: E731
    z_mean = dec.z_mean.reshape(-1)
    z_std = torch.exp(dec.z_logstd.reshape(-1))
    zb = torch.as_tensor(rng.standard_normal((R, dim)), device=dev, dtype=torch.float32)
    x = z_mean + z_std * zb
    logpz = (-0.5 * zb * zb - torch.log(z_std) - 0.5 * math.log(2 * math.pi)).sum(-1)
    logp_acc = torch.zeros(R, device=dev)
    n_hist_t = torch.as_tensor(n_hist, device=dev)
    nmax = int(n_hist.max())
    eps = torch.finfo(torch.float32).eps

    # g_0: query hidden evolved back over I_0 = [tau_{-1}=0, tau_0]
    all_idx = torch.arange(R, device=dev)
    j0 = torch.zeros(R, dtype=torch.long, device=dev)
    lo = tau_at_sub(hist_tau, n_hist_t, tau_q, all_idx, j0 - 1)
    hi = tau_at_sub(hist_tau, n_hist_t, tau_q, all_idx, j0)
    g = _solve_rows(hdyn, hi, lo, h_q, tol)
    for stage in range(nmax + 1):
        active = stage <= n_hist_t
        if not bool(active.any()):
            break
        idx = torch.nonzero(active, as_tuple=False).reshape(-1)
        jv = torch.full((idx.shape[0],), stage, device=dev, dtype=torch.long)
        lo_a = tau_at_sub(hist_tau, n_hist_t, tau_q, idx, jv - 1)
        hi_a = tau_at_sub(hist_tau, n_hist_t, tau_q, idx, jv)
        # CNF^{-1}(I_j): from lo to hi with h started at g
        dt = (hi_a - lo_a).abs()
        scale = torch.maximum(torch.maximum(lo_a.abs(), hi_a.abs()), torch.ones_like(lo_a))
        run = dt > eps * scale
        if bool(run.any()):
            ridx = idx[run]
            xin = torch.cat([x[ridx], g[ridx]], dim=1)
            xo, lp = _cnf_solve(dec.cnf, lo_a[run], hi_a[run], xin, torch.zeros(ridx.shape[0], device=dev), tol)
            x[ridx] = xo[:, :dim]
            logp_acc[ridx] = logp_acc[ridx] + lp
        # radial inverse at event `stage` for rows with stage < n_hist
        more = stage < n_hist_t[idx]
        if bool(more.any()):
            midx = idx[more]
            jv2 = torch.full((midx.shape[0],), stage + 1, device=dev, dtype=torch.long)
            lo2 = tau_at_sub(hist_tau, n_hist_t, tau_q, midx, jv2 - 1)
            hi2 = tau_at_sub(hist_tau, n_hist_t, tau_q, midx, jv2)
            g_next = _solve_rows(hdyn, hi2, lo2, h_q[midx], tol)
            obs_t = lo2.reshape(-1, 1)  # = tau_stage
            obs_x = hist_s[midx, stage]
            cond = torch.cat([obs_t, obs_x, dec._select_aux(g_next)], dim=1)
            xp, ld = _radial_inverse(dec.inst_flow, x[midx], cond)
            x[midx] = xp
            logp_acc[midx] = logp_acc[midx] + ld
            g[midx] = g_next
    return x, logpz + logp_acc


def tau_at_sub(hist_tau, n_hist_t, tau_q, idx, j_vec):
    nh = n_hist_t[idx]
    out = torch.zeros(idx.shape[0], device=hist_tau.device)
    valid = (j_vec >= 0) & (j_vec < nh)
    jj = j_vec.clamp(min=0, max=hist_tau.shape[1] - 1)
    out = torch.where(valid, hist_tau[idx].gather(1, jj[:, None])[:, 0], out)
    out = torch.where(j_vec == nh, tau_q[idx], out)
    return out


# ---------------------------------------------------------------------------
# Attn-CNF
# ---------------------------------------------------------------------------


def _attn_inputs(dec, times, locs_norm, aux_full):
    """Rows exactly as ``sequence_logprob`` builds them: [x, aux, t_embed]."""
    t_embed = dec.t_embedding(times) / math.sqrt(dec.t_embedding_dim)
    return torch.cat([locs_norm, aux_full, t_embed], dim=-1), t_embed


def _attn_end_states(ad, prep, chunk: int = 16):
    """Forward-transport all rows of every sequence in ``prep``; cached (B, T, D).

    Rows of one sequence are coupled by causal attention, so each sequence is
    one batch element of a joint solve (shape (T, B, D)), chunked over B.
    """
    c = prep.cache
    if "attn_end_all" in c:
        return c["attn_end_all"]
    dec = ad.dec
    B, T = c["t_raw"].shape
    outs = []
    for sl in chunked(B, chunk):
        times = c["t_raw"][sl]
        x_in, _ = _attn_inputs(dec, times, c["s_norm"][sl], c["h_pre"][sl])
        xt = x_in.transpose(0, 1)  # (T, b, D)
        dec.odefunc.set_shape(xt.shape)
        flat = xt.reshape(-1, xt.shape[-1])
        tf = times.transpose(0, 1).reshape(-1)
        x_end, _ = _cnf_solve(dec.cnf, tf + dec.time_offset, torch.zeros_like(tf) + dec.time_offset,
                              flat, torch.zeros_like(tf), dec.cnf.tol)
        outs.append(x_end.reshape(T, -1, xt.shape[-1]).transpose(0, 1))
    c["attn_end_all"] = torch.cat(outs, 0)
    return c["attn_end_all"]


def _attncnf_sample_group(ad, prep, b_list, k_list, t_q, h_q, rng, tol):
    """Sample one location per row; row r = context k_r of sequence b_r.

    The query of row r sits at position k_r + 1 of a joint system with
    S = max(k)+2 positions; positions > k_r + 1 are causally irrelevant
    dummies, positions <= k_r start from the history rows' forward end states.
    """
    dec = ad.dec
    dev = ad.device
    c = prep.cache
    x_end = _attn_end_states(ad, prep)  # (B, T, D)
    R = len(k_list)
    T = prep.T
    S = int(max(k_list)) + 2
    dim = dec.dim
    kk = torch.as_tensor(np.asarray(k_list), device=dev)
    bb = torch.as_tensor(np.asarray(b_list), device=dev)
    tq = torch.as_tensor(t_q, device=dev, dtype=torch.float32)
    t_emb_q = dec.t_embedding(tq[None])[0] / math.sqrt(dec.t_embedding_dim)  # (R, E)
    cond = torch.cat([dec._select_aux_tail(h_q), t_emb_q], dim=-1)
    zp = dec.base_dist_params(cond)
    z_mean, z_logstd = torch.split(zp, dim, dim=-1)
    eps_n = torch.as_tensor(rng.standard_normal((R, dim)), device=dev, dtype=torch.float32)
    z = z_mean + torch.exp(z_logstd) * eps_n
    logpz = (-0.5 * eps_n * eps_n - z_logstd - 0.5 * math.log(2 * math.pi)).sum(-1)
    zero = torch.zeros(R, device=dev)
    z1, r_base = _cnf_solve(dec.base_cnf, zero, zero + dec.time_offset, z, zero, dec.base_cnf.tol)
    pos = torch.arange(S, device=dev).clamp(max=T - 1)
    rows = x_end[bb[None, :], pos[:, None]].clone()  # (S, R, D)
    pos_times = c["t_raw"][bb[None, :], pos[:, None]].clone()  # (S, R)
    q_pos = kk + 1
    x_q_in, _ = _attn_inputs(dec, tq[None], z1[None], h_q[None])
    ar = torch.arange(R, device=dev)
    rows[q_pos, ar] = x_q_in[0]
    pos_times[q_pos, ar] = tq
    D = rows.shape[-1]
    dec.odefunc.set_shape(rows.shape)
    flat = rows.reshape(S * R, D)
    tf = pos_times.reshape(S * R)
    x_out, r_main = _cnf_solve(dec.cnf, torch.zeros_like(tf) + dec.time_offset, tf + dec.time_offset,
                               flat, torch.zeros_like(tf), tol)
    x_out = x_out.reshape(S, R, D)[q_pos, ar, :dim]
    r_main = r_main.reshape(S, R)[q_pos, ar]
    return x_out, logpz + r_base + r_main


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_cnf(ad, prep, b_idx, k_idx, t_query, rng, *, tol=None, group=128):
    """Exact CNF draws at raw times ``t_query`` for contexts (b, k).

    Returns raw locations (R, 2) and raw log-densities (R,).
    """
    b_idx = np.asarray(b_idx)
    k_idx = np.asarray(k_idx)
    R = len(b_idx)
    out_x = np.empty((R, 2))
    out_lp = np.empty(R)
    tol = float(tol if tol is not None else getattr(ad.dec.cnf, "tol", 1e-4))
    c = prep.cache
    if ad.spatial_sampler == "exact_inverse_flow_jumpcnf":
        for sl in chunked(R, 8192):
            bb, kk = b_idx[sl], k_idx[sl]
            h_q = ad.hidden_at(prep, bb, kk, t_query[sl])
            n_hist = kk + 1
            nmax = int(n_hist.max())
            bt = torch.as_tensor(bb, device=ad.device)
            hist_tau = c["t_raw"][bt, :nmax] + ad.dec.time_offset
            hist_s = c["s_norm"][bt, :nmax]
            tau_q = torch.as_tensor(t_query[sl], device=ad.device, dtype=torch.float32) + ad.dec.time_offset
            x, lp = _jumpcnf_sample_rows(ad, hist_tau, hist_s, n_hist, tau_q, h_q, rng, tol)
            out_x[sl] = ad._raw_locs(to_np(x))
            out_lp[sl] = to_np(lp) - ad.log_jac_s
        return out_x, out_lp
    if ad.spatial_sampler == "exact_inverse_flow_attncnf":
        order = np.argsort(k_idx, kind="stable")
        for start in range(0, R, group):
            ss = order[start : start + group]
            h_q = ad.hidden_at(prep, b_idx[ss], k_idx[ss], t_query[ss])
            x, lp = _attncnf_sample_group(ad, prep, b_idx[ss], k_idx[ss], t_query[ss], h_q, rng, tol)
            out_x[ss] = ad._raw_locs(to_np(x))
            out_lp[ss] = to_np(lp) - ad.log_jac_s
        return out_x, out_lp
    raise NotImplementedError(ad.spatial_sampler)


@torch.no_grad()
def cnf_logpdf_points_reference(ad, prep, b_idx, k_idx, t_query, points):
    """Raw log p(points | t_query, H_{b,k}) via the decoder's own forward pass.

    Uses ``conditional_logprob_fn`` exactly as the model does (it recomputes
    the log-density of *every* history row too, so the cost is O(n^2) per
    point for Jump-CNF). Used as the independent reference in tests and the
    round-trip check.
    """
    c = prep.cache
    R, M, _ = points.shape
    out = np.empty((R, M))
    for r in range(R):
        b, k = int(b_idx[r]), int(k_idx[r])
        h_q = ad.hidden_at(prep, np.array([b]), np.array([k]), np.array([t_query[r]]))
        z_hist = c["h_pre"][b, : k + 1]
        if ad.uses_aux:
            raise NotImplementedError
        z_aug = torch.cat([z_hist, h_q], dim=0)
        fn = ad.dec.conditional_logprob_fn(
            t_query=float(t_query[r]),
            event_times=c["t_raw"][b, : k + 1],
            event_locs=c["s_norm"][b, : k + 1],
            z_aug=z_aug,
        )
        pts = torch.as_tensor(ad._norm_locs(points[r]), device=ad.device, dtype=torch.float32)
        with torch.enable_grad():
            lp = fn(pts)
        out[r] = to_np(lp) - ad.log_jac_s
    return out


def check_roundtrip(ad, prep, b_idx, k_idx, t_query, rng, n: int = 8) -> dict:
    """Invertibility check: tracked log-density vs forward log-density."""
    b_idx, k_idx, t_query = b_idx[:n], k_idx[:n], t_query[:n]
    x, lp = sample_cnf(ad, prep, b_idx, k_idx, t_query, rng)
    lp_fwd = cnf_logpdf_points_reference(ad, prep, b_idx, k_idx, t_query, x[:, None, :])[:, 0]
    lp_fast = cnf_logpdf_points(ad, prep, b_idx, k_idx, t_query, x[:, None, :])[:, 0]
    err = np.abs(lp - lp_fwd)
    err2 = np.abs(lp_fast - lp_fwd)
    return {"n": int(len(lp)), "max_abs_logp_err": float(err.max()), "mean_abs_logp_err": float(err.mean()),
            "max_abs_fast_density_err": float(err2.max())}


# ---------------------------------------------------------------------------
# fast forward densities (query rows only)
# ---------------------------------------------------------------------------


def _jumpcnf_logpdf_rows(ad, hist_tau, hist_s, n_hist, tau_q, h_q, x_q, tol):
    """Forward (density) direction for query rows only; native coordinates."""
    dec = ad.dec
    dev = ad.device
    R = x_q.shape[0]
    dim = dec.spatial_dim
    aux_f = dec._aux_odefunc_ref
    hdyn = lambda t, h: aux_f(t - dec.time_offset, h)  # noqa: E731
    n_hist_t = torch.as_tensor(n_hist, device=dev)
    x = x_q.clone()
    dlogp = torch.zeros(R, device=dev)
    eps = torch.finfo(torch.float32).eps
    nmax = int(n_hist.max())
    for m in range(nmax + 1):
        # stage m processes interval I_{n-m} (from tau_{n-m} down to tau_{n-m-1}), then
        # the radial flow at event n-m-1 (if it exists)
        active = m <= n_hist_t
        idx = torch.nonzero(active, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            break
        j = n_hist_t[idx] - m
        hi = tau_at_sub(hist_tau, n_hist_t, tau_q, idx, j)
        lo = tau_at_sub(hist_tau, n_hist_t, tau_q, idx, j - 1)
        dt = (hi - lo).abs()
        scale = torch.maximum(torch.maximum(lo.abs(), hi.abs()), torch.ones_like(lo))
        run = dt > eps * scale
        h_after = h_q[idx].clone()
        if bool(run.any()):
            ridx = idx[run]
            xin = torch.cat([x[ridx], h_q[ridx]], dim=1)
            xo, lp = _cnf_solve(dec.cnf, hi[run], lo[run], xin, dlogp[ridx], tol)
            x[ridx] = xo[:, :dim]
            dlogp[ridx] = lp
            h_after[run] = xo[:, dim:]
        more = j >= 1
        if bool(more.any()):
            midx = idx[more]
            cond = torch.cat([lo[more].reshape(-1, 1), hist_s[midx, (j[more] - 1)], dec._select_aux(h_after[more])], dim=1)
            xr, lr = dec.inst_flow(x[midx], logpx=dlogp[midx], cond=cond)
            x[midx] = xr
            dlogp[midx] = lr
    logpz = (-0.5 * ((x - dec.z_mean) / torch.exp(dec.z_logstd)) ** 2 - dec.z_logstd
             - 0.5 * math.log(2 * math.pi)).sum(-1)
    return logpz - dlogp


@torch.no_grad()
def cnf_logpdf_points(ad, prep, b_idx, k_idx, t_query, points, *, tol=None, attn_chunk=128):
    """Raw log p(points | t_query, H_{b,k}) for CNF decoders (query rows only)."""
    c = prep.cache
    pts = np.asarray(points, float)
    R, M, _ = pts.shape
    tol = float(tol if tol is not None else getattr(ad.dec.cnf, "tol", 1e-4))
    out = np.empty((R, M))
    if ad.spatial_sampler == "exact_inverse_flow_jumpcnf":
        bb = np.repeat(np.asarray(b_idx), M)
        kk = np.repeat(np.asarray(k_idx), M)
        tq_all = np.repeat(np.asarray(t_query, float), M)
        xq_all = ad._norm_locs(pts.reshape(R * M, 2))
        res = np.empty(R * M)
        for sl in chunked(R * M, 16384):
            h_q = ad.hidden_at(prep, bb[sl], kk[sl], tq_all[sl])
            n_hist = kk[sl] + 1
            nmax = int(n_hist.max())
            bt = torch.as_tensor(bb[sl], device=ad.device)
            hist_tau = c["t_raw"][bt, :nmax] + ad.dec.time_offset
            hist_s = c["s_norm"][bt, :nmax]
            tau_q = torch.as_tensor(tq_all[sl], device=ad.device, dtype=torch.float32) + ad.dec.time_offset
            xq = torch.as_tensor(xq_all[sl], device=ad.device, dtype=torch.float32)
            res[sl] = to_np(_jumpcnf_logpdf_rows(ad, hist_tau, hist_s, n_hist, tau_q, h_q, xq, tol)) - ad.log_jac_s
        return res.reshape(R, M)
    if ad.spatial_sampler == "exact_inverse_flow_attncnf":
        dec = ad.dec
        for r in range(R):
            b, k = int(b_idx[r]), int(k_idx[r])
            h_q = ad.hidden_at(prep, np.array([b]), np.array([k]), np.array([t_query[r]]))
            for sl in chunked(M, attn_chunk):
                m = sl.stop - sl.start
                S = k + 2
                times = torch.cat([c["t_raw"][b, : k + 1], torch.as_tensor([t_query[r]], device=ad.device, dtype=torch.float32)])
                xq = torch.as_tensor(ad._norm_locs(pts[r, sl]), device=ad.device, dtype=torch.float32)  # (m,2)
                locs = torch.cat([c["s_norm"][b, : k + 1][None].expand(m, -1, -1), xq[:, None, :]], 1)
                aux = torch.cat([c["h_pre"][b, : k + 1], h_q], 0)[None].expand(m, -1, -1)
                with torch.enable_grad():
                    lp = dec.sequence_logprob(event_times=times[None].expand(m, -1), spatial_locations=locs,
                                              input_mask=None, aux_state=aux)[:, -1]
                out[r, sl] = to_np(lp) - ad.log_jac_s
        return out
    raise NotImplementedError(ad.spatial_sampler)
