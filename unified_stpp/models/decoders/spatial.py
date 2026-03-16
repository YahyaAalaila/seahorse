"""
Active spatial-decoder components retained after coarse-framework cleanup.

This module intentionally keeps only pieces used by active presets:
  - CNFVelocityField (+ ConcatSquash) for neural_stpp_*_sc spatial decoders
  - DeepSTPPDecoder for deep_stpp
  - _euler_solve_simple fallback utility
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Divergence helpers (used by CNFVelocityField)
# ============================================================================

def divergence_bf(f: Tensor, y: Tensor, training: bool) -> Tensor:
    """
    Exact divergence via brute-force Jacobian diagonal. O(d) backward passes.
    """
    sum_diag = 0.0
    for i in range(f.shape[1]):
        retain = training or (i < f.shape[1] - 1)
        grad_i = torch.autograd.grad(
            f[:, i].sum(), y, create_graph=training, retain_graph=retain
        )[0]
        sum_diag = sum_diag + grad_i[:, i]
    return sum_diag


def _hutchinson_trace(v: Tensor, x: Tensor, e: Tensor) -> Tensor:
    """Hutchinson trace estimator: E[eps^T J eps] = tr(J)."""
    vjp = torch.autograd.grad(v, x, e, create_graph=True, retain_graph=True)[0]
    return (vjp * e).sum(dim=-1)


# ============================================================================
# CNF velocity field pieces (used by neural_stpp_spatial.py)
# ============================================================================

class ConcatSquash(nn.Module):
    """ConcatSquash hidden layer from FFJORD / Neural STPP."""

    def __init__(self, in_dim: int, out_dim: int, ctx_dim: int):
        super().__init__()
        self.lin_z = nn.Linear(in_dim, out_dim)
        self.lin_t = nn.Linear(ctx_dim, out_dim, bias=False)
        self.lin_tb = nn.Linear(ctx_dim, out_dim)

    def forward(self, z: Tensor, ctx: Tensor) -> Tensor:
        return self.lin_z(z) * torch.sigmoid(self.lin_t(ctx)) + torch.tanh(
            self.lin_tb(ctx)
        )


class CNFVelocityField(nn.Module):
    """Velocity field v(s, tau; z, X) for spatial CNF-style decoders."""

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        field_cov_dim: int = 0,
        layer_type: str = "concat",
        n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.layer_type = layer_type
        ctx_dim = hidden_dim + 1 + field_cov_dim  # z + tau [+ X_field]

        if layer_type == "concat":
            cs_layers = []
            in_dim = spatial_dim
            for _ in range(n_hidden_layers):
                cs_layers.append(ConcatSquash(in_dim, hidden_dim, ctx_dim))
                in_dim = hidden_dim
            self.hidden_layers = nn.ModuleList(cs_layers)
            self.output_layer = nn.Linear(hidden_dim, spatial_dim)
            self.mlp = None
        else:
            act_cls = nn.Softplus if layer_type == "softplus" else nn.Tanh
            in_d = spatial_dim + ctx_dim
            mlp_layers: list = []
            for _ in range(n_hidden_layers):
                mlp_layers += [nn.Linear(in_d, hidden_dim), act_cls()]
                in_d = hidden_dim
            mlp_layers.append(nn.Linear(hidden_dim, spatial_dim))
            self.mlp = nn.Sequential(*mlp_layers)
            self.hidden_layers = None
            self.output_layer = None

        self._z_cond: Optional[Tensor] = None
        self._x_cond: Optional[Tensor] = None
        self._e: Optional[Tensor] = None

    def _compute_velocity(self, s: Tensor, ctx: Tensor) -> Tensor:
        if self.layer_type == "concat":
            h = s
            for layer in self.hidden_layers:
                h = layer(h, ctx)
            return self.output_layer(h)
        return self.mlp(torch.cat([s, ctx], dim=-1))

    def forward(self, tau: Tensor, s: Tensor) -> Tensor:
        d = self.spatial_dim
        s_actual = s[:, :d] if s.shape[-1] > d else s

        bsz = s_actual.shape[0]
        tau_expand = tau.expand(bsz, 1)
        ctx_parts = [self._z_cond, tau_expand]
        if self._x_cond is not None:
            ctx_parts.append(self._x_cond)
        ctx = torch.cat(ctx_parts, dim=-1)

        v = self._compute_velocity(s_actual, ctx)

        if s.shape[-1] > d:
            with torch.enable_grad():
                s_leaf = s_actual.detach().requires_grad_(True)
                v_div = self._compute_velocity(s_leaf, ctx.detach())

                if self.spatial_dim <= 3:
                    trace = divergence_bf(v_div, s_leaf, self.training)
                else:
                    if self._e is None:
                        self._e = torch.randn_like(s_leaf)
                    trace = _hutchinson_trace(v_div, s_leaf, self._e)

            return torch.cat([v, -trace.unsqueeze(-1)], dim=-1)
        return v


# ============================================================================
# DeepSTPPDecoder — faithful coupled Hawkes temporal-spatial decoder
# ============================================================================

class DeepSTPPDecoder(nn.Module):
    """
    Faithful joint temporal-spatial decoder for DeepSTPP (Lin et al. 2021).

    Implements the coupled Hawkes intensity:
        lambda(s, t) = lambda_t(t) * f(s|t)
    """

    requires_time_history: bool = True

    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        seq_len: int = 20,
        num_points: int = 20,
        sigma_min: float = 1e-4,
        n_layers: int = 3,
        field_cov_dim: int = 0,
        **kwargs,
    ):
        del field_cov_dim, kwargs
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        self.seq_len = seq_len
        self.num_points = num_points
        self.sigma_min = sigma_min
        m = seq_len + num_points

        def _mlp(out_dim: int) -> nn.Sequential:
            layers: list = [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        self.w_dec = _mlp(m)
        self.b_dec = _mlp(m)
        self.s_dec = _mlp(m * spatial_dim)

        if num_points > 0:
            self.background = nn.Parameter(torch.rand(num_points, spatial_dim) * 0.1)
        else:
            self.register_parameter("background", None)

    @property
    def requires_history(self) -> bool:
        return True

    @property
    def history_window_size(self) -> int:
        return self.seq_len

    def _decode(self, z: Tensor):
        """Decode (w_i, b_i, sigma, inv_var) from latent z."""
        bsz = z.shape[0]
        m = self.seq_len + self.num_points
        d = self.spatial_dim

        w_i = F.softplus(self.w_dec(z)) + 1e-5
        b_i = F.softplus(self.b_dec(z)) + 1e-5
        sigma = F.softplus(self.s_dec(z).reshape(bsz, m, d)) + self.sigma_min
        inv_var = 1.0 / sigma
        return w_i, b_i, sigma, inv_var

    @staticmethod
    def _log_ft(
        w_i: Tensor,
        b_i: Tensor,
        tn_ti: Tensor,
        t_ti: Tensor,
    ) -> Tensor:
        """Temporal log-density term used by DeepSTPP."""
        exp_t_ti = torch.exp(-b_i * t_ti)
        exp_tn_ti = torch.exp(-b_i * tn_ti.clamp(min=0.0))
        log_w_i = torch.log(w_i)
        log_v_i = log_w_i - b_i * t_ti
        log_lamb_t = torch.logsumexp(log_v_i, dim=-1)
        comp = (w_i / b_i * (exp_t_ti - exp_tn_ti)).sum(dim=-1)
        return log_lamb_t + comp

    @staticmethod
    def _log_s_intensity(
        w_i: Tensor,
        b_i: Tensor,
        t_ti: Tensor,
        s_diff: Tensor,
        inv_var: Tensor,
    ) -> Tensor:
        """Spatial log-density term used by DeepSTPP."""
        d = s_diff.shape[-1]
        log_w_i = torch.log(w_i)
        log_v_i = log_w_i - b_i * t_ti
        log_lamb_t = torch.logsumexp(log_v_i, dim=-1)
        log_v_norm = log_v_i - log_lamb_t.unsqueeze(-1)
        log_gauss = (
            0.5 * inv_var.prod(dim=-1).clamp(min=1e-12).log()
            - (d / 2.0) * math.log(2.0 * math.pi)
            - 0.5 * (s_diff.pow(2) * inv_var).sum(dim=-1)
        )
        return torch.logsumexp(log_v_norm + log_gauss, dim=-1)

    def event_terms(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
        return_decoded: bool = False,
    ) -> dict:
        """Compute DeepSTPP temporal/spatial log terms and per-event NLL."""
        bsz = z.shape[0]
        d = self.spatial_dim
        device = z.device

        t = t.reshape(bsz)
        t_prev = t_prev.reshape(bsz)
        dt = (t - t_prev).clamp(min=1e-6)

        if x_field is not None:
            t_hist = x_field[:, : self.seq_len]
            s_hist = x_field[:, self.seq_len :].reshape(bsz, self.seq_len, d)
            tn_ti_h = (t_prev.unsqueeze(-1) - t_hist).clamp(min=0.0)
        else:
            tn_ti_h = torch.zeros(bsz, self.seq_len, device=device)
            s_hist = torch.zeros(bsz, self.seq_len, d, device=device)

        tn_ti_bg = torch.zeros(bsz, self.num_points, device=device)
        tn_ti = torch.cat([tn_ti_h, tn_ti_bg], dim=-1)
        t_ti = (tn_ti + dt.unsqueeze(-1)).clamp(min=1e-6)

        if self.background is not None:
            bg = self.background.unsqueeze(0).expand(bsz, -1, -1)
            centers = torch.cat([s_hist, bg], dim=1)
        else:
            centers = s_hist
        s_diff = s.unsqueeze(1) - centers

        w_i, b_i, _sigma, inv_var = self._decode(z)
        log_temporal = self._log_ft(w_i, b_i, tn_ti, t_ti)
        log_spatial = self._log_s_intensity(w_i, b_i, t_ti, s_diff, inv_var)
        out = {
            "log_temporal": log_temporal,
            "log_spatial": log_spatial,
            "nll": -(log_temporal + log_spatial),
        }
        if return_decoded:
            out["w_i"] = w_i
            out["b_i"] = b_i
            out["inv_var"] = inv_var
        return out

    def nll(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Joint per-event NLL = -(log temporal + log spatial)."""
        return self.event_terms(z, t, s, t_prev, x_field=x_field, return_decoded=False)[
            "nll"
        ]

    def spatial_intensity(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Return true conditional intensity lambda_t(t) * f(s|t)."""
        bsz = z.shape[0]
        d = self.spatial_dim
        device = z.device

        t = t.reshape(bsz)
        t_prev = t_prev.reshape(bsz)
        dt = (t - t_prev).clamp(min=1e-6)

        if x_field is not None:
            t_hist = x_field[:, : self.seq_len]
            s_hist = x_field[:, self.seq_len :].reshape(bsz, self.seq_len, d)
            tn_ti_h = (t_prev.unsqueeze(-1) - t_hist).clamp(min=0.0)
        else:
            tn_ti_h = torch.zeros(bsz, self.seq_len, device=device)
            s_hist = torch.zeros(bsz, self.seq_len, d, device=device)

        tn_ti_bg = torch.zeros(bsz, self.num_points, device=device)
        tn_ti = torch.cat([tn_ti_h, tn_ti_bg], dim=-1)
        t_ti = (tn_ti + dt.unsqueeze(-1)).clamp(min=1e-6)

        if self.background is not None:
            bg = self.background.unsqueeze(0).expand(bsz, -1, -1)
            centers = torch.cat([s_hist, bg], dim=1)
        else:
            centers = s_hist

        w_i, b_i, sigma, inv_var = self._decode(z)

        log_w_i = torch.log(w_i)
        log_v_i = log_w_i - b_i * t_ti
        log_lamb_t = torch.logsumexp(log_v_i, dim=-1)
        log_v_norm = log_v_i - log_lamb_t.unsqueeze(-1)

        s_diff = s.unsqueeze(1) - centers
        log_gauss = (
            0.5 * inv_var.prod(dim=-1).clamp(min=1e-12).log()
            - (d / 2.0) * math.log(2.0 * math.pi)
            - 0.5 * (s_diff.pow(2) * inv_var).sum(dim=-1)
        )
        log_spatial = torch.logsumexp(log_v_norm + log_gauss, dim=-1)

        return torch.exp(log_lamb_t + log_spatial)

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        return -self.nll(z, t, s, t_prev, x_field=x_field)

    def sample(self, z: Tensor, t_prev: Tensor, x_field_fn=None):
        """Approximate sample: mean inter-arrival time + mixture location."""
        del x_field_fn
        bsz = z.shape[0]
        d = self.spatial_dim
        m = self.seq_len + self.num_points
        dev = z.device

        w_i, b_i, sigma, _ = self._decode(z)

        lambda0 = w_i.sum(dim=-1).clamp(min=1e-6)
        dt = (1.0 / lambda0).unsqueeze(-1)
        t_new = t_prev + dt

        v_norm = w_i / w_i.sum(-1, keepdim=True)
        k = torch.multinomial(v_norm, 1).squeeze(-1)

        centers = torch.zeros(bsz, m, d, device=dev)
        if self.background is not None:
            centers[:, self.seq_len :] = self.background.unsqueeze(0).expand(bsz, -1, -1)

        idx = torch.arange(bsz, device=dev)
        mu_k = centers[idx, k]
        sig_k = sigma[idx, k]
        s_new = mu_k + sig_k * torch.randn(bsz, d, device=dev)

        return t_new, s_new


# ============================================================================
# Utility
# ============================================================================

def _euler_solve_simple(func, y0, t_span, n_steps: int = 50):
    """Simple Euler solver fallback."""
    dt = (t_span[-1] - t_span[0]) / n_steps
    y = y0
    t = t_span[0]
    for _ in range(n_steps):
        y = y + dt * func(t, y)
        t = t + dt
    return y
