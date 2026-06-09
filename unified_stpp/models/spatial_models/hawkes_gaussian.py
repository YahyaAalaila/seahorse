"""
Hawkes-Gaussian decoder — coupled Hawkes temporal + Gaussian spatial likelihood model.

Parameterises the joint conditional intensity:
    lambda(s, t) = lambda_t(t) * f(s|t)

where lambda_t is a sum-of-exponentials Hawkes temporal kernel and f(s|t) is
an isotropic Gaussian spatial mixture with Hawkes-weighted components.

Used by the deep_stpp preset via DeepSTPPEventModel.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# HawkesGaussianDecoder — coupled Hawkes temporal + Gaussian spatial decoder
# ============================================================================

class HawkesGaussianDecoder(nn.Module):
    """
    Joint temporal-spatial decoder: sum-of-exponentials Hawkes temporal kernel
    × isotropic Gaussian spatial kernel.

    Implements the coupled intensity:
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
        constrain_b: str | bool = False,
        b_max: float = 20.0,
        s_max: Optional[float] = None,
        **kwargs,
    ):
        del field_cov_dim, kwargs
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        self.seq_len = seq_len
        self.num_points = num_points
        self.sigma_min = sigma_min
        self.constrain_b = constrain_b
        self.b_max = float(b_max)
        self.s_max = None if s_max is None else float(s_max)
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
            self.background = nn.Parameter(torch.rand(num_points, spatial_dim))
        else:
            self.register_parameter("background", None)

    @property
    def requires_history(self) -> bool:
        return True

    @property
    def history_window_size(self) -> int:
        return self.seq_len

    def decode(self, z: Tensor):
        """Decode (w_i, b_i, sigma, inv_var) from latent z."""
        bsz = z.shape[0]
        m = self.seq_len + self.num_points
        d = self.spatial_dim

        w_i = F.softplus(self.w_dec(z)) + 1e-5
        b_raw = self.b_dec(z)
        mode = self.constrain_b
        if isinstance(mode, str):
            mode = mode.strip().lower()
        if mode in (False, None, "false", "none", "unconstrained"):
            b_i = b_raw
        elif mode in (True, "softplus"):
            b_i = F.softplus(b_raw) + 1e-5
        elif mode == "tanh":
            b_i = torch.tanh(b_raw) * self.b_max
        elif mode == "sigmoid":
            b_i = torch.sigmoid(b_raw) * self.b_max
        elif mode == "neg-sigmoid":
            b_i = -torch.sigmoid(b_raw) * self.b_max
        elif mode == "clamp":
            b_i = torch.clamp(b_raw, -self.b_max, self.b_max)
        else:
            raise ValueError(f"Unsupported DeepSTPP constrain_b mode: {self.constrain_b!r}")

        sigma = F.softplus(self.s_dec(z).reshape(bsz, m, d)) + self.sigma_min
        if self.s_max is not None:
            sigma = torch.sigmoid(sigma) * self.s_max
        inv_var = 1.0 / sigma
        return w_i, b_i, sigma, inv_var

    @staticmethod
    def _safe_decay(b_i: Tensor) -> Tensor:
        eps = torch.full_like(b_i, 1e-6)
        signed_eps = torch.where(b_i < 0, -eps, eps)
        return torch.where(b_i.abs() < 1e-6, signed_eps, b_i)

    @staticmethod
    def log_ft(
        w_i: Tensor,
        b_i: Tensor,
        tn_ti: Tensor,
        t_ti: Tensor,
    ) -> Tensor:
        """Temporal log-density term used by DeepSTPP."""
        b_safe = HawkesGaussianDecoder._safe_decay(b_i)
        exp_t_ti = torch.exp(-b_safe * t_ti)
        exp_tn_ti = torch.exp(-b_safe * tn_ti.clamp(min=0.0))
        log_w_i = torch.log(w_i)
        log_v_i = log_w_i - b_safe * t_ti
        log_lamb_t = torch.logsumexp(log_v_i, dim=-1)
        comp = (w_i / b_safe * (exp_t_ti - exp_tn_ti)).sum(dim=-1)
        return log_lamb_t + comp

    @staticmethod
    def log_s_intensity(
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

        w_i, b_i, _sigma, inv_var = self.decode(z)
        log_temporal = self.log_ft(w_i, b_i, tn_ti, t_ti)
        log_spatial = self.log_s_intensity(w_i, b_i, t_ti, s_diff, inv_var)
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

        w_i, b_i, sigma, inv_var = self.decode(z)

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

        w_i, b_i, sigma, _ = self.decode(z)

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
