"""Independent paper-based GLIDE event diffusion model.

The reported variational quantity is deliberately *not* exposed as point-process
NLL: GLIDE operates on min--max transformed next-event tokens and does not
define the finite-window normalized conditional-intensity likelihood Seahorse
uses for likelihood ranking.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext
from ..model_registry import register_event


def _cosine_betas(steps: int) -> Tensor:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float32)
    alpha = torch.cos(((x / steps) + 0.008) / 1.008 * math.pi / 2).square()
    alpha = alpha / alpha[0]
    return (1 - alpha[1:] / alpha[:-1]).clamp(1e-4, .9999)


def _take(buf: Tensor, t: Tensor, like: Tensor) -> Tensor:
    return buf.gather(0, t).reshape(t.shape[0], *([1] * (like.ndim - 1))).to(like)


class _TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        x = t.float().unsqueeze(-1) * freq
        emb = torch.cat((x.sin(), x.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class _MeanAnchor(nn.Module):
    """GLIDE's deterministic conditional mean predictor."""

    def __init__(self, cond_dim: int, hidden: int, token_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(cond_dim, num_heads=4, batch_first=True)
        self.net = nn.Sequential(nn.Linear(2 * cond_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.time_head = nn.Linear(hidden, 1)
        self.space_head = nn.Linear(hidden, token_dim - 1)
        # The upstream model initializes deterministic heads at zero.
        nn.init.zeros_(self.time_head.weight); nn.init.zeros_(self.time_head.bias)
        nn.init.zeros_(self.space_head.weight); nn.init.zeros_(self.space_head.bias)

    def forward(self, cond: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        key_padding = None if mask is None else ~mask
        attended, _ = self.attention(cond, cond, cond, key_padding_mask=key_padding, need_weights=False)
        h = self.net(torch.cat((cond, attended), dim=-1))
        return torch.sigmoid(torch.cat((self.time_head(h), self.space_head(h)), dim=-1))


class _DualStreamDenoiser(nn.Module):
    """Time and spatial streams with learned conditional mixing."""

    def __init__(self, cond_dim: int, hidden: int, token_dim: int):
        super().__init__()
        self.cond_dim, self.token_dim = cond_dim, token_dim
        self.timestep = _TimestepEmbedding(hidden)
        self.time = nn.Sequential(nn.Linear(1 + hidden + 2 * cond_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.space = nn.Sequential(nn.Linear(token_dim - 1 + hidden + 2 * cond_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.time_out, self.space_out = nn.Linear(hidden, 1), nn.Linear(hidden, token_dim - 1)
        self.time_mix, self.space_mix = nn.Linear(cond_dim + hidden, 2), nn.Linear(cond_dim + hidden, 2)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        d = self.cond_dim
        ct, cs, cj = cond[..., :d], cond[..., d:2*d], cond[..., 2*d:]
        e = self.timestep(t).unsqueeze(1).expand(-1, x.shape[1], -1)
        ht = self.time(torch.cat((x[..., :1], e, ct, cj), dim=-1))
        hs = self.space(torch.cat((x[..., 1:], e, cs, cj), dim=-1))
        wt = self.time_mix(torch.cat((ct, e), dim=-1)).softmax(-1)
        ws = self.space_mix(torch.cat((cs, e), dim=-1)).softmax(-1)
        # GLIDE uses learned dual-stream attention rather than a single joint MLP.
        time = self.time_out(ht) * wt[..., :1] + self.time_out(hs)[..., :1] * wt[..., 1:]
        space_source = self.space_out(hs) * ws[..., :1] + self.space_out(ht)[..., : self.token_dim - 1] * ws[..., 1:]
        return torch.cat((time, space_source), dim=-1)


class GLIDEEMA(nn.Module):
    """Explicit, checkpointable EMA state.

    The training loop must call :meth:`update` after the optimizer step.  The
    shadows are buffers rather than a Python dictionary so normal Seahorse/
    Lightning checkpoints preserve the sampler state on resume.
    """

    def __init__(self, module: nn.Module, decay: float = .999):
        super().__init__()
        self.decay = float(decay)
        self.parameter_names = tuple(name for name, value in module.named_parameters() if value.requires_grad)
        for idx, name in enumerate(self.parameter_names):
            value = dict(module.named_parameters())[name]
            self.register_buffer(f"shadow_{idx}", value.detach().clone())

    def _shadows(self) -> Dict[str, Tensor]:
        return {name: getattr(self, f"shadow_{idx}") for idx, name in enumerate(self.parameter_names)}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        shadow = self._shadows()
        for name, value in module.named_parameters():
            if name in shadow:
                shadow[name].lerp_(value.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def store_and_copy_to(self, module: nn.Module) -> Dict[str, Tensor]:
        shadow = self._shadows()
        stored = {name: value.detach().clone() for name, value in module.named_parameters() if name in shadow}
        for name, value in module.named_parameters():
            if name in shadow:
                value.copy_(shadow[name])
        return stored

    @torch.no_grad()
    def restore(self, module: nn.Module, stored: Dict[str, Tensor]) -> None:
        for name, value in module.named_parameters():
            if name in stored:
                value.copy_(stored[name])


@register_event("glide")
class GLIDEEventModel(EventModel):
    """Weighted diffusion + L1 GLIDE objective and leap sampler."""

    def __init__(self, *, cond_dim: int = 64, hidden_units: int = 64, spatial_dim: int = 2,
                 timesteps: int = 500, start_ratio: float = .5, guidance: float = .2,
                 ema_decay: float = .999):
        super().__init__()
        self.token_dim, self.timesteps = 1 + int(spatial_dim), int(timesteps)
        self.start_ratio, self.guidance = float(start_ratio), float(guidance)
        self.anchor = _MeanAnchor(3 * int(cond_dim), hidden_units, self.token_dim)
        self.denoiser = _DualStreamDenoiser(int(cond_dim), hidden_units, self.token_dim)
        betas = _cosine_betas(self.timesteps)
        alphas = 1. - betas
        ac = alphas.cumprod(0); ac_prev = F.pad(ac[:-1], (1, 0), value=1.)
        for name, value in {
            "betas": betas, "alphas_cumprod": ac, "alphas_cumprod_prev": ac_prev,
            "sqrt_ac": ac.sqrt(), "sqrt_om_ac": (1 - ac).sqrt(),
            "sqrt_recip_ac": (1 / ac).sqrt(), "sqrt_recipm1_ac": (1 / ac - 1).sqrt(),
            "posterior_var": betas * (1 - ac_prev) / (1 - ac),
            "posterior_mean_coef1": betas * ac_prev.sqrt() / (1 - ac),
            "posterior_mean_coef2": (1 - ac_prev) * alphas.sqrt() / (1 - ac),
        }.items(): self.register_buffer(name, value)
        self.ema = GLIDEEMA(self, ema_decay)

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="weighted_diffusion_l1", metric_key="glide_loss",
            objective_description="GLIDE weighted denoising loss plus deterministic mean-anchor L1",
            nll_kind="none", nll_description="No finite-window point-process NLL; transformed-coordinate diffusion VLB is diagnostic only.",
            nll_footnote="not likelihood-ranked", has_native_sampler=True,
        )

    def update_ema(self) -> None:
        """Call once after every optimizer step (runner hook required)."""
        self.ema.update(self)

    def _q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return _take(self.sqrt_ac, t, x0) * x0 + _take(self.sqrt_om_ac, t, x0) * noise

    def _predict_x0(self, xt: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return _take(self.sqrt_recip_ac, t, xt) * xt - _take(self.sqrt_recipm1_ac, t, xt) * noise

    @staticmethod
    def _mean(x: Tensor, mask: Tensor) -> Tensor:
        return (x * mask.unsqueeze(-1)).sum() / (mask.sum().clamp_min(1) * x.shape[-1])

    def training_loss(self, *, times: Tensor, locations: Tensor, lengths: Tensor, state: StateContext,
                      state_regularization_terms: Optional[Dict[str, Tensor]] = None,
                      x_field_at_events: Optional[Tensor] = None, marks: Optional[Tensor] = None,
                      device=None) -> Dict[str, Tensor]:
        del times, locations, lengths, state_regularization_terms, x_field_at_events, marks, device
        p = self._coerce_state_context(state).payload
        token, cond, mask = p["glide_tokens"], p["glide_cond"], p["glide_mask"]
        total = mask.sum().to(token.dtype)
        if not bool(mask.any()):
            zero = token.sum() * 0.
            return {"loss": zero, "nll": zero, "total_events": total, "extra_metrics": {}}
        x0 = token * 2 - 1
        t = torch.randint(self.timesteps, (token.shape[0],), device=token.device)
        noise = torch.randn_like(x0); xt = self._q_sample(x0, t, noise)
        pred = self.denoiser(xt, t, cond)
        diff_t = self._mean((pred[..., :1] - noise[..., :1]).square(), mask)
        diff_s = self._mean((pred[..., 1:] - noise[..., 1:]).square(), mask)
        anchor = self.anchor(cond, mask)
        aux_t = self._mean((anchor[..., :1] - token[..., :1]).abs(), mask)
        aux_s = self._mean((anchor[..., 1:] - token[..., 1:]).abs(), mask)
        loss = (diff_t + 2 * diff_s) + .1 * (aux_t + 2 * aux_s)
        return {"loss": loss, "nll": loss, "total_events": total,
                "extra_metrics": {"glide_diffusion_temporal": diff_t.detach(),
                                  "glide_diffusion_spatial": diff_s.detach(), "glide_anchor_l1_temporal": aux_t.detach(),
                                  "glide_anchor_l1_spatial": aux_s.detach()}}

    @torch.no_grad()
    def transformed_vlb(self, state: StateContext) -> Tensor:
        """Full DDPM VLB in transformed token coordinates, for diagnostics only."""
        p = self._coerce_state_context(state).payload; token, cond, mask = p["glide_tokens"], p["glide_cond"], p["glide_mask"]
        if not bool(mask.any()): return token.sum() * 0.
        x0 = token * 2 - 1; total = x0.new_zeros(x0.shape[:2])
        # This is the standard fixed-variance DDPM variational decomposition,
        # evaluated at every timestep. It remains a transformed-coordinate
        # diagnostic, with no point-process likelihood interpretation.
        for step in range(self.timesteps):
            t = torch.full((x0.shape[0],), step, dtype=torch.long, device=x0.device)
            xt = self._q_sample(x0, t, torch.randn_like(x0))
            noise = self.denoiser(xt, t, cond)
            pred_x0 = self._predict_x0(xt, t, noise).clamp(-1., 1.)
            true_mean = _take(self.posterior_mean_coef1, t, xt) * x0 + _take(self.posterior_mean_coef2, t, xt) * xt
            model_mean = _take(self.posterior_mean_coef1, t, xt) * pred_x0 + _take(self.posterior_mean_coef2, t, xt) * xt
            logvar = _take(self.posterior_var.clamp_min(1e-20).log(), t, xt)
            kl = .5 * ((true_mean - model_mean).square() * (-logvar).exp()).sum(-1)
            decoder_nll = .5 * (((x0 - model_mean).square() * (-logvar).exp()) + logvar + math.log(2 * math.pi)).sum(-1)
            total = total + torch.where(t[:, None] == 0, decoder_nll, kl)
        t_last = torch.full((x0.shape[0],), self.timesteps - 1, dtype=torch.long, device=x0.device)
        mean_last = _take(self.sqrt_ac, t_last, x0) * x0
        var_last = (1 - _take(self.alphas_cumprod, t_last, x0)).clamp_min(1e-20)
        prior_kl = .5 * (mean_last.square() + var_last - 1 - var_last.log()).sum(-1)
        total = total + prior_kl
        return (total * mask).sum() / mask.sum().clamp_min(1)

    def eval_nll(self, **kwargs):
        # Deliberately callable only by explicit diagnostics; Unified likelihood routing observes nll_kind='none'.
        state = kwargs["state"]; value = self.transformed_vlb(state)
        p = self._coerce_state_context(state).payload
        return {"loss": value, "nll": value, "total_events": p["glide_total_events"],
                "extra_metrics": {"glide_transformed_vlb_diagnostic": value}}

    @torch.no_grad()
    def sample_native(self, *, state: StateContext, batch_size: Optional[int] = None, use_ema: bool = True, **kwargs) -> Tensor:
        del kwargs
        p = self._coerce_state_context(state).payload
        cond = p["glide_cond_last"]
        if batch_size is not None and batch_size != cond.shape[0]:
            if cond.shape[0] != 1: raise ValueError("batch_size may only expand a single GLIDE history state.")
            cond = cond.expand(batch_size, -1, -1)
        stored = self.ema.store_and_copy_to(self) if use_ema else None
        try:
            anchor = self.anchor(cond).clamp(0., 1.) * 2 - 1
            start = max(0, min(self.timesteps - 1, int(self.timesteps * self.start_ratio)))
            t0 = torch.full((cond.shape[0],), start, dtype=torch.long, device=cond.device)
            x = self._q_sample(anchor, t0, torch.randn_like(anchor))
            # The released ``sample_leapfrog`` initializes x at ``start_t`` and
            # then runs p_sample for start_t-1, ..., 0.
            for step in range(start - 1, -1, -1):
                t = torch.full((cond.shape[0],), step, dtype=torch.long, device=cond.device)
                eps = self.denoiser(x, t, cond)
                x0 = self._predict_x0(x, t, eps).clamp(-1., 1.)
                if step == 0:
                    x = x0; continue
                beta = _take(self.betas, t, x)
                ac = _take(self.alphas_cumprod, t, x); ap = _take(self.alphas_cumprod_prev, t, x)
                mean = (beta * ap.sqrt() / (1 - ac)) * x0 + ((1 - ap) * (1 - beta).sqrt() / (1 - ac)) * x
                var = _take(self.posterior_var, t, x).clamp_min(1e-20)
                x = mean + var.sqrt() * torch.randn_like(x)
                x = (1 - self.guidance * step / max(start, 1)) * x + (self.guidance * step / max(start, 1)) * anchor
            # The author sampler unnormalizes without clipping.  A sample may be
            # outside the train min--max range and must retain that native unit.
            return (x + 1) * .5
        finally:
            if stored is not None: self.ema.restore(self, stored)
