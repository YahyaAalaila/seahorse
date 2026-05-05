"""EventModel for Diffusion STPP (GaussianDiffusion_ST objective and native sampling).

Implements the denoising diffusion objective from Spatio-Temporal Diffusion
Point Processes. Closely mirrors the SMASH integration pattern.

Capabilities
------------
- training_objective : approx_nll  (ELBO lower bound on NLL)
- has_eval_nll       : True        (approximate VB via NLL_cal — expensive)
- has_native_sampler : True        (DDIM / DDPM sampling)
- has_intensity      : False
- has_density        : False
- has_score          : False
"""

from __future__ import annotations

import math
from random import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext
from ..model_registry import register_event


# ---------------------------------------------------------------------------
# Normalisation helpers (same convention as SMASH)
# ---------------------------------------------------------------------------

def normalize_to_neg_one_to_one(x: Tensor) -> Tensor:
    return x * 2 - 1


def unnormalize_to_zero_to_one(x: Tensor) -> Tensor:
    return (x + 1) * 0.5


def exists(x) -> bool:
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def mean_flat(tensor: Tensor) -> Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def discretized_gaussian_log_likelihood(z: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    c = torch.tensor([math.log(2 * math.pi)], device=z.device, dtype=z.dtype)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    log_probs = -0.5 * (tmp * tmp + 2 * log_std + c)
    assert log_probs.shape == z.shape
    return log_probs


def normal_kl(mean1, logvar1, mean2, logvar2) -> Tensor:
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    if tensor is None:
        raise AssertionError("at least one argument must be a Tensor")

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x, device=tensor.device, dtype=tensor.dtype)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

def _linear_beta_schedule(timesteps: int, max_beta: float = 0.01) -> Tensor:
    return torch.linspace(1e-4, max_beta, timesteps)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(1e-4, 0.9999)


# ---------------------------------------------------------------------------
# Sinusoidal position embedding (for diffusion timestep)
# ---------------------------------------------------------------------------

class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ---------------------------------------------------------------------------
# Denoising network
# ---------------------------------------------------------------------------

class STDiffusionNet(nn.Module):
    """DSTPP denoiser."""

    def __init__(
        self,
        n_steps: int,
        dim: int,
        num_units: int = 64,
        self_condition: bool = False,
        condition: bool = True,
        cond_dim: int = 64,
    ):
        del n_steps
        super().__init__()
        self.channels = 1
        self.self_condition = self_condition
        self.condition = condition
        self.cond_dim = int(cond_dim)
        self.dim = int(dim)

        sinu_pos_emb = _SinusoidalPosEmb(num_units)
        time_dim = num_units
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(num_units, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.linears_spatial = nn.ModuleList(
            [
                nn.Linear(dim - 1, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
            ]
        )
        self.linears_temporal = nn.ModuleList(
            [
                nn.Linear(1, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
            ]
        )

        self.output_spatial = nn.Sequential(
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, dim - 1),
        )
        self.output_temporal = nn.Sequential(
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, 1),
        )

        self.linear_t = nn.Sequential(
            nn.Linear(num_units * 2, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, 2),
        )
        self.linear_s = nn.Sequential(
            nn.Linear(num_units * 2, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, 2),
        )

        self.cond_all = nn.Sequential(
            nn.Linear(cond_dim * 3, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
        )
        self.cond_temporal = nn.ModuleList([nn.Linear(cond_dim, num_units) for _ in range(3)])
        self.cond_spatial = nn.ModuleList([nn.Linear(cond_dim, num_units) for _ in range(3)])
        self.cond_joint = nn.ModuleList([nn.Linear(cond_dim, num_units) for _ in range(3)])

    def get_attn(
        self,
        x: Tensor,
        t: Tensor,
        x_self_cond: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        del x, x_self_cond
        if cond is None:
            raise ValueError("STDiffusionNet requires cond for DSTPP attention weights.")
        cond_all = self.cond_all(cond)
        t_embedding = self.time_mlp(t).unsqueeze(dim=1)
        cond_all = torch.cat((cond_all, t_embedding), dim=-1)
        alpha_s = F.softmax(self.linear_s(cond_all), dim=-1).squeeze(dim=1)
        alpha_t = F.softmax(self.linear_t(cond_all), dim=-1).squeeze(dim=1)
        return alpha_s, alpha_t

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        x_self_cond: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        del x_self_cond
        if cond is None:
            raise ValueError("STDiffusionNet requires cond for DSTPP conditioning.")

        x_spatial = x[:, :, 1:].clone()
        x_temporal = x[:, :, :1].clone()

        hidden_dim = self.cond_dim
        cond_temporal = cond[:, :, :hidden_dim]
        cond_spatial = cond[:, :, hidden_dim : 2 * hidden_dim]
        cond_joint = cond[:, :, 2 * hidden_dim :]

        cond_all = self.cond_all(cond)
        t_embedding = self.time_mlp(t).unsqueeze(dim=1)
        cond_all = torch.cat((cond_all, t_embedding), dim=-1)

        alpha_s = F.softmax(self.linear_s(cond_all), dim=-1).squeeze(dim=1).unsqueeze(dim=2)
        alpha_t = F.softmax(self.linear_t(cond_all), dim=-1).squeeze(dim=1).unsqueeze(dim=2)

        for idx in range(3):
            x_spatial = self.linears_spatial[2 * idx](x_spatial)
            x_temporal = self.linears_temporal[2 * idx](x_temporal)
            x_spatial = x_spatial + t_embedding
            x_temporal = x_temporal + t_embedding

            cond_joint_emb = self.cond_joint[idx](cond_joint)
            cond_temporal_emb = self.cond_temporal[idx](cond_temporal)
            cond_spatial_emb = self.cond_spatial[idx](cond_spatial)

            x_spatial = x_spatial + cond_joint_emb + cond_spatial_emb
            x_temporal = x_temporal + cond_joint_emb + cond_temporal_emb

            x_spatial = self.linears_spatial[2 * idx + 1](x_spatial)
            x_temporal = self.linears_temporal[2 * idx + 1](x_temporal)

        x_spatial = self.linears_spatial[-1](x_spatial)
        x_temporal = self.linears_temporal[-1](x_temporal)

        x_output = torch.cat((x_temporal, x_spatial), dim=1)
        x_output_t = (x_output * alpha_t).sum(dim=1, keepdim=True)
        x_output_s = (x_output * alpha_s).sum(dim=1, keepdim=True)
        return torch.cat(
            (self.output_temporal(x_output_t), self.output_spatial(x_output_s)),
            dim=-1,
        )


# ---------------------------------------------------------------------------
# Gaussian diffusion process wrapper
# ---------------------------------------------------------------------------

class GaussianDiffusionST(nn.Module):
    """DSTPP Gaussian diffusion wrapper.

    Notes on units
    --------------
    The variational-bound terms are effectively nats per token dimension.
    DiffusionEventModel then
    converts those into the benchmark-facing ``test_nll`` in nats per event
    token and preserves the per-dim values in ``extra_metrics``.
    """

    def __init__(
        self,
        model: STDiffusionNet,
        *,
        seq_length: int,
        timesteps: int = 100,
        sampling_timesteps: int = 100,
        objective: str = "pred_noise",
        beta_schedule: str = "cosine",
        loss_type: str = "l2",
        p2_loss_weight_gamma: float = 0.0,
        p2_loss_weight_k: float = 1.0,
        ddim_sampling_eta: float = 1.0,
    ):
        super().__init__()
        assert objective in {"pred_noise", "pred_x0", "pred_v"}, (
            "objective must be 'pred_noise', 'pred_x0', or 'pred_v'"
        )
        assert loss_type in {"l1", "l2", "Euclid"}, f"Unknown loss_type: {loss_type!r}"

        self.model = model
        self.seq_length = int(seq_length)
        self.channels = self.model.channels
        self.self_condition = getattr(self.model, "self_condition", False)
        self.objective = objective
        self.loss_type = loss_type
        self.ddim_sampling_eta = float(ddim_sampling_eta)

        if beta_schedule == "linear":
            betas = _linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = _cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule!r}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = int(sampling_timesteps)
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=posterior_variance[1])),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, a: Tensor, t: Tensor, x_shape: Tuple) -> Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * noise

    def _predict_noise_from_start(self, x_t: Tensor, t: Tensor, x_0: Tensor) -> Tensor:
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (sqrt_recip * x_t - x_0) / sqrt_recipm1

    def _predict_v(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def _predict_start_from_v(self, x_t: Tensor, t: Tensor, v: Tensor) -> Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_mean_variance(self, x_start: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def model_predictions(
        self,
        x: Tensor,
        t: Tensor,
        x_self_cond: Optional[Tensor] = None,
        clip_x_start: bool = False,
        cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        model_out = self.model(x, t, x_self_cond, cond=cond)
        maybe_clip = (lambda z: z.clamp(-1.0, 1.0)) if clip_x_start else (lambda z: z)
        if self.objective == "pred_noise":
            pred_noise = model_out
            x_start = self._predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == "pred_x0":
            x_start = maybe_clip(model_out)
            pred_noise = self._predict_noise_from_start(x, t, x_start)
        else:
            v = model_out
            x_start = self._predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self._predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def q_posterior(
        self, x_start: Tensor, x_t: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        x: Tensor,
        t: Tensor,
        x_self_cond: Optional[Tensor] = None,
        clip_denoised: bool = True,
        cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        _, x_start = self.model_predictions(
            x,
            t,
            x_self_cond,
            clip_x_start=clip_denoised,
            cond=cond,
        )
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start,
            x_t=x,
            t=t,
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        b, _c, _n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                _pred_noise, x_self_cond = self.model_predictions(x, t)
                x_self_cond.detach_()

        model_out = self.model(x, t, x_self_cond, cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            target = self._predict_v(x_start, t, noise)

        if self.loss_type in {"l1", "l2"}:
            loss = F.l1_loss(model_out, target, reduction="none") if self.loss_type == "l1" else F.mse_loss(model_out, target, reduction="none")
        else:
            loss = F.pairwise_distance(model_out, target)

        if self.loss_type in {"l1", "l2"}:
            loss = loss.view(b, -1).mean(dim=1)
        loss = loss * self._extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img: Tensor, cond: Optional[Tensor]) -> Tensor:
        b = img.shape[0]
        img = normalize_to_neg_one_to_one(img)
        t = torch.randint(0, self.num_timesteps, (b,), device=img.device).long()
        return self.p_losses(img, t, cond=cond)

    # ------------------------------------------------------------------
    # VB / NLL approximation
    # ------------------------------------------------------------------

    def _vb_terms_bpd(
        self,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        cond: Optional[Tensor],
        clip_denoised: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        true_mean, _true_variance, true_log_variance_clipped = self.q_posterior(
            x_start=x_start,
            x_t=x_t,
            t=t,
        )
        model_mean, _model_variance, model_log_variance, pred_xstart = self.p_mean_variance(
            x=x_t,
            t=t,
            clip_denoised=clip_denoised,
            cond=cond,
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl_all = mean_flat(kl)
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start,
            model_mean,
            0.5 * model_log_variance,
        )
        decoder_nll_all = mean_flat(decoder_nll)

        kl_temporal = mean_flat(kl[:, :, :1])
        kl_spatial = mean_flat(kl[:, :, -(self.seq_length - 1) :])
        decoder_nll_temporal = mean_flat(decoder_nll[:, :, :1])
        decoder_nll_spatial = mean_flat(decoder_nll[:, :, -(self.seq_length - 1) :])

        output = torch.where(t == 0, decoder_nll_all, kl_all)
        output_temporal = torch.where(t == 0, decoder_nll_temporal, kl_temporal)
        output_spatial = torch.where(t == 0, decoder_nll_spatial, kl_spatial)
        return output, output_temporal, output_spatial, pred_xstart

    def _prior_bpd(self, x_start: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        b = x_start.shape[0]
        t = torch.full((b,), self.num_timesteps - 1, device=x_start.device, dtype=torch.long)
        qt_mean, _qt_variance, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean,
            logvar1=qt_log_variance,
            mean2=0.0,
            logvar2=0.0,
        )
        return (
            mean_flat(kl_prior),
            mean_flat(kl_prior[:, :, :1]),
            mean_flat(kl_prior[:, :, -(self.seq_length - 1) :]),
        )

    @torch.no_grad()
    def NLL_cal(
        self,
        x_start: Tensor,
        cond: Optional[Tensor],
        clip_denoised: bool = True,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return per-dim VB terms for total/temporal/spatial NLL."""
        x_start = normalize_to_neg_one_to_one(x_start)
        device = x_start.device
        batch_size = x_start.shape[0]

        vb_all: list = []
        vb_t_all: list = []
        vb_s_all: list = []

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            vb, vb_t, vb_s, _ = self._vb_terms_bpd(
                x_start=x_start,
                x_t=x_t,
                t=t_batch,
                cond=cond,
                clip_denoised=clip_denoised,
            )
            vb_all.append(vb.unsqueeze(1))
            vb_t_all.append(vb_t.unsqueeze(1))
            vb_s_all.append(vb_s.unsqueeze(1))

        vb_sum = torch.cat(vb_all, dim=1).sum(dim=1)
        vb_t_sum = torch.cat(vb_t_all, dim=1).sum(dim=1)
        vb_s_sum = torch.cat(vb_s_all, dim=1).sum(dim=1)

        prior, prior_t, prior_s = self._prior_bpd(x_start)
        return vb_sum + prior, vb_t_sum + prior_t, vb_s_sum + prior_s

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _p_sample(
        self,
        x: Tensor,
        t: int,
        cond: Optional[Tensor],
        x_self_cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        b = x.shape[0]
        batched_t = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _variance, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_t,
            x_self_cond=x_self_cond,
            clip_denoised=True,
            cond=cond,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def _p_sample_loop(self, shape: Tuple, cond: Optional[Tensor]) -> Tensor:
        device = self.betas.device
        x = torch.randn(shape, device=device)
        x_start = None
        for t in reversed(range(self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            x, x_start = self._p_sample(x, t, cond, x_self_cond=self_cond)
        return unnormalize_to_zero_to_one(x)

    @torch.no_grad()
    def _ddim_sample(self, shape: Tuple, cond: Optional[Tensor]) -> Tensor:
        batch = shape[0]
        device = self.betas.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start = self.model_predictions(
                x,
                time_cond,
                self_cond,
                clip_x_start=True,
                cond=cond,
            )

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x)
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return unnormalize_to_zero_to_one(x)

    @torch.no_grad()
    def sample(self, batch_size: int, cond: Optional[Tensor]) -> Tensor:
        """Generate samples. Returns tensor in [0, 1] range."""
        shape = (batch_size, self.channels, self.seq_length)
        if self.is_ddim_sampling:
            return self._ddim_sample(shape, cond)
        return self._p_sample_loop(shape, cond)


# ---------------------------------------------------------------------------
# EventModel adapter
# ---------------------------------------------------------------------------

@register_event("diffusion_stpp")
class DiffusionEventModel(EventModel):
    """Coarse EventModel wrapper for Diffusion STPP.

    Training loss
    -------------
    Single-step ELBO Monte Carlo estimate (fast, used in training/validation
    steps via UnifiedSTPP.forward → training_loss).

    eval_nll
    --------
    Full variational bound via NLL_cal.  NOTE: this loops over all
    ``timesteps`` diffusion steps and is expensive.  It is NOT called by the
    Lightning validation loop (which uses training_loss); it is intended for
    explicit post-training evaluation.  Returns NLL in approximate nats per
    event token.

    sample_native
    -------------
    DDIM (or DDPM) sampling conditioned on the last encoder state.
    """

    def __init__(
        self,
        *,
        denoising_model: STDiffusionNet,
        seq_length: int,
        timesteps: int,
        sampling_timesteps: int,
        objective: str,
        beta_schedule: str,
        loss_type: str,
    ):
        super().__init__()
        self.diffusion = GaussianDiffusionST(
            model=denoising_model,
            seq_length=seq_length,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=objective,
            beta_schedule=beta_schedule,
            loss_type=loss_type,
        )

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="elbo",
            metric_key="elbo",
            objective_description="variational ELBO (1-step)",
            nll_kind="approx",
            nll_description=(
                "full DSTPP variational bound; benchmark-facing test_nll is approximate "
                "nats/event-token, with per-dim diagnostics saved in extra_metrics"
            ),
            nll_footnote="‡ approx NLL (full VB)",
            has_native_sampler=True,
        )

    @staticmethod
    def _get(state_ctx: StateContext, key: str):
        if key not in state_ctx.payload:
            raise ValueError(f"DiffusionEventModel requires state['{key}'].")
        return state_ctx.payload[key]

    def training_loss(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms=None,
        x_field_at_events=None,
        marks=None,
        device=None,
    ) -> Dict[str, Tensor]:
        del times, locations, lengths, state_regularization_terms, x_field_at_events, marks

        img = self._get(state, "diff_img")
        cond = self._get(state, "diff_cond")
        total_events = self._get(state, "diff_total_events")

        if device is None:
            device = cond.device

        if img.shape[0] == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "loss": zero,
                "nll": zero,
                "nll_per_event": zero,
                "total_events": zero,
                "objective": zero,
            }

        loss = self.diffusion(img, cond)

        if not isinstance(total_events, Tensor):
            total_events = torch.as_tensor(total_events, device=loss.device, dtype=loss.dtype)
        total_events = total_events.to(device=loss.device, dtype=loss.dtype)

        return {
            "loss": loss,
            "nll": loss,
            "nll_per_event": loss,
            "total_events": total_events,
            "objective": loss,
            "objective_name": "diffusion_elbo",
        }

    def eval_nll(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms=None,
        x_field_at_events=None,
        marks=None,
        device=None,
    ) -> Dict[str, Tensor]:
        """Approximate NLL via full variational bound (NLL_cal).

        WARNING: loops over all ``timesteps`` diffusion steps per call.
        """
        del times, locations, lengths, state_regularization_terms, x_field_at_events, marks

        img = self._get(state, "diff_img")
        cond = self._get(state, "diff_cond")
        total_events = self._get(state, "diff_total_events")

        if device is None:
            device = cond.device

        if img.shape[0] == 0:
            zero = torch.tensor(0.0, device=device)
            return {"nll": zero, "total_events": zero}

        total_per_dim, temporal_per_dim, spatial_per_dim = self.diffusion.NLL_cal(img, cond)
        token_dim = self.diffusion.seq_length
        spatial_dim = token_dim - 1

        # Upstream numerics are effectively in nats per token dimension.
        total_upstream_per_dim = total_per_dim.mean()
        temporal_upstream_per_dim = temporal_per_dim.mean()
        spatial_upstream_per_dim = spatial_per_dim.mean()

        # Benchmark-facing metric: approximate nats per event token.
        nll_nats = total_upstream_per_dim * token_dim
        temporal_nll = temporal_upstream_per_dim
        spatial_nll = spatial_upstream_per_dim * spatial_dim

        if not isinstance(total_events, Tensor):
            total_events = torch.as_tensor(total_events, device=nll_nats.device, dtype=nll_nats.dtype)

        return {
            "nll": nll_nats.to(device=device),
            "temporal_nll": temporal_nll.to(device=device),
            "spatial_nll": spatial_nll.to(device=device),
            "total_events": total_events,
            "extra_metrics": {
                "test_nll_upstream_per_dim": total_upstream_per_dim.to(device=device),
                "temporal_nll_upstream_per_dim": temporal_upstream_per_dim.to(device=device),
                "spatial_nll_upstream_per_dim": spatial_upstream_per_dim.to(device=device),
            },
        }

    def sample_native(
        self,
        *,
        state: StateContext,
        batch_size: Optional[int] = None,
        device=None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Sample next events using DDIM (or DDPM) diffusion sampling.

        Returns
        -------
        Dict with key ``"samples"`` of shape (B, 1, seq_length) in [0, 1]
        range, where the last dimension is [delta_time, *location].
        """
        del kwargs
        cond = self._get(state, "diff_cond_last")
        if device is not None:
            cond = cond.to(device)

        b = batch_size or cond.shape[0]
        if cond.shape[0] != b:
            if cond.shape[0] == 1:
                cond = cond.expand(b, -1, -1)
            else:
                raise ValueError(
                    f"DiffusionEventModel.sample_native: cond batch {cond.shape[0]} != batch_size {b}."
                )

        samples = self.diffusion.sample(batch_size=b, cond=cond)
        return {"samples": samples}

    def query_surface(
        self,
        *,
        state: StateContext,
        grid_times: Tensor,
        grid_locs: Tensor,
        n_samples: int = 500,
        **kwargs,
    ) -> Tensor:
        """Surface query contract: proxy KDE from sample_native() for Diffusion STPP.

        Diffusion samples are in [0,1] normalized space; the spatial component
        (columns 1:) is extracted and evaluated via scipy gaussian_kde.
        """
        try:
            from scipy.stats import gaussian_kde
        except ImportError as exc:
            raise ImportError("scipy required for Diffusion proxy_kde surface.") from exc

        import torch

        device = grid_locs.device
        sample_out = self.sample_native(
            state=state,
            batch_size=n_samples,
            device=device,
        )
        samples = sample_out["samples"]                          # (B, 1, 1+d) or (B*N, 1+d)
        if samples.ndim == 3:
            B, N, dim = samples.shape
            samples = samples.reshape(B * N, dim)
        samples = samples[:n_samples]
        spatial_np = samples[:, 1:].cpu().detach().numpy().astype("float64")   # (N, d)
        grid_np    = grid_locs.cpu().numpy().astype("float64")                 # (G, d)
        kde = gaussian_kde(spatial_np.T)
        values = kde(grid_np.T).astype("float32")
        return torch.from_numpy(values).to(device=device, dtype=torch.float32)
