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
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext


# ---------------------------------------------------------------------------
# Normalisation helpers (same convention as SMASH)
# ---------------------------------------------------------------------------

def normalize_to_neg_one_to_one(x: Tensor) -> Tensor:
    return x * 2 - 1


def unnormalize_to_zero_to_one(x: Tensor) -> Tensor:
    return (x + 1) * 0.5


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

def _linear_beta_schedule(timesteps: int) -> Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)


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
    """MLP denoising network for ST diffusion.

    Accepts a noisy event token x (B, 1, seq_length), a diffusion step t (B,),
    and an optional conditioning vector cond (B, 1, cond_dim).  Returns the
    model prediction (B, 1, seq_length) — either predicted noise or predicted
    x_0 depending on the parent GaussianDiffusionST's objective.
    """

    def __init__(
        self,
        seq_length: int,
        hidden_units: int = 64,
        condition: bool = True,
        cond_dim: int = 64,
    ):
        super().__init__()
        self.seq_length = int(seq_length)
        self.condition = bool(condition)

        time_dim = hidden_units
        self.time_mlp = nn.Sequential(
            _SinusoidalPosEmb(hidden_units),
            nn.Linear(hidden_units, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        in_dim = seq_length + time_dim + (cond_dim if condition else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_units * 2),
            nn.SiLU(),
            nn.Linear(hidden_units * 2, hidden_units * 2),
            nn.SiLU(),
            nn.Linear(hidden_units * 2, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, seq_length),
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        B = x.shape[0]
        x_flat = x.reshape(B, self.seq_length)
        t_emb = self.time_mlp(t.float())  # (B, time_dim)

        if self.condition and cond is not None:
            cond_flat = cond.reshape(B, -1)
            h = torch.cat([x_flat, t_emb, cond_flat], dim=-1)
        else:
            h = torch.cat([x_flat, t_emb], dim=-1)

        return self.net(h).unsqueeze(1)  # (B, 1, seq_length)


# ---------------------------------------------------------------------------
# Gaussian diffusion process wrapper
# ---------------------------------------------------------------------------

class GaussianDiffusionST(nn.Module):
    """DDPM / DDIM wrapper for spatiotemporal event tokens.

    Faithfully re-implements the GaussianDiffusion_ST machinery from the
    Diffusion STPP paper.  Event tokens have shape (B, 1, seq_length) where
    seq_length = 1 + spatial_dim (inter-event time + location coordinates).

    NLL_cal() computes the approximate variational bound in bits-per-dim.  It
    loops over all *timesteps* timesteps and is intentionally expensive —
    consistent with the original paper evaluation protocol.  Use
    sampling_timesteps for fast DDIM sampling instead.
    """

    def __init__(
        self,
        model: STDiffusionNet,
        *,
        seq_length: int,
        timesteps: int = 1000,
        sampling_timesteps: int = 50,
        objective: str = "pred_x0",
        beta_schedule: str = "cosine",
        loss_type: str = "l2",
    ):
        super().__init__()
        assert objective in ("pred_x0", "pred_noise"), f"Unknown objective: {objective!r}"
        assert loss_type in ("l1", "l2"), f"Unknown loss_type: {loss_type!r}"

        self.model = model
        self.seq_length = int(seq_length)
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = int(sampling_timesteps)
        self.objective = objective
        self.loss_type = loss_type
        self.is_ddim_sampling = sampling_timesteps < timesteps
        self.channels = 1

        if beta_schedule == "linear":
            betas = _linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = _cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule!r}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Forward process q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("log_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).log())
        self.register_buffer("sqrt_recip_alphas_cumprod", (1.0 / alphas_cumprod).sqrt())
        self.register_buffer("sqrt_recipm1_alphas_cumprod", (1.0 / alphas_cumprod - 1).sqrt())

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            posterior_variance.clamp(min=1e-20).log(),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, a: Tensor, t: Tensor, x_shape: Tuple) -> Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """Sample x_t ~ q(x_t | x_0) (forward diffusion)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    def _predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * noise

    def _predict_noise_from_start(self, x_t: Tensor, t: Tensor, x_0: Tensor) -> Tensor:
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (sqrt_recip * x_t - x_0) / sqrt_recipm1.clamp(min=1e-8)

    def model_predictions(
        self, x: Tensor, t: Tensor, cond: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Run denoising model; return (pred_noise, pred_x_start)."""
        model_out = self.model(x, t, cond)
        if self.objective == "pred_noise":
            pred_noise = model_out
            x_start = self._predict_start_from_noise(x, t, pred_noise)
        else:  # pred_x0
            x_start = model_out.clamp(-1.0, 1.0)
            pred_noise = self._predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def q_posterior(
        self, x_start: Tensor, x_t: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Posterior q(x_{t-1} | x_t, x_0): mean and log-variance."""
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, log_var

    def p_mean_variance(
        self, x: Tensor, t: Tensor, cond: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Model posterior: (model_mean, log_var, x_start)."""
        _, x_start = self.model_predictions(x, t, cond)
        model_mean, log_var = self.q_posterior(x_start, x, t)
        return model_mean, log_var, x_start

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        x_start: Tensor,
        cond: Optional[Tensor],
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Denoising loss (ELBO Monte Carlo estimate for a single random t)."""
        b = x_start.shape[0]
        if t is None:
            t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        model_out = self.model(x_t, t, cond)
        target = noise if self.objective == "pred_noise" else x_start
        if self.loss_type == "l1":
            return F.l1_loss(model_out, target)
        return F.mse_loss(model_out, target)

    def forward(self, img: Tensor, cond: Optional[Tensor]) -> Tensor:
        """Training entry point: normalize, sample t, compute denoising loss."""
        b = img.shape[0]
        img = normalize_to_neg_one_to_one(img)
        t = torch.randint(0, self.num_timesteps, (b,), device=img.device).long()
        return self.p_losses(img, cond, t)

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
        """Per-batch variational bound terms in bits-per-dim.

        Returns (vb, vb_temporal, vb_spatial, model_mean).
        """
        # True posterior
        true_mean, true_log_var = self.q_posterior(x_start, x_t, t)
        # Model posterior
        model_mean, log_var, _ = self.p_mean_variance(x_t, t, cond)

        # Expand scalar log-variances (B, 1, 1) → (B, 1, seq_len) for per-dim decomposition
        log_var = log_var.expand_as(x_t)
        true_log_var = true_log_var.expand_as(x_t)

        def _kl(mu1, lv1, mu2, lv2):
            return 0.5 * (-1.0 + lv2 - lv1 + (lv1.exp() + (mu1 - mu2) ** 2) / lv2.exp())

        dims = list(range(1, x_start.ndim))
        kl = _kl(true_mean, true_log_var, model_mean, log_var)
        kl_bpd = kl.mean(dim=dims) / math.log(2.0)

        # Decoder NLL at t=0 (Gaussian log-likelihood)
        dec_nll = 0.5 * (
            (x_start - model_mean) ** 2 / (log_var.exp() + 1e-8)
            + log_var + math.log(2.0 * math.pi)
        )
        dec_nll_bpd = dec_nll.mean(dim=dims) / math.log(2.0)

        # Temporal / spatial decomposition (first dim = time, rest = space)
        if x_start.shape[-1] > 1:
            kl_t = _kl(
                true_mean[..., :1], true_log_var[..., :1],
                model_mean[..., :1], log_var[..., :1],
            )
            kl_s = _kl(
                true_mean[..., 1:], true_log_var[..., 1:],
                model_mean[..., 1:], log_var[..., 1:],
            )
            kl_t_bpd = kl_t.mean(dim=dims) / math.log(2.0)
            kl_s_bpd = kl_s.mean(dim=dims) / math.log(2.0)
        else:
            kl_t_bpd = kl_bpd
            kl_s_bpd = torch.zeros_like(kl_bpd)

        # t=0 → use decoder NLL; t>0 → use KL
        output = torch.where(t == 0, dec_nll_bpd, kl_bpd)
        output_t = torch.where(t == 0, dec_nll_bpd, kl_t_bpd)
        output_s = torch.where(t == 0, torch.zeros_like(kl_s_bpd), kl_s_bpd)

        return output, output_t, output_s, model_mean

    def _prior_bpd(self, x_start: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """KL(q(x_T | x_0) || N(0,I)) in bits-per-dim — prior term."""
        b = x_start.shape[0]
        t = torch.full((b,), self.num_timesteps - 1, device=x_start.device, dtype=torch.long)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        log_var = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)

        # Expand to full token shape for per-dim decomposition
        mean = mean.expand_as(x_start)
        log_var = log_var.expand_as(x_start)

        dims = list(range(1, x_start.ndim))
        kl = 0.5 * (-1.0 - log_var + mean ** 2 + log_var.exp())
        kl_bpd = kl.mean(dim=dims) / math.log(2.0)

        if x_start.shape[-1] > 1:
            kl_t = 0.5 * (-1.0 - log_var[..., :1] + mean[..., :1] ** 2 + log_var[..., :1].exp())
            kl_s = 0.5 * (-1.0 - log_var[..., 1:] + mean[..., 1:] ** 2 + log_var[..., 1:].exp())
            kl_t_bpd = kl_t.mean(dim=dims) / math.log(2.0)
            kl_s_bpd = kl_s.mean(dim=dims) / math.log(2.0)
        else:
            kl_t_bpd, kl_s_bpd = kl_bpd, torch.zeros_like(kl_bpd)

        return kl_bpd, kl_t_bpd, kl_s_bpd

    @torch.no_grad()
    def NLL_cal(
        self,
        x_start: Tensor,
        cond: Optional[Tensor],
        clip_denoised: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Approximate NLL via variational bound (faithful to the original paper).

        Loops over ALL num_timesteps timesteps — intentionally expensive.
        Returns (total_bpd, temporal_bpd, spatial_bpd) as (B,) tensors
        where B = batch size (number of event tokens = total_events).

        Notes
        -----
        x_start must already be in [0, 1] (pre-minmax-normalized).
        Internally normalised to [-1, 1] as per the DDPM convention.
        """
        x_start = normalize_to_neg_one_to_one(x_start)
        device = x_start.device
        batch_size = x_start.shape[0]

        vb_all: list = []
        vb_t_all: list = []
        vb_s_all: list = []

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_t = self.q_sample(x_start=x_start, t=t_batch)
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
    def _p_sample(self, x: Tensor, t: int, cond: Optional[Tensor]) -> Tensor:
        b = x.shape[0]
        batched_t = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, log_var, _ = self.p_mean_variance(x, batched_t, cond)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def _p_sample_loop(self, shape: Tuple, cond: Optional[Tensor]) -> Tensor:
        device = self.betas.device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            x = self._p_sample(x, t, cond)
        return unnormalize_to_zero_to_one(x)

    @torch.no_grad()
    def _ddim_sample(self, shape: Tuple, cond: Optional[Tensor]) -> Tensor:
        device = self.betas.device
        # times goes from T-1 down to -1; t_now is always ≥ 0, t_next may be -1 (sentinel)
        times = torch.linspace(self.num_timesteps - 1, -1, self.sampling_timesteps + 1)
        time_pairs = list(zip(times[:-1].int().tolist(), times[1:].int().tolist()))

        x = torch.randn(shape, device=device)
        for t_now, t_next in time_pairs:
            t_batch = torch.full((shape[0],), t_now, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(x, t_batch, cond)
            x_start.clamp_(-1.0, 1.0)

            if t_next < 0:
                x = x_start
                continue

            alpha_bar = self.alphas_cumprod[t_now]
            alpha_bar_next = self.alphas_cumprod[t_next]
            c = (1.0 - alpha_bar_next).sqrt()
            x = x_start * alpha_bar_next.sqrt() + c * pred_noise

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
            training_objective="approx_nll",
            has_eval_nll=True,
            has_intensity=False,
            has_density=False,
            has_score=False,
            has_native_sampler=True,
            exposes_eventwise_terms=False,
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

        # NLL_cal expects x_start in [0, 1] (it normalises to [-1,1] internally).
        total_bpd, temporal_bpd, spatial_bpd = self.diffusion.NLL_cal(img, cond)

        # Convert bits-per-dim (per-event) → nats per event.
        # total_bpd is (N_flat,); mean over events then convert.
        nll_nats = total_bpd.mean() * self.diffusion.seq_length * math.log(2.0)

        if not isinstance(total_events, Tensor):
            total_events = torch.as_tensor(total_events, device=nll_nats.device, dtype=nll_nats.dtype)

        return {
            "nll": nll_nats.to(device=device),
            "nll_temporal_bpd": temporal_bpd.mean(),
            "nll_spatial_bpd": spatial_bpd.mean(),
            "total_events": total_events,
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
