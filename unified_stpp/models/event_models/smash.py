"""EventModel for SMASH score-matching objective and native sampling."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext
from ..model_registry import register_event


def normalize_to_neg_one_to_one(img: Tensor) -> Tensor:
    img_new = img * 2 - 1
    if img.size(-1) == 4:
        # Keep discrete mark channel unchanged.
        img_new[:, :, 1] = img[:, :, 1]
    return img_new


def unnormalize_to_zero_to_one(img: Tensor) -> Tensor:
    img_new = (img + 1) * 0.5
    if img.size(-1) == 4:
        img_new[:, :, 1] = img[:, :, 1]
    return img_new


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
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


class ScoreNet(nn.Module):
    """Score network: predicts score ∇ log p(x|H) and intensity λ."""

    def __init__(
        self,
        dim: int,
        num_units: int = 64,
        self_condition: bool = False,
        condition: bool = True,
        cond_dim: int = 0,
        num_types: int = 1,
    ):
        del dim
        super().__init__()
        self.channels = 1
        self.self_condition = bool(self_condition)
        self.condition = bool(condition)
        self.cond_dim = int(cond_dim)
        self.num_types = int(num_types)

        sinu_pos_emb = SinusoidalPosEmb(num_units)
        fourier_dim = num_units
        time_dim = num_units

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.linears_spatial = nn.ModuleList(
            [
                nn.Linear(2, num_units),
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

        self.output_intensity = nn.Sequential(
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_types),
            nn.Softplus(beta=1),
        )

        self.output_score = nn.Sequential(
            nn.Linear(num_units * 2, num_units),
            nn.ReLU(),
            nn.Linear(num_units, 2),
        )

        self.linear_t = nn.Sequential(
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, 2),
        )

        self.linear_s = nn.Sequential(
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, 2),
        )

        cond_all_dim = cond_dim * (4 if num_types > 1 else 3)
        self.cond_all = nn.Sequential(
            nn.Linear(cond_all_dim, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
        )

        self.cond_temporal = nn.ModuleList(
            [nn.Linear(cond_dim, num_units), nn.Linear(cond_dim, num_units), nn.Linear(cond_dim, num_units)]
        )
        self.cond_spatial = nn.ModuleList(
            [nn.Linear(cond_dim, num_units), nn.Linear(cond_dim, num_units), nn.Linear(cond_dim, num_units)]
        )
        self.cond_joint = nn.ModuleList(
            [nn.Linear(cond_dim, num_units), nn.Linear(cond_dim, num_units), nn.Linear(cond_dim, num_units)]
        )

    def _split_cond(self, cond: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden_dim = self.cond_dim
        cond_temporal = cond[:, :, :hidden_dim]
        cond_spatial = cond[:, :, hidden_dim : 2 * hidden_dim]
        cond_joint = cond[:, :, 2 * hidden_dim : 3 * hidden_dim]
        cond_mark = cond[:, :, 3 * hidden_dim :]
        return cond_temporal, cond_spatial, cond_joint, cond_mark

    def get_intensity(self, t: Tensor, cond: Tensor) -> Tensor:
        x_temporal = t
        cond_temporal, _cond_spatial, cond_joint, cond_mark = self._split_cond(cond)
        _ = self.cond_all(cond)

        for idx in range(3):
            x_temporal = self.linears_temporal[2 * idx](x_temporal)
            cond_joint_emb = self.cond_joint[idx](cond_joint)
            cond_temporal_base = (cond_temporal + cond_mark) if self.num_types > 1 else cond_temporal
            cond_temporal_emb = self.cond_temporal[idx](cond_temporal_base)
            x_temporal = x_temporal + cond_joint_emb + cond_temporal_emb
            x_temporal = self.linears_temporal[2 * idx + 1](x_temporal)

        x_temporal = self.linears_temporal[-1](x_temporal)
        return self.output_intensity(x_temporal)

    def get_score_loc(self, x: Tensor, cond: Tensor) -> Tensor:
        x_spatial, x_temporal = x[:, :, 1:], x[:, :, :1]

        cond_temporal, cond_spatial, cond_joint, cond_mark = self._split_cond(cond)
        cond_all = self.cond_all(cond)

        alpha_s = F.softmax(self.linear_s(cond_all), dim=-1).squeeze(1).unsqueeze(2)
        alpha_t = F.softmax(self.linear_t(cond_all), dim=-1).squeeze(1).unsqueeze(2)

        for idx in range(3):
            x_spatial = self.linears_spatial[2 * idx](x_spatial)
            x_temporal = self.linears_temporal[2 * idx](x_temporal)

            cond_joint_emb = self.cond_joint[idx](cond_joint)
            cond_temporal_base = (cond_temporal + cond_mark) if self.num_types > 1 else cond_temporal
            cond_temporal_emb = self.cond_temporal[idx](cond_temporal_base)
            cond_spatial_emb = self.cond_spatial[idx](cond_spatial)

            x_spatial = x_spatial + cond_joint_emb + cond_spatial_emb
            x_temporal = x_temporal + cond_joint_emb + cond_temporal_emb

            x_spatial = self.linears_spatial[2 * idx + 1](x_spatial)
            x_temporal = self.linears_temporal[2 * idx + 1](x_temporal)

        x_spatial = self.linears_spatial[-1](x_spatial)
        x_temporal = self.linears_temporal[-1](x_temporal)

        x_output_t = x_temporal * alpha_t[:, :1, :] + x_spatial * alpha_t[:, 1:2, :]
        x_output_s = x_temporal * alpha_s[:, :1, :] + x_spatial * alpha_s[:, 1:2, :]

        return self.output_score(torch.cat((x_output_t, x_output_s), dim=-1))

    def get_score(self, x: Tensor, cond: Tensor, sample: bool = True) -> Tensor:
        t = torch.autograd.Variable(x[:, :, :1], requires_grad=True)
        intensity = self.get_intensity(t, cond)
        intensity_log = (intensity + 1e-10).log()

        intensity_grad_t = torch.autograd.grad(
            intensity_log.sum(),
            t,
            retain_graph=True,
            create_graph=sample,
        )[0]
        score_t = intensity_grad_t - intensity
        score_loc = self.get_score_loc(x, cond)
        return torch.cat((score_t, score_loc), dim=-1)

    def get_score_mark(
        self,
        x: Tensor,
        mark: Tensor,
        cond: Tensor,
        sample: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        t = torch.autograd.Variable(x[:, :, :1], requires_grad=True)
        intensity = self.get_intensity(t, cond)

        mark_onehot = F.one_hot(mark.long(), num_classes=self.num_types)
        intensity_mark = (mark_onehot * intensity).sum(-1)
        intensity_mark_log = (intensity_mark + 1e-10).log()

        intensity_grad_t = torch.autograd.grad(
            intensity_mark_log.sum(),
            t,
            retain_graph=True,
            create_graph=sample,
        )[0]
        score_t = intensity_grad_t - intensity.sum(-1, keepdim=True)
        score_loc = self.get_score_loc(x, cond)
        score_mark = intensity / (intensity.sum(-1, keepdim=True) + 1e-10)

        return torch.cat((score_t, score_loc), dim=-1), score_mark


class ScoreMatchingProcess(nn.Module):
    """Score-matching training objective + annealed Langevin sampler."""

    def __init__(
        self,
        model: ScoreNet,
        sigma: Tuple[float, float],
        seq_length: int,
        num_noise: int = 50,
        sampling_timesteps: int = 500,
        langevin_step: float = 0.05,
        n_samples: int = 300,
        sampling_method: str = "normal",
        num_types: int = 1,
        loss_lambda: float = 1.0,
        loss_lambda2: float = 1.0,
        smooth: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.channels = int(n_samples)
        self.num_noise = int(num_noise)
        self.self_condition = self.model.self_condition
        self.is_marked = num_types > 1
        self.num_types = int(num_types)
        self.loss_lambda = float(loss_lambda)
        self.smooth = float(smooth)

        self.seq_length = int(seq_length)
        self.sampling_timesteps = int(sampling_timesteps)
        self.langevin_step = float(langevin_step)
        self.n_samples = int(n_samples)
        self.sampling_method = str(sampling_method)

        self.register_buffer(
            "loss_lambda2",
            torch.tensor([1.0, float(loss_lambda2), float(loss_lambda2)], dtype=torch.float32),
        )
        self.register_buffer(
            "sigma",
            torch.tensor([float(sigma[0]), float(sigma[1]), float(sigma[1])], dtype=torch.float32),
        )

    def sample_from_last(
        self,
        batch_size: int = 16,
        step: int = 100,
        is_last: bool = False,
        cond: Optional[Tensor] = None,
        last_sample: Optional[Tuple[Tensor, Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if cond is None:
            raise ValueError("SMASH.sample_from_last requires conditioning tensor `cond`.")

        shape = (batch_size, self.channels, 3)
        e = self.langevin_step
        sqrt_e = math.sqrt(e)

        sigma = self.sigma.to(device=cond.device, dtype=cond.dtype).view(1, 1, 3)

        if not self.is_marked:
            if last_sample is not None:
                x = normalize_to_neg_one_to_one(last_sample[0])
            else:
                x = torch.randn(shape, device=cond.device)

            if self.sampling_method == "normal":
                for _ in range(step):
                    z = torch.randn_like(x)
                    score = self.model.get_score(x, cond, False)
                    x = x + 0.5 * e * score.detach() + sqrt_e * z
                    x.clamp_(-1.0, 1.0)

            if is_last:
                score = self.model.get_score(x, cond, False)
                x_final = x + sigma.pow(2) * score.detach()
            else:
                x_final = x

            x_final.clamp_(-1.0, 1.0)
            img = unnormalize_to_zero_to_one(x_final)
            return img.detach(), None

        if last_sample is not None:
            x, score_mark = last_sample
            x = normalize_to_neg_one_to_one(x)
            if score_mark is None:
                raise ValueError("Marked SMASH sampling requires mark probabilities.")
            mark = torch.multinomial(
                score_mark.reshape(-1, self.num_types) + 1e-10,
                1,
                replacement=False,
            ).reshape(batch_size, self.n_samples)
        else:
            x = 0.5 * torch.randn(shape, device=cond.device)
            mark = torch.multinomial(
                torch.ones(self.num_types, device=cond.device),
                batch_size * self.n_samples,
                replacement=True,
            ).reshape(batch_size, self.n_samples)
            score_mark = None

        if self.sampling_method == "normal":
            for _ in range(step):
                z = torch.randn_like(x)
                score, score_mark = self.model.get_score_mark(x, mark, cond, False)
                x = x + 0.5 * e * score.detach() + sqrt_e * z
                x.clamp_(-1.0, 1.0)
                mark = torch.multinomial(
                    score_mark.detach().reshape(-1, self.num_types) + 1e-10,
                    1,
                    replacement=False,
                ).reshape(batch_size, self.n_samples)

        if score_mark is None:
            _, score_mark = self.model.get_score_mark(x, mark, cond, False)

        if is_last:
            score, _ = self.model.get_score_mark(x, mark, cond, False)
            x_final = x + sigma.pow(2) * score.detach()
            _, score_mark = self.model.get_score_mark(x_final, mark, cond, False)
            mark = torch.multinomial(
                score_mark.detach().reshape(-1, self.num_types) + 1e-10,
                1,
                replacement=False,
            ).reshape(batch_size, self.n_samples)
            for _ in range(200):
                z = torch.randn_like(x)
                score, score_mark = self.model.get_score_mark(x_final, mark, cond, False)
                x_final[:, :, 1:] = (
                    x_final[:, :, 1:]
                    + 0.5 * e * score.detach()[:, :, 1:]
                    + sqrt_e * z[:, :, 1:]
                )
        else:
            x_final = x

        x_final.clamp_(-1.0, 1.0)
        img = unnormalize_to_zero_to_one(x_final)
        return img.detach(), score_mark.detach()

    def p_losses(self, x_start: Tensor, noise: Optional[Tensor] = None, cond: Optional[Tensor] = None) -> Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start.repeat(1, self.num_noise, 1)))
        sigma = self.sigma.to(device=x_start.device, dtype=x_start.dtype).view(1, 1, 3)
        x = x_start + sigma * noise
        score = self.model.get_score(x, cond)
        loss = self.get_obj_denoise(x_start, x, score)
        return loss.mean()

    def p_losses_mark(self, x_start: Tensor, noise: Optional[Tensor] = None, cond: Optional[Tensor] = None) -> Tensor:
        x_mark = x_start[:, :, 1]
        x_start_cont = torch.cat((x_start[:, :, :1], x_start[:, :, 2:]), dim=-1)

        noise = default(noise, lambda: torch.randn_like(x_start_cont.repeat(1, self.num_noise, 1)))
        sigma = self.sigma.to(device=x_start.device, dtype=x_start.dtype).view(1, 1, 3)
        x = x_start_cont + sigma * noise

        score, score_mark = self.model.get_score_mark(x, x_mark - 1, cond)

        loss = self.get_obj_denoise(x_start_cont, x, score)
        loss = loss * self.loss_lambda2.to(device=loss.device, dtype=loss.dtype).view(1, 1, 3)
        loss_mark = self.get_obj_mark(x_mark, score_mark, smooth=self.smooth)

        return loss.mean() + self.loss_lambda * loss_mark.mean()

    def get_obj_denoise(self, x_start: Tensor, x: Tensor, score: Tensor) -> Tensor:
        sigma = self.sigma.to(device=x_start.device, dtype=x_start.dtype).view(1, 1, 3)
        target = (x_start - x) / sigma.pow(2)
        obj = 0.5 * (score - target).pow(2)
        obj = obj * sigma.pow(2)
        return obj

    def get_obj_mark(self, x_mark: Tensor, score_mark: Tensor, smooth: float = 0.0) -> Tensor:
        truth = x_mark.long() - 1
        one_hot = F.one_hot(truth, num_classes=self.num_types).float()
        one_hot = one_hot * (1 - smooth) + (1 - one_hot) * smooth / self.num_types
        log_prb = (score_mark + 1e-10).log()
        return -(one_hot * log_prb).sum(dim=-1)

    @torch.no_grad()
    def NLL_cal(
        self,
        x_start: Tensor,
        cond: Tensor,
        n_quad: int = 20,
    ) -> Tuple[float, float, float]:
        """Approximate NLL via native SMASH mechanics.

        Temporal: exact log-density of the model's implicit TPP (intensity quadrature).
        Spatial:  Tweedie approximation at the model's fixed sigma_s.

        Neither uses sampling, KDE, or external density estimators.

        Parameters
        ----------
        x_start : (N_total, 1, 1+loc_dim) — [delta_t, s1, s2] from smash_img
        cond    : (N_total, 1, cond_dim) — per-event conditioning from smash_cond
        n_quad  : number of quadrature points for the compensator integral

        Returns
        -------
        (nll_total, nll_temporal, nll_spatial) — floats, nat-sums over N_total events
        """
        N_total = x_start.shape[0]
        device = x_start.device
        dtype = x_start.dtype

        # ---- Temporal NLL via compensator integral --------------------------------
        delta_t = x_start[:, 0, 0]  # (N_total,) — inter-arrival times

        # t_grid[i, j] = delta_t[i] * j / (n_quad-1)  — (N_total, n_quad)
        t_frac = torch.linspace(0.0, 1.0, n_quad, device=device, dtype=dtype)
        t_grid = delta_t.unsqueeze(1) * t_frac.unsqueeze(0)  # (N_total, n_quad)

        # Batched intensity query: reshape to (N_total*n_quad, 1, 1)
        t_flat = t_grid.reshape(N_total * n_quad, 1, 1)
        cond_flat = cond.repeat_interleave(n_quad, dim=0)  # (N_total*n_quad, 1, cond_dim)
        lambda_flat = self.model.get_intensity(t_flat, cond_flat)  # (..., 1, num_types)
        lambda_flat = lambda_flat.sum(-1).squeeze(1)              # (N_total*n_quad,)
        lambda_grid = lambda_flat.reshape(N_total, n_quad)        # (N_total, n_quad)

        # Trapezoid rule: compensator[i] = integral_0^{delta_t[i]} lambda(u) du
        widths = t_grid[:, 1:] - t_grid[:, :-1]                  # (N_total, n_quad-1)
        compensator = (0.5 * (lambda_grid[:, :-1] + lambda_grid[:, 1:]) * widths).sum(dim=1)

        # log lambda at the actual event time delta_t_i
        lambda_at_dt = self.model.get_intensity(
            delta_t.view(N_total, 1, 1), cond
        )  # (N_total, 1, num_types)
        lambda_at_dt = lambda_at_dt.sum(-1).squeeze(1)            # (N_total,)
        log_lambda = torch.log(lambda_at_dt.clamp(min=1e-10))     # (N_total,)

        temporal_logp = log_lambda - compensator                  # (N_total,)

        # ---- Spatial NLL via Tweedie approximation at fixed sigma_s ---------------
        # Evaluate score_loc at the clean (unperturbed) event location.
        # Tweedie approximation: log p_{sigma_s}(s | H) ≈
        #     -sigma_s^2 * ||score_loc(s, cond)||^2 / 2 - (d/2) * log(2*pi*sigma_s^2)
        sigma_s = self.sigma[1].to(device=device, dtype=dtype)    # scalar

        score_loc = self.model.get_score_loc(x_start, cond)       # (N_total, 1, loc_dim)
        score_s = score_loc.squeeze(1)                             # (N_total, loc_dim)
        loc_dim = score_s.shape[-1]

        score_s_norm_sq = (score_s ** 2).sum(dim=-1)              # (N_total,)
        sigma_s_sq = sigma_s ** 2
        spatial_logp = (
            -sigma_s_sq * score_s_norm_sq / 2.0
            - (loc_dim / 2.0) * math.log(2.0 * math.pi * float(sigma_s_sq.item()))
        )  # (N_total,)

        nll_temporal = float(-temporal_logp.sum().item())
        nll_spatial  = float(-spatial_logp.sum().item())
        return nll_temporal + nll_spatial, nll_temporal, nll_spatial

    def forward(self, img: Tensor, cond: Tensor, *args, **kwargs) -> Tensor:
        _b, _c, n = img.shape
        if n != self.seq_length:
            raise ValueError(f"SMASH seq length must be {self.seq_length}, got {n}.")
        img = normalize_to_neg_one_to_one(img)

        if not self.is_marked:
            return self.p_losses(img, cond=cond, *args, **kwargs)
        return self.p_losses_mark(img, cond=cond, *args, **kwargs)


@register_event("smash")
class SMASHEventModel(EventModel):
    """Coarse EventModel wrapper for SMASH."""

    def __init__(
        self,
        *,
        score_net: ScoreNet,
        sigma_time: float,
        sigma_loc: float,
        seq_length: int,
        num_noise: int,
        sampling_timesteps: int,
        langevin_step: float,
        n_samples: int,
        sampling_method: str,
        num_types: int,
        loss_lambda: float,
        loss_lambda2: float,
        smooth: float,
    ):
        super().__init__()
        self.score_net = score_net
        self.score_matching = ScoreMatchingProcess(
            model=score_net,
            sigma=(sigma_time, sigma_loc),
            seq_length=seq_length,
            num_noise=num_noise,
            sampling_timesteps=sampling_timesteps,
            langevin_step=langevin_step,
            n_samples=n_samples,
            sampling_method=sampling_method,
            num_types=num_types,
            loss_lambda=loss_lambda,
            loss_lambda2=loss_lambda2,
            smooth=smooth,
        )
        self.num_types = int(num_types)

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="score_matching",
            metric_key="sm",
            objective_description="denoising score matching",
            nll_kind="approx",
            nll_description=(
                "framework-added approx NLL (non-upstream): exact temporal "
                "(intensity quadrature) + Tweedie spatial at fixed σ_s"
            ),
            nll_footnote="‡ framework-added approx NLL (non-upstream)",
            has_score=True,
            has_native_sampler=True,
        )

    @staticmethod
    def _get_state_term(state_ctx: StateContext, key: str):
        if key not in state_ctx.payload:
            raise ValueError(f"SMASHEventModel requires state['{key}'].")
        return state_ctx.payload[key]

    @staticmethod
    def _broadcast_cond(cond: Tensor, batch_size: int) -> Tensor:
        if cond.shape[0] == batch_size:
            return cond
        if cond.shape[0] == 1:
            return cond.expand(batch_size, -1, -1)
        raise ValueError(
            f"SMASH conditioning batch mismatch: cond={cond.shape[0]} query={batch_size}."
        )

    def training_loss(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms: Optional[Dict[str, Tensor]] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        del times, locations, lengths, state_regularization_terms, x_field_at_events, marks
        img = self._get_state_term(state, "smash_img")
        cond = self._get_state_term(state, "smash_cond")
        total_events = self._get_state_term(state, "smash_total_events")

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

        # Score-matching objective needs autograd inside score computation.
        with torch.enable_grad():
            loss = self.score_matching(img, cond)

        if not isinstance(total_events, Tensor):
            total_events = torch.as_tensor(total_events, device=loss.device, dtype=loss.dtype)
        total_events = total_events.to(device=loss.device, dtype=loss.dtype)

        return {
            "loss": loss,
            # Compatibility key for current trainer logging/checkpoint flow.
            "nll": loss,
            "nll_per_event": loss,
            "total_events": total_events,
            "objective": loss,
            "objective_name": "score_matching",
        }

    def eval_nll(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms: Optional[Dict[str, Tensor]] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        """Framework-added approximate per-event NLL for benchmark reporting.

        Temporal NLL: exact log-density of the model's implicit TPP
            (numerical quadrature of the compensator integral).
        Spatial NLL:  Tweedie approximation at the model's fixed sigma_s.

        This metric is not part of the upstream SMASH training/evaluation code.
        Neither component uses sampling, KDE, or any external density estimator.
        val/sm (validation/training) remains the score-matching objective;
        this method is invoked only at test time for benchmark reporting.
        """
        del times, locations, lengths, state_regularization_terms, x_field_at_events, marks

        smash_img  = self._get_state_term(state, "smash_img")   # (N_total, 1, 1+loc_dim)
        smash_cond = self._get_state_term(state, "smash_cond")  # (N_total, 1, cond_dim)

        if device is None:
            device = smash_cond.device

        if smash_img.shape[0] == 0:
            z = torch.tensor(0.0, device=device)
            return {"nll": z, "loss": z, "temporal_nll": 0.0, "spatial_nll": 0.0, "total_events": z}

        nll_total, nll_temporal, nll_spatial = self.score_matching.NLL_cal(smash_img, smash_cond)
        n = float(smash_img.shape[0])
        nll_per_event = torch.tensor(nll_total / n, device=device)
        return {
            "nll":          nll_per_event,
            "loss":         nll_per_event,
            "temporal_nll": nll_temporal / n,
            "spatial_nll":  nll_spatial / n,
            "total_events": torch.tensor(n, device=device),
        }

    def score(
        self,
        *,
        state: StateContext,
        query_times: Tensor,
        query_locations: Tensor,
        query_lengths: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Tensor:
        del query_lengths, x_field_at_events
        if query_times.ndim == 1:
            query_times = query_times.unsqueeze(-1)
        if query_locations.ndim == 3 and query_locations.shape[1] == 1:
            query_locations = query_locations.squeeze(1)
        if query_locations.ndim != 2:
            raise ValueError("SMASH score expects query_locations shape (B, 2).")

        if device is None:
            device = query_times.device

        cond = self._get_state_term(state, "smash_cond_last").to(device=device)
        batch_size = query_times.shape[0]
        cond = self._broadcast_cond(cond, batch_size)

        x = torch.cat(
            [
                query_times.to(device=device, dtype=cond.dtype),
                query_locations.to(device=device, dtype=cond.dtype),
            ],
            dim=-1,
        ).unsqueeze(1)

        if self.num_types > 1:
            if marks is None:
                raise ValueError("Marked SMASH score() requires marks.")
            mark = marks.long()
            if mark.ndim == 2 and mark.shape[1] == 1:
                mark = mark.squeeze(1)
            if mark.min() >= 1:
                mark = mark - 1
            score, _score_mark = self.score_net.get_score_mark(
                x,
                mark.unsqueeze(1),
                cond,
                sample=False,
            )
            return score.squeeze(1)

        score = self.score_net.get_score(x, cond, sample=False)
        return score.squeeze(1)

    def sample_native(
        self,
        *,
        state: StateContext,
        step: int = 100,
        is_last: bool = False,
        last_sample: Optional[Tuple[Tensor, Optional[Tensor]]] = None,
        n_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        device=None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        del kwargs
        cond = self._get_state_term(state, "smash_cond_last")
        if device is None:
            device = cond.device
        cond = cond.to(device=device)

        if batch_size is None:
            batch_size = cond.shape[0]
        cond = self._broadcast_cond(cond, batch_size)

        prev_n_samples = self.score_matching.n_samples
        prev_channels = self.score_matching.channels
        if n_samples is not None:
            self.score_matching.n_samples = int(n_samples)
            self.score_matching.channels = int(n_samples)

        try:
            samples, score_mark = self.score_matching.sample_from_last(
                batch_size=batch_size,
                step=step,
                is_last=is_last,
                cond=cond,
                last_sample=last_sample,
            )
        finally:
            if n_samples is not None:
                self.score_matching.n_samples = prev_n_samples
                self.score_matching.channels = prev_channels

        out: Dict[str, Tensor] = {"samples": samples}
        if score_mark is not None:
            mark = torch.multinomial(
                score_mark.reshape(-1, self.num_types) + 1e-10,
                1,
                replacement=False,
            ).reshape(score_mark.shape[0], score_mark.shape[1])
            out["mark_probs"] = score_mark
            # Return one-based marks to match SMASH preprocessing convention.
            out["marks"] = mark + 1
        return out

    def sample_upstream_flattened(
        self,
        *,
        state: StateContext,
        per_step: int = 250,
        total_steps: Optional[int] = None,
        n_samples: Optional[int] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        """Faithful upstream SMASH sampling over all flattened valid histories.

        This mirrors the upstream decoder sampling semantics more closely than
        sample_native(), which is a framework convenience for next-event queries.
        """
        cond = self._get_state_term(state, "smash_cond")
        if device is None:
            device = cond.device
        cond = cond.to(device=device)

        batch_size = cond.shape[0]
        target_steps = int(total_steps or self.score_matching.sampling_timesteps)
        if batch_size == 0:
            channels = int(n_samples or self.score_matching.n_samples)
            samples = torch.zeros(
                0,
                channels,
                3,
                device=device,
                dtype=cond.dtype,
            )
            return {"samples": samples}

        prev_n_samples = self.score_matching.n_samples
        prev_channels = self.score_matching.channels
        if n_samples is not None:
            self.score_matching.n_samples = int(n_samples)
            self.score_matching.channels = int(n_samples)

        current_step = 0
        last_sample: Optional[Tuple[Tensor, Optional[Tensor]]] = None
        samples: Optional[Tensor] = None
        score_mark: Optional[Tensor] = None
        try:
            while current_step < target_steps:
                step = min(int(per_step), target_steps - current_step)
                is_last = current_step + step >= target_steps
                samples, score_mark = self.score_matching.sample_from_last(
                    batch_size=batch_size,
                    step=step,
                    is_last=is_last,
                    cond=cond,
                    last_sample=last_sample,
                )
                last_sample = (samples, score_mark)
                current_step += step
        finally:
            if n_samples is not None:
                self.score_matching.n_samples = prev_n_samples
                self.score_matching.channels = prev_channels

        if samples is None:
            raise RuntimeError("SMASH upstream sampling produced no samples.")

        out: Dict[str, Tensor] = {"samples": samples}
        if score_mark is not None:
            mark = torch.multinomial(
                score_mark.reshape(-1, self.num_types) + 1e-10,
                1,
                replacement=False,
            ).reshape(score_mark.shape[0], score_mark.shape[1])
            out["mark_probs"] = score_mark
            out["marks"] = mark + 1
        return out

    def query_surface(
        self,
        *,
        state: StateContext,
        grid_times: Tensor,
        grid_locs: Tensor,
        n_samples: int = 500,
        **kwargs,
    ) -> Tensor:
        """Surface query contract: proxy KDE from sample_native() for SMASH.

        SMASH score-based sampling uses torch.autograd.grad internally, so this
        method uses torch.enable_grad() to allow gradient computation even when
        called inside a torch.no_grad() context.
        """
        try:
            from scipy.stats import gaussian_kde
        except ImportError as exc:
            raise ImportError("scipy required for SMASH proxy_kde surface.") from exc

        device = grid_locs.device
        import torch

        with torch.enable_grad():
            sample_out = self.sample_native(
                state=state,
                n_samples=n_samples,
                batch_size=n_samples,
                device=device,
            )

        samples = sample_out["samples"]                          # (B, N, 1+d) or (B*N, 1+d)
        if samples.ndim == 3:
            B, N, dim = samples.shape
            samples = samples.reshape(B * N, dim)
        samples = samples[:n_samples]
        spatial_np = samples[:, 1:].cpu().detach().numpy().astype("float64")   # (N, d)
        grid_np    = grid_locs.cpu().numpy().astype("float64")                 # (G, d)
        kde = gaussian_kde(spatial_np.T)
        values = kde(grid_np.T).astype("float32")
        return torch.from_numpy(values).to(device=device, dtype=torch.float32)
