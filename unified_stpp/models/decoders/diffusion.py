"""
Diffusion Decoder — score-based model for joint f*(t, s | H).

Used by DSTPP. Models the joint density of (τ, s) where τ = t - t_n
using a denoising score matching objective.

Training: minimize E_{σ, x, ε}[||s_θ(x + σε; z, σ) - (-ε/σ)||²]
Inference: probability flow ODE or variational lower bound.

This implementation uses a simplified variance-exploding SDE
with discrete noise levels for tractability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math

from ..base import Decoder


class ScoreNetwork(nn.Module):
    """
    Score network s_θ(x_noisy, σ; z, X_field) that estimates ∇_x log p_σ(x).
    
    x = (τ, s) ∈ R^{1+d}, conditioned on latent state z and noise level σ.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        field_cov_dim: int = 0,
        n_layers: int = 3,
    ):
        super().__init__()
        self.data_dim = data_dim
        # Input: x (1+d) + σ_embed (hidden) + z (hidden) + X_field (r)
        cond_dim = hidden_dim + hidden_dim + field_cov_dim

        layers = []
        in_dim = data_dim + cond_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else data_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        # Final layer without activation
        layers[-1] = nn.Linear(hidden_dim, data_dim)
        self.net = nn.Sequential(*layers)

        # Noise level embedding
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x_noisy: Tensor,
        sigma: Tensor,
        z: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x_noisy: (B, 1+d) — noisy (τ, s)
            sigma: (B, 1) — noise level
            z: (B, h) — latent state
            x_field: (B, r) optional
        Returns:
            score: (B, 1+d) — estimated ∇_x log p_σ(x)
        """
        sigma_emb = self.sigma_embed(sigma)  # (B, h)
        parts = [x_noisy, sigma_emb, z]
        if x_field is not None:
            parts.append(x_field)
        inp = torch.cat(parts, dim=-1)
        return self.net(inp)


class DiffusionDecoder(Decoder):
    """
    Joint spatiotemporal density via score-based diffusion.
    
    Noise schedule: geometric series σ_min ... σ_max with L levels.
    Training loss: denoising score matching.
    Log-likelihood: variational bound via discrete noise levels.
    """

    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        field_cov_dim: int = 0,
        n_noise_levels: int = 50,
        sigma_min: float = 0.01,
        sigma_max: float = 5.0,
        n_score_layers: int = 3,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim, spatial_dim=spatial_dim)
        self.data_dim = 1 + spatial_dim  # (τ, s)
        self.n_levels = n_noise_levels

        # Geometric noise schedule
        sigmas = torch.exp(
            torch.linspace(math.log(sigma_min), math.log(sigma_max), n_noise_levels)
        )
        self.register_buffer("sigmas", sigmas)

        self.score_net = ScoreNetwork(
            data_dim=self.data_dim,
            hidden_dim=hidden_dim,
            field_cov_dim=field_cov_dim,
            n_layers=n_score_layers,
        )

    def _dsm_loss(
        self,
        x: Tensor,
        z: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Denoising score matching loss.
        
        Args:
            x: (B, 1+d) — clean data (τ, s)
            z: (B, h) — conditioning
        Returns:
            loss: scalar
        """
        B = x.shape[0]
        device = x.device

        # Sample random noise levels
        idx = torch.randint(0, self.n_levels, (B,), device=device)
        sigma = self.sigmas[idx].unsqueeze(-1)  # (B, 1)

        # Add noise
        eps = torch.randn_like(x)
        x_noisy = x + sigma * eps

        # Predict score
        score_pred = self.score_net(x_noisy, sigma, z, x_field)

        # Target score: -eps / sigma
        target = -eps / sigma

        # Weighted MSE (weight by σ² for balanced training across noise levels)
        loss = ((score_pred - target) ** 2).sum(dim=-1)  # (B,)
        weight = sigma.squeeze(-1) ** 2
        return (loss * weight).mean()

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Approximate log f*(t, s) via the ELBO / sliced score matching bound.
        
        Uses the identity:
          log p(x) ≥ -Σ_{l=0}^{L-1} (σ_{l+1}² - σ_l²)/(2σ_l²) · E[||s_θ(x+σε)||²]
        
        This is an approximation. For exact likelihood, use probability flow ODE.
        Here we use a simpler MC estimate for the proof of concept.
        """
        B = z.shape[0]
        tau = (t - t_prev).clamp(min=1e-6)  # (B, 1)
        x = torch.cat([tau, s], dim=-1)  # (B, 1+d)

        # Evaluate score at the smallest noise level (approximation)
        sigma_min = self.sigmas[0].unsqueeze(0).unsqueeze(0).expand(B, 1)
        score = self.score_net(x, sigma_min, z, x_field)

        # Approximate log p ≈ -0.5 ||x||² + score-based correction
        # This is a rough approximation; proper implementation would use
        # the probability flow ODE or the variational bound.
        # For training, we use the DSM loss directly (see nll method).
        log_base = -0.5 * self.data_dim * math.log(2 * math.pi) - 0.5 * (x ** 2).sum(-1)
        correction = 0.5 * sigma_min.squeeze(-1) ** 2 * (score ** 2).sum(-1)
        return log_base - correction

    def nll(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Training surrogate via denoising score matching (DSM) loss.

        .. warning::
            **This is NOT the true negative log-likelihood.**

            The returned value is the weighted DSM objective
            ``E[σ²||s_θ(x+σε) − (−ε/σ)||²]``, which is used as a
            training surrogate for the intractable exact NLL.  It is NOT
            comparable to the per-event NLL returned by FactorizedDecoder
            or AutoIntDecoder, and MUST NOT be reported alongside those
            values in likelihood tables.

            True per-event NLL for a diffusion model requires integrating
            the probability-flow ODE (see Song et al., 2021, "Score-Based
            Generative Modeling through SDEs").  That path is not yet
            implemented here.

        TODO: implement probability-flow ODE evaluation to obtain a true,
        comparable per-event NLL for DiffusionDecoder / DSTPP.
        """
        tau = (t - t_prev).clamp(min=1e-6)  # (B, 1)
        x = torch.cat([tau, s], dim=-1)  # (B, 1+d)
        return self._dsm_loss(x, z, x_field)

    def sample(
        self,
        z: Tensor,
        t_prev: Tensor,
        x_field_fn=None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample (t, s) via annealed Langevin dynamics.
        """
        B = z.shape[0]
        device = z.device

        # Initialize from prior
        x = torch.randn(B, self.data_dim, device=device) * self.sigmas[-1]

        # Annealed Langevin dynamics (reverse noise levels)
        n_steps_per_level = 5
        for i in reversed(range(self.n_levels)):
            sigma = self.sigmas[i]
            step_size = 0.5 * (sigma / self.sigmas[0]) ** 2 * 0.01

            for _ in range(n_steps_per_level):
                sigma_inp = sigma.unsqueeze(0).unsqueeze(0).expand(B, 1)
                x_field = None
                if x_field_fn is not None:
                    x_field = x_field_fn(x[:, 0:1] + t_prev)  # τ → t
                score = self.score_net(x, sigma_inp, z, x_field)
                noise = torch.randn_like(x)
                x = x + step_size * score + (2 * step_size).sqrt() * noise

        # Ensure τ > 0
        tau = F.softplus(x[:, 0:1])
        s = x[:, 1:]
        t = tau + t_prev
        return t, s
