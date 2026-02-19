"""
Temporal Decoders — model f*(t | H) or equivalently λ*(t) and S*(t).

1. CumulativeHazardTemporal (NeuralSTPP):
   - λ*(t) = softplus(w · z(t) + b)
   - Λ*(t) = ∫_{t_n}^{t} λ*(τ) dτ  (via numerical quadrature)
   - f*(t) = λ*(t) · exp(-Λ*(t))

2. LogNormalMixtureTemporal (DeepSTPP):
   - f*(τ) = Σ_k π_k · LogNormal(τ; μ_k, σ_k)  where τ = t - t_n
   - Parameters (π, μ, σ) predicted from z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math


class CumulativeHazardTemporal(nn.Module):
    """
    NeuralSTPP-style temporal density via cumulative hazard.

    Given z(t) from the dynamics module, computes:
      λ*(t) = softplus(linear(z(t)))
      Λ*(t) = ∫_{t_n}^{t} λ*(τ) dτ   (Monte Carlo / quadrature)
      log f*(t) = log λ*(t) - Λ*(t)

    When dynamics = Identity, z(t) = z_n is constant and we parameterize
    λ*(t) = softplus(net(z_n, t - t_n)) to retain temporal expressiveness.

    Augmented ODE side channel
    --------------------------
    When NeuralODEDynamics runs in augmented mode it jointly integrates Λ
    alongside z(t) and caches the result. UnifiedSTPP._forward_sequential()
    detects this and sets self._precomputed_lambda before calling log_prob().
    log_prob() consumes and clears _precomputed_lambda, skipping quadrature.
    """

    def __init__(
        self,
        hidden_dim: int,
        field_cov_dim: int = 0,
        n_quad_points: int = 20,
        **kwargs,
    ):
        super().__init__()
        input_dim = hidden_dim + 1 + field_cov_dim  # z + elapsed time + covariates
        self.intensity_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self.n_quad = n_quad_points
        self.field_cov_dim = field_cov_dim
        # Side channel: set by UnifiedSTPP when augmented ODE provides Λ.
        self._precomputed_lambda: Optional[Tensor] = None

    def _intensity(
        self, z: Tensor, elapsed: Tensor, x_field: Optional[Tensor] = None
    ) -> Tensor:
        """Compute λ*(t) given state z and elapsed time."""
        parts = [z, elapsed]
        if x_field is not None and self.field_cov_dim > 0:
            parts.append(x_field)
        inp = torch.cat(parts, dim=-1)
        return self.intensity_net(inp).squeeze(-1)  # (B,)

    def _cumulative_hazard(
        self, z: Tensor, dt: Tensor, x_field: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute Λ*(t) = ∫_0^{dt} λ*(z, τ) dτ via Gauss-Legendre quadrature.
        """
        B = z.shape[0]
        device = z.device

        # Gauss-Legendre points on [0, 1], scaled to [0, dt]
        points, weights = _gauss_legendre(self.n_quad, device)
        # points: (Q,), weights: (Q,)

        # Scale to [0, dt]: τ_q = dt * points_q
        dt_col = dt.unsqueeze(-1)  # (B, 1)
        tau = dt_col * points.unsqueeze(0)  # (B, Q)
        w_scaled = dt_col * weights.unsqueeze(0)  # (B, Q)

        # Evaluate intensity at each quadrature point
        z_expand = z.unsqueeze(1).expand(B, self.n_quad, -1)  # (B, Q, h)
        z_flat = z_expand.reshape(B * self.n_quad, -1)
        tau_flat = tau.reshape(B * self.n_quad, 1)

        x_field_flat = None
        if x_field is not None and self.field_cov_dim > 0:
            # Assume field covariate is constant or we use the value at event time
            x_field_expand = x_field.unsqueeze(1).expand(B, self.n_quad, -1)
            x_field_flat = x_field_expand.reshape(B * self.n_quad, -1)

        lam = self._intensity(z_flat, tau_flat, x_field_flat)  # (B*Q,)
        lam = lam.reshape(B, self.n_quad)

        # Weighted sum
        Lambda = (lam * w_scaled).sum(dim=-1)  # (B,)
        return Lambda

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        log f*(t) = log λ*(t) - Λ*(t)

        If _precomputed_lambda has been set (augmented ODE side channel),
        it is used directly and then cleared; otherwise quadrature is run.
        """
        dt = (t - t_prev).squeeze(-1)  # (B,)
        lam = self._intensity(z, dt.unsqueeze(-1), x_field)

        if self._precomputed_lambda is not None:
            Lambda = self._precomputed_lambda
            self._precomputed_lambda = None  # consume once
        else:
            Lambda = self._cumulative_hazard(z, dt, x_field)

        return torch.log(lam + 1e-8) - Lambda

    def sample(
        self,
        z: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
        dt_max: float = 20.0,
        n_bisect: int = 30,
    ) -> Tensor:
        """
        Sample t via inverse CDF (bisection on the survival function).
        
        S*(t) = exp(-Λ*(t)), so for U ~ Uniform(0,1):
          t* = S*^{-1}(U)  ⟺  Λ*(t*) = -log(U)
        
        We find t* by bisection: search for dt s.t. Λ*(dt) = target.
        """
        B = z.shape[0]
        device = z.device

        # Target: -log(U), U ~ Uniform(0,1)
        u = torch.rand(B, device=device).clamp(min=1e-6)
        target = -torch.log(u)  # (B,)

        # Bisection on dt ∈ [0, dt_max]
        lo = torch.zeros(B, device=device)
        hi = torch.full((B,), dt_max, device=device)

        for _ in range(n_bisect):
            mid = (lo + hi) / 2  # (B,)
            Lambda_mid = self._cumulative_hazard(z, mid, x_field)  # (B,)
            # If Λ(mid) < target, we need to go further right
            lo = torch.where(Lambda_mid < target, mid, lo)
            hi = torch.where(Lambda_mid >= target, mid, hi)

        dt_sample = (lo + hi) / 2
        return t_prev.squeeze(-1) + dt_sample  # (B,)


class LogNormalMixtureTemporal(nn.Module):
    """
    DeepSTPP-style temporal density: mixture of log-normals.
    
    f*(τ) = Σ_k π_k · LogNormal(τ; μ_k, σ_k²)
    where τ = t - t_n > 0.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_components: int = 16,
        field_cov_dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        input_dim = hidden_dim + field_cov_dim
        self.n_components = n_components
        # Predict mixture parameters from hidden state
        self.param_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_components * 3),  # logits, mu, log_sigma
        )

    def _get_params(self, z: Tensor, x_field: Optional[Tensor] = None):
        if x_field is not None:
            inp = torch.cat([z, x_field], dim=-1)
        else:
            inp = z
        params = self.param_net(inp)  # (B, K*3)
        K = self.n_components
        logits = params[:, :K]
        mu = params[:, K : 2 * K]
        log_sigma = params[:, 2 * K : 3 * K]
        sigma = F.softplus(log_sigma) + 1e-4
        return logits, mu, sigma

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        log f*(τ) where τ = t - t_prev.
        """
        tau = (t - t_prev).squeeze(-1).clamp(min=1e-6)  # (B,)
        logits, mu, sigma = self._get_params(z, x_field)

        # Log-normal: log p(τ) = -log(τ) - log(σ√2π) - (log τ - μ)²/(2σ²)
        log_tau = torch.log(tau).unsqueeze(-1)  # (B, 1)
        normal_ll = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - 0.5 * ((log_tau - mu) / sigma) ** 2
        )  # (B, K)
        # Subtract log(τ) for the Jacobian of the log transform
        lognormal_ll = normal_ll - log_tau  # (B, K)

        # Mixture: log Σ_k π_k p_k(τ)
        log_pi = F.log_softmax(logits, dim=-1)  # (B, K)
        return torch.logsumexp(log_pi + lognormal_ll, dim=-1)  # (B,)

    def sample(
        self,
        z: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample τ ~ Σ_k π_k LogNormal(μ_k, σ_k²), return t = t_prev + τ.
        
        Direct ancestral sampling:
          1. Sample component k ~ Categorical(π)
          2. Sample τ ~ LogNormal(μ_k, σ_k²)
        """
        B = z.shape[0]
        logits, mu, sigma = self._get_params(z, x_field)

        # 1. Sample component indices
        pi = F.softmax(logits, dim=-1)  # (B, K)
        k = torch.multinomial(pi, num_samples=1).squeeze(-1)  # (B,)

        # 2. Gather parameters for selected components
        mu_k = mu[torch.arange(B, device=z.device), k]       # (B,)
        sigma_k = sigma[torch.arange(B, device=z.device), k]  # (B,)

        # 3. Sample: log(τ) ~ N(μ_k, σ_k²)  ⟹  τ = exp(μ_k + σ_k · ε)
        eps = torch.randn(B, device=z.device)
        tau = torch.exp(mu_k + sigma_k * eps)  # (B,)

        return t_prev.squeeze(-1) + tau  # (B,)


def _gauss_legendre(n: int, device) -> tuple:
    """Gauss-Legendre quadrature points and weights on [0, 1]."""
    # Use numpy for the nodes/weights, convert to torch
    import numpy as np
    points_np, weights_np = np.polynomial.legendre.leggauss(n)
    # Transform from [-1, 1] to [0, 1]
    points_np = 0.5 * (points_np + 1)
    weights_np = 0.5 * weights_np
    points = torch.tensor(points_np, dtype=torch.float32, device=device)
    weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    return points, weights
