"""
Spatial Decoders — model f*(s | t, H).

1. CNFSpatial (NeuralSTPP):
   - Continuous Normalizing Flow conditioned on z(t)
   - Base distribution: N(0, I) on R^d
   - Flow: ds/dτ = v_θ(s, τ; z(t), X_field) for τ ∈ [0, 1]
   - log f*(s|t) via change of variables

2. GaussianMixtureSpatial (DeepSTPP):
   - f*(s|t) = Σ_k w_k · N(s; μ_k, σ_k² I)
   - Parameters predicted from z(t)

Optimizations in CNFSpatial:
- use_adjoint=True: uses odeint_adjoint for O(1)-memory backprop.
- divergence_bf: exact Jacobian diagonal for spatial_dim ≤ 3 (d backward
  passes instead of stochastic Hutchinson).
- Noise caching: the Hutchinson noise vector e is sampled once before the ODE
  solve and reused across all evaluations within that solve, reducing variance
  and saving compute.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math

try:
    from torchdiffeq import odeint as _odeint_std
    from torchdiffeq import odeint_adjoint as _odeint_adj
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


# ============================================================================
# Divergence helpers
# ============================================================================

def divergence_bf(f: Tensor, y: Tensor, training: bool) -> Tensor:
    """
    Exact divergence via brute-force Jacobian diagonal. O(d) backward passes.

    For d=2 (the common case) this costs 2 passes vs. one stochastic
    Hutchinson pass — but is exact and bias-free.

    Args:
        f: velocity field output (B, d) — must share a graph with y.
        y: spatial position (B, d) with requires_grad=True in the graph.
        training: if True, build second-order graph for backprop through loss.
    Returns:
        divergence: (B,)
    """
    sum_diag = 0.0
    for i in range(f.shape[1]):
        retain = training or (i < f.shape[1] - 1)
        grad_i = torch.autograd.grad(
            f[:, i].sum(), y,
            create_graph=training,
            retain_graph=retain,
        )[0]
        sum_diag = sum_diag + grad_i[:, i]
    return sum_diag


def _hutchinson_trace(v: Tensor, x: Tensor, e: Tensor) -> Tensor:
    """
    Hutchinson trace estimator: E[ε^T (dv/dx) ε] = tr(dv/dx).

    Uses a pre-sampled noise vector e (same e for all ODE steps in one solve).
    """
    vjp = torch.autograd.grad(v, x, e, create_graph=True, retain_graph=True)[0]
    return (vjp * e).sum(dim=-1)


# ============================================================================
# CNF Spatial Decoder (NeuralSTPP)
# ============================================================================


class CNFVelocityField(nn.Module):
    """Velocity field v(s, τ; z, X) for the spatial CNF."""

    def __init__(self, spatial_dim: int, hidden_dim: int, field_cov_dim: int = 0):
        super().__init__()
        # Input: s (d) + τ (1) + z (h) + X_field (r)
        input_dim = spatial_dim + 1 + hidden_dim + field_cov_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, spatial_dim),
        )
        self.spatial_dim = spatial_dim
        self._z_cond = None   # conditioning state, set before each ODE solve
        self._x_cond = None   # conditioning covariates
        self._e: Optional[Tensor] = None  # cached Hutchinson noise (reset each solve)

    def forward(self, tau: Tensor, s: Tensor) -> Tensor:
        """
        Args:
            tau: scalar tensor — flow time
            s: (B, d) or (B, d+1) if augmented with log-det accumulator
        Returns:
            ds/dτ: same shape as s
        """
        d = self.spatial_dim
        if s.shape[-1] > d:
            # Augmented state: last dim is log-det accumulator
            s_actual = s[:, :d]
        else:
            s_actual = s

        B = s_actual.shape[0]
        tau_expand = tau.expand(B, 1)
        parts = [s_actual, tau_expand, self._z_cond]
        if self._x_cond is not None:
            parts.append(self._x_cond)
        inp = torch.cat(parts, dim=-1)
        v = self.net(inp)  # (B, d)

        if s.shape[-1] > d:
            # Compute div(v) for the log-det accumulator.
            #
            # When odeint_adjoint runs its *forward* ODE integration it operates
            # inside a no-grad context, so intermediate states lose requires_grad.
            # We therefore always use torch.enable_grad() + a fresh leaf for
            # s_actual so that autograd.grad() can differentiate v w.r.t. s.
            # This is the standard pattern in CNF+adjoint implementations.
            # During the adjoint *backward* re-evaluation torchdiffeq itself
            # provides y with requires_grad=True, so using a detached leaf is
            # a minor approximation (trace Jacobian w.r.t. s is not propagated
            # through the adjoint state, but velocity Jacobian is correct).
            with torch.enable_grad():
                s_leaf = s_actual.detach().requires_grad_(True)
                parts_div = [s_leaf, tau_expand.detach(), self._z_cond.detach()]
                if self._x_cond is not None:
                    parts_div.append(self._x_cond.detach())
                v_div = self.net(torch.cat(parts_div, dim=-1))

                if self.spatial_dim <= 3:
                    trace = divergence_bf(v_div, s_leaf, self.training)
                else:
                    # Sample noise once per solve, reuse across ODE steps.
                    if self._e is None:
                        self._e = torch.randn_like(s_leaf)
                    trace = _hutchinson_trace(v_div, s_leaf, self._e)

            return torch.cat([v, -trace.unsqueeze(-1)], dim=-1)
        return v


class CNFSpatial(nn.Module):
    """
    Continuous Normalizing Flow for spatial density f*(s | t, H).

    Forward (density evaluation):
      Solve ODE backward: s(1) = s_observed → s(0) ~ base_dist
      log f*(s) = log p_0(s(0)) + ∫_0^1 tr(dv/ds) dτ

    Parameters
    ----------
    use_adjoint : bool
        Use odeint_adjoint for O(1)-memory backprop. Default True.
    n_steps : int
        For the 'euler' solver: fixed number of steps over [0,1]. 0 = adaptive.
    """

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        field_cov_dim: int = 0,
        solver: str = "dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        use_adjoint: bool = True,
        n_steps: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.velocity = CNFVelocityField(spatial_dim, hidden_dim, field_cov_dim)
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.use_adjoint = use_adjoint
        self.n_steps = n_steps

    def _run_odeint(self, t_span: Tensor, s_aug: Tensor) -> Tensor:
        """Run ODE integration, dispatching between adjoint and standard."""
        if not HAS_TORCHDIFFEQ:
            return _euler_solve_simple(self.velocity, s_aug, t_span,
                                       n_steps=self.n_steps if self.n_steps > 0 else 50)

        ode_kwargs: dict = {
            "method": self.solver,
            "atol": self.atol,
            "rtol": self.rtol,
        }

        if self.solver == "euler" and self.n_steps > 0:
            dt_total = abs((t_span[-1] - t_span[0]).item())
            if dt_total > 1e-8:
                ode_kwargs["options"] = {"step_size": dt_total / self.n_steps}

        if self.use_adjoint:
            traj = _odeint_adj(self.velocity, s_aug, t_span, **ode_kwargs)
        else:
            traj = _odeint_std(self.velocity, s_aug, t_span, **ode_kwargs)

        return traj[-1]  # state at last time point

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute log f*(s | z(t)).

        Integrate backward from τ=1 (observed s) to τ=0 (base space),
        accumulating the log-det Jacobian.
        """
        B, d = s.shape
        self.velocity._z_cond = z
        self.velocity._x_cond = x_field
        self.velocity._e = None   # reset cached noise — sample fresh each solve

        # Augmented state: [s, log_det] where log_det starts at 0
        s_aug = torch.cat([s, torch.zeros(B, 1, device=s.device)], dim=-1)

        # Need gradients w.r.t. s for the divergence computation
        s_aug = s_aug.requires_grad_(True)

        # Integrate backward: τ from 1 → 0
        t_span = torch.tensor([1.0, 0.0], device=s.device)
        s_aug_0 = self._run_odeint(t_span, s_aug)

        s_0     = s_aug_0[:, :d]
        # Backward solve stores the accumulated change term over τ: 1 -> 0.
        # To recover log p_data(s), subtract this accumulator from log p0(s0).
        log_det = s_aug_0[:, d]

        # Base distribution log-prob: standard normal
        log_p0 = -0.5 * (d * math.log(2 * math.pi) + (s_0 ** 2).sum(dim=-1))

        return log_p0 - log_det  # (B,)

    def sample(
        self,
        z: Tensor,
        t: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample s ~ f*(s | z(t)) by running the CNF forward.

        Draw z_0 ~ N(0, I), then solve ODE forward: τ from 0 → 1.
        No log-det tracking needed for sampling.
        """
        B = z.shape[0]
        d = self.spatial_dim
        device = z.device

        self.velocity._z_cond = z
        self.velocity._x_cond = x_field
        self.velocity._e = None  # no noise needed for sampling (no trace)

        # Draw from base distribution
        s_0 = torch.randn(B, d, device=device)

        # Integrate forward: τ from 0 → 1
        t_span = torch.tensor([0.0, 1.0], device=device)

        if HAS_TORCHDIFFEQ:
            ode_kwargs: dict = {
                "method": self.solver,
                "atol": self.atol,
                "rtol": self.rtol,
            }
            if self.solver == "euler" and self.n_steps > 0:
                ode_kwargs["options"] = {"step_size": 1.0 / self.n_steps}

            if self.use_adjoint:
                s_1 = _odeint_adj(self.velocity, s_0, t_span, **ode_kwargs)[-1]
            else:
                s_1 = _odeint_std(self.velocity, s_0, t_span, **ode_kwargs)[-1]
        else:
            n = self.n_steps if self.n_steps > 0 else 50
            s_1 = _euler_solve_simple(self.velocity, s_0, t_span, n_steps=n)

        return s_1  # (B, d)


# ============================================================================
# Gaussian Mixture Spatial Decoder (DeepSTPP)
# ============================================================================


class GaussianMixtureSpatial(nn.Module):
    """
    Gaussian mixture spatial density: f*(s | t, H) = Σ_k w_k N(s; μ_k, σ_k² I).

    Mixture parameters are predicted from the latent state z.
    Covariates can modify the parameters (Proposition 1 safe: modifying μ, σ).
    """

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        n_components: int = 16,
        field_cov_dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.n_components = n_components
        K = n_components
        d = spatial_dim

        input_dim = hidden_dim + field_cov_dim
        self.param_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, K * (1 + d + 1)),  # logits, means, log_vars
        )

    def _get_params(self, z: Tensor, x_field: Optional[Tensor] = None):
        if x_field is not None:
            inp = torch.cat([z, x_field], dim=-1)
        else:
            inp = z
        params = self.param_net(inp)
        K, d = self.n_components, self.spatial_dim

        logits  = params[:, :K]                              # (B, K)
        means   = params[:, K : K + K * d].reshape(-1, K, d) # (B, K, d)
        log_vars = params[:, K + K * d :]                    # (B, K)
        vars_   = F.softplus(log_vars) + 1e-4                # (B, K)

        return logits, means, vars_

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """log f*(s | z)."""
        d = self.spatial_dim
        logits, means, vars_ = self._get_params(z, x_field)

        # s: (B, d) → (B, 1, d), means: (B, K, d)
        diff = s.unsqueeze(1) - means  # (B, K, d)
        log_gauss = (
            -0.5 * d * math.log(2 * math.pi)
            - 0.5 * d * torch.log(vars_)
            - 0.5 * (diff ** 2).sum(dim=-1) / vars_
        )  # (B, K)

        log_pi = F.log_softmax(logits, dim=-1)  # (B, K)
        return torch.logsumexp(log_pi + log_gauss, dim=-1)  # (B,)

    def sample(
        self,
        z: Tensor,
        t: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample s ~ Σ_k w_k N(μ_k, σ_k² I).

        Ancestral: sample component k ~ Categorical(w), then s ~ N(μ_k, σ_k² I).
        """
        B = z.shape[0]
        d = self.spatial_dim
        device = z.device

        logits, means, vars_ = self._get_params(z, x_field)

        # 1. Sample component
        pi = F.softmax(logits, dim=-1)  # (B, K)
        k = torch.multinomial(pi, num_samples=1).squeeze(-1)  # (B,)

        # 2. Gather selected component params
        batch_idx = torch.arange(B, device=device)
        mu_k  = means[batch_idx, k]   # (B, d)
        var_k = vars_[batch_idx, k]    # (B,)

        # 3. Sample: s ~ N(μ_k, σ_k² I)
        std_k = var_k.sqrt().unsqueeze(-1)  # (B, 1)
        eps   = torch.randn(B, d, device=device)
        return mu_k + std_k * eps  # (B, d)


# ============================================================================
# DataCenteredGaussianSpatial (DeepSTPP faithful — Lin et al. 2021)
# ============================================================================

class DataCenteredGaussianSpatial(nn.Module):
    """
    Spatial Gaussian mixture whose centers are pinned to past event locations
    (+ learnable background anchors), matching Lin et al. 2021 DeepSTPP.

    The decoder predicts ONLY mixture logits and per-component diagonal
    log-sigma from z.  Event-location centers are received via the x_field
    argument as a flattened (seq_len * spatial_dim,) vector.

    Args:
        spatial_dim:   d, dimension of spatial coordinates.
        hidden_dim:    dimension of latent z.
        seq_len:       number of most-recent history events used as centers.
        num_points:    number of learnable background anchor centers.
        sigma_min:     minimum allowed Gaussian sigma (model-space units);
                       prevents excessively sharp intensity peaks.
        field_cov_dim: ignored (kept for API compatibility).
    """

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        seq_len: int = 20,
        num_points: int = 20,
        sigma_min: float = 0.3,
        field_cov_dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.seq_len = seq_len
        self.num_points = num_points
        self.sigma_min = sigma_min
        M = seq_len + num_points

        if num_points > 0:
            self.background = nn.Parameter(
                torch.randn(num_points, spatial_dim) * 0.01
            )
        else:
            self.register_parameter("background", None)

        # Predict logits (M) + log-sigma per dim (M*d) from z
        self.param_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, M + M * spatial_dim),
        )

    @property
    def requires_history(self) -> bool:
        return True

    @property
    def history_window_size(self) -> int:
        return self.seq_len

    def _get_centers(self, x_field: Optional[Tensor], B: int, device) -> Tensor:
        """Build (B, M, d) centers from history locs + background anchors."""
        if x_field is not None:
            hist = x_field.reshape(B, self.seq_len, self.spatial_dim)
        else:
            hist = torch.zeros(B, self.seq_len, self.spatial_dim, device=device)

        if self.background is not None:
            bg = self.background.unsqueeze(0).expand(B, -1, -1)  # (B, P, d)
            return torch.cat([hist, bg], dim=1)                   # (B, M, d)
        return hist

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        B = z.shape[0]
        M = self.seq_len + self.num_points
        d = self.spatial_dim

        centers = self._get_centers(x_field, B, z.device)   # (B, M, d)

        params   = self.param_net(z)                                   # (B, M + M*d)
        logits   = params[:, :M]                                       # (B, M)
        log_sig  = params[:, M:].reshape(B, M, d)                     # (B, M, d)
        sigma    = F.softplus(log_sig) + self.sigma_min                # (B, M, d)
        inv_var  = 1.0 / sigma.pow(2).clamp(min=1e-6)                  # (B, M, d)

        diff    = s.unsqueeze(1) - centers                             # (B, M, d)
        log_det = 0.5 * inv_var.prod(dim=-1).clamp(min=1e-12).log()   # (B, M)
        quad    = (diff * inv_var * diff).sum(dim=-1)                  # (B, M)
        log_g   = log_det - math.log(2.0 * math.pi) * (d / 2.0) - 0.5 * quad  # (B, M)
        log_pi  = F.log_softmax(logits, dim=-1)                        # (B, M)

        return torch.logsumexp(log_pi + log_g, dim=-1)                 # (B,)

    def nll(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        return -self.log_prob(z, t, s, t_prev, x_field=x_field)

    def sample(
        self,
        z: Tensor,
        t: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """Ancestral sample: draw component k ~ Cat(w), then s ~ N(μ_k, σ_k² I)."""
        B = z.shape[0]
        d = self.spatial_dim
        M = self.seq_len + self.num_points
        device = z.device

        centers = self._get_centers(x_field, B, device)               # (B, M, d)

        params   = self.param_net(z)
        logits   = params[:, :M]
        log_sig  = params[:, M:].reshape(B, M, d)
        sigma    = F.softplus(log_sig) + self.sigma_min                # (B, M, d)

        pi = F.softmax(logits, dim=-1)                                 # (B, M)
        k  = torch.multinomial(pi, num_samples=1).squeeze(-1)          # (B,)
        batch_idx = torch.arange(B, device=device)
        mu_k    = centers[batch_idx, k]                                # (B, d)
        sigma_k = sigma[batch_idx, k]                                  # (B, d)
        eps     = torch.randn(B, d, device=device)
        return mu_k + sigma_k * eps                                    # (B, d)


# ============================================================================
# Utilities
# ============================================================================

def _euler_solve_simple(func, y0, t_span, n_steps=50):
    """Simple Euler solver fallback."""
    dt = (t_span[-1] - t_span[0]) / n_steps
    y = y0
    t = t_span[0]
    for _ in range(n_steps):
        y = y + dt * func(t, y)
        t = t + dt
    return y
