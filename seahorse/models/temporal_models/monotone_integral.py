"""
Monotone integral decoder — intensity-based spatiotemporal point process decoder.

Implements the auto-differentiable integration approach from:
  "AutoSTPP: Efficient Spatiotemporal Point Process with Automatic Integration"
  (Zhou & Yu, NeurIPS 2023)

Following the official Rose-STL-Lab implementation:
  src/integration/autoint.py
  src/models/lightning/prodnet_cuboid.py

Architecture
------------
  λ*(t, s | H) = μ + f(s_x, s_y, τ; z)

  μ  = softplus(log_mu)      — learnable base rate (>0)
  τ  = t − t_prev            — inter-event time
  z                          — latent history state from encoder
  f  = ∂³F / (∂s_x ∂s_y ∂τ) — ProdNet mixed partial derivative (≥0)

ProdNet defines the 3-D anti-derivative:
  F(x, y, τ; z) = Σ_r  G_r^x(x; z) · G_r^y(y; z) · G_r^τ(τ; z)

Each G_r is a MonotoneNet: weights along the scalar path are non-negative
(via F.relu projection, matching NonNegLinear from the official repo).

Compensator
-----------
  Λ(t_prev, t) = μ·|S|·τ + [box_integral(τ) − box_integral(0)]

where box_integral(τ) = ∫_{x_lo}^{x_hi} ∫_{y_lo}^{y_hi} F(x,y,τ;z) dx dy
computed via 4-corner inclusion-exclusion (Cuboid.int_lamb pattern).

Constraints
-----------
  - spatial_dim must equal 2
  - sample() raises NotImplementedError (thinning not yet supported)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from ..base import Decoder


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class NonNegLinear(nn.Linear):
    """
    Linear layer with non-negative weights via F.relu projection.

    Matches NonNegLinear from src/integration/autoint.py in the official repo.
    Weight positivity guarantees monotonicity along the scalar-input path.
    """

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, F.relu(self.weight), self.bias)


class MonotoneNet(nn.Module):
    """
    Monotone network  G(u; h): R × R^{d_h} → R^{n_components}.

    Monotone in scalar u: all weight matrices applied to the u-path use
    NonNegLinear so dG/du ≥ 0 pointwise.  The conditioning signal h enters
    at the first layer without a positivity constraint.

    Architecture
    ~~~~~~~~~~~~
      first layer : u → internal_dim  (NonNegLinear, no bias)
                    h → internal_dim  (Linear, with bias)
                    combined by addition then Tanh
      hidden      : n_layers × [Tanh → NonNegLinear(internal_dim, internal_dim)]
      output      : NonNegLinear(internal_dim, n_components)

    Matches BaselineSequential + NonNegLinear from the official AutoSTPP repo.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_components: int,
        n_layers: int = 2,
        internal_dim: int = 64,
    ):
        super().__init__()
        self.n_components = n_components

        # First layer: u (scalar) and h projected to the same internal_dim
        self.u_proj = NonNegLinear(1, internal_dim, bias=False)
        self.h_proj = nn.Linear(hidden_dim, internal_dim, bias=True)

        # Hidden layers (all non-negative to preserve monotonicity in u)
        layers: list = []
        for _ in range(n_layers):
            layers.append(nn.Tanh())
            layers.append(NonNegLinear(internal_dim, internal_dim))
        self.hidden = nn.Sequential(*layers)

        # Output (non-negative weights)
        self.out = NonNegLinear(internal_dim, n_components)

    def forward(self, u: Tensor, h: Tensor) -> Tensor:
        """
        Args:
            u : (B, 1)          — scalar input (τ or spatial coordinate)
            h : (B, hidden_dim) — conditioning state
        Returns:
            (B, n_components)   — anti-derivative values
        """
        x = self.u_proj(u) + self.h_proj(h)
        x = self.hidden(x)
        return self.out(x)


# ---------------------------------------------------------------------------
# ProdNet
# ---------------------------------------------------------------------------

class ProdNet(nn.Module):
    """
    Separable product network for spatiotemporal intensity.

    Defines the 3-D anti-derivative:
        F(x, y, t; h) = Σ_r  G_r^x(x;h) · G_r^y(y;h) · G_r^t(t;h)

    The conditional intensity is the mixed partial derivative:
        f(x,y,t;h) = ∂³F / (∂x ∂y ∂t)
                   = Σ_r  (∂G_r^x/∂x) · (∂G_r^y/∂y) · (∂G_r^t/∂t)  ≥ 0

    The mixed partial is computed by three successive torch.autograd.grad
    calls (matching the dnforward / Cuboid pattern in the official repo).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_components: int = 8,
        n_layers: int = 2,
        internal_dim: int = 64,
    ):
        super().__init__()
        kw = dict(
            hidden_dim=hidden_dim,
            n_components=n_components,
            n_layers=n_layers,
            internal_dim=internal_dim,
        )
        self.net_x = MonotoneNet(**kw)
        self.net_y = MonotoneNet(**kw)
        self.net_t = MonotoneNet(**kw)

    def integral(self, x: Tensor, y: Tensor, t: Tensor, h: Tensor) -> Tensor:
        """
        F(x, y, t; h) = Σ_r G_r^x · G_r^y · G_r^t  (plain forward pass).

        Args:
            x, y, t : (B, 1)
            h       : (B, hidden_dim)
        Returns:
            (B,) — anti-derivative value at (x, y, t)
        """
        Gx = self.net_x(x, h)   # (B, n_components)
        Gy = self.net_y(y, h)
        Gt = self.net_t(t, h)
        return (Gx * Gy * Gt).sum(-1)   # (B,)

    def intensity(self, x: Tensor, y: Tensor, t: Tensor, h: Tensor) -> Tensor:
        """
        f(x, y, t; h) = ∂³F / (∂x ∂y ∂t) via iterated autograd.

        Scalar inputs are detached and have grad re-enabled so that
        autograd.grad can differentiate through them.  Network parameters
        (accessed via h and the MonotoneNet weights) are NOT detached, so
        gradients w.r.t. model parameters flow correctly during training.

        `create_graph=self.training` mirrors the official repo pattern.

        Args:
            x, y, t : (B, 1)
            h       : (B, hidden_dim)
        Returns:
            (B,) — conditional intensity (non-negative by construction)
        """
        x_ = x.detach().requires_grad_(True)
        y_ = y.detach().requires_grad_(True)
        t_ = t.detach().requires_grad_(True)
        # Intermediate steps ALWAYS need create_graph=True: each grad call
        # must produce a tensor with a grad_fn so the next call can
        # differentiate through it.  Only the final step respects
        # self.training (False during eval = no further differentiation needed).
        cg_final = self.training

        with torch.enable_grad():
            F = (
                self.net_x(x_, h) * self.net_y(y_, h) * self.net_t(t_, h)
            ).sum(-1)  # (B,)

            # ∂F / ∂t  — must keep graph so d²/dy can differentiate through this
            dF_dt = torch.autograd.grad(
                F.sum(), t_, create_graph=True, retain_graph=True
            )[0]  # (B, 1)

            # ∂²F / (∂y ∂t) — must keep graph so d³/dx can differentiate through this
            d2F_dydt = torch.autograd.grad(
                dF_dt.sum(), y_, create_graph=True, retain_graph=True
            )[0]  # (B, 1)

            # ∂³F / (∂x ∂y ∂t) — final step: only keep graph during training
            d3F = torch.autograd.grad(
                d2F_dydt.sum(), x_, create_graph=cg_final
            )[0]  # (B, 1)

        return d3F.squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# MonotoneIntegralDecoder
# ---------------------------------------------------------------------------

class MonotoneIntegralDecoder(Decoder):
    """
    AutoSTPP intensity-based decoder using analytic integration.

    Models the conditional intensity as:
        λ*(t, s | H) = μ + f(s_x, s_y, τ; z)

    NLL contribution per event:
        −log λ*(t_{n+1}, s_{n+1}) + Λ(t_n, t_{n+1})

    where Λ is the compensator:
        Λ = μ·|S|·τ + [box_integral(τ) − box_integral(0)]

    and box_integral(τ) = ∫∫_S F(x, y, τ; z) dx dy is evaluated via
    4-corner inclusion-exclusion (matching Cuboid.int_lamb from the official repo).

    spatial_dim must equal 2.
    sample() is not implemented.
    """

    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        n_components: int = 8,
        n_layers: int = 2,
        internal_dim: int = 64,
        field_cov_dim: int = 0,
        x_lo: float = -5.0,
        x_hi: float = 5.0,
        y_lo: float = -5.0,
        y_hi: float = 5.0,
        **kwargs,
    ):
        assert spatial_dim == 2, (
            f"AutoIntDecoder requires spatial_dim == 2, got {spatial_dim}"
        )
        super().__init__(hidden_dim=hidden_dim, spatial_dim=spatial_dim)

        self.x_lo = x_lo
        self.x_hi = x_hi
        self.y_lo = y_lo
        self.y_hi = y_hi

        # Learnable base rate (log-parameterised; softplus keeps μ > 0)
        self.log_mu = nn.Parameter(torch.zeros(1))

        # History-conditioned spatiotemporal intensity
        self.prodnet = ProdNet(
            hidden_dim=hidden_dim,
            n_components=n_components,
            n_layers=n_layers,
            internal_dim=internal_dim,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def mu(self) -> Tensor:
        """Base rate μ = softplus(log_mu) > 0 (scalar)."""
        return F.softplus(self.log_mu)

    def _raw_intensity(self, z: Tensor, s: Tensor, tau: Tensor) -> Tensor:
        """
        f(s, τ; z) from ProdNet.

        Args:
            z   : (B, hidden_dim)
            s   : (B, 2)  — [s_x, s_y]
            tau : (B, 1)  — t − t_prev
        Returns:
            (B,)
        """
        return self.prodnet.intensity(s[:, 0:1], s[:, 1:2], tau, z)

    def _box_integral(self, tau: Tensor, z: Tensor) -> Tensor:
        """
        ∫_{x_lo}^{x_hi} ∫_{y_lo}^{y_hi} F(x, y, τ; z) dx dy
        via 4-corner inclusion-exclusion.

        Returns: (B,)
        """
        B, dev = z.shape[0], z.device
        xlo = torch.full((B, 1), self.x_lo, device=dev)
        xhi = torch.full((B, 1), self.x_hi, device=dev)
        ylo = torch.full((B, 1), self.y_lo, device=dev)
        yhi = torch.full((B, 1), self.y_hi, device=dev)
        I = self.prodnet.integral
        return I(xhi, yhi, tau, z) - I(xhi, ylo, tau, z) \
             - I(xlo, yhi, tau, z) + I(xlo, ylo, tau, z)

    def compensator(self, z: Tensor, tau: Tensor) -> Tensor:
        """
        Λ(t_prev, t) = μ·|S|·τ + [box_integral(τ) − box_integral(0)].

        Args:
            z   : (B, hidden_dim)
            tau : (B, 1)  — t − t_prev (≥ 0)
        Returns:
            (B,)
        """
        B, dev = z.shape[0], z.device
        t0 = torch.zeros(B, 1, device=dev)

        prodnet_comp = self._box_integral(tau, z) - self._box_integral(t0, z)

        area = (self.x_hi - self.x_lo) * (self.y_hi - self.y_lo)
        base_comp = self.mu() * area * tau.squeeze(-1)   # (B,)

        return prodnet_comp + base_comp   # (B,)

    # ------------------------------------------------------------------
    # Decoder interface
    # ------------------------------------------------------------------

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        log λ*(t, s | H) = log(μ + f(s, τ; z)).

        Args:
            z      : (B, hidden_dim)
            t      : (B, 1)
            s      : (B, 2)
            t_prev : (B, 1)
            x_field: ignored (AutoSTPP does not use field covariates)
        Returns:
            (B,) — log-intensity
        """
        tau = (t - t_prev).clamp(min=1e-6)                   # (B, 1)
        f   = self._raw_intensity(z, s, tau).clamp(min=0.0)  # (B,)
        return torch.log(self.mu() + f + 1e-8)              # (B,)

    def nll(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        NLL = −log λ*(t, s) + Λ(t_prev, t).

        Args:
            z      : (B, hidden_dim)
            t      : (B, 1)
            s      : (B, 2)
            t_prev : (B, 1)
            x_field: ignored
        Returns:
            (B,) — per-event NLL contribution
        """
        log_lam = self.log_prob(z, t, s, t_prev, x_field)  # (B,)
        tau     = (t - t_prev).clamp(min=1e-6)             # (B, 1)
        comp    = self.compensator(z, tau)                 # (B,)
        return -log_lam + comp

    def sample(
        self,
        z: Tensor,
        t_prev: Tensor,
        x_field_fn=None,
    ) -> Tuple[Tensor, Tensor]:
        """Not implemented — use thinning-based sampling externally."""
        raise NotImplementedError(
            "AutoIntDecoder does not support sampling. "
            "Use thinning-based sampling externally."
        )
