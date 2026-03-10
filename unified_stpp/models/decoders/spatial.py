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


class ConcatSquash(nn.Module):
    """
    ConcatSquash hidden layer from FFJORD / Neural STPP (Chen et al. 2021).

        output = lin_z(z) * σ(lin_t(ctx)) + tanh(lin_tb(ctx))

    Each spatial-domain linear transform is multiplicatively gated by the
    conditioning context (hidden state + flow time).  This is more expressive
    than simple concat+Tanh because the context scales the spatial transform.

    References: Grathwohl et al. 2019 (FFJORD), Chen et al. 2021 (NeuralSTPP).
    """

    def __init__(self, in_dim: int, out_dim: int, ctx_dim: int):
        super().__init__()
        self.lin_z  = nn.Linear(in_dim, out_dim)           # spatial → output
        self.lin_t  = nn.Linear(ctx_dim, out_dim, bias=False)  # gate (no bias)
        self.lin_tb = nn.Linear(ctx_dim, out_dim)           # context bias

    def forward(self, z: Tensor, ctx: Tensor) -> Tensor:
        return self.lin_z(z) * torch.sigmoid(self.lin_t(ctx)) + torch.tanh(self.lin_tb(ctx))


class CNFVelocityField(nn.Module):
    """Velocity field v(s, τ; z, X) for the spatial CNF.

    Parameters
    ----------
    layer_type : ``"concat"`` | ``"mlp"``
        ``"concat"`` (paper default): ConcatSquash gated layers.
        ``"mlp"``: plain concat + Tanh MLP.
    n_hidden_layers : int
        Depth. Default 3 matches the ``hdims="64-64-64"`` default in the
        original Neural STPP code.
    """

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
        ctx_dim = hidden_dim + 1 + field_cov_dim  # z + τ [+ X_field]

        if layer_type == "concat":
            # ConcatSquash stack: s(d) → h → … → h, then output h → d
            cs_layers = []
            in_dim = spatial_dim
            for _ in range(n_hidden_layers):
                cs_layers.append(ConcatSquash(in_dim, hidden_dim, ctx_dim))
                in_dim = hidden_dim
            self.hidden_layers = nn.ModuleList(cs_layers)
            self.output_layer  = nn.Linear(hidden_dim, spatial_dim)
            self.mlp = None
        else:  # "mlp" or "softplus"
            # Plain concat+activation: [s, τ, z, X] → Linear → Act → … → Linear
            # "softplus" uses nn.Softplus (matches NeuralSTPP base_cnf).
            # "mlp" uses nn.Tanh (legacy default).
            act_cls = nn.Softplus if layer_type == "softplus" else nn.Tanh
            in_d = spatial_dim + ctx_dim
            mlp_layers: list = []
            for _ in range(n_hidden_layers):
                mlp_layers += [nn.Linear(in_d, hidden_dim), act_cls()]
                in_d = hidden_dim
            mlp_layers.append(nn.Linear(hidden_dim, spatial_dim))
            self.mlp = nn.Sequential(*mlp_layers)
            self.hidden_layers = None
            self.output_layer  = None

        self._z_cond: Optional[Tensor] = None  # set before each ODE solve
        self._x_cond: Optional[Tensor] = None  # field covariates (optional)
        self._e: Optional[Tensor] = None        # cached Hutchinson noise

    def _compute_velocity(self, s: Tensor, ctx: Tensor) -> Tensor:
        """Compute v(s; ctx) without divergence tracking."""
        if self.layer_type == "concat":
            h = s
            for layer in self.hidden_layers:
                h = layer(h, ctx)
            return self.output_layer(h)
        else:
            return self.mlp(torch.cat([s, ctx], dim=-1))

    def forward(self, tau: Tensor, s: Tensor) -> Tensor:
        """
        Args:
            tau: scalar tensor — flow time in [0, 1]
            s: (B, d) or (B, d+1) if augmented with log-det accumulator
        Returns:
            ds/dτ: same shape as s
        """
        d = self.spatial_dim
        s_actual = s[:, :d] if s.shape[-1] > d else s

        B = s_actual.shape[0]
        tau_expand = tau.expand(B, 1)
        ctx_parts = [self._z_cond, tau_expand]
        if self._x_cond is not None:
            ctx_parts.append(self._x_cond)
        ctx = torch.cat(ctx_parts, dim=-1)  # (B, h+1[+r])

        v = self._compute_velocity(s_actual, ctx)  # (B, d)

        if s.shape[-1] > d:
            # Compute div(v) for the log-det accumulator.
            # Always use enable_grad + fresh leaf so that autograd.grad works
            # correctly inside no_grad / inference_mode contexts (adjoint ODE,
            # Lightning val/test steps with inference_mode=False).
            with torch.enable_grad():
                s_leaf = s_actual.detach().requires_grad_(True)
                v_div  = self._compute_velocity(s_leaf, ctx.detach())

                if self.spatial_dim <= 3:
                    trace = divergence_bf(v_div, s_leaf, self.training)
                else:
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
    layer_type : ``"concat"`` | ``"mlp"``
        Velocity-field architecture.  ``"concat"`` = ConcatSquash (paper default).
    n_hidden_layers : int
        Depth of the velocity field. Default 3.
    base_type : ``"standard"`` | ``"self_attentive"``
        ``"standard"`` (default): base distribution N(0, I).
        ``"self_attentive"``: base distribution N(μ_attn, I) where μ_attn is an
        attention-weighted mean of the K most recent event locations.  Implements
        the SelfAttentiveCNF variant from Chen et al. 2021 (NeuralSTPP).
        Requires ``x_field`` to be the K past locations (B, K*d).
    history_k : int
        Window size for ``base_type="self_attentive"``. Default 20.
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
        layer_type: str = "concat",
        n_hidden_layers: int = 3,
        base_type: str = "standard",
        history_k: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.base_type = base_type
        self.history_k = history_k
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.use_adjoint = use_adjoint
        self.n_steps = n_steps

        # In self_attentive mode x_field carries past locations for the base
        # distribution mean — it is NOT forwarded to the velocity field.
        vf_field_cov_dim = 0 if base_type == "self_attentive" else field_cov_dim
        self.velocity = CNFVelocityField(
            spatial_dim, hidden_dim, vf_field_cov_dim, layer_type, n_hidden_layers
        )

        if base_type == "self_attentive":
            self.query_proj = nn.Linear(hidden_dim, spatial_dim)
            self.key_proj   = nn.Linear(spatial_dim, spatial_dim)
        else:
            self.query_proj = None
            self.key_proj   = None

    # ------------------------------------------------------------------ #
    # Properties used by UnifiedSTPP for sliding-window routing            #
    # ------------------------------------------------------------------ #

    @property
    def requires_history(self) -> bool:
        """True when x_field_spatial must carry K past event locations."""
        return self.base_type == "self_attentive"

    @property
    def history_window_size(self) -> int:
        return self.history_k

    # ------------------------------------------------------------------ #
    # Self-attentive base helper                                           #
    # ------------------------------------------------------------------ #

    def _compute_mu_attn(self, z: Tensor, x_field: Optional[Tensor]) -> Tensor:
        """Attention-weighted mean of past event locations.

        Implements: μ = Σ_i α_i s_i  where  α = softmax(Q(z)·K(s_i)^T / √d).

        Args:
            z:       (B, h) current hidden state.
            x_field: (B, K*d) flattened past locations, or None.
        Returns:
            (B, d) — zeros when x_field is None (no history).
        """
        B, d = z.shape[0], self.spatial_dim
        if x_field is None:
            return torch.zeros(B, d, device=z.device, dtype=z.dtype)

        locs   = x_field.reshape(B, self.history_k, d)        # (B, K, d)
        q      = self.query_proj(z)                            # (B, d)
        k      = self.key_proj(locs)                           # (B, K, d)
        scores = (q.unsqueeze(1) * k).sum(-1) / math.sqrt(d)  # (B, K)
        alpha  = F.softmax(scores, dim=-1)                     # (B, K)
        return (alpha.unsqueeze(-1) * locs).sum(dim=1)         # (B, d)

    # ------------------------------------------------------------------ #
    # ODE integration                                                      #
    # ------------------------------------------------------------------ #

    def _run_odeint(self, t_span: Tensor, s_aug: Tensor) -> Tensor:
        """Run ODE integration, dispatching between adjoint and standard."""
        if not HAS_TORCHDIFFEQ:
            return _euler_solve_simple(self.velocity, s_aug, t_span,
                                       n_steps=self.n_steps if self.n_steps > 0 else 50)

        # torchdiffeq adaptive solvers default to dtype=float64 for time-like
        # tensors (rtol, atol, step sizes).  MPS does not support float64, so
        # we pin all solver scalars to s_aug's dtype (float32 on MPS).
        options: dict = {"dtype": s_aug.dtype}
        if self.solver == "euler" and self.n_steps > 0:
            dt_total = abs((t_span[-1] - t_span[0]).item())
            if dt_total > 1e-8:
                options["step_size"] = dt_total / self.n_steps

        ode_kwargs: dict = {
            "method": self.solver,
            "atol": self.atol,
            "rtol": self.rtol,
            "options": options,
        }

        if self.use_adjoint:
            traj = _odeint_adj(self.velocity, s_aug, t_span, **ode_kwargs)
        else:
            traj = _odeint_std(self.velocity, s_aug, t_span, **ode_kwargs)

        return traj[-1]

    # ------------------------------------------------------------------ #
    # Density evaluation                                                   #
    # ------------------------------------------------------------------ #

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute log f*(s | z(t)).

        Integrates the CNF backward (τ: 1 → 0) accumulating log-det.
        For ``base_type="self_attentive"``, the base is N(μ_attn(z, hist), I).
        """
        B, d = s.shape
        self.velocity._z_cond = z
        # x_field feeds the velocity only in standard mode with field covariates
        self.velocity._x_cond = None if self.base_type == "self_attentive" else x_field
        self.velocity._e = None  # fresh Hutchinson noise per solve

        s_aug = torch.cat([s, torch.zeros(B, 1, device=s.device)], dim=-1)
        s_aug = s_aug.requires_grad_(True)

        t_span = torch.tensor([1.0, 0.0], device=s.device, dtype=s.dtype)
        s_aug_0 = self._run_odeint(t_span, s_aug)

        s_0     = s_aug_0[:, :d]
        log_det = s_aug_0[:, d]

        if self.base_type == "self_attentive":
            mu     = self._compute_mu_attn(z, x_field)   # (B, d)
            log_p0 = -0.5 * (d * math.log(2 * math.pi) + ((s_0 - mu) ** 2).sum(-1))
        else:
            log_p0 = -0.5 * (d * math.log(2 * math.pi) + (s_0 ** 2).sum(-1))

        return log_p0 - log_det

    # ------------------------------------------------------------------ #
    # Sampling                                                             #
    # ------------------------------------------------------------------ #

    def sample(
        self,
        z: Tensor,
        t: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample s ~ f*(s | z(t)) via forward ODE.

        For ``base_type="self_attentive"``: draw from N(μ_attn, I) then ODE forward.
        """
        B = z.shape[0]
        d = self.spatial_dim
        device = z.device

        self.velocity._z_cond = z
        self.velocity._x_cond = None if self.base_type == "self_attentive" else x_field
        self.velocity._e = None

        s_0 = torch.randn(B, d, device=device)
        if self.base_type == "self_attentive":
            s_0 = s_0 + self._compute_mu_attn(z, x_field)

        t_span = torch.tensor([0.0, 1.0], device=device, dtype=s_0.dtype)
        return self._run_odeint(t_span, s_0)


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
# DeepSTPPDecoder — faithful coupled Hawkes temporal-spatial decoder
# ============================================================================

class DeepSTPPDecoder(nn.Module):
    """
    Faithful joint temporal-spatial decoder for DeepSTPP (Lin et al. 2021).

    Implements the original coupled Hawkes intensity:
        λ(s, t) = λ_t(t) · f(s|t)
        λ_t(t)  = Σ_i w_i · exp(−b_i · (t − t_i))          [Hawkes temporal]
        f(s|t)  = Σ_i v_i/Σv_j · N(s; s_i, diag(1/inv_var))  [weighted GMM]

    where v_i = w_i · exp(−b_i · (t − t_i)) **ties spatial weights to temporal**.
    This coupling is the paper's core contribution: the component contributing
    most to the temporal intensity also attracts the spatial mass.

    M = seq_len + num_points kernel components:
        seq_len:    Hawkes kernels centred at the most-recent history events.
        num_points: Hawkes kernels centred at learnable background anchors.

    x_field layout (B, seq_len + seq_len * spatial_dim):
        x_field[:, :seq_len]          — absolute event times of lookback window
        x_field[:, seq_len:]          — event locations, flattened row-major

    Temporal compensator (closed form):
        ∫_{t_n}^{t} λ_t(τ) dτ = Σ_i w_i/b_i · (exp(−b_i·tn_ti) − exp(−b_i·t_ti))
    so log f*(dt) = log λ_t(t) − ∫ λ dτ  is computed exactly.

    Args:
        hidden_dim:   Encoder output / latent state dimension.
        spatial_dim:  d (typically 2).
        seq_len:      Lookback window size (paper: 20).
        num_points:   Learnable background anchors (paper: 20).
        sigma_min:    Minimum Gaussian σ (paper: 1e-4 in MinMax [0,1] space).
        n_layers:     MLP depth per decoder (paper: decoder_n_layer=3).
    """

    # Class-level flag read by _forward_batched to route history + time windows
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
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.spatial_dim = spatial_dim
        self.seq_len     = seq_len
        self.num_points  = num_points
        self.sigma_min   = sigma_min
        M = seq_len + num_points

        def _mlp(out_dim: int) -> nn.Sequential:
            """MLP matching paper: Linear→ELU repeated n_layers times, final Linear."""
            layers: list = [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        # Three separate MLPs as in the original (w_dec, b_dec, s_dec)
        self.w_dec = _mlp(M)               # → softplus → positive weights
        self.b_dec = _mlp(M)               # → softplus → positive decay rates
        self.s_dec = _mlp(M * spatial_dim) # → softplus + σ_min → spatial variances

        if num_points > 0:
            self.background = nn.Parameter(torch.rand(num_points, spatial_dim) * 0.1)
        else:
            self.register_parameter("background", None)

    # ------------------------------------------------------------------ #
    # Properties for _forward_batched routing                             #
    # ------------------------------------------------------------------ #

    @property
    def requires_history(self) -> bool:
        return True

    @property
    def history_window_size(self) -> int:
        return self.seq_len

    # ------------------------------------------------------------------ #
    # Kernel parameter decoding                                            #
    # ------------------------------------------------------------------ #

    def _decode(self, z: Tensor):
        """Decode (w_i, b_i, sigma, inv_var) from latent z.

        Returns:
            w_i:     (B, M)    — positive weights
            b_i:     (B, M)    — positive decay rates
            sigma:   (B, M, d) — Gaussian σ per dim (≥ sigma_min)
            inv_var: (B, M, d) — 1 / sigma
        """
        B = z.shape[0]
        M = self.seq_len + self.num_points
        d = self.spatial_dim

        w_i   = F.softplus(self.w_dec(z)) + 1e-5                         # (B, M)
        b_i   = F.softplus(self.b_dec(z)) + 1e-5                         # (B, M)
        sigma = F.softplus(self.s_dec(z).reshape(B, M, d)) + self.sigma_min  # (B, M, d)
        inv_var = 1.0 / sigma                                              # (B, M, d)
        return w_i, b_i, sigma, inv_var

    # ------------------------------------------------------------------ #
    # NLL (training loss)                                                  #
    # ------------------------------------------------------------------ #

    def nll(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Joint NLL = −(log λ_t(t) + compensator + log f(s|t)).

        Args:
            z:       (B, h)                           — latent state
            t:       (B, 1) or (B,)                   — target event absolute time
            s:       (B, d)                           — target event location
            t_prev:  (B, 1) or (B,)                   — previous event absolute time
            x_field: (B, seq_len + seq_len*d)         — history window (times then locs)

        Returns:
            nll: (B,) — per-event negative log-likelihood
        """
        B = z.shape[0]
        d = self.spatial_dim
        M = self.seq_len + self.num_points
        device = z.device

        t      = t.reshape(B)       # (B,)
        t_prev = t_prev.reshape(B)  # (B,)
        dt     = (t - t_prev).clamp(min=1e-6)  # (B,)

        # ---- Build time differences ----------------------------------------
        if x_field is not None:
            t_hist = x_field[:, :self.seq_len]                           # (B, seq_len)
            s_hist = x_field[:, self.seq_len:].reshape(B, self.seq_len, d)  # (B, seq_len, d)
            tn_ti_h = (t_prev.unsqueeze(-1) - t_hist).clamp(min=0.0)    # (B, seq_len)
        else:
            # Fallback: assume history events all happened at t_prev
            tn_ti_h = torch.zeros(B, self.seq_len, device=device)
            s_hist  = torch.zeros(B, self.seq_len, d, device=device)

        tn_ti_bg = torch.zeros(B, self.num_points, device=device)  # background at t_n
        tn_ti    = torch.cat([tn_ti_h, tn_ti_bg], dim=-1)          # (B, M)
        t_ti     = (tn_ti + dt.unsqueeze(-1)).clamp(min=1e-6)       # (B, M)

        # ---- Build Gaussian centers ----------------------------------------
        if self.background is not None:
            bg      = self.background.unsqueeze(0).expand(B, -1, -1)  # (B, P, d)
            centers = torch.cat([s_hist, bg], dim=1)                   # (B, M, d)
        else:
            centers = s_hist  # (B, seq_len, d)

        # ---- Decode kernel parameters ----------------------------------------
        w_i, b_i, sigma, inv_var = self._decode(z)

        # ---- Temporal: Hawkes intensity + closed-form compensator ------------
        exp_t_ti  = torch.exp(-b_i * t_ti)                        # (B, M)
        exp_tn_ti = torch.exp(-b_i * tn_ti.clamp(min=0.0))       # (B, M)

        # Work in log-space to avoid NaN from log(0) when exp underflows.
        # w_i > 0 always (softplus + eps), so log(w_i) is well-defined.
        log_w_i    = torch.log(w_i)                                # (B, M)
        log_v_i    = log_w_i - b_i * t_ti                         # (B, M) = log(v_i)
        log_lamb_t = torch.logsumexp(log_v_i, dim=-1)             # (B,)   stable log λ_t

        # Compensator: ∫_{t_n}^{t} λ_t(τ) dτ (≥ 0, so log f* = log λ_t − compensator)
        # Direct-space OK here: when exp→0, contribution is 0 (correct).
        comp = (w_i / b_i * (exp_t_ti - exp_tn_ti)).sum(dim=-1)   # (B,)

        log_temporal = log_lamb_t + comp  # log f*(dt) = log λ_t − compensator

        # ---- Spatial: weighted GMM, weights = v_i --------------------------
        # log-normalised weights: log(v_i/Σv_j) = log_v_i − log_lamb_t
        log_v_norm = log_v_i - log_lamb_t.unsqueeze(-1)            # (B, M) no log(0)

        s_diff    = s.unsqueeze(1) - centers                              # (B, M, d)
        # log N(s; μ_i, diag(σ_i²)): uses exact diagonal Gaussian formula
        log_gauss = (
            0.5 * inv_var.prod(dim=-1).clamp(min=1e-12).log()            # log sqrt(|Σ^{-1}|)
            - (d / 2.0) * math.log(2.0 * math.pi)
            - 0.5 * (s_diff.pow(2) * inv_var).sum(dim=-1)                # (B, M)
        )  # (B, M)

        log_spatial = torch.logsumexp(log_v_norm + log_gauss, dim=-1)    # (B,)

        return -(log_temporal + log_spatial)  # (B,)

    def spatial_intensity(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """λ_t(t) · f(s|t) — true conditional intensity for visualization.

        Unlike ``nll()`` / ``log_prob()``, this does NOT include the
        compensator term (∫ λ dτ), so it returns the actual point-process
        intensity rather than the joint density over (dt, s).
        """
        B = z.shape[0]
        d = self.spatial_dim
        device = z.device

        t      = t.reshape(B)
        t_prev = t_prev.reshape(B)
        dt     = (t - t_prev).clamp(min=1e-6)

        if x_field is not None:
            t_hist  = x_field[:, :self.seq_len]
            s_hist  = x_field[:, self.seq_len:].reshape(B, self.seq_len, d)
            tn_ti_h = (t_prev.unsqueeze(-1) - t_hist).clamp(min=0.0)
        else:
            tn_ti_h = torch.zeros(B, self.seq_len, device=device)
            s_hist  = torch.zeros(B, self.seq_len, d, device=device)

        tn_ti_bg = torch.zeros(B, self.num_points, device=device)
        tn_ti    = torch.cat([tn_ti_h, tn_ti_bg], dim=-1)
        t_ti     = (tn_ti + dt.unsqueeze(-1)).clamp(min=1e-6)

        if self.background is not None:
            bg      = self.background.unsqueeze(0).expand(B, -1, -1)
            centers = torch.cat([s_hist, bg], dim=1)
        else:
            centers = s_hist

        w_i, b_i, sigma, inv_var = self._decode(z)

        log_w_i    = torch.log(w_i)
        log_v_i    = log_w_i - b_i * t_ti
        log_lamb_t = torch.logsumexp(log_v_i, dim=-1)
        log_v_norm = log_v_i - log_lamb_t.unsqueeze(-1)

        s_diff    = s.unsqueeze(1) - centers
        log_gauss = (
            0.5 * inv_var.prod(dim=-1).clamp(min=1e-12).log()
            - (d / 2.0) * math.log(2.0 * math.pi)
            - 0.5 * (s_diff.pow(2) * inv_var).sum(dim=-1)
        )
        log_spatial = torch.logsumexp(log_v_norm + log_gauss, dim=-1)

        return torch.exp(log_lamb_t + log_spatial)  # λ_t(t) · f(s|t), no compensator

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

    def sample(
        self,
        z: Tensor,
        t_prev: Tensor,
        x_field_fn=None,
    ):
        """Approximate sample: mean inter-arrival time + mixture location."""
        from typing import Tuple as _Tuple
        B   = z.shape[0]
        d   = self.spatial_dim
        M   = self.seq_len + self.num_points
        dev = z.device

        w_i, b_i, sigma, _ = self._decode(z)

        # Approximate dt via mean of exponential baseline intensity
        lambda0 = w_i.sum(dim=-1).clamp(min=1e-6)  # (B,)
        dt = (1.0 / lambda0).unsqueeze(-1)           # (B, 1)
        t_new = t_prev + dt

        # Sample spatial component ~ Cat(w_i/Σw_i) then s ~ N(μ_k, σ_k)
        v_norm = w_i / w_i.sum(-1, keepdim=True)
        k      = torch.multinomial(v_norm, 1).squeeze(-1)  # (B,)

        # Centers: use background; no history x_field available here
        centers = torch.zeros(B, M, d, device=dev)
        if self.background is not None:
            centers[:, self.seq_len:] = self.background.unsqueeze(0).expand(B, -1, -1)

        idx    = torch.arange(B, device=dev)
        mu_k   = centers[idx, k]           # (B, d)
        sig_k  = sigma[idx, k]             # (B, d)
        s_new  = mu_k + sig_k * torch.randn(B, d, device=dev)

        return t_new, s_new


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
