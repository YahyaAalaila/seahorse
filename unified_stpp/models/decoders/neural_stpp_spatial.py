"""Sequence-coupled spatial decoders for Neural STPP.

JumpCNFSpatial (SEQUENCE_COUPLED=True):
    Backward-chained radial flow decoder.
    log f*(s_i | z_i, {t_j, s_j}_{j<i}) computed by composing i radial flows,
    each conditioned on one past event (t_j, s_j, z_i). O(T²) per sequence.

SelfAttentiveCNFSpatial (SEQUENCE_COUPLED=True):
    Cross-event attention-conditioned CNF.
    Augments each z_i with global sequence context via self-attention over all L
    hidden states, then runs L independent ConcatSquash CNFs in one batched ODE
    solve. SEQUENCE_COUPLED because computing context c_i = Attn(z_i | z_{1..L})
    requires all L hidden states simultaneously.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torchdiffeq import odeint as _odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

from ..base import Decoder
from .spatial import CNFVelocityField, _euler_solve_simple

__all__ = [
    "HypernetworkRadialFlow",
    "JumpCNFSpatial",
    "EventTimeEncoding",
    "ActNorm",
    "SelfAttentiveODEFunc",
    "SelfAttentiveCNFSpatial",
]


# ============================================================================
# JumpCNFSpatial: backward-chained radial flow decoder
# ============================================================================

class HypernetworkRadialFlow(nn.Module):
    """Stack of ``n_flows`` radial flow layers parameterised by a hypernetwork.

    The hypernetwork takes context ``(z, t_obs, s_obs)`` and predicts the
    reference point x₀, log-scale α, and shift β_raw for each radial flow layer.

    Radial flow (Rezende & Mohamed 2015):
        h  = β / (r + α),  r = ‖s − x₀‖
        s' = s + h·(s − x₀)
        log|det J| = (d−1)·log|1+h| + log|1 + h − β·r/(r+α)²|

    Invertibility: β ≥ −α enforced via β = −α + softplus(β_raw).
    """

    def __init__(self, spatial_dim: int, hidden_dim: int, n_flows: int = 4):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.n_flows = n_flows
        ctx_dim = hidden_dim + 1 + spatial_dim   # z ‖ t_obs ‖ s_obs
        out_dim = n_flows * (spatial_dim + 2)    # per flow: x₀(d), log_α(1), β_raw(1)
        self.hyper = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        s: Tensor,
        t_obs: Tensor,
        s_obs: Tensor,
        z: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            s:     (B, d) spatial position to transform.
            t_obs: (B, 1) observed event time.
            s_obs: (B, d) observed event location.
            z:     (B, h) conditioning hidden state.
        Returns:
            s_out:   (B, d) transformed position.
            log_det: (B,)   log |det Jacobian|.
        """
        B, d = s.shape
        ctx    = torch.cat([z, t_obs, s_obs], dim=-1)  # (B, ctx_dim)
        params = self.hyper(ctx)                        # (B, n_flows*(d+2))
        params = params.reshape(B, self.n_flows, d + 2)

        s_cur       = s
        log_det_sum = s.new_zeros(B)

        for k in range(self.n_flows):
            x0_k     = params[:, k, :d]           # (B, d)
            log_ak   = params[:, k, d]             # (B,)
            beta_raw = params[:, k, d + 1]         # (B,)

            alpha  = F.softplus(log_ak) + 1e-5     # (B,) > 0
            beta   = -alpha + F.softplus(beta_raw)  # (B,) ≥ -alpha

            diff   = s_cur - x0_k                  # (B, d)
            r      = diff.norm(dim=-1)              # (B,)
            r_safe = r + 1e-8

            h      = beta / (r_safe + alpha)        # (B,)
            s_cur  = s_cur + h.unsqueeze(-1) * diff

            # log-det (Rezende & Mohamed 2015, Appendix)
            h2  = h - beta * r / (r_safe + alpha).pow(2)  # (B,)
            ld  = (d - 1) * torch.log((1.0 + h).clamp(min=1e-8)) \
                + torch.log((1.0 + h2).clamp(min=1e-8))
            log_det_sum = log_det_sum + ld

        return s_cur, log_det_sum


class JumpCNFSpatial(Decoder):
    """Backward-chained radial flow spatial decoder (SEQUENCE_COUPLED=True).

    For event i, the spatial log-density is computed via:
        s_base, log_det = F_{i-1}(F_{i-2}(…F_0(s_i)))
        log f*(s_i) = log N(s_base; 0, I) + log_det

    where each flow F_j is a HypernetworkRadialFlow conditioned on (t_j, s_j, z_i).

    Base distribution: isotropic Gaussian N(0, I).
    Complexity: O(T²) radial flow evaluations per sequence.
    """

    SEQUENCE_COUPLED = True

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        n_flows: int = 4,
        field_cov_dim: int = 0,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim, spatial_dim=spatial_dim)
        self.jump_flow = HypernetworkRadialFlow(spatial_dim, hidden_dim, n_flows)

    def log_prob(self, z, t, s, t_prev, x_field=None):
        raise NotImplementedError(
            "JumpCNFSpatial.SEQUENCE_COUPLED=True; use sequence_nll()."
        )

    def nll(self, z, t, s, t_prev, x_field=None):
        raise NotImplementedError(
            "JumpCNFSpatial.SEQUENCE_COUPLED=True; use sequence_nll()."
        )

    def sequence_nll(
        self,
        z_seq: Tensor,
        t_seq: Tensor,
        s_seq: Tensor,
        t_prev_seq: Tensor,
        lengths: Tensor,
        mask: Tensor,
        x_field_seq: Optional[Tensor] = None,
    ) -> Tensor:
        """Per-event NLL via backward-chained radial flows.

        For event i: apply i radial flows (one per past event j < i),
        each conditioned on (t_j, s_j, z_i). Evaluate N(0, I) at the
        transformed location.

        Note: O(T²) complexity — each event i requires i flow evaluations.
        For very long sequences consider SelfAttentiveCNFSpatial instead.

        Returns: (B, L) unmasked NLL.
        """
        B, L, h = z_seq.shape
        d = self.spatial_dim
        log_probs = z_seq.new_zeros(B, L)

        for i in range(L):
            s_i   = s_seq[:, i, :]   # (B, d)
            z_i   = z_seq[:, i, :]   # (B, h)
            s_cur = s_i
            log_det = z_seq.new_zeros(B)

            for j in range(i):
                t_j = t_seq[:, j, :]   # (B, 1)
                s_j = s_seq[:, j, :]   # (B, d)
                s_cur, ld = self.jump_flow(s_cur, t_j, s_j, z_i)
                log_det = log_det + ld

            # Base log-density N(0, I)
            log_p0 = -0.5 * (d * math.log(2 * math.pi) + (s_cur ** 2).sum(-1))
            log_probs[:, i] = log_p0 + log_det

        return -log_probs  # (B, L) NLL


# ============================================================================
# SelfAttentiveCNFSpatial: faithful reimplementation of NeuralSTPP
# ============================================================================

class EventTimeEncoding(nn.Module):
    """Sinusoidal encoding for scalar event times.

    Encodes each scalar time value as a ``d_model``-dimensional vector using
    a geometric frequency progression (log-spaced from 1 to max_freq).
    """

    def __init__(self, d_model: int, max_freq: float = 1000.0):
        super().__init__()
        if d_model % 2 != 0:
            d_model += 1  # pad to even
        self.d_model = d_model
        half = d_model // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * (-math.log(max_freq) / max(half - 1, 1))
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (*, 1) event times.
        Returns:
            (*, d_model) sinusoidal encoding.
        """
        t_val  = t.squeeze(-1)                        # (*,)
        phases = t_val.unsqueeze(-1) * self.freqs     # (*, half)
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)


class ActNorm(nn.Module):
    """ActNorm affine normalisation (Kingma & Dhariwal 2018).

    On the first training forward pass, initialises ``log_scale`` and ``bias``
    from batch statistics so the output has zero mean and unit variance.
    Subsequent passes apply the affine transform with learnable parameters.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("_initialized", torch.zeros(1, dtype=torch.bool))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias      = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, dim) input tensor.
        Returns:
            y:       (B, dim) normalised output.
            log_det: (B,)     log |det Jacobian| = sum(log_scale) per sample.
        """
        if not self._initialized.item() and self.training:
            with torch.no_grad():
                mean = x.mean(0)
                std  = x.std(0).clamp(min=1e-5)
                self.bias.data      = (-mean / std).to(self.bias.dtype)
                self.log_scale.data = (-std.log()).to(self.log_scale.dtype)
                self._initialized.fill_(True)

        scale   = self.log_scale.exp()
        y       = x * scale + self.bias
        log_det = self.log_scale.sum().expand(x.shape[0])
        return y, log_det


class SelfAttentiveODEFunc(nn.Module):
    """CNF velocity field that runs self-attention at every ODE function evaluation.

    Faithful to the original NeuralSTPP SelfAttentiveCNF (Chen et al. 2021):
    attention is inside the ODE, not a one-time pre-processing step.

    Implements **TimeVariableCNF**: each event carries its own occurrence time
    ``t_event`` in the ODE state (constant slot, derivative = 0).  The ODE is
    integrated over dummy time ``s ∈ [1, 0]`` (backward), where the actual time
    for event ``l`` at step ``s`` is ``t_actual = s · t_event[l]``.  The velocity
    in dummy time scales by ``t_event``: ``v_dummy = v_actual · t_event``.

    State shape: ``(B, L, d + 3)``

    * ``[..., :d]``   — current spatial locations z (transform with ODE)
    * ``[...,  d]``   — log-det accumulator (integrated)
    * ``[..., d+1]``  — per-event occurrence time ``t_event`` (constant, d/ds = 0)
    * ``[..., d+2]``  — kinetic energy accumulator ∫‖v_dummy‖²/d ds (for --otreg_strength)

    At each evaluation:

    1. Compute per-event actual time ``t_actual = s · t_event``.
    2. Project ``cat([z (d), h_backbone (h), t_emb(t_actual) (te)])`` → ``hidden_dim``.
    3. Causal upper-triangular self-attention (event i attends only to 0..i-1).
    4. Softplus FC block → ``v_actual`` (B, L, d); scale to ``v_dummy = v_actual · t_event``.
    5. Hutchinson per-event divergence trace (unbiased for coupled attention).

    **Why Hutchinson is unbiased for coupled attention**: the noise ``e`` is i.i.d.
    across events, so cross-event terms ``E[e_{b,l} · e_{b,l'}] = 0`` for l ≠ l',
    making the per-event estimator ``(vjp * e).sum(-1)`` unbiased for the local
    diagonal ``Σ_k ∂v_{b,l,k}/∂z_{b,l,k}`` even when attention couples events.
    """

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        num_heads: int,
        n_hidden_layers: int,
        backbone_dim: int,
        t_embed_dim: int = 16,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.hidden_dim  = hidden_dim
        self.t_embed_dim = t_embed_dim

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads

        self.in_proj  = nn.Linear(spatial_dim + backbone_dim + t_embed_dim, hidden_dim)
        # Manual multi-head attention projections (no bias, matching original).
        # We avoid nn.MultiheadAttention because its Flash-Attention CPU kernel
        # does not support create_graph=True (needed for Hutchinson via autograd.grad).
        self.q_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        fc_layers: list = []
        for _ in range(n_hidden_layers):
            fc_layers += [nn.Linear(hidden_dim, hidden_dim), nn.Softplus()]
        fc_layers.append(nn.Linear(hidden_dim, spatial_dim))
        self.velocity_net = nn.Sequential(*fc_layers)

        # Set before each ODE solve
        self._h_backbone:   Optional[Tensor] = None  # (B, L, backbone_dim)
        self._e:            Optional[Tensor] = None  # (B, L, d) Hutchinson noise
        self._causal_mask:  Optional[Tensor] = None  # (L, L) upper-triangular -inf

    def _time_embed(self, t: Tensor) -> Tensor:
        """Sinusoidal encoding of per-event actual times.

        Args:
            t: ``(B, L)`` per-event actual times.
        Returns:
            ``(B, L, t_embed_dim)`` sinusoidal encodings.
        """
        half  = self.t_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )  # (half,)
        angles = t.unsqueeze(-1) * freqs  # (B, L, half)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, L, t_embed_dim)

    def _velocity(self, z: Tensor, h: Tensor, t_emb: Tensor) -> Tensor:
        """z: (B,L,d), h: (B,L,H), t_emb: (B,L,te) → (B,L,d).

        Manual multi-head attention so that create_graph=True works in the
        Hutchinson divergence path (Flash-Attention CPU does not support it).
        Sequences are end-padded so the causal mask is sufficient — valid event i
        only attends to positions 0..i-1, all of which are also valid.
        Padded events' outputs are discarded in sequence_nll via ``valid`` mask.
        """
        x = self.in_proj(torch.cat([z, h, t_emb], dim=-1))  # (B, L, H)
        B, L, H = x.shape
        nh, hd  = self.num_heads, self.head_dim

        def _split_heads(t: Tensor) -> Tensor:
            # (B, L, H) → (B, nh, L, hd)
            return t.reshape(B, L, nh, hd).permute(0, 2, 1, 3)

        Q = _split_heads(self.q_proj(x))  # (B, nh, L, hd)
        K = _split_heads(self.k_proj(x))
        V = _split_heads(self.v_proj(x))

        scale  = math.sqrt(hd)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, nh, L, L)
        if self._causal_mask is not None:
            scores = scores + self._causal_mask                 # broadcast over B, nh
        weights = torch.softmax(scores, dim=-1)                 # (B, nh, L, L)
        ctx     = torch.matmul(weights, V)                      # (B, nh, L, hd)
        ctx     = ctx.permute(0, 2, 1, 3).reshape(B, L, H)     # (B, L, H)
        x_attn  = self.out_proj(ctx)
        x_attn  = self.attn_norm(x + x_attn)
        return self.velocity_net(x_attn)                        # (B, L, d)

    def forward(self, s: Tensor, state: Tensor) -> Tensor:
        """
        Args:
            s:     scalar dummy time (integration goes 1 → 0 for density eval).
            state: ``(B, L, d + 2)`` ODE state — see class docstring.
        Returns:
            ``d(state)/ds`` of the same shape.
        """
        d = self.spatial_dim
        B, L, _ = state.shape
        z       = state[..., :d]      # (B, L, d) current spatial locations
        t_event = state[..., d + 1]   # (B, L)    per-event occurrence time (constant)
        h       = self._h_backbone    # (B, L, H)

        # TimeVariableCNF: at dummy step s, actual time = s * t_event
        # (s=1 → t_event = data time; s=0 → 0 = base time)
        t_actual = s * t_event                  # (B, L)
        t_emb    = self._time_embed(t_actual)   # (B, L, te)

        # ---- main velocity path (grad flows for ODE solver backprop) -------
        v_actual = self._velocity(z, h, t_emb)              # (B, L, d)
        v        = v_actual * t_event.unsqueeze(-1)         # (B, L, d)  scale to dummy

        # ---- divergence via Hutchinson (separate detached path) ------------
        # Mirrors CNFVelocityField in spatial.py: detach z → fresh leaf → grad.
        # Hutchinson is applied to v_dummy (= v_actual * t_event), which is
        # correct: Tr(∂v_dummy/∂z) = t_event · Tr(∂v_actual/∂z) per event.
        with torch.enable_grad():
            z_leaf   = z.detach().requires_grad_(True)
            ta_det   = t_actual.detach()
            te_det   = self._time_embed(ta_det)
            h_det    = h.detach()
            v_act_d  = self._velocity(z_leaf, h_det, te_det)          # (B, L, d)
            v_div    = v_act_d * t_event.detach().unsqueeze(-1)       # (B, L, d)

            e = self._e
            if e is not None:
                e_dot_v = (e * v_div).sum(-1)               # (B, L)
                vjp, = torch.autograd.grad(
                    e_dot_v.sum(), z_leaf,
                    create_graph=True, retain_graph=True,
                )                                           # (B, L, d)
                div = (vjp * e).sum(-1)                     # (B, L)
            else:
                div = torch.zeros(B, L, device=z.device, dtype=z.dtype)

        # d(t_event)/ds = 0  (constant slot)
        zeros_t = torch.zeros_like(state[..., d + 1 : d + 2])  # (B, L, 1)
        # Kinetic energy rate (detached): ‖v_dummy‖² / d per event per dummy-time step.
        # Integrated over s ∈ [1→0] gives the OT regularization term (--otreg_strength).
        energy_rate = v.detach().pow(2).sum(-1, keepdim=True) / max(d, 1)  # (B, L, 1)
        return torch.cat([v, -div.unsqueeze(-1), zeros_t, energy_rate], dim=-1)  # (B, L, d+3)


class SelfAttentiveCNFSpatial(Decoder):
    """Faithful reimplementation of the SelfAttentiveCNF from NeuralSTPP (Chen et al. 2021).

    Two-stage normalizing flow:

    1. **Attentive CNF**: self-attention runs *inside* the ODE at each function
       evaluation over current spatial locations + backbone hidden states.
       Causal upper-triangular mask: event i attends only to events 0..i-1.
       (Implemented by :class:`SelfAttentiveODEFunc`.)

    2. **Base FC CNF**: simple ConcatSquash/MLP flow, no attention.
       Conditioned on backbone hidden states per event.
       (Uses the existing :class:`~.spatial.CNFVelocityField`.)

    Learned base distribution: ``N(μ(h), σ(h)²)`` per event, predicted from the
    backbone hidden states.  ActNorm pre-normalises spatial locations.

    ``SEQUENCE_COUPLED = True``: the attentive CNF requires all L events
    simultaneously (attention over the full sequence).
    """

    SEQUENCE_COUPLED = True

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        solver: str = "dopri5",
        atol: float = 1e-4,
        rtol: float = 1e-4,
        num_heads: int = 4,
        n_hidden_layers: int = 3,
        layer_type: str = "concat",
        field_cov_dim: int = 0,
        t_embed_dim: int = 16,
        otreg_strength: float = 0.0,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim, spatial_dim=spatial_dim)
        self.spatial_dim    = spatial_dim
        self.hidden_dim     = hidden_dim
        self.solver         = solver
        self.atol           = atol
        self.rtol           = rtol
        self.otreg_strength = otreg_strength
        self._energy_reg: torch.Tensor | float = 0.0  # populated each forward pass
        self._debug_nstpp = os.getenv("UNIFIED_STPP_DEBUG_NSTPP", "0") == "1"
        self._debug_nstpp_max_calls = max(
            1, int(os.getenv("UNIFIED_STPP_DEBUG_NSTPP_MAX_CALLS", "10"))
        )
        self._debug_nstpp_calls = 0

        # 1. Attentive CNF ODE function (attention inside ODE)
        self.attn_ode_func = SelfAttentiveODEFunc(
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            n_hidden_layers=n_hidden_layers,
            backbone_dim=hidden_dim,
            t_embed_dim=t_embed_dim,
        )

        # 2. Base FC CNF velocity field: FC + Softplus (faithful to original base_cnf).
        # Hardcoded "softplus" regardless of layer_type, which applies only to the
        # main attentive velocity net inside SelfAttentiveODEFunc.
        self.base_velocity = CNFVelocityField(
            spatial_dim, hidden_dim, field_cov_dim, "softplus", n_hidden_layers
        )

        # 3. Learned base distribution: backbone h → (μ, log σ) per spatial dim
        self.base_dist_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, spatial_dim * 2),  # → (μ_d, log_σ_d)
        )

        # 4. Data-driven input normalisation
        self.act_norm = ActNorm(spatial_dim)

    def log_prob(self, z, t, s, t_prev, x_field=None):
        raise NotImplementedError(
            "SelfAttentiveCNFSpatial.SEQUENCE_COUPLED=True; use sequence_nll."
        )

    def nll(self, z, t, s, t_prev, x_field=None):
        raise NotImplementedError(
            "SelfAttentiveCNFSpatial.SEQUENCE_COUPLED=True; use sequence_nll."
        )

    def sequence_nll(
        self,
        z_seq: Tensor,
        t_seq: Tensor,
        s_seq: Tensor,
        t_prev_seq: Tensor,
        lengths: Tensor,
        mask: Tensor,
        x_field_seq: Optional[Tensor] = None,
    ) -> Tensor:
        """Per-event NLL via two-stage attentive CNF.

        Pipeline:

        1. ActNorm normalise spatial locations.
        2. **Attentive CNF** (``SelfAttentiveODEFunc``): ODE over ``(B, L, d+2)``
           state (TimeVariableCNF); attention inside the ODE at each step; causal mask
           applied; per-event actual time ``t_event`` carried as a constant slot.
        3. **Base FC CNF** (``CNFVelocityField``): ODE over ``(B*L, d+1)`` state;
           each event is independent; conditioned on backbone hidden states.
        4. Evaluate learned base distribution ``N(μ(h), σ(h)²)`` at base samples.
        5. Sum log-probs and return per-event NLL.

        Returns:
            Tensor of shape ``(B, L)`` — NLL for valid events, 0 for padding.
        """
        B, L, h = z_seq.shape
        d = self.spatial_dim
        valid     = mask > 0                     # (B, L)
        valid_flat = valid.reshape(B * L)

        # ------------------------------------------------------------------ #
        # 1. ActNorm normalise
        # ------------------------------------------------------------------ #
        s_flat = s_seq.reshape(B * L, d)
        if (
            self.training
            and not bool(self.act_norm._initialized.item())
            and valid_flat.any()
        ):
            _ = self.act_norm(s_flat[valid_flat])
        s_normed, ld_actn = self.act_norm(s_flat)   # (B*L, d), (B*L,)
        s_normed_seq = s_normed.reshape(B, L, d)
        ld_actn_seq  = ld_actn.reshape(B, L)

        t_span = torch.tensor(
            [1.0, 0.0], device=z_seq.device, dtype=z_seq.dtype
        )

        # ------------------------------------------------------------------ #
        # 2. Attentive CNF: state (B, L, d+2) with TimeVariableCNF
        #    State layout: [spatial(d) | logdet(1) | t_event(1)]
        #    t_event is constant during integration (derivative = 0).
        #    At dummy step s, actual time per event = s * t_event.
        # ------------------------------------------------------------------ #
        causal_mask  = torch.triu(
            torch.full(
                (L, L), float("-inf"), device=z_seq.device, dtype=z_seq.dtype
            ),
            diagonal=1,
        )
        t_event_seq = t_seq.squeeze(-1)   # (B, L) actual event occurrence times

        self.attn_ode_func._h_backbone  = z_seq
        self.attn_ode_func._causal_mask = causal_mask
        self.attn_ode_func._e           = torch.randn_like(s_normed_seq)

        y0_attn = torch.cat(
            [
                s_normed_seq,                          # (B, L, d)
                s_normed_seq.new_zeros(B, L, 1),      # logdet = 0
                t_event_seq.unsqueeze(-1),             # (B, L, 1) t_event (constant)
                s_normed_seq.new_zeros(B, L, 1),      # energy = 0
            ],
            dim=-1,
        ).requires_grad_(True)                         # (B, L, d+3)

        if HAS_TORCHDIFFEQ:
            traj_attn = _odeint(
                self.attn_ode_func, y0_attn, t_span,
                method=self.solver, atol=self.atol, rtol=self.rtol,
                options={"dtype": z_seq.dtype},
            )
            y_attn = traj_attn[-1]                    # (B, L, d+2)
        else:
            y_attn = _euler_solve_simple(
                self.attn_ode_func, y0_attn, t_span, n_steps=50
            )

        s_mid   = y_attn[..., :d]                     # (B, L, d)
        ld_attn = y_attn[...,  d]                     # (B, L)
        # y_attn[..., d+1] = t_event (constant, unchanged)
        energy_attn = y_attn[..., d + 2]              # (B, L) ∫‖v_dummy‖²/d ds
        # Store OT regularization scalar (--otreg_strength); read by UnifiedSTPP.
        # Integration runs s: 1→0 so ds<0, making energy_attn negative;
        # negate to recover the positive integral ∫_0^1 ‖v‖²/d ds.
        n_valid = valid.float().sum().clamp(min=1)
        self._energy_reg = self.otreg_strength * ((-energy_attn) * valid.float()).sum() / n_valid

        # ------------------------------------------------------------------ #
        # 3. Base FC CNF: state (B*L, d+1), independent per event
        # ------------------------------------------------------------------ #
        self.base_velocity._z_cond = z_seq.reshape(B * L, h)
        self.base_velocity._x_cond = None
        self.base_velocity._e      = None                    # fresh Hutchinson noise

        s_mid_flat = s_mid.reshape(B * L, d)
        y0_base = torch.cat(
            [s_mid_flat, s_mid_flat.new_zeros(B * L, 1)], dim=-1
        ).requires_grad_(True)                               # (B*L, d+1)

        if HAS_TORCHDIFFEQ:
            traj_base = _odeint(
                self.base_velocity, y0_base, t_span,
                method=self.solver, atol=self.atol, rtol=self.rtol,
                options={"dtype": z_seq.dtype},
            )
            y_base = traj_base[-1]                           # (B*L, d+1)
        else:
            y_base = _euler_solve_simple(
                self.base_velocity, y0_base, t_span, n_steps=50
            )

        s_base_flat = y_base[:, :d]                          # (B*L, d)
        ld_base     = y_base[:,  d].reshape(B, L)            # (B, L)

        # ------------------------------------------------------------------ #
        # 4. Learned base distribution: N(μ(h), σ(h)²) per event per dim
        # ------------------------------------------------------------------ #
        params    = self.base_dist_net(z_seq.reshape(B * L, h))  # (B*L, 2d)
        mu        = params[:, :d].reshape(B, L, d)
        log_sigma = params[:, d:].reshape(B, L, d)
        s_base    = s_base_flat.reshape(B, L, d)

        # log N(s; μ, σ²) = -d/2 log(2π) - Σ_k log(σ_k) - Σ_k (s_k-μ_k)²/(2σ_k²)
        log_p0 = (
            -0.5 * d * math.log(2.0 * math.pi)
            - log_sigma.sum(-1)                              # -Σ_k log σ_k  (B, L)
            - 0.5 * ((s_base - mu) * (-log_sigma).exp()).pow(2).sum(-1)  # (B, L)
        )

        # ------------------------------------------------------------------ #
        # 5. Total log-prob and NLL
        # ------------------------------------------------------------------ #
        # CNF convention (same as CNFSpatial.log_prob in spatial.py):
        #   log p(data) = log p(base) - log_det_flow
        # where log_det_flow is accumulated by integrating d(log_det)/dt = -div(v).
        # Therefore both attentive and base CNF log-det accumulators are SUBTRACTED.
        # ActNorm contributes +log|det(ds_normed/ds_obs)| (forward affine map).
        log_fs = log_p0 - ld_attn - ld_base + ld_actn_seq   # (B, L)
        nll    = -log_fs
        debug_this_call = (
            self._debug_nstpp and self._debug_nstpp_calls < self._debug_nstpp_max_calls
        )
        if debug_this_call:
            valid_2d = valid
            valid_3d = valid.unsqueeze(-1).expand_as(log_sigma)

            def _print_stats(name: str, tensor: Tensor, mask_debug: Tensor) -> None:
                vals = tensor.detach()[mask_debug]
                if vals.numel() == 0:
                    print(f"[NSTPP-DEBUG][spatial] {name}: no valid values")
                    return
                finite = torch.isfinite(vals)
                if not bool(finite.all().item()):
                    n_bad = int((~finite).sum().item())
                    n_nan = int(torch.isnan(vals).sum().item())
                    n_inf = int(torch.isinf(vals).sum().item())
                    print(
                        f"[NSTPP-DEBUG][spatial][nonfinite] {name}: "
                        f"bad={n_bad}/{vals.numel()} nan={n_nan} inf={n_inf}"
                    )
                vals = vals[finite]
                if vals.numel() == 0:
                    print(f"[NSTPP-DEBUG][spatial] {name}: no finite values")
                    return
                print(
                    f"[NSTPP-DEBUG][spatial] {name}(min/mean/max)="
                    f"{vals.min().item():.6f}/{vals.mean().item():.6f}/{vals.max().item():.6f}"
                )

            _print_stats("log_sigma", log_sigma, valid_3d)
            _print_stats("log_p0", log_p0, valid_2d)
            _print_stats("ld_attn", ld_attn, valid_2d)
            _print_stats("ld_base", ld_base, valid_2d)
            _print_stats("log_fs", log_fs, valid_2d)
            _print_stats("nll", nll, valid_2d)
            print(
                f"[NSTPP-DEBUG][spatial] energy_reg="
                f"{float(torch.as_tensor(self._energy_reg).detach().item()):.6f}"
            )
            self._debug_nstpp_calls += 1
        return torch.where(valid, nll, torch.zeros_like(nll))
