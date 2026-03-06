"""
Faithful reimplementation of DeepSTPP (Lin et al. 2021).

Implements the original Hawkes-based spatio-temporal point process model:
  - Sinusoidal positional encoding keyed on actual cumulative event time
  - Transformer encoder → optional VAE (qm, qv) → deterministic z
  - Hawkes exponential kernel: λ(t) = Σ_i w_i exp(-b_i (t - t_i))
  - Data-centered spatial Gaussians: f*(s|t) = Σ_i v_i(t) N(s; s_i, Λ_i^{-1})
  - Learnable background spatial anchors
  - Optional KL term when sample=True

This is a standalone nn.Module exposing the same forward() interface as
UnifiedSTPP: forward(times, locations, lengths, **kwargs) → {"nll", ...}.

Under protocol="paper_autostpp_sthp":
  times[:, :seq_len]      = history cumulative MinMax-scaled times
  times[:, seq_len]       = target cumulative time (single target per window)
  locations[:, :seq_len]  = history MinMax-scaled [x, y]
  locations[:, seq_len]   = target location
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Sinusoidal time-based positional encoding
# ---------------------------------------------------------------------------

class SinusoidalTimePE(nn.Module):
    """
    Fixed sinusoidal positional encoding keyed on actual elapsed time values.

    Matches the original DeepSTPP PositionalEncoding which computes
    t = cumsum(delta_t) and applies sin/cos at frequencies
    omega_k = 1 / max_freq^(2k/d_model).
    """

    def __init__(self, d_model: int, max_freq: float = 1e4):
        super().__init__()
        self.d_model = d_model
        k = torch.arange(0, d_model // 2, dtype=torch.float32)
        omega = 1.0 / (max_freq ** (2.0 * k / d_model))  # (d//2,)
        self.register_buffer("omega", omega)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, d_model) — token embeddings
            t: (B, N)          — cumulative event times (e.g. cumsum(MinMax(delta_t)))
        Returns:
            x + PE: (B, N, d_model)
        """
        # angles: (B, N, d//2)
        angles = t.unsqueeze(-1) * self.omega.unsqueeze(0).unsqueeze(0)
        pe = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, N, d_model or d_model±1)
        return x + pe[..., : self.d_model]


# ---------------------------------------------------------------------------
# Reusable MLP decoder block (matches original Decoder class)
# ---------------------------------------------------------------------------

class _HawkesMLPDecoder(nn.Module):
    """
    MLP decoder matching the original DeepSTPP Decoder class:
      Linear(in) → ELU, [Linear(mlp) → ELU] * (n_layers-2), Linear(out) → [Softplus]

    With n_layers=2: Linear(in→mlp) → ELU → Linear(mlp→out) → [Softplus]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int = 2,
        mlp_dim: int = 64,
        softplus_out: bool = False,
    ):
        super().__init__()
        layers: list = []
        cur = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(cur, mlp_dim), nn.ELU()]
            cur = mlp_dim
        layers.append(nn.Linear(cur, out_dim))
        if softplus_out:
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FaithfulDeepSTPP(nn.Module):
    """
    Faithful DeepSTPP reimplementation.

    Args:
        hidden_dim:   Transformer d_model (= latent z dimension).
        spatial_dim:  Dimensionality of event locations (2 for lat/lon).
        seq_len:      History window length (must match paper_lookback).
        num_points:   Number of learnable background spatial anchors.
        num_heads:    Transformer attention heads.
        num_layers:   Transformer encoder depth.
        dropout:      Attention dropout.
        n_mlp_layers: Depth of each MLP decoder branch (w, b, s).
        mlp_dim:      Hidden width of MLP decoders.
        constrain_b:  How to constrain Hawkes decay rates:
                      "softplus" | "sigmoid" | "clamp" | "tanh"
        b_max:        Upper bound used by "sigmoid", "clamp", "tanh" constraints.
        sample:       If True, sample z from VAE posterior (adds KL to loss).
        beta:         LL coefficient in NELBO when sample=True.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        spatial_dim: int = 2,
        seq_len: int = 10,
        num_points: int = 50,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_mlp_layers: int = 2,
        mlp_dim: int = 64,
        constrain_b: str = "softplus",
        b_max: float = 20.0,
        sample: bool = False,
        beta: float = 0.1,
        **kwargs,  # absorb unused registry keys
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        self.seq_len = seq_len
        self.num_points = num_points
        self.constrain_b = constrain_b
        self.b_max = b_max
        self.sample = sample
        self.beta = beta

        M = seq_len + num_points  # total number of Hawkes kernels

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        # Input: [x, y, delta_t] (spatial_dim + 1 features)
        self.event_embed = nn.Linear(spatial_dim + 1, hidden_dim)

        # Time-based sinusoidal PE (uses cumulative times, not token indices)
        self.pos_enc = SinusoidalTimePE(hidden_dim)

        # Causal transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # VAE projection heads (last transformer token → qm, qv)
        self.vae_mean   = nn.Linear(hidden_dim, hidden_dim)
        self.vae_logvar = nn.Linear(hidden_dim, hidden_dim)

        # ------------------------------------------------------------------
        # Background anchors
        # ------------------------------------------------------------------
        self.background = nn.Parameter(torch.randn(num_points, spatial_dim) * 0.01)

        # ------------------------------------------------------------------
        # Hawkes kernel decoders  (z → kernel parameters over M terms)
        # ------------------------------------------------------------------
        # w_dec: positive weights (softplus output)
        self.w_dec = _HawkesMLPDecoder(
            hidden_dim, M, n_layers=n_mlp_layers, mlp_dim=mlp_dim, softplus_out=True
        )
        # b_dec: decay rates (constrained in _decode)
        self.b_dec = _HawkesMLPDecoder(
            hidden_dim, M, n_layers=n_mlp_layers, mlp_dim=mlp_dim, softplus_out=False
        )
        # s_dec: inverse variances for spatial Gaussians (softplus, reshaped)
        self.s_dec = _HawkesMLPDecoder(
            hidden_dim, M * spatial_dim, n_layers=n_mlp_layers, mlp_dim=mlp_dim, softplus_out=True
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(
        self,
        h_times: Tensor,
        h_locs: Tensor,
        h_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Encode history window into a latent vector z.

        Args:
            h_times:   (B, seq_len) — cumulative scaled times for history events
            h_locs:    (B, seq_len, d) — scaled spatial coordinates for history
            h_lengths: (B,) — actual history lengths (for padding mask)

        Returns:
            z:          (B, h) — latent code (qm if not sampling)
            qm:         (B, h) — VAE posterior mean
            qv:         (B, h) — VAE posterior log-variance
            all_states: (B, seq_len, h) — transformer output states
        """
        B, N = h_times.shape
        device = h_times.device

        # Reconstruct delta_t from cumulative times (matches original st_x[:,:,2])
        delta_t = torch.cat(
            [h_times[:, :1], h_times[:, 1:] - h_times[:, :-1]], dim=-1
        )  # (B, seq_len)

        # Stack [x, y, delta_t] — same order as original st_x
        events = torch.cat([h_locs, delta_t.unsqueeze(-1)], dim=-1)  # (B, N, d+1)

        # Embed and add time-based sinusoidal PE
        x = self.event_embed(events)     # (B, N, h)
        x = self.pos_enc(x, h_times)     # (B, N, h)

        # Causal mask: upper-triangular, True = masked (future)
        causal_mask = torch.triu(
            torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1
        )

        # Padding mask: True = position is padding (beyond actual length)
        arange = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
        pad_mask = arange >= h_lengths.unsqueeze(1)            # (B, N)

        all_states = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        all_states = self.norm(all_states)  # (B, N, h)

        # Extract last valid token per sequence
        idx = (h_lengths - 1).clamp(min=0).long()              # (B,)
        last = all_states[torch.arange(B, device=device), idx]  # (B, h)

        # VAE projection
        qm = self.vae_mean(last)    # (B, h)
        qv = self.vae_logvar(last)  # (B, h)  — log variance

        if self.sample and self.training:
            eps = torch.randn_like(qm)
            z = qm + torch.exp(0.5 * qv) * eps
        else:
            z = qm

        return z, qm, qv, all_states

    def _decode(
        self, z: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Decode latent vector into Hawkes kernel parameters.

        Args:
            z: (B, h)

        Returns:
            w_i:    (B, M)    — positive kernel weights (softplus)
            b_i:    (B, M)    — positive decay rates (constrained)
            inv_var:(B, M, d) — positive inverse variances for spatial Gaussians
        """
        B = z.shape[0]

        w_i = self.w_dec(z)   # (B, M) — already softplus-positive

        # Decay rates: constrained positive
        b_raw = self.b_dec(z)  # (B, M)
        if self.constrain_b == "sigmoid":
            b_i = self.b_max * torch.sigmoid(b_raw)
        elif self.constrain_b == "clamp":
            b_i = b_raw.clamp(min=0.0, max=self.b_max)
        elif self.constrain_b == "tanh":
            b_i = self.b_max * (1.0 + torch.tanh(b_raw)) / 2.0
        else:  # "softplus" (default)
            b_i = F.softplus(b_raw)

        # Inverse variances: softplus-positive, reshape to (B, M, d)
        inv_var = self.s_dec(z).reshape(B, self.seq_len + self.num_points, self.spatial_dim)

        return w_i, b_i, inv_var

    # ------------------------------------------------------------------
    # Hawkes intensity functions (static, match original exactly)
    # ------------------------------------------------------------------

    @staticmethod
    def _t_intensity(w_i: Tensor, b_i: Tensor, t_ti: Tensor) -> Tensor:
        """
        Temporal intensity: λ(t) = Σ_i w_i exp(-b_i * t_ti)

        t_ti = t - t_i  (elapsed time from history event i to query time t).
        """
        return (w_i * torch.exp(-b_i * t_ti)).sum(dim=-1)  # (B,)

    @staticmethod
    def _ll_no_events(
        w_i: Tensor, b_i: Tensor, tn_ti: Tensor, t_ti: Tensor
    ) -> Tensor:
        """
        Negative compensator:  Σ_i w_i/b_i * (exp(-b_i*t_ti) - exp(-b_i*tn_ti))

        This equals  -∫_{t_n}^{t} λ(s) ds  and is ≤ 0 when t > t_n.

        tn_ti = t_n - t_i  (elapsed from event i to last history event t_n).
        t_ti  = tn_ti + delta_t_y  (elapsed from event i to target time t).
        """
        return (w_i / b_i * (torch.exp(-b_i * t_ti) - torch.exp(-b_i * tn_ti))).sum(dim=-1)

    @staticmethod
    def _log_ft(
        w_i: Tensor, b_i: Tensor, tn_ti: Tensor, t_ti: Tensor
    ) -> Tensor:
        """
        Log conditional density of next event time:
          log f*(t) = log λ(t) - ∫_{t_n}^{t} λ(s) ds
                    = log(t_intensity) + ll_no_events
        """
        lam = FaithfulDeepSTPP._t_intensity(w_i, b_i, t_ti).clamp(min=1e-8)
        return FaithfulDeepSTPP._ll_no_events(w_i, b_i, tn_ti, t_ti) + lam.log()

    @staticmethod
    def _log_s_intensity(
        w_i: Tensor,
        b_i: Tensor,
        t_ti: Tensor,
        s_diff: Tensor,
        inv_var: Tensor,
    ) -> Tensor:
        """
        Log conditional spatial density given time t:
          f*(s|t) = Σ_i v_i(t) * g(s - s_i, Λ_i)

        where v_i(t) = w_i exp(-b_i t_ti) / Σ_j v_j(t)  (normalized weights),
        and   g(s_diff, Λ) = sqrt(det Λ) / (2π) * exp(-0.5 s_diff^T Λ s_diff)
              is the diagonal-Gaussian density with precision Λ = diag(inv_var).

        Computed in log-space for numerical stability.

        Args:
            w_i:    (B, M)
            b_i:    (B, M)
            t_ti:   (B, M)
            s_diff: (B, M, d) — s_target - [history_locs, background]
            inv_var:(B, M, d) — diagonal precision per kernel

        Returns: (B,) log f*(s|t)
        """
        v_i   = w_i * torch.exp(-b_i * t_ti)                        # (B, M)
        v_norm = v_i / v_i.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, M)

        # Diagonal Gaussian log-density
        # log g = 0.5 * log(prod(inv_var)) - log(2π) - 0.5 * Σ_d s_diff_d * inv_var_d * s_diff_d
        log_det = 0.5 * inv_var.prod(dim=-1).clamp(min=1e-12).log()  # (B, M)
        quad    = (s_diff * inv_var * s_diff).sum(dim=-1)             # (B, M)
        log_g   = log_det - math.log(2.0 * math.pi) - 0.5 * quad    # (B, M)

        log_v_norm = v_norm.clamp(min=1e-8).log()  # (B, M)
        return torch.logsumexp(log_v_norm + log_g, dim=-1)  # (B,)

    @staticmethod
    def _kl_normal(qm: Tensor, qv: Tensor) -> Tensor:
        """
        KL divergence KL(N(qm, exp(qv)) || N(0, I)):
          = 0.5 * Σ_d (qm_d^2 + exp(qv_d) - qv_d - 1)

        qv is log-variance (not log-std).
        """
        return 0.5 * (qm.pow(2) + qv.exp() - qv - 1).sum(dim=-1)  # (B,)

    # ------------------------------------------------------------------
    # Main forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        **kwargs,  # accept (and ignore) marks, x_event, x_field_at_events, etc.
    ) -> dict:
        """
        Compute NLL for a batch of fixed-window sequences.

        Under protocol="paper_autostpp_sthp" with lookback=seq_len, lookahead=1:
          times[:, :seq_len]     = history cumulative scaled times
          times[:, seq_len]      = target cumulative time (one target per window)
          locations[:, :seq_len] = history scaled locations
          locations[:, seq_len]  = target location

        Args:
            times:     (B, seq_len+1) — cumulative scaled times
            locations: (B, seq_len+1, d) — scaled spatial coordinates
            lengths:   (B,) — actual sequence lengths (accepted for API compatibility)

        Returns:
            dict with keys:
              "nll"           — scalar mean NLL over batch
              "nll_per_event" — (B,) per-sequence NLL
              "total_events"  — scalar number of target events (= B)
        """
        B = times.shape[0]
        device = times.device
        seq_len = self.seq_len

        # --- Split history and target ---
        h_times  = times[:, :seq_len]           # (B, seq_len)
        h_locs   = locations[:, :seq_len, :]    # (B, seq_len, d)
        t_target = times[:, seq_len]             # (B,)
        s_target = locations[:, seq_len, :]      # (B, d)

        # History lengths: clamp to seq_len (padding beyond that is ignored)
        h_lengths = lengths.clamp(max=seq_len).to(device)

        # --- Encode ---
        z, qm, qv, _ = self._encode(h_times, h_locs, h_lengths)

        # --- Decode Hawkes parameters ---
        w_i, b_i, inv_var = self._decode(z)
        # w_i, b_i: (B, M)   inv_var: (B, M, d)

        # --- Temporal: compute tn_ti and t_ti ---
        # tn_ti_hist = t_n - t_i   (time from each history event to last history event t_n)
        t_n = h_times[:, -1]  # (B,) last history event cumulative time
        tn_ti_hist = t_n.unsqueeze(-1) - h_times                         # (B, seq_len)
        tn_ti_bg   = torch.zeros(B, self.num_points, device=device)      # (B, num_points) — bg has no arrival time
        tn_ti = torch.cat([tn_ti_hist, tn_ti_bg], dim=-1)                # (B, M)

        delta_t_y = (t_target - t_n).unsqueeze(-1)   # (B, 1)  time from t_n to target
        t_ti = tn_ti + delta_t_y                      # (B, M)  time from each event to target

        # --- Spatial: compute s_diff ---
        bg_exp = self.background.unsqueeze(0).expand(B, -1, -1)           # (B, num_points, d)
        s_x    = torch.cat([h_locs, bg_exp], dim=1)                       # (B, M, d)
        s_diff = s_target.unsqueeze(1) - s_x                              # (B, M, d)

        # --- Log-likelihoods ---
        sll = self._log_s_intensity(w_i, b_i, t_ti, s_diff, inv_var)    # (B,)
        tll = self._log_ft(w_i, b_i, tn_ti, t_ti)                        # (B,)

        # --- Loss ---
        if self.sample and self.training:
            kl = self._kl_normal(qm, qv).mean()
            mean_nll = kl - self.beta * (sll.mean() + tll.mean())
        else:
            mean_nll = -(sll + tll).mean()

        nll_per_event = -(sll + tll)  # (B,) — always the plain NLL for logging

        return {
            "nll":            mean_nll,
            "nll_per_event":  nll_per_event,
            "total_events":   torch.tensor(float(B), device=device),
        }

    # ------------------------------------------------------------------
    # Intensity evaluation (for visualization)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_intensity(
        self,
        h_times: Tensor,
        h_locs: Tensor,
        t_query: Tensor,
        s_grid: Tensor,
    ) -> Tensor:
        """
        Evaluate joint intensity λ(t_query, s) on a spatial grid.

        λ(t, s) = λ_temporal(t) * f*(s|t)
               = t_intensity(t) * exp(log_s_intensity(t, s))

        Args:
            h_times:  (B, seq_len) — history cumulative times (model space)
            h_locs:   (B, seq_len, d) — history locations (model space)
            t_query:  (B,) — query time in model space
            s_grid:   (G, d) — spatial query grid

        Returns:
            lam: (B, G) — joint intensity at each grid point
        """
        # Accept either batched (B, N, ...) or unbatched (N, ...) inputs.
        if h_times.dim() == 1:
            h_times = h_times.unsqueeze(0)
            h_locs  = h_locs.unsqueeze(0)

        B = h_times.shape[0]
        G = s_grid.shape[0]
        device = h_times.device

        # The model is always trained with exactly seq_len history events.
        # When the experiment passes more events (all events up to t_query),
        # keep only the most recent seq_len to match the decoder output size
        # M = seq_len + num_points.
        if h_times.shape[1] > self.seq_len:
            h_times = h_times[:, -self.seq_len:]
            h_locs  = h_locs[:, -self.seq_len:]

        h_lengths = torch.full((B,), self.seq_len, device=device, dtype=torch.long)
        z, _, _, _ = self._encode(h_times, h_locs, h_lengths)
        w_i, b_i, inv_var = self._decode(z)

        t_n = h_times[:, -1]  # (B,)
        tn_ti_hist = t_n.unsqueeze(-1) - h_times                 # (B, seq_len)
        tn_ti_bg   = torch.zeros(B, self.num_points, device=device)
        tn_ti = torch.cat([tn_ti_hist, tn_ti_bg], dim=-1)  # (B, M)

        delta_t_q = (t_query - t_n).unsqueeze(-1)  # (B, 1)
        t_ti = tn_ti + delta_t_q                    # (B, M)

        # Temporal intensity: (B,)
        lam_t = self._t_intensity(w_i, b_i, t_ti)  # (B,)

        # Spatial for each grid point
        bg_exp = self.background.unsqueeze(0).expand(B, -1, -1)  # (B, P, d)
        s_x    = torch.cat([h_locs, bg_exp], dim=1)              # (B, M, d)

        # Evaluate at each grid point: expand s_grid over batch
        s_q = s_grid.unsqueeze(0).expand(B, -1, -1)  # (B, G, d)

        # s_diff for each grid point vs each kernel: (B, G, M, d)
        s_diff_grid = s_q.unsqueeze(2) - s_x.unsqueeze(1)  # (B, G, M, d)

        # Expand kernel params for G points
        w_i_g   = w_i.unsqueeze(1).expand(-1, G, -1)      # (B, G, M)
        b_i_g   = b_i.unsqueeze(1).expand(-1, G, -1)      # (B, G, M)
        t_ti_g  = t_ti.unsqueeze(1).expand(-1, G, -1)     # (B, G, M)
        iv_g    = inv_var.unsqueeze(1).expand(-1, G, -1, -1)  # (B, G, M, d)

        # Flatten B*G for the static function call
        BG = B * G
        log_fs_flat = self._log_s_intensity(
            w_i_g.reshape(BG, -1),
            b_i_g.reshape(BG, -1),
            t_ti_g.reshape(BG, -1),
            s_diff_grid.reshape(BG, self.seq_len + self.num_points, self.spatial_dim),
            iv_g.reshape(BG, self.seq_len + self.num_points, self.spatial_dim),
        )  # (B*G,)
        log_fs = log_fs_flat.reshape(B, G)  # (B, G)

        lam = lam_t.unsqueeze(-1) * log_fs.exp()  # (B, G)
        return lam
