"""
General-purpose sampling utilities for spatiotemporal point processes.

Provides:
1. Thinning algorithm — works with ANY model that can evaluate λ*(t, s).
   This is the fallback for intensity-based decoders and a validation
   reference for direct samplers.
   
2. Intensity evaluation wrapper — extracts λ*(t, s) from either
   density-based or intensity-based decoders for visualization/analysis.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Callable
import math


def get_event_model_capabilities(model):
    """Return EventModel capabilities when available, else None."""
    event_model = getattr(model, "event_model", None)
    if event_model is None:
        return None
    return getattr(event_model, "capabilities", None)


def supports_native_sampling(model) -> bool:
    """Whether model exposes native event-model sampling."""
    caps = get_event_model_capabilities(model)
    return bool(caps is not None and getattr(caps, "has_native_sampler", False))


def supports_intensity_query(model) -> bool:
    """Whether model exposes an event-model intensity interface."""
    caps = get_event_model_capabilities(model)
    return bool(caps is not None and getattr(caps, "has_intensity", False))


def thinning_sample(
    intensity_fn: Callable,
    t_start: Tensor,
    t_max: float,
    spatial_bounds: Tuple[Tensor, Tensor],
    lambda_bar: float = 10.0,
    max_events: int = 100,
    adaptive: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Ogata's thinning algorithm for spatiotemporal point processes.
    
    Works with ANY model that provides an intensity function λ*(t, s).
    
    Algorithm:
      1. Propose candidate time: Δt ~ Exp(λ̄)
      2. Propose candidate location: s ~ Uniform(S)
      3. Accept with probability λ*(t, s) / λ̄
      4. If adaptive: update λ̄ based on observed intensity
    
    Args:
        intensity_fn: callable (t, s) → λ*(t, s)
            t: (B, 1), s: (B, d) → returns (B,)
            Must account for history internally (e.g., via closure over z).
        t_start: (B, 1) — start time
        t_max: float — time horizon
        spatial_bounds: (s_min, s_max) — each (d,) tensor
            Uniform proposal region for spatial coordinates.
        lambda_bar: float — upper bound on intensity (initial)
        max_events: int — maximum events to sample
        adaptive: bool — whether to adapt λ̄
        
    Returns:
        times: (B, max_events) — sampled times (padded with t_max+1)
        locations: (B, max_events, d) — sampled locations (padded with 0)
        counts: (B,) — actual number of events per sequence
    """
    B = t_start.shape[0]
    device = t_start.device
    d = spatial_bounds[0].shape[0]
    s_min, s_max = spatial_bounds

    times = torch.full((B, max_events), t_max + 1, device=device)
    locations = torch.zeros(B, max_events, d, device=device)
    counts = torch.zeros(B, dtype=torch.long, device=device)
    
    # Spatial volume for proposal
    vol = (s_max - s_min).prod().item()

    t_current = t_start.squeeze(-1).clone()  # (B,)
    lam_bar = torch.full((B,), lambda_bar, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)

    for event_idx in range(max_events):
        if not active.any():
            break

        # Propose inter-arrival time: Δt ~ Exp(λ̄ · vol)
        rate = (lam_bar * vol).clamp(min=1e-6)
        dt = torch.distributions.Exponential(rate).sample()  # (B,)
        t_proposal = t_current + dt

        # Check time bound
        still_active = active & (t_proposal < t_max)
        if not still_active.any():
            active = still_active
            break

        # Propose location: s ~ Uniform(s_min, s_max)
        u_s = torch.rand(B, d, device=device)
        s_proposal = s_min.unsqueeze(0) + u_s * (s_max - s_min).unsqueeze(0)

        # Evaluate intensity
        with torch.no_grad():
            lam = intensity_fn(
                t_proposal.unsqueeze(-1),
                s_proposal,
            )  # (B,)

        # Accept/reject
        u = torch.rand(B, device=device)
        accept = still_active & (u < lam / lam_bar.clamp(min=1e-8))

        # Store accepted events
        for b in range(B):
            if accept[b]:
                idx = counts[b].item()
                if idx < max_events:
                    times[b, idx] = t_proposal[b]
                    locations[b, idx] = s_proposal[b]
                    counts[b] += 1

        # Update current time (move forward regardless of accept/reject)
        t_current = torch.where(still_active, t_proposal, t_current)

        # Adaptive λ̄
        if adaptive:
            lam_bar = torch.where(
                still_active,
                torch.maximum(lam_bar, lam * 1.5),
                lam_bar,
            )

        active = still_active & (counts < max_events)

    return times, locations, counts


class IntensityEvaluator:
    """
    Wraps a trained UnifiedSTPP model to expose λ*(t, s | H) for
    arbitrary (t, s) queries. Useful for:
      - Visualization (intensity heatmaps)
      - Thinning-based sampling
      - Goodness-of-fit diagnostics
    
    For density-based decoders:
      λ*(t, s) = f*(t, s) / S*(t)
      where S*(t) = 1 - F*(t) is the temporal survival function.
      
      In practice, for a single query time with the factorized decoder:
      λ*(t, s) = λ*(t) · f*(s | t)
      where λ*(t) = f*(t) / S*(t) is the temporal intensity.
      
    For intensity-based decoders:
      λ*(t, s) is directly available.
    """

    def __init__(
        self,
        model,
        z: Tensor,
        t_prev: Tensor,
        history_locs_norm: Optional[Tensor] = None,
        history_times_norm: Optional[Tensor] = None,
    ):
        """
        Args:
            model: trained UnifiedSTPP
            z: (B, h) — latent state after encoding history
            t_prev: (B, 1) — time of last event
            history_locs_norm: (N, d) — normalized event locations (history)
            history_times_norm: (N,) — normalized event times (history),
                required for DeepSTPPDecoder which needs both.
        """
        self.model = model
        self.z = z
        self.t_prev = t_prev
        self.history_locs_norm = history_locs_norm
        self.history_times_norm = history_times_norm

    @torch.no_grad()
    def intensity(self, t: Tensor, s: Tensor, x_field: Optional[Tensor] = None) -> Tensor:
        """
        Evaluate λ*(t, s | H).
        
        Args:
            t: (B, 1) — query time
            s: (B, d) — query location
            x_field: (B, r) optional
        Returns:
            lambda_star: (B,)
        """
        decoder = self.model.decoder

        # Evolve state to query time.
        # Dynamics expects x_field shaped (B, 1, r); unsqueeze the middle dim.
        dt = (t - self.t_prev).clamp(min=1e-6)
        x_field_dyn = x_field.unsqueeze(1) if x_field is not None else None
        z_t = self.model.dynamics(self.z, dt, x_field=x_field_dyn)
        z_t = z_t.squeeze(1)  # (B, h)

        # Check if decoder is density-based (has temporal + spatial sub-decoders)
        if hasattr(decoder, 'temporal') and hasattr(decoder, 'spatial'):
            return self._factorized_intensity(z_t, t, s, x_field)
        elif getattr(decoder, 'requires_time_history', False):
            # DeepSTPPDecoder: build x_field from history times + locations,
            # then call spatial_intensity() which returns λ_t(t)·f(s|t) directly
            # (without the compensator that log_prob includes).
            x_field_dec = self._build_time_history_x_field(decoder, s.shape[0])
            if hasattr(decoder, 'spatial_intensity'):
                return decoder.spatial_intensity(z_t, t, s, self.t_prev, x_field=x_field_dec)
            else:
                log_val = decoder.log_prob(z_t, t, s, self.t_prev, x_field_dec)
                return torch.exp(log_val)
        else:
            # Intensity or joint density decoder (e.g. diffusion)
            log_val = decoder.log_prob(z_t, t, s, self.t_prev, x_field)
            return torch.exp(log_val)

    def _build_time_history_x_field(self, decoder, B: int) -> Optional[Tensor]:
        """Build x_field = [abs_times, locs] for DeepSTPPDecoder grid evaluation.

        Returns (B, seq_len + seq_len*d) with all rows identical (same history
        for every grid point).  Returns None if no history is available.
        """
        seq_len = decoder.history_window_size
        d       = decoder.spatial_dim
        device  = self.z.device

        has_times = self.history_times_norm is not None
        has_locs  = self.history_locs_norm  is not None
        if not (has_times and has_locs):
            return None

        times = self.history_times_norm  # (N,)
        locs  = self.history_locs_norm   # (N, d)
        N     = locs.shape[0]

        if N >= seq_len:
            t_win = times[-seq_len:]
            s_win = locs[-seq_len:]
        else:
            t_pad = times[:1].expand(seq_len - N)
            t_win = torch.cat([t_pad, times], dim=0)
            s_pad = locs[:1].expand(seq_len - N, d)
            s_win = torch.cat([s_pad, locs], dim=0)

        # Pack: [t_0,...,t_{seq-1}, s_0x,s_0y,...] — same layout as nll() expects
        return torch.cat([
            t_win.unsqueeze(0).expand(B, -1),     # (B, seq_len)
            s_win.reshape(1, -1).expand(B, -1),   # (B, seq_len*d)
        ], dim=-1)

    def _factorized_intensity(
        self, z: Tensor, t: Tensor, s: Tensor, x_field: Optional[Tensor]
    ) -> Tensor:
        """
        For factorized decoder: λ*(t, s) = λ*(t) · f*(s|t).
        
        λ*(t) = f*(t) / S*(t), computed from the temporal decoder.
        
        For LogNormalMixture: S*(t) = Σ_k π_k (1 - Φ((log τ - μ_k)/σ_k))
        For CumulativeHazard: S*(t) = exp(-Λ*(t))
        """
        decoder = self.model.decoder

        # Temporal intensity
        temporal = decoder.temporal
        if hasattr(temporal, '_cumulative_hazard'):
            # CumulativeHazardTemporal: λ*(t) is directly available
            dt = (t - self.t_prev).squeeze(-1)
            lam_t = temporal._intensity(z, dt.unsqueeze(-1), x_field)  # (B,)
        else:
            # LogNormalMixture: λ*(t) = f*(t) / S*(t)
            # Approximate S*(t) via 1 - CDF
            log_ft = temporal.log_prob(z, t, self.t_prev, x_field)
            ft = torch.exp(log_ft)  # (B,)
            # Compute S*(t) by numerical integration or approximation
            # S*(t) ≈ 1 - ∫_0^τ f*(u) du
            # For efficiency, use the MC approximation:
            # S*(t) = 1 - F*(t), estimated via the CDF of the mixture
            St = self._lognormal_mixture_survival(temporal, z, t, self.t_prev, x_field)
            lam_t = ft / St.clamp(min=1e-8)  # (B,)

        # Spatial density
        # For data-centered decoders, override x_field with the history window.
        spatial_dec = decoder.spatial
        if getattr(spatial_dec, "requires_history", False) and self.history_locs_norm is not None:
            seq_len = spatial_dec.history_window_size
            hist = self.history_locs_norm  # (N, d) normalized
            N = hist.shape[0]
            if N >= seq_len:
                window = hist[-seq_len:]               # (seq_len, d)
            else:
                pad = hist[:1].expand(seq_len - N, -1) # (seq_len-N, d)
                window = torch.cat([pad, hist], dim=0) # (seq_len, d)
            # Expand to (B, seq_len*d)
            B = z.shape[0]
            x_field_spatial = window.reshape(1, -1).expand(B, -1)
        else:
            x_field_spatial = x_field

        log_fs = spatial_dec.log_prob(z, t, s, self.t_prev, x_field_spatial)
        fs = torch.exp(log_fs)  # (B,)

        return lam_t * fs

    @staticmethod
    def _lognormal_mixture_survival(temporal, z, t, t_prev, x_field) -> Tensor:
        """S*(τ) = 1 - Σ_k π_k Φ((log τ - μ_k) / σ_k) for log-normal mixture."""
        tau = (t - t_prev).squeeze(-1).clamp(min=1e-6)  # (B,)
        logits, mu, sigma = temporal._get_params(z, x_field)
        log_tau = torch.log(tau).unsqueeze(-1)  # (B, 1)
        # Φ((log τ - μ) / σ) for each component
        normal_cdf = 0.5 * (1 + torch.erf((log_tau - mu) / (sigma * math.sqrt(2))))
        pi = torch.softmax(logits, dim=-1)
        Ft = (pi * normal_cdf).sum(dim=-1)  # (B,)
        return (1 - Ft).clamp(min=1e-8)

    @torch.no_grad()
    def intensity_grid(
        self,
        t: float,
        s_min: Tensor,
        s_max: Tensor,
        n_grid: int = 50,
        x_field_fn=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate intensity on a spatial grid at fixed time t.
        Returns grid coordinates and intensity values for visualization.
        
        Args:
            t: float — query time
            s_min: (d,) — spatial lower bounds
            s_max: (d,) — spatial upper bounds
            n_grid: int — grid resolution per dimension
            x_field_fn: optional callable (t, s) → (1, r)
        Returns:
            grid_x: (n_grid,)
            grid_y: (n_grid,) (only for d=2)
            intensity: (n_grid, n_grid) for d=2, or (n_grid,) for d=1
        """
        device = self.z.device
        B = self.z.shape[0]
        assert B == 1, "intensity_grid only supports B=1"

        d = s_min.shape[0]
        assert d in (1, 2), "Grid visualization supports d=1 or d=2"

        if d == 2:
            x = torch.linspace(s_min[0].item(), s_max[0].item(), n_grid, device=device)
            y = torch.linspace(s_min[1].item(), s_max[1].item(), n_grid, device=device)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            s_flat = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (n², 2)
            n_pts = s_flat.shape[0]

            t_tensor = torch.full((n_pts, 1), t, device=device)
            # Expand z and t_prev for batch evaluation
            z_exp = self.z.expand(n_pts, -1)
            t_prev_exp = self.t_prev.expand(n_pts, -1)

            # Temporarily override for batch eval
            orig_z, orig_tp = self.z, self.t_prev
            self.z, self.t_prev = z_exp, t_prev_exp

            x_field = None
            if x_field_fn is not None:
                x_field = x_field_fn(t_tensor, s_flat)

            lam = self.intensity(t_tensor, s_flat, x_field)  # (n²,)
            self.z, self.t_prev = orig_z, orig_tp

            return x, y, lam.reshape(n_grid, n_grid)
        else:
            x = torch.linspace(s_min[0].item(), s_max[0].item(), n_grid, device=device)
            s_flat = x.unsqueeze(-1)  # (n, 1)
            n_pts = n_grid
            t_tensor = torch.full((n_pts, 1), t, device=device)

            z_exp = self.z.expand(n_pts, -1)
            t_prev_exp = self.t_prev.expand(n_pts, -1)
            orig_z, orig_tp = self.z, self.t_prev
            self.z, self.t_prev = z_exp, t_prev_exp

            lam = self.intensity(t_tensor, s_flat)
            self.z, self.t_prev = orig_z, orig_tp

            return x, None, lam
