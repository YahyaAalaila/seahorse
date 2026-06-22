"""
Gaussian mixture spatial model for factorized STPP baselines.

Models the spatial log-density of each event conditioned on past event locations,
using a time-decay weighted mixture of Gaussian kernels centred at past events.

    p(s_i | s_{<i}, t_{<i}) = Σ_{j<i} w_{ij} * N(s_i | s_j, σ_kernel²)
    w_{ij} ∝ exp(-τ * (t_i - t_j)) * mask_j

For the first event in a sequence (i=0), an isotropic Gaussian prior N(0, σ_prior²) is used.

Interface:

    logprob(event_times, locations, input_mask) -> (B, T)

Returns per-event spatial log-probabilities. Padding positions return 0.0.
Fully vectorized (no Python loop over events).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GaussianMixtureSpatialModel(nn.Module):
    """
    Time-decay Gaussian mixture spatial density.

    Parameters (all log-parameterized, initialized from constructor args):
        log_sigma_prior  : controls isotropic prior std for the first event
        log_sigma_kernel : controls Gaussian kernel bandwidth
        log_tau          : controls exponential time-decay of mixture weights

    Args:
        sigma_prior  : initial value of softplus(log_sigma_prior)
        sigma_kernel : initial value of softplus(log_sigma_kernel)
        tau          : initial value of softplus(log_tau)
    """

    def __init__(
        self,
        sigma_prior: float = 1.0,
        sigma_kernel: float = 0.5,
        tau: float = 1.0,
    ):
        super().__init__()
        # Inverse-softplus initialization so softplus(param) ≈ given value
        def _inv_softplus(x: float) -> float:
            return math.log(math.expm1(x)) if x > 0.5 else x

        self.log_sigma_prior  = nn.Parameter(torch.tensor(_inv_softplus(sigma_prior)))
        self.log_sigma_kernel = nn.Parameter(torch.tensor(_inv_softplus(sigma_kernel)))
        self.log_tau          = nn.Parameter(torch.tensor(_inv_softplus(tau)))

    def logprob(
        self,
        event_times: Tensor,
        locations: Tensor,
        input_mask: Tensor,
    ) -> Tensor:
        """
        Compute per-event spatial log-probabilities.

        Args:
            event_times : (B, T)    — event times
            locations   : (B, T, D) — spatial event locations
            input_mask  : (B, T)    float — 1 for valid events, 0 for padding
        Returns:
            (B, T) — per-event log-prob; 0.0 at padding positions
        """
        sigma_prior  = F.softplus(self.log_sigma_prior)   # scalar
        sigma_kernel = F.softplus(self.log_sigma_kernel)  # scalar
        tau          = F.softplus(self.log_tau)           # scalar

        _, T, D = locations.shape
        device = locations.device

        # ---------------------------------------------------------------
        # Mixture weights in log-space — avoids log(0) in backward.
        # On MPS, log(x.clamp(min=ε)) when x=0 computes 1/x=inf then
        # multiplies by clamp_grad=0, yielding inf*0=NaN.
        # masked_fill with -1e9 blocks non-causal positions cleanly.
        # ---------------------------------------------------------------
        # dt[b, i, j] = t_i - t_j  (non-negative for causal pairs)
        dt = (event_times.unsqueeze(-1) - event_times.unsqueeze(-2)).clamp(min=0.0)  # (B,T,T)

        # Causal mask: 1 where j < i (strictly past events)
        causal = torch.tril(
            torch.ones(T, T, device=device, dtype=input_mask.dtype), diagonal=-1
        )  # (T, T)

        # causal_valid[b, i, j] = 1 iff j < i AND event j is valid
        causal_valid = input_mask.unsqueeze(-2) * causal.unsqueeze(0)  # (B, T, T)

        # Log unnormalized weights: -τ*(t_i-t_j) for valid causal pairs; -1e9 elsewhere
        log_w_unnorm = -tau * dt                                           # (B, T, T)
        log_w_unnorm = log_w_unnorm.masked_fill(causal_valid == 0, -1e9)  # (B, T, T)

        # Log-normalize over j
        log_norm = torch.logsumexp(log_w_unnorm, dim=-1, keepdim=True)    # (B, T, 1)
        log_w    = log_w_unnorm - log_norm                                 # (B, T, T)

        # ---------------------------------------------------------------
        # Gaussian log-density: log N(s_i | s_j, σ_kernel²)
        # diff[b, i, j, d] = locations[b, i, d] - locations[b, j, d]
        # ---------------------------------------------------------------
        diff = locations.unsqueeze(2) - locations.unsqueeze(1)  # (B, T, T, D)
        log_gauss = (
            -0.5 * (diff ** 2).sum(dim=-1) / (sigma_kernel ** 2)
            - (D / 2.0) * math.log(2.0 * math.pi)
            - D * torch.log(sigma_kernel)
        )  # (B, T, T)

        # ---------------------------------------------------------------
        # Mixture log-prob for events i >= 1:
        # log Σ_j w[b,i,j] * N(s_i | s_j, σ_kernel)
        # ---------------------------------------------------------------
        log_mix = torch.logsumexp(log_w + log_gauss, dim=-1)  # (B, T)

        # ---------------------------------------------------------------
        # Isotropic Gaussian prior for event i=0: N(s_0 | 0, σ_prior²)
        # ---------------------------------------------------------------
        log_prior = (
            -0.5 * (locations[:, 0, :] ** 2).sum(dim=-1) / (sigma_prior ** 2)
            - (D / 2.0) * math.log(2.0 * math.pi)
            - D * torch.log(sigma_prior)
        )  # (B,)

        # Override index 0 with the prior (log_mix[:, 0] is numerically garbage
        # since there are no causal past events for i=0)
        result = log_mix.clone()
        result[:, 0] = log_prior

        # Zero out padding
        return result * input_mask

    def log_spatial_density_at(
        self,
        t_query: Tensor,
        s_query: Tensor,
        history_times: Tensor,
        history_locs: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        """Log spatial density of a query point given history.

        Evaluates log p(s_query | t_query, history) as a time-decay weighted
        Gaussian mixture centred at history event locations:

            p(s | t, H) = Σ_j w_j * N(s | s_j, σ_kernel²)
            w_j ∝ exp(-τ * (t - t_j)) * mask_j

        The prior N(0, σ_prior²) is used when history_mask.sum() == 0 (no past events).

        Args:
            t_query      : (M,) query times (shifted, ≥ 0)
            s_query      : (M, D) query spatial locations
            history_times: (M, T) history times (shifted), expanded to M
            history_locs : (M, T, D) history locations, expanded to M
            history_mask : (M, T) float mask (1 = valid, 0 = padding)
        Returns:
            (M,) log p(s_query | t_query, history)
        """
        sigma_prior  = F.softplus(self.log_sigma_prior)
        sigma_kernel = F.softplus(self.log_sigma_kernel)
        tau          = F.softplus(self.log_tau)

        M, D = s_query.shape
        device = s_query.device

        # dt[m, j] = t_query[m] - history_times[m, j] (non-negative)
        dt = (t_query.unsqueeze(1) - history_times).clamp(min=0.0)  # (M, T)

        # Log unnormalized weights; mask out invalid history positions
        log_w_unnorm = -tau * dt                                                  # (M, T)
        log_w_unnorm = log_w_unnorm.masked_fill(history_mask == 0, -1e9)         # (M, T)

        log_norm = torch.logsumexp(log_w_unnorm, dim=-1, keepdim=True)           # (M, 1)
        log_w    = log_w_unnorm - log_norm                                        # (M, T)

        # Gaussian log-density: log N(s_query | s_j, σ_kernel²)
        # diff[m, j, d] = s_query[m, d] - history_locs[m, j, d]
        diff = s_query.unsqueeze(1) - history_locs                               # (M, T, D)
        log_gauss = (
            -0.5 * (diff ** 2).sum(dim=-1) / (sigma_kernel ** 2)
            - (D / 2.0) * math.log(2.0 * math.pi)
            - D * torch.log(sigma_kernel)
        )  # (M, T)

        log_mix = torch.logsumexp(log_w + log_gauss, dim=-1)  # (M,)

        # Fall back to prior where there are no valid history events
        log_prior = (
            -0.5 * (s_query ** 2).sum(dim=-1) / (sigma_prior ** 2)
            - (D / 2.0) * math.log(2.0 * math.pi)
            - D * torch.log(sigma_prior)
        )  # (M,)

        has_history = (history_mask.sum(dim=-1) > 0).float()  # (M,)
        return has_history * log_mix + (1.0 - has_history) * log_prior
