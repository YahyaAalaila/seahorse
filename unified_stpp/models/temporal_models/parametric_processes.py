"""
Parametric temporal point process models.

Three closed-form temporal processes for use as factorized baselines:
  - HomogeneousPoissonProcess: constant rate λ
  - HawkesProcess: sum-of-exponentials self-exciting process
  - SelfCorrectingProcess: exp(μt − βN(t)) inhibitory process

All implement the same interface:

    logprob(event_times, locations, input_mask, t0, t1) -> (B,)

where:
    event_times : (B, T)  — event times, padded
    locations   : (B, T, D) — spatial locations (unused by temporal models)
    input_mask  : (B, T)  float — 1 for valid events, 0 for padding
    t0          : (B,)  — start of observation window
    t1          : (B,)  — end of observation window
    return      : (B,)  — per-sequence total temporal log-prob
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HomogeneousPoissonProcess(nn.Module):
    """
    Homogeneous Poisson process with learnable constant rate λ.

    Log-likelihood:
        log p = N * log(λ) - (t1 - t0) * λ
    where N = number of valid events in the sequence.
    """

    def __init__(self):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.randn(1) * 0.2 - 2.0)

    def intensity_at(
        self,
        t_query: Tensor,
        history_times: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        """Evaluate λ(t_query | H) = λ (constant).

        Args:
            t_query      : (M,) query times (unused; rate is constant)
            history_times: (M, T) — ignored
            history_mask : (M, T) — ignored
        Returns:
            (M,) constant intensity values
        """
        del history_times, history_mask
        return F.softplus(self.log_lambda).expand(t_query.shape[0])

    def logprob(
        self,
        event_times: Tensor,
        locations: Tensor,
        input_mask: Tensor,
        t0: Tensor,
        t1: Tensor,
    ) -> Tensor:
        """
        Args:
            event_times : (B, T) — event times (unused; count from mask)
            locations   : (B, T, D) — ignored
            input_mask  : (B, T) float
            t0          : (B,) — observation window start
            t1          : (B,) — observation window end
        Returns:
            (B,) per-sequence log-prob
        """
        lamb = F.softplus(self.log_lambda).squeeze()  # scalar
        n_events = input_mask.sum(dim=-1)             # (B,)
        window = (t1 - t0).clamp(min=1e-7)
        return n_events * torch.log(lamb + 1e-20) - window * lamb


class HawkesProcess(nn.Module):
    """
    Sum-of-exponentials Hawkes self-exciting process.

    Intensity: λ(t) = μ + α * Σ_{j: t_j < t} exp(-β*(t - t_j))

    Log-likelihood:
        log p = Σ_i log λ(t_i) - Λ(t0, t1)

    Compensator:
        Λ(t0, t1) = μ*(t1-t0) + (α/β) * Σ_i [1 - exp(-β*(t1 - t_i))] * mask_i

    Vectorized via lower-triangular time differences (no Python loop).
    """

    def __init__(self):
        super().__init__()
        self.log_mu    = nn.Parameter(torch.zeros(1))
        self.log_alpha = nn.Parameter(torch.full((1,), -1.0))
        self.log_beta  = nn.Parameter(torch.zeros(1))

    def intensity_at(
        self,
        t_query: Tensor,
        history_times: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        """Evaluate λ(t_query | H) = μ + α Σ_{j: t_j < t} exp(-β*(t-t_j)).

        Args:
            t_query      : (M,) query times (shifted, ≥ 0)
            history_times: (M, T) history times (shifted), already expanded to M
            history_mask : (M, T) float mask
        Returns:
            (M,) intensity values
        """
        mu    = F.softplus(self.log_mu).squeeze()
        alpha = F.softplus(self.log_alpha).squeeze()
        beta  = F.softplus(self.log_beta).squeeze()

        # dt[m, j] = t_query[m] - history_times[m, j]
        dt = (t_query.unsqueeze(1) - history_times).clamp(min=0.0)  # (M, T)
        hawkes_sum = (alpha * torch.exp(-beta * dt) * history_mask).sum(dim=-1)  # (M,)
        return mu + hawkes_sum

    def logprob(
        self,
        event_times: Tensor,
        locations: Tensor,
        input_mask: Tensor,
        t0: Tensor,
        t1: Tensor,
    ) -> Tensor:
        """
        Args:
            event_times : (B, T)
            locations   : (B, T, D) — ignored
            input_mask  : (B, T) float
            t0          : (B,)
            t1          : (B,)
        Returns:
            (B,) per-sequence log-prob
        """
        mu    = F.softplus(self.log_mu).squeeze()     # scalar
        alpha = F.softplus(self.log_alpha).squeeze()  # scalar
        beta  = F.softplus(self.log_beta).squeeze()   # scalar

        T = event_times.shape[1]
        device = event_times.device

        # dt[b, i, j] = t_i - t_j  (B, T, T)
        dt = event_times.unsqueeze(-1) - event_times.unsqueeze(-2)  # (B, T, T)

        # Hawkes excitation: Σ_{j<i} α * exp(-β*(t_i - t_j)) * mask_j
        # Use strict lower triangle mask on the *result* instead of filling dt with inf.
        # Filling dt with inf causes exp(-β*inf)=0 forward but (-inf)*0=NaN on MPS backward.
        # With clamp(min=0): upper-triangle dt≤0 → clamped to 0 → exp(0)=1 → masked to 0.
        lower = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=-1)
        hawkes_contrib = (
            alpha * torch.exp(-beta * dt.clamp(min=0.0))  # (B, T, T)
            * lower.unsqueeze(0)                           # causal mask
            * input_mask.unsqueeze(-2)                     # valid-j mask
        )
        hawkes_sum = hawkes_contrib.sum(dim=-1)  # (B, T)

        log_intensity = torch.log(mu + hawkes_sum + 1e-20)  # (B, T)

        # Log-likelihood: sum over valid events
        ll_events = (log_intensity * input_mask).sum(dim=-1)  # (B,)

        # Compensator: Λ = μ*(t1-t0) + (α/β)*Σ_i (1 - exp(-β*(t1-t_i))) * mask_i
        t1_expanded = t1.unsqueeze(-1)  # (B, 1)
        comp_hawkes = (alpha / (beta + 1e-20)) * (
            (1.0 - torch.exp(-beta * (t1_expanded - event_times).clamp(min=0.0))) * input_mask
        ).sum(dim=-1)  # (B,)
        comp_base = mu * (t1 - t0).clamp(min=1e-7)  # (B,)
        compensator = comp_base + comp_hawkes  # (B,)

        return ll_events - compensator


class SelfCorrectingProcess(nn.Module):
    """
    Self-correcting (inhibitory) process.

    Intensity: λ(t) = exp(μ*t − β*N(t))
    where N(t) = number of events that have occurred strictly before t.

    Log-likelihood:
        log p = Σ_i log λ(t_i) - Λ(t0, t1)

    Compensator is computed as a sum over inter-event intervals:
        ∫_{t_i}^{t_{i+1}} exp(μ*t - β*N_i) dt
        = exp(-β*N_i) / μ * [exp(μ*t_{i+1}) - exp(μ*t_i)]

    where N_i = number of valid events up to and including index i.
    """

    def __init__(self):
        super().__init__()
        self.log_mu   = nn.Parameter(torch.zeros(1))
        self.log_beta = nn.Parameter(torch.zeros(1))

    def intensity_at(
        self,
        t_query: Tensor,
        history_times: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        """Evaluate λ(t_query | H) = exp(μ*t_query - β*N(t_query)).

        Args:
            t_query      : (M,) query times (shifted, ≥ 0)
            history_times: (M, T) history times (shifted), expanded to M
            history_mask : (M, T) float mask
        Returns:
            (M,) intensity values
        """
        mu   = F.softplus(self.log_mu).squeeze()
        beta = F.softplus(self.log_beta).squeeze()

        # N(t_query[m]) = events strictly before t_query[m]
        n_before = (
            (history_times < t_query.unsqueeze(1)).float() * history_mask
        ).sum(dim=-1)  # (M,)
        return torch.exp(mu * t_query - beta * n_before)

    def logprob(
        self,
        event_times: Tensor,
        locations: Tensor,
        input_mask: Tensor,
        t0: Tensor,
        t1: Tensor,
    ) -> Tensor:
        """
        Args:
            event_times : (B, T)
            locations   : (B, T, D) — ignored
            input_mask  : (B, T) float
            t0          : (B,)
            t1          : (B,)
        Returns:
            (B,) per-sequence log-prob
        """
        mu   = F.softplus(self.log_mu).squeeze()    # scalar
        beta = F.softplus(self.log_beta).squeeze()  # scalar

        # N(t_i-) = number of valid events strictly before position i
        # = cumsum of mask up to (but not including) position i
        n_before = torch.cumsum(input_mask, dim=-1) - input_mask  # (B, T)

        # log λ(t_i) = μ*t_i - β*N(t_i-)
        log_intensity = mu * event_times - beta * n_before  # (B, T)
        ll_events = (log_intensity * input_mask).sum(dim=-1)  # (B,)

        # Compensator: sum over consecutive-event intervals [t_i, t_{i+1}]
        # N_i = cumulative count through index i (i.e., after event i has occurred)
        # ∫_{t_i}^{t_{i+1}} exp(μ*t - β*N_i) dt
        #   = exp(-β*N_i) / μ * exp(μ*t_i) * expm1(μ*(t_{i+1}-t_i))
        #
        # Computed in log-space to avoid exp(μ*t) overflow when μ is large:
        #   log_comp_i = μ*t_i - β*N_i - log(μ) + log(expm1(μ*dt_i) + ε)
        # Then clamp log_comp_i before exponentiating to stay within float32 range.
        N_i = torch.cumsum(input_mask[:, :-1], dim=-1)    # (B, T-1), count after event at i
        dt_intervals = (event_times[:, 1:] - event_times[:, :-1]).clamp(min=1e-7)  # (B, T-1)
        # log of each interval's compensator contribution:
        log_comp_i = (
            mu * event_times[:, :-1]                          # μ * t_i  (≥ 0 after shifting)
            - beta * N_i                                       # -β * N_i (≤ 0)
            - torch.log(mu + 1e-20)                           # -log μ
            + torch.log(torch.expm1((mu * dt_intervals)) + 1e-20)  # log(expm1(μ*Δt))
        )  # (B, T-1)
        # Clamp to float32-safe range before exp; 80 ≈ log(5.5e34), below float32 max 3.4e38
        comp_intervals = torch.exp(log_comp_i.clamp(max=80.0))  # (B, T-1)
        # Only include intervals where both endpoints are valid events
        interval_mask = input_mask[:, :-1] * input_mask[:, 1:]
        compensator = (comp_intervals * interval_mask).sum(dim=-1)  # (B,)

        return ll_events - compensator
