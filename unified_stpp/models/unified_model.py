"""
Unified Neural STPP Model.

Composes Encoder, Dynamics, Updater, Decoder with systematic covariate injection.

M_X = (E_θ, D_θ, U_θ, G_θ; X^field, X^event, L_θ)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any
from .base import Encoder, Dynamics, Updater, Decoder
from .dynamics.identity import IdentityDynamics


class UnifiedSTPP(nn.Module):
    """
    The unified Neural STPP framework.

    Composes four modular components with covariate injection at each point.

    Forward pass for a sequence of events:
    1. Encode full history → z_0, all_states
    2. For each event n:
       a. Dynamics: z(t) = D(z_n, t - t_n, X_field)
       b. Decoder: log f*(t_{n+1}, s_{n+1} | z(t_{n+1}))
       c. Update: z_{n+1} = U(z(t⁻), t_{n+1}, s_{n+1}, X)
    3. Sum log-likelihoods

    In training mode (full sequence known), we can batch efficiently:
    - Encode entire sequence at once (causal masking)
    - Compute all decoder losses in parallel (for Identity dynamics)
    """

    def __init__(
        self,
        encoder: Encoder,
        dynamics: Dynamics,
        updater: Updater,
        decoder: Decoder,
        lifting_map: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.updater = updater
        self.decoder = decoder
        self.lifting_map = lifting_map

    def forward(
        self,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        x_event: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
        x_field_fn=None,
    ) -> Dict[str, Tensor]:
        """
        Compute NLL for a batch of sequences.

        Args:
            times: (B, N) — event times (padded, sorted)
            locations: (B, N, d) — event locations
            lengths: (B,) — actual sequence lengths
            x_event: (B, N, p) optional — event-level covariates
            x_field_at_events: (B, N, r) optional — field covariates pre-evaluated at events
            x_field_fn: callable (t, s) → (B, r) optional — field covariate function

        Returns:
            dict with 'nll' (scalar), 'nll_per_event' (B,), and diagnostics
        """
        B, N = times.shape
        device = times.device

        # ====================================================================
        # Step 1: Encode
        # ====================================================================
        events = torch.cat([times.unsqueeze(-1), locations], dim=-1)
        z_final, all_states = self.encoder(events, lengths, x_event=x_event)

        # ====================================================================
        # Step 2: Dispatch to batched or sequential forward
        # ====================================================================
        # For Identity dynamics, all events can be processed in a single batched
        # decoder call (no ODE to solve — state is unchanged by dynamics).
        # For ODE dynamics, we must process events sequentially because each
        # event has a different dt and initial state.
        if isinstance(self.dynamics, IdentityDynamics):
            return self._forward_batched(
                times, locations, lengths, all_states, x_field_at_events, device
            )
        else:
            return self._forward_sequential(
                times, locations, lengths, all_states, x_field_at_events, device
            )

    def _forward_batched(
        self,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        all_states: Tensor,
        x_field_at_events: Optional[Tensor],
        device,
    ) -> Dict[str, Tensor]:
        """
        Batched forward for Identity dynamics.

        Since z(t) = z_n for all t, we can gather all conditioning states and
        target events at once, flatten to (B*L, ...), and call the decoder in
        one pass — eliminating the Python loop over events.
        """
        B = times.shape[0]
        max_len = int(lengths.max().item())

        if max_len < 2:
            return {
                "nll": torch.tensor(0.0, device=device),
                "nll_per_event": torch.zeros(B, device=device),
                "total_events": torch.tensor(0.0, device=device),
            }

        L = max_len - 1  # number of prediction positions per sequence

        # Conditioning states: all_states[:, n, :] predicts event n+1
        z_cond = all_states[:, :L, :]          # (B, L, h)

        # Target events
        t_target = times[:, 1:1 + L].unsqueeze(-1)    # (B, L, 1)
        s_target = locations[:, 1:1 + L, :]            # (B, L, d)
        t_prev   = times[:, :L].unsqueeze(-1)          # (B, L, 1)

        # Mask: event n+1 is valid iff n+1 < lengths[b]
        n_indices = torch.arange(L, device=device)                         # (L,)
        mask = (n_indices.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()  # (B, L)

        # Field covariates at position n (not n+1, to avoid future leakage)
        x_field_dec = None
        if x_field_at_events is not None:
            x_field_dec = x_field_at_events[:, :L, :]  # (B, L, r)

        # Flatten to (B*L, ...) for a single batched decoder call
        h = z_cond.shape[-1]
        d = s_target.shape[-1]
        z_flat      = z_cond.reshape(B * L, h)
        t_flat      = t_target.reshape(B * L, 1)
        s_flat      = s_target.reshape(B * L, d)
        t_prev_flat = t_prev.reshape(B * L, 1)
        x_field_flat = (
            x_field_dec.reshape(B * L, -1) if x_field_dec is not None else None
        )

        # Single batched decoder call
        nll_flat = self.decoder.nll(
            z_flat, t_flat, s_flat, t_prev_flat, x_field=x_field_flat
        )  # (B*L,)
        nll_all = nll_flat.reshape(B, L)  # (B, L)

        # Mask padded positions and aggregate
        nll_masked = nll_all * mask        # (B, L)
        total_nll  = nll_masked.sum(dim=1) # (B,)
        n_events   = mask.sum(dim=1)       # (B,)

        mean_nll = total_nll.sum() / n_events.sum().clamp(min=1)

        return {
            "nll": mean_nll,
            "nll_per_event": total_nll / n_events.clamp(min=1),
            "total_events": n_events.sum(),
        }

    def _forward_sequential(
        self,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        all_states: Tensor,
        x_field_at_events: Optional[Tensor],
        device,
    ) -> Dict[str, Tensor]:
        """
        Sequential forward for ODE (or any non-Identity) dynamics.

        Each event step requires an ODE solve from the previous state,
        so we process events one at a time. Supports the augmented ODE
        side channel: if dynamics stores _cached_Lambda after each solve,
        we pass it to the temporal decoder to skip quadrature.
        """
        B = times.shape[0]
        max_len = int(lengths.max().item())

        total_nll = torch.zeros(B, device=device)
        n_events  = torch.zeros(B, device=device)

        if max_len < 2:
            return {
                "nll": torch.tensor(0.0, device=device),
                "nll_per_event": total_nll,
                "total_events": torch.tensor(0.0, device=device),
            }

        for n in range(max_len - 1):
            active = (lengths > n + 1).float()  # (B,)
            if active.sum() == 0:
                break

            z_n      = all_states[:, n, :]              # (B, h)
            t_target = times[:, n + 1].unsqueeze(-1)    # (B, 1)
            s_target = locations[:, n + 1, :]           # (B, d)
            t_prev   = times[:, n].unsqueeze(-1)        # (B, 1)
            dt       = (t_target - t_prev).clamp(min=1e-6)  # (B, 1)

            # Field covariates at last-observed event (index n), not target (n+1),
            # to avoid leaking the target's location through s-dependent features.
            x_field_dyn = None
            if x_field_at_events is not None:
                x_field_dyn = x_field_at_events[:, n, :].unsqueeze(1)  # (B, 1, r)

            z_t = self.dynamics(z_n, dt, x_field_dyn)  # (B, 1, h)
            z_t = z_t.squeeze(1)                        # (B, h)

            # Side channel: augmented ODE pre-computes Λ*(t_{n+1}) and caches it
            # on the dynamics module. Pass it to the temporal decoder so it can
            # skip its own quadrature for this step.
            cached_Lambda = getattr(self.dynamics, '_cached_Lambda', None)
            if cached_Lambda is not None:
                temporal = getattr(self.decoder, 'temporal', None)
                if temporal is not None and hasattr(temporal, '_precomputed_lambda'):
                    temporal._precomputed_lambda = cached_Lambda[:, 0]  # (B,)

            x_field_dec = None
            if x_field_at_events is not None:
                x_field_dec = x_field_at_events[:, n, :]  # (B, r)

            nll_n = self.decoder.nll(
                z_t, t_target, s_target, t_prev, x_field=x_field_dec
            )
            total_nll = total_nll + nll_n * active
            n_events  = n_events  + active

        mean_nll = total_nll.sum() / n_events.sum().clamp(min=1)

        return {
            "nll": mean_nll,
            "nll_per_event": total_nll / n_events.clamp(min=1),
            "total_events": n_events.sum(),
        }

    @torch.no_grad()
    def sample(
        self,
        history_times: Tensor,
        history_locations: Tensor,
        history_lengths: Tensor,
        n_samples: int = 1,
        t_max: float = float("inf"),
        x_event: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
        x_field_fn=None,
    ):
        """
        Autoregressively sample future events given history.

        For each step:
          1. Evolve state via dynamics (identity or ODE)
          2. Sample (t, s) from decoder
          3. Update state with new event
          4. Repeat

        Works for all decoder types:
          - FactorizedDecoder: temporal.sample() then spatial.sample()
          - DiffusionDecoder: joint sample via annealed Langevin

        Args:
            history_times: (B, N) — past event times
            history_locations: (B, N, d) — past event locations
            history_lengths: (B,) — actual lengths
            n_samples: Number of future events to sample
            t_max: Maximum time horizon
            x_event: (B, N, p) optional — event covariates for history
            x_field_at_events: (B, N, r) optional — field covariates at history events
            x_field_fn: callable (t, s) → (B, r) optional — field covariate function

        Returns:
            sampled_times: (B, n_samples)
            sampled_locations: (B, n_samples, d)
        """
        B = history_times.shape[0]
        device = history_times.device
        d = history_locations.shape[-1]

        # Encode history
        events = torch.cat([history_times.unsqueeze(-1), history_locations], dim=-1)
        z, all_states = self.encoder(events, history_lengths, x_event=x_event)

        sampled_t = []
        sampled_s = []

        # Last event time per sequence
        t_prev = history_times[
            torch.arange(B, device=device), (history_lengths - 1).long()
        ].unsqueeze(-1)  # (B, 1)

        for step in range(n_samples):
            # Dynamics: for sampling, we don't know dt yet.
            # With identity dynamics: z(t) = z for any t, so no issue.
            # With ODE dynamics: we'd need to integrate to each candidate t.
            # Practical approach: use z directly (post-update state).
            # This is exact for identity dynamics and approximate for ODE dynamics.
            # (NeuralSTPP's original code also uses this approximation for sampling.)

            # Sample next event from decoder
            t_new, s_new = self.decoder.sample(z, t_prev, x_field_fn)
            # t_new: (B, 1), s_new: (B, d)

            # Ensure t_new is (B, 1)
            if t_new.dim() == 1:
                t_new = t_new.unsqueeze(-1)

            # Mask events beyond t_max
            valid = (t_new.squeeze(-1) <= t_max)

            sampled_t.append(t_new.squeeze(-1))
            sampled_s.append(s_new)

            # Update state for next step
            z = self.updater(z, t_new, s_new)
            t_prev = t_new

        sampled_times = torch.stack(sampled_t, dim=1)      # (B, n_samples)
        sampled_locs = torch.stack(sampled_s, dim=1)        # (B, n_samples, d)

        # Mask out events beyond t_max
        mask = sampled_times <= t_max  # (B, n_samples)

        return sampled_times, sampled_locs, mask
