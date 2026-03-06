"""
Unified Neural STPP Model.

Composes Encoder, Dynamics, Updater, Decoder with systematic covariate injection.

M_X = (E_θ, D_θ, U_θ, G_θ, G^m_θ; X^field, X^event, L_θ)

G^m is optional (None = unmarked). When present, marks enter via the encoder
and updater as embedded event-level covariates; the mark NLL adds additively to
the ground process NLL.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any
from .base import Encoder, Dynamics, Updater, Decoder, MarkDecoder
from .dynamics.identity import IdentityDynamics


def _sliding_history_windows(locations: Tensor, L: int, seq_len: int) -> Tensor:
    """
    Build (B, L, seq_len * d) history windows for prediction positions 0..L-1.

    Window n contains the seq_len events immediately preceding prediction
    position n+1 (events max(0, n+1-seq_len)..n), left-padded with the
    first observed event when fewer than seq_len events are available.
    """
    B, _, d = locations.shape
    # Pad seq_len-1 copies of the first event on the left
    first = locations[:, :1, :].expand(-1, seq_len - 1, -1)          # (B, seq_len-1, d)
    padded = torch.cat([first, locations[:, :L + 1, :]], dim=1)       # (B, seq_len-1+L+1, d)
    windows = torch.stack(
        [padded[:, n : n + seq_len, :] for n in range(L)], dim=1
    )                                                                   # (B, L, seq_len, d)
    return windows.reshape(B, L, seq_len * d)


class UnifiedSTPP(nn.Module):
    """
    The unified Neural STPP framework.

    Composes four modular components with covariate injection at each point.
    Optionally models discrete marks via G^m (mark_decoder).

    Forward pass for a sequence of events:
    1. Encode full history (including mark embeddings) → z_0, all_states
    2. For each event n:
       a. Dynamics: z(t) = D(z_n, t - t_n, X_field)
       b. Decoder: log f*(t_{n+1}, s_{n+1} | z(t_{n+1}))
       c. Mark decoder (if present): log p*(k_{n+1} | z(t_{n+1}), t, s)
       d. Update: z_{n+1} = U(z(t⁻), t_{n+1}, s_{n+1}, X)
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
        mark_decoder: Optional[MarkDecoder] = None,
        mark_embedding: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.updater = updater
        self.decoder = decoder
        self.lifting_map = lifting_map
        self.mark_decoder = mark_decoder
        self.mark_embedding = mark_embedding

    def forward(
        self,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor] = None,
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
            marks: (B, N) LongTensor optional — discrete mark indices
            x_event: (B, N, p) optional — event-level covariates
            x_field_at_events: (B, N, r) optional — field covariates pre-evaluated at events
            x_field_fn: callable (t, s) → (B, r) optional — field covariate function

        Returns:
            dict with 'nll' (scalar), 'nll_per_event' (B,), and diagnostics
        """
        device = times.device

        # ====================================================================
        # Step 1: Embed marks as event-level covariates (if mark_embedding set)
        # ====================================================================
        if marks is not None and self.mark_embedding is not None:
            x_event_marks = self.mark_embedding(marks)  # (B, N, embed_dim)
            if x_event is not None:
                x_event = torch.cat([x_event, x_event_marks], dim=-1)
            else:
                x_event = x_event_marks

        # ====================================================================
        # Step 2: Encode
        # ====================================================================
        events = torch.cat([times.unsqueeze(-1), locations], dim=-1)
        z_final, all_states = self.encoder(events, lengths, x_event=x_event)

        # ====================================================================
        # Step 3: Dispatch to batched or sequential forward
        # ====================================================================
        if isinstance(self.dynamics, IdentityDynamics):
            return self._forward_batched(
                times, locations, lengths, all_states, x_field_at_events, device,
                marks=marks,
            )
        else:
            return self._forward_sequential(
                times, locations, lengths, all_states, x_field_at_events, device,
                marks=marks,
            )

    def _forward_batched(
        self,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        all_states: Tensor,
        x_field_at_events: Optional[Tensor],
        device,
        marks: Optional[Tensor] = None,
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

        # Field covariates / history windows for decoder.
        # When the spatial sub-decoder needs history windows (data-centered
        # Gaussians), build them from the event locations and pass via the
        # dedicated x_field_spatial kwarg so the temporal decoder is unaffected.
        spatial_dec = getattr(self.decoder, "spatial", None)
        x_field_flat = None
        x_field_spatial_flat = None
        if spatial_dec is not None and getattr(spatial_dec, "requires_history", False):
            seq_len = spatial_dec.history_window_size
            hist_windows = _sliding_history_windows(locations, L, seq_len)   # (B, L, seq_len*d)
            x_field_spatial_flat = hist_windows.reshape(B * L, -1)
            if x_field_at_events is not None:
                x_field_flat = x_field_at_events[:, :L, :].reshape(B * L, -1)
        elif x_field_at_events is not None:
            x_field_flat = x_field_at_events[:, :L, :].reshape(B * L, -1)

        # Flatten to (B*L, ...) for a single batched decoder call
        h = z_cond.shape[-1]
        d = s_target.shape[-1]
        z_flat      = z_cond.reshape(B * L, h)
        t_flat      = t_target.reshape(B * L, 1)
        s_flat      = s_target.reshape(B * L, d)
        t_prev_flat = t_prev.reshape(B * L, 1)

        # Single batched ground decoder call
        if x_field_spatial_flat is not None:
            # FactorizedDecoder accepts x_field_spatial to route history windows
            # exclusively to the spatial sub-decoder.
            nll_flat = self.decoder.nll(
                z_flat, t_flat, s_flat, t_prev_flat,
                x_field=x_field_flat,
                x_field_spatial=x_field_spatial_flat,
            )  # (B*L,)
        else:
            nll_flat = self.decoder.nll(
                z_flat, t_flat, s_flat, t_prev_flat,
                x_field=x_field_flat,
            )  # (B*L,)
        nll_all = nll_flat.reshape(B, L)  # (B, L)

        # Mark NLL (additive, same batched pattern)
        if self.mark_decoder is not None and marks is not None:
            k_target = marks[:, 1:1 + L]          # (B, L)
            k_flat   = k_target.reshape(B * L)    # (B*L,)
            mark_nll_flat = self.mark_decoder.nll(
                z_flat, t_flat, s_flat, k_flat, x_field=x_field_flat
            )  # (B*L,)
            nll_all = nll_all + mark_nll_flat.reshape(B, L)

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
        marks: Optional[Tensor] = None,
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

            # Mark NLL (additive)
            if self.mark_decoder is not None and marks is not None:
                k_target = marks[:, n + 1]  # (B,) LongTensor
                mark_nll = self.mark_decoder.nll(
                    z_t, t_target, s_target, k_target, x_field=x_field_dec
                )
                nll_n = nll_n + mark_nll

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
        history_marks: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
        x_field_fn=None,
    ):
        """
        Autoregressively sample future events given history.

        For each step:
          1. Evolve state via dynamics (identity or ODE)
          2. Sample (t, s) from decoder
          3. Sample mark k from mark_decoder (if present)
          4. Update state with new event
          5. Repeat

        Args:
            history_times: (B, N) — past event times
            history_locations: (B, N, d) — past event locations
            history_lengths: (B,) — actual lengths
            n_samples: Number of future events to sample
            t_max: Maximum time horizon
            x_event: (B, N, p) optional — event covariates for history
            history_marks: (B, N) LongTensor optional — mark indices for history
            x_field_at_events: (B, N, r) optional — field covariates at history events
            x_field_fn: callable (t, s) → (B, r) optional — field covariate function

        Returns:
            sampled_times: (B, n_samples)
            sampled_locations: (B, n_samples, d)
            mask: (B, n_samples) — True for events within t_max
            sampled_marks: (B, n_samples) LongTensor or None
        """
        B = history_times.shape[0]
        device = history_times.device

        # Embed history marks if the model has a mark embedding.
        # Always provide embeddings (zeros if no marks given) so the encoder
        # sees the expected input dimension (input_dim + embed_dim).
        x_event_hist = x_event
        if self.mark_embedding is not None:
            B_h, N_h = history_times.shape
            if history_marks is not None:
                x_event_marks = self.mark_embedding(history_marks)  # (B, N, embed_dim)
            else:
                x_event_marks = torch.zeros(
                    B_h, N_h, self.mark_embedding.embed_dim, device=device
                )
            if x_event_hist is not None:
                x_event_hist = torch.cat([x_event_hist, x_event_marks], dim=-1)
            else:
                x_event_hist = x_event_marks

        # Encode history
        events = torch.cat([history_times.unsqueeze(-1), history_locations], dim=-1)
        z, all_states = self.encoder(events, history_lengths, x_event=x_event_hist)

        sampled_t = []
        sampled_s = []
        sampled_k = []

        # Last event time per sequence
        t_prev = history_times[
            torch.arange(B, device=device), (history_lengths - 1).long()
        ].unsqueeze(-1)  # (B, 1)

        for step in range(n_samples):
            # Sample next event from ground decoder
            t_new, s_new = self.decoder.sample(z, t_prev, x_field_fn)
            # t_new: (B, 1), s_new: (B, d)

            # Ensure t_new is (B, 1)
            if t_new.dim() == 1:
                t_new = t_new.unsqueeze(-1)

            sampled_t.append(t_new.squeeze(-1))
            sampled_s.append(s_new)

            # Sample mark if mark decoder is present
            k_new = None
            if self.mark_decoder is not None:
                log_probs = self.mark_decoder.log_prob(z, t_new, s_new)  # (B, K)
                k_new = torch.distributions.Categorical(logits=log_probs).sample()  # (B,)
                sampled_k.append(k_new)

            # Update state for next step, embedding the sampled mark for the updater.
            # If mark_embedding is set, the updater expects x_event of size embed_dim
            # (possibly concatenated with other event covariates, but here we only
            # have mark embeddings). Pass zeros when no marks were sampled.
            x_event_update = None
            if self.mark_embedding is not None:
                if k_new is not None:
                    x_event_update = self.mark_embedding(k_new.unsqueeze(1)).squeeze(1)  # (B, embed_dim)
                else:
                    x_event_update = torch.zeros(B, self.mark_embedding.embed_dim, device=device)
            z = self.updater(z, t_new, s_new, x_event=x_event_update)
            t_prev = t_new

        sampled_times = torch.stack(sampled_t, dim=1)   # (B, n_samples)
        sampled_locs  = torch.stack(sampled_s, dim=1)   # (B, n_samples, d)
        mask = sampled_times <= t_max                    # (B, n_samples)

        sampled_marks = None
        if sampled_k:
            sampled_marks = torch.stack(sampled_k, dim=1)  # (B, n_samples)

        return sampled_times, sampled_locs, mask, sampled_marks
