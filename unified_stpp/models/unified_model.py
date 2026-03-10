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
import os
from .base import Encoder, Dynamics, Updater, Decoder, MarkDecoder
from .abstractions import StateModel, EventModel
from .adapters import LegacyPipelineStateAdapter, LegacyPipelineEventAdapter
from .dynamics.identity import IdentityDynamics
from .neural_tpp_backbone import NeuralTPPBackbone


def _sliding_history_windows_with_times(
    times: Tensor, locations: Tensor, L: int, seq_len: int
) -> Tensor:
    """
    Build (B, L, seq_len + seq_len * d) windows packed as [times | locations].

    Packed format per window:
        [t_0, ..., t_{seq_len-1},  s_0_x, s_0_y, ..., s_{seq_len-1}_y]

    Window n contains the seq_len events immediately preceding prediction
    position n+1, left-padded with the first observed event when fewer than
    seq_len events are available.  Times are absolute (cumulative), matching
    the ``times`` tensor format used throughout the framework.
    """
    B, _, d = locations.shape

    # Pad seq_len-1 copies of the first event
    first_t = times[:, :1].expand(-1, seq_len - 1)                    # (B, seq_len-1)
    padded_t = torch.cat([first_t, times[:, : L + 1]], dim=1)         # (B, seq_len-1+L+1)

    first_s  = locations[:, :1, :].expand(-1, seq_len - 1, -1)        # (B, seq_len-1, d)
    padded_s = torch.cat([first_s, locations[:, : L + 1, :]], dim=1)  # (B, seq_len-1+L+1, d)

    t_windows = torch.stack(
        [padded_t[:, n : n + seq_len] for n in range(L)], dim=1
    )  # (B, L, seq_len)

    s_windows = torch.stack(
        [padded_s[:, n : n + seq_len, :] for n in range(L)], dim=1
    )  # (B, L, seq_len, d)

    # Pack as (B, L, seq_len + seq_len*d) = [times | locs]
    return torch.cat(
        [t_windows, s_windows.reshape(B, L, seq_len * d)], dim=-1
    )  # (B, L, seq_len*(1+d))


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
        encoder: Optional[Encoder] = None,
        dynamics: Optional[Dynamics] = None,
        updater: Optional[Updater] = None,
        decoder: Optional[Decoder] = None,
        lifting_map: Optional[nn.Module] = None,
        mark_decoder: Optional[MarkDecoder] = None,
        mark_embedding: Optional[nn.Module] = None,
        vae: bool = False,
        hidden_dim: int = 128,
        backbone: Optional[NeuralTPPBackbone] = None,
        backbone_spatial: Optional[nn.Module] = None,
        state_model: Optional[StateModel] = None,
        event_model: Optional[EventModel] = None,
        use_state_event_path: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.updater = updater
        self.decoder = decoder
        self.lifting_map = lifting_map
        self.mark_decoder = mark_decoder
        self.mark_embedding = mark_embedding
        self.vae = vae
        # NeuralTPP faithful backbone — bypasses encoder/dynamics/updater/decoder
        self.backbone = backbone
        self.backbone_spatial = backbone_spatial
        self.state_model = state_model
        self.event_model = event_model
        self.use_state_event_path = use_state_event_path
        if vae:
            self.vae_mu_proj = nn.Linear(hidden_dim, hidden_dim)
            self.vae_logvar_proj = nn.Linear(hidden_dim, hidden_dim)
        if self.state_model is None and self.encoder is not None:
            self.state_model = LegacyPipelineStateAdapter(
                encode_fn=self._encode_legacy_history,
                mark_embed_fn=self._embed_marks if self.mark_embedding is not None else None,
                vae_reparameterize_fn=self._vae_reparameterize if self.vae else None,
            )
        if self.event_model is None and self.decoder is not None and self.dynamics is not None:
            self.event_model = LegacyPipelineEventAdapter(
                forward_batched_fn=self._forward_batched,
                forward_sequential_fn=self._forward_sequential,
            )
        self._debug_nstpp = os.getenv("UNIFIED_STPP_DEBUG_NSTPP", "0") == "1"
        self._debug_nstpp_max_calls = max(
            1, int(os.getenv("UNIFIED_STPP_DEBUG_NSTPP_MAX_CALLS", "10"))
        )
        self._debug_nstpp_calls = 0

    def _embed_marks(self, marks: Tensor) -> Tensor:
        if self.mark_embedding is None:
            raise RuntimeError("mark_embedding is not configured.")
        return self.mark_embedding(marks)

    def _encode_legacy_history(
        self,
        events: Tensor,
        lengths: Tensor,
        x_event: Optional[Tensor] = None,
    ):
        if self.encoder is None:
            raise RuntimeError("encoder is not configured.")
        return self.encoder(events, lengths, x_event=x_event)

    def encode(self, events: Tensor, lengths: Tensor, x_event=None):
        """Encode events to (z_final, all_states), applying VAE bottleneck if enabled.

        Use this instead of ``model.encoder(...)`` so that the VAE projection
        (mu_proj / logvar_proj) is always applied during inference/visualization.
        During eval mode, returns ``mu`` deterministically (no sampling noise).
        """
        z_final, all_states = self.encoder(events, lengths, x_event=x_event)
        if self.vae:
            z_final, _ = self._vae_reparameterize(z_final)
        return z_final, all_states

    def _vae_reparameterize(self, z: Tensor):
        """VAE bottleneck: project to (mu, log_var), sample during training.

        Works on any shape (..., H). Returns (z_out, kl_loss).
        During eval, returns mu (deterministic).
        """
        mu = self.vae_mu_proj(z)
        log_var = self.vae_logvar_proj(z).clamp(min=-10, max=4)
        if self.training:
            z_out = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        else:
            z_out = mu
        kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean()
        return z_out, kl

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
        # Coarse StateModel/EventModel path
        # ====================================================================
        if (
            self.use_state_event_path
            and self.state_model is not None
            and self.event_model is not None
        ):
            return self._forward_state_event(
                times=times,
                locations=locations,
                lengths=lengths,
                marks=marks,
                x_event=x_event,
                x_field_at_events=x_field_at_events,
                device=device,
            )

        # ====================================================================
        # Fast path fallback: legacy NeuralTPP backbone execution
        # ====================================================================
        if self.backbone is not None:
            return self._forward_neural_tpp(times, locations, lengths, device)

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
        # Step 2b: VAE bottleneck (optional)
        # ====================================================================
        kl_loss = None
        if self.vae:
            all_states, kl_loss = self._vae_reparameterize(all_states)

        # ====================================================================
        # Step 3: Dispatch to batched or sequential forward
        # ====================================================================
        if isinstance(self.dynamics, IdentityDynamics):
            result = self._forward_batched(
                times, locations, lengths, all_states, x_field_at_events, device,
                marks=marks,
            )
        else:
            result = self._forward_sequential(
                times, locations, lengths, all_states, x_field_at_events, device,
                marks=marks,
            )
        if kl_loss is not None:
            result["kl_loss"] = kl_loss
        return result

    def _forward_state_event(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor],
        x_event: Optional[Tensor],
        x_field_at_events: Optional[Tensor],
        device,
    ) -> Dict[str, Tensor]:
        """
        Coarse StateModel/EventModel execution path.

        This path covers both compatibility adapters (Stage 1) and native
        StateModel/EventModel implementations (Stage 2+).
        """
        state_ctx = self.state_model.forward_history(
            times=times,
            locations=locations,
            lengths=lengths,
            marks=marks,
            x_event=x_event,
            x_field_at_events=x_field_at_events,
        )

        if isinstance(self.dynamics, IdentityDynamics):
            state = self.state_model.sequence_forward(
                state_ctx,
                times=times,
                locations=locations,
                lengths=lengths,
                x_field_at_events=x_field_at_events,
            )
            result = self.event_model.sequence_nll(
                times=times,
                locations=locations,
                lengths=lengths,
                state=state,
                x_field_at_events=x_field_at_events,
                marks=marks,
                device=device,
            )
        else:
            state = self.state_model.query(
                state_ctx,
                times=times,
                locations=locations,
                lengths=lengths,
                x_field_at_events=x_field_at_events,
            )
            result = self.event_model.nll(
                times=times,
                locations=locations,
                lengths=lengths,
                state=state,
                x_field_at_events=x_field_at_events,
                marks=marks,
                device=device,
            )

        debug_this_call = (
            self._debug_nstpp and self._debug_nstpp_calls < self._debug_nstpp_max_calls
        )
        if debug_this_call:
            mask = result.get("mask")
            temporal_nll = result.get("temporal_nll_matrix")
            spatial_nll = result.get("spatial_nll_matrix")
            nll_matrix = result.get("nll_matrix")
            if (
                isinstance(mask, Tensor)
                and isinstance(temporal_nll, Tensor)
                and isinstance(spatial_nll, Tensor)
                and isinstance(nll_matrix, Tensor)
                and mask.numel() > 0
                and (mask > 0).any()
            ):
                valid = mask > 0
                t_vals = temporal_nll.detach()[valid]
                s_vals = spatial_nll.detach()[valid]
                j_vals = nll_matrix.detach()[valid]
                energy_reg = result.get("temporal_energy_reg", 0.0)
                spatial_reg = result.get("spatial_reg", 0.0)
                print(
                    "[NSTPP-DEBUG][loss] "
                    f"temporal_nll(min/mean/max)="
                    f"{t_vals.min().item():.6f}/{t_vals.mean().item():.6f}/{t_vals.max().item():.6f} "
                    f"spatial_nll(min/mean/max)="
                    f"{s_vals.min().item():.6f}/{s_vals.mean().item():.6f}/{s_vals.max().item():.6f} "
                    f"joint_nll(min/mean/max)="
                    f"{j_vals.min().item():.6f}/{j_vals.mean().item():.6f}/{j_vals.max().item():.6f} "
                    f"energy_reg={float(torch.as_tensor(energy_reg).detach().item()):.6f} "
                    f"spatial_reg={float(torch.as_tensor(spatial_reg).detach().item()):.6f}"
                )
                self._debug_nstpp_calls += 1

        kl_loss = getattr(state_ctx, "kl_loss", None)
        if kl_loss is not None:
            result["kl_loss"] = kl_loss
        return result

    def _forward_neural_tpp(
        self,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        device,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for backbone-based presets (neural_stpp_jump_sc, _attn_sc).

        The NeuralTPPBackbone owns hidden-state dynamics, intensity, and the joint
        [h, Λ] ODE.  It returns pre-jump hidden states h_seq_pre which are fed
        directly to the SEQUENCE_COUPLED spatial decoder.

        Bypasses: encoder, NeuralODEDynamics, updater, CumulativeHazardTemporal.
        """
        B = times.shape[0]
        max_len = int(lengths.max().item())

        if max_len < 2:
            return {
                "nll": torch.tensor(0.0, device=device),
                "nll_per_event": torch.zeros(B, device=device),
                "total_events": torch.tensor(0.0, device=device),
            }

        L = max_len - 1

        # Pack [t, s] for backbone
        events = torch.cat([times.unsqueeze(-1), locations], dim=-1)  # (B, N, 1+d)

        # ── Step 1: temporal backbone ─────────────────────────────────────── #
        # Returns per-position temporal NLL, pre-jump hidden states, and the
        # kinetic-energy regularization term (--tpp_otreg_strength equivalent).
        temporal_nll, h_seq_pre, energy_reg = self.backbone.sequence_nll_and_states(
            events, lengths
        )
        # temporal_nll: (B, L),  h_seq_pre: (B, L, hidden_dim),  energy_reg: scalar

        # ── Step 2: spatial decoder ───────────────────────────────────────── #
        t_seq = times[:, 1 : 1 + L].unsqueeze(-1)   # (B, L, 1)
        s_seq = locations[:, 1 : 1 + L, :]           # (B, L, d)
        t_prev_seq = times[:, :L].unsqueeze(-1)       # (B, L, 1)

        n_idx = torch.arange(L, device=device)
        mask = (n_idx.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()  # (B, L)

        spatial_nll = self.backbone_spatial.sequence_nll(
            z_seq=h_seq_pre,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        )  # (B, L)

        # ── Aggregate ─────────────────────────────────────────────────────── #
        nll_all = temporal_nll + spatial_nll   # (B, L)
        nll_masked = nll_all * mask
        total_nll = nll_masked.sum(dim=1)      # (B,)
        n_events = mask.sum(dim=1)             # (B,)
        mean_nll = total_nll.sum() / n_events.sum().clamp(min=1)
        debug_this_call = (
            self._debug_nstpp and self._debug_nstpp_calls < self._debug_nstpp_max_calls
        )

        # Add temporal kinetic energy reg (already scaled by energy_regularization).
        # Zero when backbone.energy_regularization=0.0 (default).
        mean_nll = mean_nll + energy_reg

        # Add spatial OT regularization (--otreg_strength), if spatial decoder computed it.
        # SelfAttentiveCNFSpatial stores _energy_reg after each sequence_nll call.
        # Zero for decoders that do not implement this attribute (JumpCNFSpatial, etc.).
        spatial_reg = getattr(self.backbone_spatial, "_energy_reg", 0.0)
        mean_nll = mean_nll + spatial_reg

        if debug_this_call:
            valid = mask > 0
            t_vals = temporal_nll.detach()[valid]
            s_vals = spatial_nll.detach()[valid]
            j_vals = nll_all.detach()[valid]
            print(
                "[NSTPP-DEBUG][loss] "
                f"temporal_nll(min/mean/max)="
                f"{t_vals.min().item():.6f}/{t_vals.mean().item():.6f}/{t_vals.max().item():.6f} "
                f"spatial_nll(min/mean/max)="
                f"{s_vals.min().item():.6f}/{s_vals.mean().item():.6f}/{s_vals.max().item():.6f} "
                f"joint_nll(min/mean/max)="
                f"{j_vals.min().item():.6f}/{j_vals.mean().item():.6f}/{j_vals.max().item():.6f} "
                f"energy_reg={float(torch.as_tensor(energy_reg).detach().item()):.6f} "
                f"spatial_reg={float(torch.as_tensor(spatial_reg).detach().item()):.6f}"
            )
            self._debug_nstpp_calls += 1

        return {
            "nll": mean_nll,
            "nll_per_event": total_nll / n_events.clamp(min=1),
            "total_events": n_events.sum(),
        }

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

        if getattr(self.decoder, "SEQUENCE_COUPLED", False):
            # Sequence-coupled decoder: pass full (B, L, ...) tensors directly.
            # History-window routing (DeepSTPP, DataCenteredGaussian) is not used
            # by sequence-coupled decoders.
            nll_all = self.decoder.sequence_nll(
                z_cond, t_target, s_target, t_prev, lengths, mask,
            )  # (B, L)

            # Mark NLL: still per-event — computed from flattened states.
            if self.mark_decoder is not None and marks is not None:
                _h = z_cond.shape[-1]
                _d = s_target.shape[-1]
                z_flat = z_cond.reshape(B * L, _h)
                t_flat = t_target.reshape(B * L, 1)
                s_flat = s_target.reshape(B * L, _d)
                k_flat = marks[:, 1:1 + L].reshape(B * L)
                nll_all = nll_all + self.mark_decoder.nll(
                    z_flat, t_flat, s_flat, k_flat,
                ).reshape(B, L)
        else:
            # Field covariates / history windows for decoder.
            #
            # Three routing cases:
            #   1. DeepSTPPDecoder (requires_time_history=True): needs absolute times
            #      AND locations for each history window event.  These are packed into
            #      x_field as (B*L, seq_len + seq_len*d) and passed as the sole x_field.
            #
            #   2. FactorizedDecoder with DataCenteredGaussianSpatial (requires_history
            #      on the spatial sub-decoder, no time history): build location-only
            #      windows and route via the x_field_spatial kwarg so the temporal
            #      sub-decoder is unaffected.
            #
            #   3. Standard case: pass x_field_at_events directly (no history window).
            spatial_dec = getattr(self.decoder, "spatial", None)
            x_field_flat = None
            x_field_spatial_flat = None

            if getattr(self.decoder, "requires_time_history", False):
                # Case 1: DeepSTPPDecoder — build (time, location) windows
                seq_len = self.decoder.history_window_size
                hist_windows = _sliding_history_windows_with_times(
                    times, locations, L, seq_len
                )  # (B, L, seq_len + seq_len*d)
                x_field_spatial_flat = hist_windows.reshape(B * L, -1)
                if x_field_at_events is not None:
                    x_field_flat = x_field_at_events[:, :L, :].reshape(B * L, -1)
            elif spatial_dec is not None and getattr(spatial_dec, "requires_history", False):
                # Case 2: FactorizedDecoder + DataCenteredGaussianSpatial
                seq_len = spatial_dec.history_window_size
                hist_windows = _sliding_history_windows(locations, L, seq_len)  # (B, L, seq_len*d)
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
            if getattr(self.decoder, "requires_time_history", False):
                # DeepSTPPDecoder: history (times+locs) as the sole x_field argument
                nll_flat = self.decoder.nll(
                    z_flat, t_flat, s_flat, t_prev_flat,
                    x_field=x_field_spatial_flat,
                )  # (B*L,)
            elif x_field_spatial_flat is not None:
                # FactorizedDecoder: route location windows exclusively to spatial sub-decoder
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

        if getattr(self.decoder, "SEQUENCE_COUPLED", False):
            # Two-pass approach for sequence-coupled decoders (e.g. JumpCNF,
            # SelfAttentiveCNF):
            # Pass 1 — run dynamics, collect z_t / t / s / t_prev for every step.
            # Pass 2 — call decoder.sequence_nll once with (B, L, ...) tensors.
            # NOTE: the augmented ODE side-channel (_precomputed_lambda) is not
            # forwarded here; CumulativeHazardTemporal falls back to quadrature.
            z_t_list, t_list, s_list, tp_list = [], [], [], []
            for n in range(max_len - 1):
                active = lengths > n + 1
                if not active.any():
                    break

                z_n  = all_states[:, n, :]
                t_n1 = times[:, n + 1].unsqueeze(-1)    # (B, 1)
                s_n1 = locations[:, n + 1, :]           # (B, d)
                t_n  = times[:, n].unsqueeze(-1)        # (B, 1)
                dt   = (t_n1 - t_n).clamp(min=1e-6)

                x_fd = None
                if x_field_at_events is not None:
                    x_fd = x_field_at_events[:, n, :].unsqueeze(1)

                z_t = self.dynamics(z_n, dt, x_fd).squeeze(1)  # (B, h)
                z_t_list.append(z_t)
                t_list.append(t_n1)
                s_list.append(s_n1)
                tp_list.append(t_n)

            L_seq = len(z_t_list)
            if L_seq == 0:
                return {
                    "nll": torch.tensor(0.0, device=device),
                    "nll_per_event": torch.zeros(B, device=device),
                    "total_events": torch.tensor(0.0, device=device),
                }

            z_seq  = torch.stack(z_t_list, dim=1)   # (B, L_seq, h)
            t_seq  = torch.stack(t_list,   dim=1)   # (B, L_seq, 1)
            s_seq  = torch.stack(s_list,   dim=1)   # (B, L_seq, d)
            tp_seq = torch.stack(tp_list,  dim=1)   # (B, L_seq, 1)

            n_idx = torch.arange(L_seq, device=device)
            mask  = (n_idx.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()

            nll_all = self.decoder.sequence_nll(
                z_seq, t_seq, s_seq, tp_seq, lengths, mask,
            )  # (B, L_seq)

            if self.mark_decoder is not None and marks is not None:
                _h = z_seq.shape[-1]
                _d = s_seq.shape[-1]
                z_flat = z_seq.reshape(B * L_seq, _h)
                t_flat = t_seq.reshape(B * L_seq, 1)
                s_flat = s_seq.reshape(B * L_seq, _d)
                k_flat = marks[:, 1:1 + L_seq].reshape(B * L_seq)
                nll_all = nll_all + self.mark_decoder.nll(
                    z_flat, t_flat, s_flat, k_flat,
                ).reshape(B, L_seq)

            nll_masked = nll_all * mask
            total_nll  = nll_masked.sum(dim=1)
            n_events   = mask.sum(dim=1)

        else:
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

        # Encode history (encode() applies VAE bottleneck if enabled)
        events = torch.cat([history_times.unsqueeze(-1), history_locations], dim=-1)
        z, all_states = self.encode(events, history_lengths, x_event=x_event_hist)

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
