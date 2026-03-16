"""
Unified STPP model with coarse StateModel/EventModel dispatch.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .abstractions import EventModel, StateModel
from .base import Decoder, Dynamics, Encoder, MarkDecoder, Updater
from .neural_tpp_backbone import NeuralTPPBackbone


class UnifiedSTPP(nn.Module):
    """Top-level model container for coarse state/event execution."""

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
        use_state_event_path: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.updater = updater
        self.decoder = decoder
        self.lifting_map = lifting_map
        self.mark_decoder = mark_decoder
        self.mark_embedding = mark_embedding

        # Faithful Neural STPP backbone components (used by *_sc presets).
        self.backbone = backbone
        self.backbone_spatial = backbone_spatial

        self.state_model = state_model
        self.event_model = event_model
        self.use_state_event_path = bool(use_state_event_path)

        self.vae = bool(vae)
        if self.vae:
            self.vae_mu_proj = nn.Linear(hidden_dim, hidden_dim)
            self.vae_logvar_proj = nn.Linear(hidden_dim, hidden_dim)

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
        """Encode history and apply VAE projection when enabled."""
        z_final, all_states = self._encode_legacy_history(events, lengths, x_event=x_event)
        if self.vae:
            z_final, _ = self._vae_reparameterize(z_final)
        return z_final, all_states

    def _vae_project(self, z: Tensor):
        """Project latent tensor to VAE posterior stats (mu, log_var)."""
        mu = self.vae_mu_proj(z)
        log_var = self.vae_logvar_proj(z).clamp(min=-10, max=4)
        return mu, log_var

    def _vae_reparameterize(self, z: Tensor):
        """VAE bottleneck (sample during training, deterministic in eval)."""
        mu, log_var = self._vae_project(z)
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
        """Compute NLL for a batch of sequences."""
        del x_field_fn

        if self.state_model is None or self.event_model is None:
            raise RuntimeError(
                "UnifiedSTPP now requires both state_model and event_model. "
                "Build the model via unified_stpp.registry.build_model."
            )
        if not self.use_state_event_path:
            raise RuntimeError(
                "Legacy non-state-event dispatch has been removed. "
                "Set use_state_event_path=True."
            )

        return self._forward_state_event(
            times=times,
            locations=locations,
            lengths=lengths,
            marks=marks,
            x_event=x_event,
            x_field_at_events=x_field_at_events,
            device=times.device,
        )

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
        state_ctx = self.state_model.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
            marks=marks,
            x_event=x_event,
            x_field_at_events=x_field_at_events,
        )

        state_for_event = self.state_model.sequence_states(
            state_ctx,
            times=times,
            locations=locations,
            lengths=lengths,
            x_field_at_events=x_field_at_events,
        )
        state_regularization_terms = self.state_model.regularization_terms(
            state_ctx,
            times=times,
            locations=locations,
            lengths=lengths,
            marks=marks,
        )

        result = self.event_model.training_loss(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state_for_event,
            state_regularization_terms=state_regularization_terms,
            x_field_at_events=x_field_at_events,
            marks=marks,
            device=device,
        )

        if "nll" not in result and "loss" in result:
            result["nll"] = result["loss"]

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
        if kl_loss is not None and "kl_loss" not in result:
            result["kl_loss"] = kl_loss

        if state_regularization_terms:
            result["state_regularization_terms"] = state_regularization_terms
            for name, term in state_regularization_terms.items():
                result.setdefault(name, term)
        return result
