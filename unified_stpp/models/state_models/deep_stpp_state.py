"""StateModel for DeepSTPP — owns TransformerEncoder and optional VAE layers."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel


class DeepSTPPStateModel(StateModel):
    """DeepSTPP state model.  Owns encoder and optional VAE bottleneck."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        spatial_dim: int,
        event_cov_dim: int = 0,
        enc_num_heads: int = 2,
        enc_num_layers: int = 3,
        enc_dropout: float = 0.0,
        vae: bool = False,
        **enc_extra,
    ):
        super().__init__()
        from ..history_encoders import TransformerEncoder
        self.encoder = TransformerEncoder(
            input_dim=1 + spatial_dim,
            hidden_dim=hidden_dim,
            event_cov_dim=event_cov_dim,
            num_heads=enc_num_heads,
            num_layers=enc_num_layers,
            dropout=enc_dropout,
            **enc_extra,
        )
        self.hidden_dim = hidden_dim
        self.vae = bool(vae)
        if self.vae:
            self.vae_mu_proj = nn.Linear(hidden_dim, hidden_dim)
            self.vae_logvar_proj = nn.Linear(hidden_dim, hidden_dim)

    @property
    def capabilities(self) -> StateCapabilities:
        return StateCapabilities(
            has_query_state=True,
            has_sequence_states=True,
            has_regularization_terms=True,
            state_kind="latent_static",
        )

    def encode_history(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor] = None,
        x_event: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
    ) -> StateContext:
        del marks, x_field_at_events
        events = torch.cat([times.unsqueeze(-1), locations], dim=-1)
        z_final, all_states = self.encoder(events, lengths, x_event=x_event)

        z = all_states
        qm = qv = kl_loss = None
        if self.vae:
            mu = self.vae_mu_proj(all_states)
            log_var = self.vae_logvar_proj(all_states).clamp(min=-10, max=4)
            qm, qv = mu, log_var
            if self.training:
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
            else:
                z = mu
            kl_loss = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean()

        return StateContext(
            payload={
                "z": z,
                "qm": qm,
                "qv": qv,
                "kl_loss": kl_loss,
                "z_final": z_final,
                "all_states": all_states,
                "times": times,
                "locations": locations,
                "lengths": lengths,
            },
            kl_loss=kl_loss,
        )

    def query_state(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        x_field_at_events: Optional[Tensor] = None,
    ) -> StateContext:
        del times, locations, lengths, x_field_at_events
        return state_ctx

    def sequence_states(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        x_field_at_events: Optional[Tensor] = None,
    ) -> StateContext:
        del times, locations, lengths, x_field_at_events
        return state_ctx

    def regularization_terms(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        del times, locations, lengths, marks
        kl_loss = state_ctx.payload.get("kl_loss")
        if isinstance(kl_loss, Tensor):
            return {"kl_loss": kl_loss}
        if kl_loss is not None:
            return {"kl_loss": torch.as_tensor(kl_loss)}
        return {}
