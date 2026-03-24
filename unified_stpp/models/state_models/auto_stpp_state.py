"""StateModel for AutoSTPP — owns TransformerEncoder and optional mark embedding."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel


class AutoSTPPStateModel(StateModel):
    """AutoSTPP state model.  Owns encoder and optional mark embedding."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        spatial_dim: int,
        event_cov_dim: int = 0,
        enc_num_heads: int = 2,
        enc_num_layers: int = 3,
        enc_dropout: float = 0.1,
        n_marks: int = 0,
        **enc_extra,
    ):
        super().__init__()
        from ..history_encoders import TransformerEncoder

        mark_embed_dim = hidden_dim if n_marks > 0 else 0
        self.mark_embedding: Optional[nn.Embedding] = (
            nn.Embedding(n_marks, hidden_dim) if n_marks > 0 else None
        )
        self.encoder = TransformerEncoder(
            input_dim=1 + spatial_dim,
            hidden_dim=hidden_dim,
            event_cov_dim=event_cov_dim + mark_embed_dim,
            num_heads=enc_num_heads,
            num_layers=enc_num_layers,
            dropout=enc_dropout,
            **enc_extra,
        )

    @property
    def capabilities(self) -> StateCapabilities:
        return StateCapabilities(
            has_query_state=True,
            has_sequence_states=True,
            has_regularization_terms=False,
            state_kind="history_passthrough",
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
        del x_field_at_events
        if marks is not None and self.mark_embedding is not None:
            x_event_marks = self.mark_embedding(marks)
            x_event = (
                torch.cat([x_event, x_event_marks], dim=-1)
                if x_event is not None
                else x_event_marks
            )

        events = torch.cat([times.unsqueeze(-1), locations], dim=-1)
        z_final, all_states = self.encoder(events, lengths, x_event=x_event)
        return StateContext(
            payload={
                "times": times,
                "locations": locations,
                "lengths": lengths,
                "z_final": z_final,
                "z_seq": all_states,
                "all_states": all_states,
            }
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
