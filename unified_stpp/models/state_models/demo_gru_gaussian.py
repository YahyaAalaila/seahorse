"""Small GRU state model used by the researcher tutorial preset."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from unified_stpp.models.abstractions import StateCapabilities, StateContext, StateModel
from unified_stpp.models.model_registry import register_state


@register_state("demo_gru_decay")
class DemoGRUDecayStateModel(StateModel):
    """Encode event histories and evolve the previous state to each event time.

    This is intentionally compact: it is a tutorial model, not a paper model.
    It demonstrates the Seahorse state contract with a familiar PyTorch GRU and
    a learnable exponential decay evolve step.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        spatial_dim: int = 2,
        decay_init: float = 0.25,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.spatial_dim = int(spatial_dim)
        self.input_proj = nn.Linear(self.spatial_dim + 1, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.log_decay = nn.Parameter(torch.log(torch.tensor(float(decay_init))))

    @property
    def capabilities(self) -> StateCapabilities:
        return StateCapabilities(
            has_query_state=False,
            has_sequence_states=True,
            has_regularization_terms=False,
            state_kind="process_backbone",
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
        del marks, x_event, x_field_at_events
        B, T = times.shape
        mask = torch.arange(T, device=times.device).unsqueeze(0) < lengths.unsqueeze(1)

        features = torch.cat([times.unsqueeze(-1), locations], dim=-1)
        features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))
        encoded = torch.tanh(self.input_proj(features))
        hidden, _ = self.gru(encoded)
        hidden = torch.where(mask.unsqueeze(-1), hidden, torch.zeros_like(hidden))

        return StateContext(
            payload={
                "encoded_history": hidden,
                "times": times,
                "locations": locations,
                "lengths": lengths,
                "mask": mask,
            }
        )

    def sequence_states(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        x_field_at_events: Optional[Tensor] = None,
    ) -> StateContext:
        del locations, x_field_at_events
        encoded = state_ctx.payload["encoded_history"]
        B, T, H = encoded.shape
        zeros = encoded.new_zeros(B, 1, H)
        previous_state = torch.cat([zeros, encoded[:, :-1, :]], dim=1)

        t0 = times.new_zeros(B, 1)
        previous_times = torch.cat([t0, times[:, :-1]], dim=1)
        delta_t = (times - previous_times).clamp_min(0.0).unsqueeze(-1)
        decay_rate = torch.nn.functional.softplus(self.log_decay) + 1e-4
        evolved = previous_state * torch.exp(-decay_rate * delta_t)

        mask = torch.arange(T, device=times.device).unsqueeze(0) < lengths.unsqueeze(1)
        evolved = torch.where(mask.unsqueeze(-1), evolved, torch.zeros_like(evolved))
        return StateContext(
            payload={
                **state_ctx.payload,
                "event_state": evolved,
                "mask": mask,
            }
        )
