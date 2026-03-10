"""
StateModel wrapper for faithful Neural STPP backbones.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from ..abstractions import StateContext, StateModel


BackboneFn = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]


class NeuralTPPBackboneStateModel(StateModel):
    """
    Coarse state interface for NeuralTPP backbone presets.

    This model consumes full event history and exposes a payload designed for
    sequence-coupled event likelihood models.
    """

    def __init__(self, *, sequence_nll_and_states_fn: BackboneFn):
        super().__init__()
        self._sequence_nll_and_states_fn = sequence_nll_and_states_fn

    def forward_history(
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

        B = times.shape[0]
        max_len = int(lengths.max().item())
        if max_len < 2:
            empty_l = 0
            return StateContext(
                payload={
                    "temporal_nll_matrix": torch.zeros(B, empty_l, device=times.device),
                    "z_seq": torch.zeros(B, empty_l, 0, device=times.device),
                    "temporal_energy_reg": torch.tensor(0.0, device=times.device),
                }
            )

        events = torch.cat([times.unsqueeze(-1), locations], dim=-1)  # (B, N, 1+d)
        temporal_nll, h_seq_pre, energy_reg = self._sequence_nll_and_states_fn(events, lengths)
        return StateContext(
            payload={
                "temporal_nll_matrix": temporal_nll,   # (B, L)
                "z_seq": h_seq_pre,                    # (B, L, h)
                "temporal_energy_reg": energy_reg,     # scalar
            }
        )

    def query(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        x_field_at_events: Optional[Tensor] = None,
    ):
        del times, locations, lengths, x_field_at_events
        return state_ctx.payload
