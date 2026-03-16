"""StateModel wrapper for faithful Neural STPP backbones."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel


BackboneFn = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]


class NeuralTPPBackboneStateModel(StateModel):
    """Coarse state interface for NeuralTPP backbone presets."""

    def __init__(self, *, sequence_nll_and_states_fn: BackboneFn):
        super().__init__()
        self._sequence_nll_and_states_fn = sequence_nll_and_states_fn

    @property
    def capabilities(self) -> StateCapabilities:
        return StateCapabilities(
            has_query_state=True,
            has_sequence_states=True,
            has_regularization_terms=True,
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

        bsz = times.shape[0]
        max_len = int(lengths.max().item())
        if max_len < 2:
            empty_l = 0
            return StateContext(
                payload={
                    "temporal_nll_matrix": torch.zeros(bsz, empty_l, device=times.device),
                    "z_seq": torch.zeros(bsz, empty_l, 0, device=times.device),
                    "temporal_energy_reg": torch.tensor(0.0, device=times.device),
                }
            )

        events = torch.cat([times.unsqueeze(-1), locations], dim=-1)  # (B, N, 1+d)
        temporal_nll, h_seq_pre, energy_reg = self._sequence_nll_and_states_fn(events, lengths)
        return StateContext(
            payload={
                "temporal_nll_matrix": temporal_nll,
                "z_seq": h_seq_pre,
                "temporal_energy_reg": energy_reg,
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
        energy = state_ctx.payload.get("temporal_energy_reg")
        if isinstance(energy, Tensor):
            return {"temporal_energy_reg": energy}
        if energy is not None:
            return {"temporal_energy_reg": torch.as_tensor(energy)}
        return {}
