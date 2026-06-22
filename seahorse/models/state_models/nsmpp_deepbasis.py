"""Thin passthrough state model for the NSMPP DeepBasis preset."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel
from ..model_registry import register_state
from seahorse.data.transforms import transform_from_spec


@register_state("nsmpp_deepbasis")
class NSMPPDeepBasisStateModel(StateModel):
    """Packages raw event history for the direct-intensity DeepBasis event model."""

    def __init__(self, *, input_transform: Optional[dict] = None):
        super().__init__()
        self._input_transform_spec = dict(input_transform or {})
        self._input_transform = transform_from_spec(self._input_transform_spec)

    @property
    def capabilities(self) -> StateCapabilities:
        return StateCapabilities(
            has_query_state=False,
            has_sequence_states=False,
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
        del marks, x_event, x_field_at_events
        transform = self._input_transform
        times_native = transform.forward_times(times, lengths) if transform is not None else times
        locations_native = (
            transform.forward_locations(locations, lengths)
            if transform is not None
            else locations
        )
        max_len = int(times.shape[1]) if times.ndim == 2 else 0
        idx = torch.arange(max_len, device=times.device)
        event_mask = idx.unsqueeze(0) < lengths.unsqueeze(1)
        event_vectors = torch.cat([times_native.unsqueeze(-1), locations_native], dim=-1)
        return StateContext(
            payload={
                "times": times_native,
                "locations": locations_native,
                "lengths": lengths,
                "event_mask": event_mask,
                "event_vectors": event_vectors,
                "times_raw": times,
                "locations_raw": locations,
                "input_transform": self._input_transform_spec,
            }
        )
