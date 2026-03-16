"""
StateModel wrapper for AutoSTPP.

This state model is intentionally thin: it packages raw history tensors and
encoder-produced conditioning states needed by the current AutoSTPP decoder
contract.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from ..abstractions import StateContext, StateModel


EncodeFn = Callable[[Tensor, Tensor, Optional[Tensor]], Tuple[Tensor, Tensor]]
MarkEmbedFn = Callable[[Tensor], Tensor]


class AutoSTPPStateModel(StateModel):
    """
    Thin state wrapper for AutoSTPP.

    Notes
    -----
    ``z_seq`` is the primary latent history payload for AutoSTPP event scoring.
    ``all_states`` is kept as a compatibility alias.
    """

    def __init__(
        self,
        *,
        encode_fn: EncodeFn,
        mark_embed_fn: Optional[MarkEmbedFn] = None,
    ):
        super().__init__()
        self._encode_fn = encode_fn
        self._mark_embed_fn = mark_embed_fn

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
        del x_field_at_events
        if marks is not None and self._mark_embed_fn is not None:
            x_event_marks = self._mark_embed_fn(marks)
            x_event = (
                torch.cat([x_event, x_event_marks], dim=-1)
                if x_event is not None
                else x_event_marks
            )

        events = torch.cat([times.unsqueeze(-1), locations], dim=-1)
        _z_final, all_states = self._encode_fn(events, lengths, x_event)
        return StateContext(
            payload={
                "times": times,
                "locations": locations,
                "lengths": lengths,
                "z_seq": all_states,
                # Compatibility alias.
                "all_states": all_states,
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
    ) -> Dict[str, Tensor]:
        del times, locations, lengths, x_field_at_events
        return state_ctx.payload
