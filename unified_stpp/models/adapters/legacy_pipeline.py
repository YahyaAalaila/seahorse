"""
Compatibility adapters for the legacy Encoder/Dynamics/Updater/Decoder stack.

These adapters provide Stage 1 ``StateModel`` / ``EventModel`` interfaces
without changing the underlying behavior.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from ..abstractions import EventModel, StateContext, StateModel


EncodeFn = Callable[[Tensor, Tensor, Optional[Tensor]], Tuple[Tensor, Tensor]]
MarkEmbedFn = Callable[[Tensor], Tensor]
VAEFn = Callable[[Tensor], Tuple[Tensor, Tensor]]
ForwardFn = Callable[..., Dict[str, Tensor]]


class LegacyPipelineStateAdapter(StateModel):
    """
    Wrap the legacy history encoding path behind the coarse StateModel API.
    """

    def __init__(
        self,
        *,
        encode_fn: EncodeFn,
        mark_embed_fn: Optional[MarkEmbedFn] = None,
        vae_reparameterize_fn: Optional[VAEFn] = None,
    ):
        super().__init__()
        self._encode_fn = encode_fn
        self._mark_embed_fn = mark_embed_fn
        self._vae_reparameterize_fn = vae_reparameterize_fn

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
            x_event = torch.cat([x_event, x_event_marks], dim=-1) if x_event is not None else x_event_marks

        events = torch.cat([times.unsqueeze(-1), locations], dim=-1)
        _z_final, all_states = self._encode_fn(events, lengths, x_event)

        kl_loss = None
        if self._vae_reparameterize_fn is not None:
            all_states, kl_loss = self._vae_reparameterize_fn(all_states)

        return StateContext(payload={"all_states": all_states}, kl_loss=kl_loss)

    def query(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        x_field_at_events: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        del times, locations, lengths, x_field_at_events
        return state_ctx.payload


class LegacyPipelineEventAdapter(EventModel):
    """
    Wrap the legacy NLL execution paths behind the coarse EventModel API.
    """

    def __init__(
        self,
        *,
        forward_batched_fn: ForwardFn,
        forward_sequential_fn: ForwardFn,
    ):
        super().__init__()
        self._forward_batched_fn = forward_batched_fn
        self._forward_sequential_fn = forward_sequential_fn

    @staticmethod
    def _extract_all_states(state: Dict[str, Any]) -> Tensor:
        all_states = state.get("all_states")
        if all_states is None:
            raise ValueError("LegacyPipelineEventAdapter requires state['all_states'].")
        return all_states

    def nll(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: Dict[str, Any],
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        all_states = self._extract_all_states(state)
        return self._forward_sequential_fn(
            times, locations, lengths, all_states, x_field_at_events, device, marks=marks
        )

    def sequence_nll(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: Dict[str, Any],
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        all_states = self._extract_all_states(state)
        return self._forward_batched_fn(
            times, locations, lengths, all_states, x_field_at_events, device, marks=marks
        )
