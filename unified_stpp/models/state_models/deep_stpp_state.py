"""StateModel wrapper for DeepSTPP history encoding / posterior sampling."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel


EncodeFn = Callable[[Tensor, Tensor, Optional[Tensor]], Tuple[Tensor, Tensor]]
VAEStatsFn = Callable[[Tensor], Tuple[Tensor, Tensor]]
VAEReparamFn = Callable[[Tensor], Tuple[Tensor, Tensor]]


class DeepSTPPStateModel(StateModel):
    """Coarse DeepSTPP state model."""

    def __init__(
        self,
        *,
        encode_fn: EncodeFn,
        vae_stats_fn: Optional[VAEStatsFn] = None,
        vae_reparameterize_fn: Optional[VAEReparamFn] = None,
    ):
        super().__init__()
        self._encode_fn = encode_fn
        self._vae_stats_fn = vae_stats_fn
        self._vae_reparameterize_fn = vae_reparameterize_fn

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
        z_final, all_states = self._encode_fn(events, lengths, x_event)

        z = all_states
        qm = None
        qv = None
        kl_loss = None
        if self._vae_stats_fn is not None:
            qm, qv = self._vae_stats_fn(all_states)
        if self._vae_reparameterize_fn is not None:
            z, kl_loss = self._vae_reparameterize_fn(all_states)

        return StateContext(
            payload={
                "z": z,
                "qm": qm,
                "qv": qv,
                "kl_loss": kl_loss,
                "z_final": z_final,
                "all_states": all_states,
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
