"""
StateModel wrapper for DeepSTPP history encoding / posterior sampling.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from ..abstractions import StateContext, StateModel


EncodeFn = Callable[[Tensor, Tensor, Optional[Tensor]], Tuple[Tensor, Tensor]]
VAEStatsFn = Callable[[Tensor], Tuple[Tensor, Tensor]]
VAEReparamFn = Callable[[Tensor], Tuple[Tensor, Tensor]]


class DeepSTPPStateModel(StateModel):
    """
    Coarse DeepSTPP state model.

    Owns:
      - history encoding
      - optional latent posterior statistics (qm, qv)
      - optional latent reparameterization (VAE path)
    """

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
