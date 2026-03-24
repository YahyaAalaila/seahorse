"""StateModel for factorized baselines — thin history passthrough with no learned encoder."""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel


class FactorizedStateModel(StateModel):
    """
    Thin state model for factorized STPP baselines.

    Has no learnable parameters. Packages the raw event tensors into
    StateContext.payload so that FactorizedEventModel can access them
    via the standard framework interface.

    In practice, FactorizedEventModel uses the `times/locations/lengths`
    kwargs passed directly to training_loss, so the payload is mostly
    for framework compatibility (e.g., query_state calls).
    """

    def __init__(self):
        super().__init__()

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
        return StateContext(payload={"times": times, "locations": locations, "lengths": lengths})
