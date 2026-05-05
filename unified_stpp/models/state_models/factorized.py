"""StateModel for factorized baselines — thin history passthrough with no learned encoder."""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel
from unified_stpp.data.transforms import transform_from_spec


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
        return StateContext(
            payload={
                "times": times_native,
                "locations": locations_native,
                "lengths": lengths,
                "times_raw": times,
                "locations_raw": locations,
                "input_transform": self._input_transform_spec,
            }
        )
