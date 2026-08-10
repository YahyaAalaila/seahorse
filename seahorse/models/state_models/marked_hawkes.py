"""StateModel for the marked Hawkes baseline — history passthrough that keeps marks.

Identical in spirit to :class:`FactorizedStateModel` (no learnable parameters,
packages the raw event tensors into ``StateContext.payload``) with one addition:
the per-event integer ``marks`` tensor is carried into the payload so that
intensity/surface queries — which receive only a ``StateContext`` — can see it.

``FactorizedStateModel`` deliberately drops ``marks``; rather than change its
behaviour, the marked family gets its own passthrough.  This keeps the change
strictly additive: no existing preset's state payload is altered.
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel
from ..model_registry import register_state
from seahorse.data.transforms import transform_from_spec


@register_state("marked_hawkes")
class MarkedHawkesStateModel(StateModel):
    """History passthrough for marked point-process baselines."""

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
        del x_event, x_field_at_events
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
                "marks": marks,
                "times_raw": times,
                "locations_raw": locations,
                "input_transform": self._input_transform_spec,
            }
        )
