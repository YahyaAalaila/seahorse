"""
Coarse-grained model abstractions for state and event modeling.

Stage 1 intentionally keeps these interfaces broad so future models are not
forced into Encoder/Dynamics/Updater phases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch.nn as nn
from torch import Tensor


@dataclass
class StateContext:
    """
    Opaque state context returned by ``StateModel.forward_history``.

    ``payload`` is intentionally unstructured in Stage 1 to avoid baking old
    decomposition assumptions into the new abstraction.
    """

    payload: Dict[str, Any] = field(default_factory=dict)
    kl_loss: Optional[Tensor] = None


class StateModel(ABC, nn.Module):
    """
    Coarse latent-state interface.

    Implementations decide how history is consumed and what query payload shape
    is exposed to an event model.
    """

    def __init__(self):
        ABC.__init__(self)
        nn.Module.__init__(self)

    @abstractmethod
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
        """Build a context object from observed history."""
        ...

    @abstractmethod
    def query(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        x_field_at_events: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Return state payload for event scoring/sampling."""
        ...

    def sequence_forward(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        x_field_at_events: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Optional sequence-wise query hook; defaults to ``query``."""
        return self.query(
            state_ctx,
            times=times,
            locations=locations,
            lengths=lengths,
            x_field_at_events=x_field_at_events,
        )


class EventModel(ABC, nn.Module):
    """
    Event likelihood interface.

    Stage 1 keeps two entry points:
      - ``nll``: generic per-batch event loss path
      - ``sequence_nll``: sequence-oriented path (including sequence-coupled
        decoders)
    """

    def __init__(self):
        ABC.__init__(self)
        nn.Module.__init__(self)

    @abstractmethod
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
        """Compute NLL through the non-sequence-forward path."""
        ...

    @abstractmethod
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
        """Compute NLL through the sequence-forward path."""
        ...
