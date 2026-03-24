"""Coarse state/event abstractions with capability declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import torch.nn as nn
from torch import Tensor


TrainingObjective = Literal[
    "exact_nll",
    "approx_nll",
    "elbo",
    "score_matching",
    "hybrid",
]


@dataclass(frozen=True)
class StateCapabilities:
    """Capability declarations for state models."""

    has_query_state: bool = True
    has_sequence_states: bool = True
    has_regularization_terms: bool = False
    state_kind: Literal["process_backbone", "latent_static", "history_passthrough"] = (
        "history_passthrough"
    )


@dataclass(frozen=True)
class EventCapabilities:
    """Capability declarations for event models."""

    training_objective: TrainingObjective = "exact_nll"
    has_eval_nll: bool = True
    has_intensity: bool = False
    has_density: bool = False
    has_score: bool = False
    has_native_sampler: bool = False
    exposes_eventwise_terms: bool = False


@dataclass
class StateContext:
    """Opaque state container shared between state and event models."""

    payload: Dict[str, Any] = field(default_factory=dict)
    # Compatibility field kept for existing VAE-style workflows.
    kl_loss: Optional[Tensor] = None


class StateModel(ABC, nn.Module):
    """Coarse latent-state interface."""

    def __init__(self):
        ABC.__init__(self)
        nn.Module.__init__(self)

    @property
    def capabilities(self) -> StateCapabilities:
        return StateCapabilities()

    @abstractmethod
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
        """Build state context from observed history."""
        ...

    def query_state(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        x_field_at_events: Optional[Tensor] = None,
    ) -> StateContext:
        """Optional point-query state hook."""
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
        """Optional sequence-query state hook."""
        return self.query_state(
            state_ctx,
            times=times,
            locations=locations,
            lengths=lengths,
            x_field_at_events=x_field_at_events,
        )

    def regularization_terms(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Explicit raw state-side regularization terms."""
        del state_ctx, times, locations, lengths, marks
        return {}

    # ------------------------------------------------------------------
    # Backward-compatible Stage-1 wrappers
    # ------------------------------------------------------------------

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
        return self.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
            marks=marks,
            x_event=x_event,
            x_field_at_events=x_field_at_events,
        )

    def query(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        x_field_at_events: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        return self.query_state(
            state_ctx,
            times=times,
            locations=locations,
            lengths=lengths,
            x_field_at_events=x_field_at_events,
        ).payload

    def sequence_forward(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        x_field_at_events: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        return self.sequence_states(
            state_ctx,
            times=times,
            locations=locations,
            lengths=lengths,
            x_field_at_events=x_field_at_events,
        ).payload


class EventModel(ABC, nn.Module):
    """Event-law interface with capability-driven optional methods."""

    def __init__(self):
        ABC.__init__(self)
        nn.Module.__init__(self)

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities()

    @staticmethod
    def _coerce_state_context(state: StateContext | Dict[str, Any]) -> StateContext:
        if isinstance(state, StateContext):
            return state
        if isinstance(state, dict):
            return StateContext(payload=state)
        raise TypeError("EventModel expected `state` to be StateContext or dict payload.")

    @abstractmethod
    def training_loss(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms: Optional[Dict[str, Tensor]] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        """Compute model-specific training loss output.

        Required output key: ``loss``.
        """
        ...

    def eval_nll(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms: Optional[Dict[str, Tensor]] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        """Compute evaluation-time NLL when supported."""
        if not self.capabilities.has_eval_nll:
            raise NotImplementedError("This EventModel does not expose eval_nll().")

        out = self.training_loss(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
            state_regularization_terms=state_regularization_terms,
            x_field_at_events=x_field_at_events,
            marks=marks,
            device=device,
        )
        if "nll" not in out and "loss" in out and self.capabilities.training_objective in {
            "exact_nll",
            "approx_nll",
            "elbo",
        }:
            out["nll"] = out["loss"]
        return out

    def intensity(
        self,
        *,
        state: StateContext,
        query_times: Tensor,
        query_locations: Tensor,
        query_lengths: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Tensor:
        del state, query_times, query_locations, query_lengths, x_field_at_events, marks, device
        raise NotImplementedError("EventModel does not expose intensity().")

    def density(
        self,
        *,
        state: StateContext,
        query_times: Tensor,
        query_locations: Tensor,
        query_lengths: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Tensor:
        del state, query_times, query_locations, query_lengths, x_field_at_events, marks, device
        raise NotImplementedError("EventModel does not expose density().")

    def score(
        self,
        *,
        state: StateContext,
        query_times: Tensor,
        query_locations: Tensor,
        query_lengths: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Tensor:
        del state, query_times, query_locations, query_lengths, x_field_at_events, marks, device
        raise NotImplementedError("EventModel does not expose score().")

    def sample_native(self, **kwargs):
        raise NotImplementedError("EventModel does not expose sample_native().")

    @property
    def surface_query_type(self) -> Literal["intensity", "density", "proxy_kde"]:
        """Semantic type of values returned by ``query_surface()``.

        Derived automatically from capability flags; override only if needed.
        """
        if self.capabilities.has_intensity:
            return "intensity"
        if self.capabilities.has_density:
            return "density"
        return "proxy_kde"

    def query_surface(
        self,
        *,
        state: StateContext,
        grid_times: Tensor,
        grid_locs: Tensor,
        **kwargs,
    ) -> Tensor:
        """Return ``(G,)`` non-negative surface values in **normalized** space.

        Contract
        --------
        - ``grid_times``: ``(G,)`` — all equal to the query t, in normalized space.
        - ``grid_locs``:  ``(G, d)`` — flattened spatial meshgrid, in normalized space.
        - ``state``: ``StateContext`` produced by ``state_model.encode_history()``.
        - Returns: ``(G,)`` non-negative values (intensity, density, or KDE proxy).
        - Normalization/denormalization is the **caller's** responsibility.
        - Models must **not** re-encode history or access raw sequence data here.
        """
        del state, grid_times, grid_locs, kwargs
        raise NotImplementedError(
            f"{type(self).__name__} must implement query_surface(). "
            "Override this method to support surface visualization."
        )

    # ------------------------------------------------------------------
    # Backward-compatible Stage-1 wrappers
    # ------------------------------------------------------------------

    def nll(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext | Dict[str, Any],
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        state_ctx = self._coerce_state_context(state)
        return self.eval_nll(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state_ctx,
            state_regularization_terms=None,
            x_field_at_events=x_field_at_events,
            marks=marks,
            device=device,
        )

    def sequence_nll(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext | Dict[str, Any],
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        return self.nll(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
            x_field_at_events=x_field_at_events,
            marks=marks,
            device=device,
        )
