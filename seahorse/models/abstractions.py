"""Coarse state/event abstractions with capability declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from seahorse.data.transforms import transform_from_spec


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
    """Capability declarations for event models.

    Three-layer metric architecture
    --------------------------------
    Layer 1 — Objective: what the model trains on; drives train/val logging and
      checkpoint selection.  Fields: training_objective, metric_key,
      objective_description, objective_includes_regularization.

    Layer 2 — NLL: a benchmark-facing likelihood quantity independent of the
      training objective.  Fields: nll_kind, nll_description, nll_footnote.

    Layer 3 — Sampling-based eval metrics: RMSE, MAE, etc. computed post-training
      from the learned model.  Hook: has_native_sampler.
      Future field: supported_eval_metrics: tuple[str, ...] = ()
    """

    # ── Layer 1: Objective ─────────────────────────────────────────────
    training_objective: str = "nll"
    # Formal name: "nll", "elbo", "score_matching", "hybrid"

    metric_key: str = "nll"
    # Display slug for train/val logging: f"train/{metric_key}", f"val/{metric_key}"

    objective_description: str = ""
    # Human-readable: "exact NLL", "variational ELBO (1-step)", "denoising score matching"

    objective_includes_regularization: bool = False
    # True when training loss ≠ pure objective metric (e.g. NeuralSTPP adds energy_reg)

    # ── Layer 2: NLL ───────────────────────────────────────────────────
    nll_kind: Literal["exact", "approx", "none"] = "exact"
    # "exact"  — eval_nll() = training_loss() = exact NLL; no separate path needed.
    # "approx" — model has a separate eval_nll() returning an approximation.
    # "none"   — model has no NLL eval capability.

    nll_description: str = "exact NLL/event (normalized space)"
    # Describes what test/nll measures (shown in benchmark reports).

    nll_footnote: str = ""
    # Superscript in benchmark tables for the NLL column (e.g. "‡ approx NLL").

    supports_raw_reporting: bool = False
    # True when the family can convert native-space exact NLL into raw/original space.

    raw_nll_description: str = ""
    # Human-readable description of raw-space ``test_nll`` when reported.

    # ── Layer 3: Sampling-based eval metrics ──────────────────────────
    has_native_sampler: bool = False
    # Hook: when True, sampling-based metrics (RMSE, MAE, …) can be computed.
    # Future: supported_eval_metrics: tuple[str, ...] = ()

    # ── Other capability flags ─────────────────────────────────────────
    has_intensity: bool = False
    has_density: bool = False
    has_score: bool = False
    exposes_eventwise_terms: bool = False

    # ── Backward-compat derived property ──────────────────────────────
    @property
    def has_eval_nll(self) -> bool:
        """True when the model can compute an NLL (exact or approx)."""
        return self.nll_kind != "none"


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

    def encode_sampling_history(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor] = None,
        x_event: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
    ) -> StateContext:
        """Optional fast path for post-hoc sampling / intensity querying.

        Default behavior falls back to the full training-time history encoding.
        Exact-intensity presets can override this to build only the query-side
        state needed by ``event_model.intensity(...)``.
        """
        return self.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
            marks=marks,
            x_event=x_event,
            x_field_at_events=x_field_at_events,
        )

    def append_sampling_event(
        self,
        state_ctx: StateContext,
        *,
        event_time_raw: Tensor,
        event_location_raw: Tensor,
    ) -> StateContext:
        """Optional incremental sampling-state update hook.

        Presets that override ``encode_sampling_history(...)`` can also override
        this method to append one accepted sampled event without rebuilding the
        full history state from scratch.
        """
        del state_ctx, event_time_raw, event_location_raw
        raise NotImplementedError(
            "StateModel does not implement append_sampling_event()."
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

    @staticmethod
    def _as_scalar_tensor(value: Tensor | float | int, *, ref: Tensor | None = None) -> Tensor:
        if isinstance(value, Tensor):
            return value.reshape(())
        if ref is None:
            return torch.tensor(float(value), dtype=torch.float32)
        return ref.new_tensor(float(value))

    def _reporting_transform(self, state: StateContext | Dict[str, Any]):
        state_ctx = self._coerce_state_context(state)
        spec = state_ctx.payload.get("input_transform")
        return transform_from_spec(spec if isinstance(spec, dict) else None)

    def raw_reporting_metrics(
        self,
        *,
        state: StateContext | Dict[str, Any],
        nll: Tensor | float,
        temporal_nll: Tensor | float | None = None,
        spatial_nll: Tensor | float | None = None,
        total_events: Tensor | float | None = None,
    ) -> Dict[str, float]:
        """Convert exact native-space NLL terms into raw/original-space metrics.

        Families opt in via ``capabilities.supports_raw_reporting`` and by
        storing a serialized ``input_transform`` artifact in the state payload.
        Approximate-NLL families intentionally return no correction here.
        """
        caps = self.capabilities
        if caps.nll_kind != "exact" or not caps.supports_raw_reporting:
            return {}
        if total_events is not None:
            total_events_t = self._as_scalar_tensor(total_events)
            if float(total_events_t.item()) <= 0.0:
                out = {"raw_space_nll": 0.0, "raw_reporting_correction": 0.0}
                if temporal_nll is not None:
                    out["raw_space_temporal_nll"] = 0.0
                if spatial_nll is not None:
                    out["raw_space_spatial_nll"] = 0.0
                return out
        transform = self._reporting_transform(state)
        if transform is None or not bool(getattr(transform, "supports_raw_reporting", False)):
            return {}

        nll_t = self._as_scalar_tensor(nll)
        corr = transform.reporting_correction(ref=nll_t)
        out: Dict[str, float] = {
            "raw_space_nll": float((nll_t + corr).item()),
            "raw_reporting_correction": float(corr.item()),
        }
        if temporal_nll is not None:
            t_nll = self._as_scalar_tensor(temporal_nll, ref=nll_t)
            t_corr = transform.temporal_reporting_correction(ref=nll_t)
            out["raw_space_temporal_nll"] = float((t_nll + t_corr).item())
        if spatial_nll is not None:
            s_nll = self._as_scalar_tensor(spatial_nll, ref=nll_t)
            s_corr = transform.spatial_reporting_correction(ref=nll_t)
            out["raw_space_spatial_nll"] = float((s_nll + s_corr).item())
        return out

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
        """Compute evaluation-time NLL when supported (Layer 2).

        For exact-NLL models (nll_kind="exact") this delegates to training_loss().
        For approx-NLL models (nll_kind="approx") subclasses override this method.
        For nll_kind="none" models this raises NotImplementedError.
        """
        if self.capabilities.nll_kind == "none":
            raise NotImplementedError("This EventModel has no NLL eval capability (nll_kind='none').")

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
        if "nll" not in out and "loss" in out:
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
