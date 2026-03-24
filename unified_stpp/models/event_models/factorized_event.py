"""EventModel for factorized STPP baselines.

Owns a temporal process model and a spatial model. Computes the factorized NLL:

    NLL = -(tll + sll) / n_events_total

where:
    tll = Σ_b temporal_logprob[b] / n_events_total   (per-event average)
    sll = Σ_{b,i} spatial_logprob[b,i] * mask[b,i] / n_events_total

Time convention — sequence-relative shifting:
    Parametric temporal models (Poisson, Hawkes, SelfCorrecting) require non-negative
    absolute times. The framework always passes z-score-normalized times which can be
    negative. To bridge this, FactorizedEventModel internally shifts times per-sequence:

        t_shifted[b, i] = times[b, i] - times[b, 0]   ≥ 0 always

    Temporal models receive t_shifted so that:
      - All times are non-negative
      - Time differences are unchanged (shift cancels)
      - The observation window is [t0, t1] in shifted coordinates

    Config parameters t0 and t1 are interpreted in this shifted frame:
      t0 = 0.0 (default) → window starts at first observed event
      t1 = None (default) → last-event convention: window ends at last observed event
      t1 = float          → fixed window end in shifted time (for right-censored data)

    The spatial model (GaussianMixtureSpatialModel) also receives shifted times.
    Its mixing weights use time differences (t_i - t_j), which are shift-invariant, so
    the shift has no semantic effect on the spatial log-probability.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext


class FactorizedEventModel(EventModel):
    """
    Factorized STPP event model.

    Owns:
        temporal_model : module with .logprob(times, locations, mask, t0, t1) → (B,)
        spatial_model  : module with .logprob(times, locations, mask) → (B, T)

    Args:
        temporal_model : parametric temporal process
        spatial_model  : parametric spatial density
        t0             : window start in sequence-relative (shifted) time (default 0.0 = first event)
        t1             : window end in shifted time; None → last-event convention (see module docstring)
    """

    def __init__(
        self,
        *,
        temporal_model: nn.Module,
        spatial_model: nn.Module,
        t0: float = 0.0,
        t1: Optional[float] = None,
    ):
        super().__init__()
        self.temporal_model = temporal_model
        self.spatial_model = spatial_model
        self.t0 = t0
        self.t1 = t1

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="exact_nll",
            has_eval_nll=True,
            has_intensity=True,
            has_density=True,
            has_score=False,
            has_native_sampler=False,
            exposes_eventwise_terms=True,
        )

    def _extract_history(self, state: StateContext, device):
        """Extract history tensors from state and apply sequence-relative time shift.

        Returns (times_shifted, locations, mask) all on device, with B=1.
        The shift maps times[:, 0] → 0 so temporal models receive non-negative inputs.
        """
        times     = state.payload["times"].to(device)      # (B, T)
        locations = state.payload["locations"].to(device)  # (B, T, D)
        lengths   = state.payload["lengths"].to(device)    # (B,)

        B, T = times.shape
        idx  = torch.arange(T, device=device)
        mask = (idx.unsqueeze(0) < lengths.unsqueeze(1)).float()  # (B, T)

        t_shift = times[:, 0:1] if T > 0 else torch.zeros(B, 1, device=device, dtype=times.dtype)
        times_shifted  = (times - t_shift).clamp(min=0.0)        # (B, T)

        return times_shifted, locations, mask, t_shift

    def intensity(
        self,
        *,
        state: StateContext,
        query_times: Tensor,
        query_locations: Tensor,
        query_lengths=None,
        x_field_at_events=None,
        marks=None,
        device=None,
    ) -> Tensor:
        """Conditional intensity λ*(t_query, s_query | H) = f*(t|H) × p(s|t,H).

        The intensity is the product of:
          - f*(t|H): exact analytic temporal intensity from the parametric process
          - p(s|t,H): exact parametric spatial density from the spatial model

        Args:
            state          : StateContext with payload {times, locations, lengths}
            query_times    : (M, 1) normalized query times
            query_locations: (M, D) normalized query locations
        Returns:
            (M,) intensity values in normalized space
        """
        if device is None:
            device = query_times.device

        times_shifted, locations, mask, t_shift = self._extract_history(state, device)
        B = times_shifted.shape[0]

        # Shift query times by the same offset used for history
        t_q_shifted = (query_times.squeeze(-1) - t_shift.squeeze()).clamp(min=0.0)  # (M,)

        M = t_q_shifted.shape[0]

        # Expand history from (B=1, T) → (M, T) for vectorized evaluation
        history_times_exp = times_shifted.expand(M, -1)     # (M, T)
        history_locs_exp  = locations.expand(M, -1, -1)     # (M, T, D)
        history_mask_exp  = mask.expand(M, -1)              # (M, T)

        # Temporal intensity f*(t | H)
        lambda_t = self.temporal_model.intensity_at(
            t_q_shifted, history_times_exp, history_mask_exp
        )  # (M,)

        # Spatial log-density log p(s | t, H)
        log_p_s = self.spatial_model.log_spatial_density_at(
            t_q_shifted, query_locations, history_times_exp, history_locs_exp, history_mask_exp
        )  # (M,)

        return lambda_t * torch.exp(log_p_s)  # (M,)

    def density(
        self,
        *,
        state: StateContext,
        query_times: Tensor,
        query_locations: Tensor,
        query_lengths=None,
        x_field_at_events=None,
        marks=None,
        device=None,
    ) -> Tensor:
        """Conditional spatial density p(s_query | t_query, H).

        Args:
            state          : StateContext with payload {times, locations, lengths}
            query_times    : (M, 1) normalized query times
            query_locations: (M, D) normalized query locations
        Returns:
            (M,) density values in normalized space
        """
        if device is None:
            device = query_times.device

        times_shifted, locations, mask, t_shift = self._extract_history(state, device)

        t_q_shifted = (query_times.squeeze(-1) - t_shift.squeeze()).clamp(min=0.0)  # (M,)
        M = t_q_shifted.shape[0]

        history_times_exp = times_shifted.expand(M, -1)
        history_locs_exp  = locations.expand(M, -1, -1)
        history_mask_exp  = mask.expand(M, -1)

        log_p_s = self.spatial_model.log_spatial_density_at(
            t_q_shifted, query_locations, history_times_exp, history_locs_exp, history_mask_exp
        )  # (M,)
        return torch.exp(log_p_s)  # (M,)

    def query_surface(
        self,
        *,
        state: StateContext,
        grid_times: "Tensor",
        grid_locs: "Tensor",
        **kwargs,
    ) -> "Tensor":
        """Surface query contract: routes to intensity() for factorized models."""
        t = grid_times.unsqueeze(-1) if grid_times.ndim == 1 else grid_times
        return self.intensity(
            state=state,
            query_times=t,
            query_locations=grid_locs,
            device=grid_times.device,
        )

    def _compute(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        device,
    ) -> Dict[str, Tensor]:
        B, T = times.shape
        n_idx = torch.arange(T, device=device)
        mask = (n_idx.unsqueeze(0) < lengths.unsqueeze(1)).float()  # (B, T)

        # Sequence-relative time shift: ensure all times are non-negative.
        # Parametric temporal models require t ≥ 0; z-score normalized times can be
        # negative. Shifting by times[:,0] maps the first event to t=0 in each sequence.
        # Time differences (used by Hawkes and GMM) are unchanged by this shift.
        t_shift = times[:, 0:1]                            # (B, 1) — first event time
        times_shifted = (times - t_shift).clamp(min=0.0)  # (B, T) — sequence-relative

        # Observation window in shifted coordinates
        t0_tensor = torch.full((B,), self.t0, device=device, dtype=times.dtype)

        if self.t1 is not None:
            # Fixed window end from config (in shifted time)
            t1_tensor = torch.full((B,), self.t1, device=device, dtype=times.dtype)
        else:
            # Last-event convention: window ends at last observed event (shifted)
            last_idx = (lengths - 1).clamp(min=0)
            t1_tensor = times_shifted[torch.arange(B, device=device), last_idx]  # (B,)
        # Temporal log-prob: (B,)
        tll_seq = self.temporal_model.logprob(
            times_shifted, locations, mask, t0_tensor, t1_tensor
        )

        # Spatial log-prob: (B, T) — uses shifted times (time diffs are shift-invariant)
        sll_mat = self.spatial_model.logprob(times_shifted, locations, mask)

        n_events_total = mask.sum().clamp(min=1)
        tll = tll_seq.sum() / n_events_total
        sll = (sll_mat * mask).sum() / n_events_total
        mean_nll = -(tll + sll)

        n_per_seq = mask.sum(dim=-1).clamp(min=1)
        nll_per_event = -(tll_seq / n_per_seq + (sll_mat * mask).sum(dim=-1) / n_per_seq)

        return {
            "loss": mean_nll,
            "nll": mean_nll,
            "nll_per_event": nll_per_event,
            "total_events": mask.sum(),
            "tll": tll,
            "sll": sll,
            "sll_matrix": sll_mat,
            "mask": mask,
        }

    def training_loss(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms=None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        del state, state_regularization_terms, x_field_at_events, marks
        if device is None:
            device = times.device
        return self._compute(times=times, locations=locations, lengths=lengths, device=device)

    def eval_nll(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms=None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        return self.training_loss(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
            state_regularization_terms=state_regularization_terms,
            x_field_at_events=x_field_at_events,
            marks=marks,
            device=device,
        )
