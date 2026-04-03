"""Capability-driven sampling and intensity utilities for coarse STPP models."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


def _require_state_event_models(model):
    state_model = getattr(model, "state_model", None)
    event_model = getattr(model, "event_model", None)
    if state_model is None or event_model is None:
        raise RuntimeError(
            "Model must expose both state_model and event_model for sampling utilities."
        )
    return state_model, event_model


def get_event_model_capabilities(model):
    """Return EventModel capabilities when available, else None."""
    event_model = getattr(model, "event_model", None)
    if event_model is None:
        return None
    return getattr(event_model, "capabilities", None)


def supports_native_sampling(model) -> bool:
    """Whether model exposes native event-model sampling."""
    caps = get_event_model_capabilities(model)
    return bool(caps is not None and getattr(caps, "has_native_sampler", False))


def supports_intensity_query(model) -> bool:
    """Whether model exposes an event-model intensity interface."""
    caps = get_event_model_capabilities(model)
    return bool(caps is not None and getattr(caps, "has_intensity", False))


def supports_density_query(model) -> bool:
    """Whether model exposes an event-model density interface."""
    caps = get_event_model_capabilities(model)
    return bool(caps is not None and getattr(caps, "has_density", False))


def thinning_sample(
    intensity_fn: Callable,
    t_start: Tensor,
    t_max: float,
    spatial_bounds: Tuple[Tensor, Tensor],
    lambda_bar: float = 10.0,
    max_events: int = 100,
    adaptive: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Ogata's thinning algorithm for spatiotemporal point processes.

    Works with any callable that evaluates lambda*(t, s).
    """
    B = t_start.shape[0]
    device = t_start.device
    d = spatial_bounds[0].shape[0]
    s_min, s_max = spatial_bounds

    times = torch.full((B, max_events), t_max + 1, device=device)
    locations = torch.zeros(B, max_events, d, device=device)
    counts = torch.zeros(B, dtype=torch.long, device=device)

    vol = (s_max - s_min).prod().item()

    t_current = t_start.squeeze(-1).clone()
    lam_bar = torch.full((B,), lambda_bar, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)

    for _event_idx in range(max_events):
        if not active.any():
            break

        rate = (lam_bar * vol).clamp(min=1e-6)
        dt = torch.distributions.Exponential(rate).sample()
        t_proposal = t_current + dt

        still_active = active & (t_proposal < t_max)
        if not still_active.any():
            active = still_active
            break

        u_s = torch.rand(B, d, device=device)
        s_proposal = s_min.unsqueeze(0) + u_s * (s_max - s_min).unsqueeze(0)

        with torch.no_grad():
            lam = intensity_fn(t_proposal.unsqueeze(-1), s_proposal)

        u = torch.rand(B, device=device)
        accept = still_active & (u < lam / lam_bar.clamp(min=1e-8))

        for b in range(B):
            if accept[b]:
                idx = counts[b].item()
                if idx < max_events:
                    times[b, idx] = t_proposal[b]
                    locations[b, idx] = s_proposal[b]
                    counts[b] += 1

        t_current = torch.where(still_active, t_proposal, t_current)

        if adaptive:
            lam_bar = torch.where(
                still_active,
                torch.maximum(lam_bar, lam * 1.5),
                lam_bar,
            )

        active = still_active & (counts < max_events)

    return times, locations, counts


# class IntensityEvaluator:
#     """Evaluate event-model intensity via StateModel/EventModel interfaces."""

#     def __init__(
#         self,
#         model,
#         *,
#         history_times: Tensor,
#         history_locations: Tensor,
#         history_lengths: Tensor,
#         marks: Optional[Tensor] = None,
#         x_event: Optional[Tensor] = None,
#         x_field_at_events: Optional[Tensor] = None,
#     ):
#         self.model = model
#         self.state_model, self.event_model = _require_state_event_models(model)

#         if history_times.ndim != 2:
#             raise ValueError("history_times must have shape (B, N).")
#         if history_locations.ndim != 3:
#             raise ValueError("history_locations must have shape (B, N, d).")
#         if history_lengths.ndim != 1:
#             raise ValueError("history_lengths must have shape (B,).")

#         self.history_times = history_times
#         self.history_locations = history_locations
#         self.history_lengths = history_lengths
#         self.marks = marks
#         self.x_event = x_event
#         self.x_field_at_events = x_field_at_events

#     def _encode_state(self):
#         return self.state_model.encode_history(
#             times=self.history_times,
#             locations=self.history_locations,
#             lengths=self.history_lengths,
#             marks=self.marks,
#             x_event=self.x_event,
#             x_field_at_events=self.x_field_at_events,
#         )

#     def intensity(
#         self,
#         query_times: Tensor,
#         query_locations: Tensor,
#         x_field_at_events: Optional[Tensor] = None,
#     ) -> Tensor:
#         """Evaluate lambda*(t, s | history) for query batches."""
#         if not supports_intensity_query(self.model):
#             name = type(self.event_model).__name__
#             raise NotImplementedError(
#                 f"EventModel '{name}' does not advertise has_intensity=True."
#             )

#         if query_times.ndim == 1:
#             query_times = query_times.unsqueeze(-1)
#         if query_times.ndim != 2 or query_times.shape[-1] != 1:
#             raise ValueError("query_times must have shape (B, 1).")

#         if query_locations.ndim == 3 and query_locations.shape[1] == 1:
#             query_locations = query_locations.squeeze(1)
#         if query_locations.ndim != 2:
#             raise ValueError("query_locations must have shape (B, d).")
#         if query_locations.shape[0] != query_times.shape[0]:
#             raise ValueError("query_times and query_locations batch sizes must match.")

#         state_ctx = self._encode_state()
#         query_lengths = torch.ones(
#             query_times.shape[0], dtype=torch.long, device=query_times.device
#         )
#         queried_state = self.state_model.query_state(
#             state_ctx,
#             times=query_times,
#             locations=query_locations.unsqueeze(1),
#             lengths=query_lengths,
#             x_field_at_events=x_field_at_events,
#         )

#         return self.event_model.intensity(
#             state=queried_state,
#             query_times=query_times,
#             query_locations=query_locations,
#             query_lengths=query_lengths,
#             x_field_at_events=x_field_at_events,
#             marks=self.marks,
#             device=query_times.device,
#         )

#     @torch.no_grad()
#     def intensity_grid(
#         self,
#         t: float,
#         s_min: Tensor,
#         s_max: Tensor,
#         n_grid: int = 50,
#         x_field_fn=None,
#     ) -> Tuple[Tensor, Tensor, Tensor]:
#         """Evaluate intensity on a spatial grid at fixed time t."""
#         if self.history_times.shape[0] != 1:
#             raise ValueError("intensity_grid currently requires history batch size B=1.")

#         device = self.history_times.device
#         d = s_min.shape[0]
#         if d not in (1, 2):
#             raise ValueError("Grid visualization supports spatial_dim in {1, 2}.")

#         if d == 2:
#             x = torch.linspace(s_min[0].item(), s_max[0].item(), n_grid, device=device)
#             y = torch.linspace(s_min[1].item(), s_max[1].item(), n_grid, device=device)
#             xx, yy = torch.meshgrid(x, y, indexing="ij")
#             s_flat = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
#             n_pts = s_flat.shape[0]

#             t_tensor = torch.full((n_pts, 1), t, device=device)
#             x_field = None
#             if x_field_fn is not None:
#                 x_field = x_field_fn(t_tensor, s_flat)
#             lam = self.intensity(t_tensor, s_flat, x_field)
#             return x, y, lam.reshape(n_grid, n_grid)

#         x = torch.linspace(s_min[0].item(), s_max[0].item(), n_grid, device=device)
#         s_flat = x.unsqueeze(-1)
#         t_tensor = torch.full((n_grid, 1), t, device=device)
#         lam = self.intensity(t_tensor, s_flat)
#         return x, None, lam
