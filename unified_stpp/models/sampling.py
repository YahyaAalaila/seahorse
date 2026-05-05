"""Capability-driven sampling and intensity utilities for coarse STPP models."""

from __future__ import annotations

from typing import Callable, Tuple

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
