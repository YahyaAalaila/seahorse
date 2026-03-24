"""
Unified STPP model with coarse StateModel/EventModel dispatch.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .abstractions import EventModel, StateModel


class UnifiedSTPP(nn.Module):
    """Top-level model container for coarse state/event execution."""

    def __init__(
        self,
        state_model: StateModel,
        event_model: EventModel,
        *,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_model = state_model
        self.event_model = event_model
        self.hidden_dim = hidden_dim

        self._debug_nstpp = os.getenv("UNIFIED_STPP_DEBUG_NSTPP", "0") == "1"
        self._debug_nstpp_max_calls = max(
            1, int(os.getenv("UNIFIED_STPP_DEBUG_NSTPP_MAX_CALLS", "10"))
        )
        self._debug_nstpp_calls = 0

    def forward(
        self,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor] = None,
        x_event: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
        x_field_fn=None,
    ) -> Dict[str, Tensor]:
        """Compute NLL for a batch of sequences."""
        del x_field_fn
        return self._forward_state_event(
            times=times,
            locations=locations,
            lengths=lengths,
            marks=marks,
            x_event=x_event,
            x_field_at_events=x_field_at_events,
            device=times.device,
        )

    def _forward_state_event(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor],
        x_event: Optional[Tensor],
        x_field_at_events: Optional[Tensor],
        device,
    ) -> Dict[str, Any]:
        state_ctx = self.state_model.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
            marks=marks,
            x_event=x_event,
            x_field_at_events=x_field_at_events,
        )

        state_for_event = self.state_model.sequence_states(
            state_ctx,
            times=times,
            locations=locations,
            lengths=lengths,
            x_field_at_events=x_field_at_events,
        )
        state_regularization_terms = self.state_model.regularization_terms(
            state_ctx,
            times=times,
            locations=locations,
            lengths=lengths,
            marks=marks,
        )

        result = self.event_model.training_loss(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state_for_event,
            state_regularization_terms=state_regularization_terms,
            x_field_at_events=x_field_at_events,
            marks=marks,
            device=device,
        )

        if "nll" not in result and "loss" in result:
            result["nll"] = result["loss"]

        debug_this_call = (
            self._debug_nstpp and self._debug_nstpp_calls < self._debug_nstpp_max_calls
        )
        if debug_this_call:
            mask = result.get("mask")
            temporal_nll = result.get("temporal_nll_matrix")
            spatial_nll = result.get("spatial_nll_matrix")
            nll_matrix = result.get("nll_matrix")
            if (
                isinstance(mask, Tensor)
                and isinstance(temporal_nll, Tensor)
                and isinstance(spatial_nll, Tensor)
                and isinstance(nll_matrix, Tensor)
                and mask.numel() > 0
                and (mask > 0).any()
            ):
                valid = mask > 0
                t_vals = temporal_nll.detach()[valid]
                s_vals = spatial_nll.detach()[valid]
                j_vals = nll_matrix.detach()[valid]
                energy_reg = result.get("temporal_energy_reg", 0.0)
                spatial_reg = result.get("spatial_reg", 0.0)
                print(
                    "[NSTPP-DEBUG][loss] "
                    f"temporal_nll(min/mean/max)="
                    f"{t_vals.min().item():.6f}/{t_vals.mean().item():.6f}/{t_vals.max().item():.6f} "
                    f"spatial_nll(min/mean/max)="
                    f"{s_vals.min().item():.6f}/{s_vals.mean().item():.6f}/{s_vals.max().item():.6f} "
                    f"joint_nll(min/mean/max)="
                    f"{j_vals.min().item():.6f}/{j_vals.mean().item():.6f}/{j_vals.max().item():.6f} "
                    f"energy_reg={float(torch.as_tensor(energy_reg).detach().item()):.6f} "
                    f"spatial_reg={float(torch.as_tensor(spatial_reg).detach().item()):.6f}"
                )
                self._debug_nstpp_calls += 1

        kl_loss = getattr(state_ctx, "kl_loss", None)
        if kl_loss is not None and "kl_loss" not in result:
            result["kl_loss"] = kl_loss

        if state_regularization_terms:
            result["state_regularization_terms"] = state_regularization_terms
            for name, term in state_regularization_terms.items():
                result.setdefault(name, term)
        return result
