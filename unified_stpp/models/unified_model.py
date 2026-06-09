"""
Unified STPP model with coarse StateModel/EventModel dispatch.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .abstractions import EventModel, StateModel


@dataclasses.dataclass
class LossResult:
    """Structured output of :meth:`UnifiedSTPP.compute_loss`.

    ``loss``         — scalar to back-propagate (primary training objective).
    ``nll``          — pure per-event NLL (selection metric + benchmark metric).
                       For regularized models (NeuralSTPP) this is the base NLL
                       without regularization; ``loss ≥ nll`` for those models.
    ``total_events`` — event count used as the ``batch_size`` weight for
                       Lightning's per-epoch metric averaging.
    ``kl``           — KL divergence term when the model emits ``kl_loss``
                       (e.g. DeepSTPP in VAE mode); ``None`` otherwise.
                       The Lightning module applies training-config-level
                       weighting (``vae_beta``) to this term if present.
    ``aux_terms``    — ``state_regularization_terms`` dict (may be empty).
    ``temporal_nll`` — mean temporal NLL/event (scalar float); ``None`` when
                       the model does not expose a temporal/spatial breakdown.
    ``spatial_nll``  — mean spatial NLL/event (scalar float); ``None`` when not
                       available or when the model uses a joint (non-factored) NLL.
    ``extra_metrics`` — additional scalar metrics that should survive into test-time
                       logging and run artifacts (for example diffusion per-dim
                       diagnostics during verification).
    """
    loss:         Tensor
    nll:          Tensor
    total_events: Tensor
    kl:           Optional[Tensor]
    aux_terms:    Dict[str, Any]
    temporal_nll: Optional[float] = None
    spatial_nll:  Optional[float] = None
    extra_metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)


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

    def eval_forward(
        self,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor] = None,
        x_event: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Like forward(), but routes through event_model.eval_nll() for test reporting.

        For exact models, eval_nll() defaults to training_loss() — identical to forward().
        For SMASH and Diffusion, this runs the respective approximate NLL eval path.
        """
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

        result = self.event_model.eval_nll(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state_for_event,
            state_regularization_terms=state_regularization_terms,
            x_field_at_events=x_field_at_events,
            marks=marks,
            device=times.device,
        )

        if "nll" not in result and "loss" in result:
            result["nll"] = result["loss"]

        kl_loss = getattr(state_ctx, "kl_loss", None)
        if kl_loss is not None and "kl_loss" not in result:
            result["kl_loss"] = kl_loss

        if state_regularization_terms:
            result["state_regularization_terms"] = state_regularization_terms
        return result

    def compute_loss(self, output: dict) -> LossResult:
        """Extract structured loss components from a :meth:`forward` output dict.

        Generic across all model families — takes no training-config parameters.
        KL weighting (``vae_beta``) is the caller's responsibility.
        """
        nll = output["nll"]
        t_nll = output.get("temporal_nll")
        s_nll = output.get("spatial_nll")
        return LossResult(
            loss=output.get("loss", nll),
            nll=nll,
            total_events=output["total_events"],
            kl=output.get("kl_loss"),
            aux_terms=output.get("state_regularization_terms") or {},
            temporal_nll=float(t_nll) if t_nll is not None else None,
            spatial_nll=float(s_nll) if s_nll is not None else None,
            extra_metrics=output.get("extra_metrics") or {},
        )

    def project_parameters(self) -> None:
        """Apply optional post-step parameter projections exposed by submodules."""
        for module in (self.state_model, self.event_model):
            projector = getattr(module, "project_parameters", None)
            if callable(projector):
                projector()
                continue
            compatibility_projector = getattr(module, "project", None)
            if callable(compatibility_projector):
                compatibility_projector()
