"""EventModel for sequence-coupled faithful Neural STPP spatial decoders."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext


SpatialSeqNLLFn = Callable[..., Tensor]
SpatialRegFn = Callable[[], Any]


class NeuralSTPPSequenceEventModel(EventModel):
    """EventModel for JumpCNFSpatial / SelfAttentiveCNFSpatial over backbone states."""

    def __init__(
        self,
        *,
        spatial_sequence_nll_fn: SpatialSeqNLLFn,
        spatial_regularization_fn: SpatialRegFn,
    ):
        super().__init__()
        self._spatial_sequence_nll_fn = spatial_sequence_nll_fn
        self._spatial_regularization_fn = spatial_regularization_fn

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="exact_nll",
            has_eval_nll=True,
            has_intensity=False,
            has_density=False,
            has_score=False,
            has_native_sampler=False,
            exposes_eventwise_terms=True,
        )

    @staticmethod
    def _as_like(x: Any, ref: Tensor) -> Tensor:
        if isinstance(x, Tensor):
            if x.device == ref.device and x.dtype == ref.dtype:
                return x
            return x.to(device=ref.device, dtype=ref.dtype)
        return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _get_state_term(state_ctx: StateContext, key: str) -> Tensor:
        val = state_ctx.payload.get(key)
        if val is None:
            raise ValueError(f"NeuralSTPPSequenceEventModel requires state['{key}'].")
        return val

    def _compute(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state_ctx: StateContext,
        state_regularization_terms: Optional[Dict[str, Tensor]],
        device,
    ) -> Dict[str, Tensor]:
        bsz = times.shape[0]
        max_len = int(lengths.max().item())
        if max_len < 2:
            zeros_per_seq = torch.zeros(bsz, device=device)
            zero_scalar = torch.tensor(0.0, device=device)
            empty_l = 0
            return {
                "loss": zero_scalar,
                "nll": zero_scalar,
                "nll_per_event": zeros_per_seq,
                "total_events": zero_scalar,
                "base_mean_nll": zero_scalar,
                "nll_matrix": torch.zeros(bsz, empty_l, device=device),
                "mask": torch.zeros(bsz, empty_l, device=device),
                "temporal_nll_matrix": torch.zeros(bsz, empty_l, device=device),
                "spatial_nll_matrix": torch.zeros(bsz, empty_l, device=device),
                "temporal_energy_reg": zero_scalar,
                "spatial_reg": zero_scalar,
                "regularization_total": zero_scalar,
            }

        l_steps = max_len - 1
        temporal_nll = self._get_state_term(state_ctx, "temporal_nll_matrix")[:, :l_steps]
        z_seq = self._get_state_term(state_ctx, "z_seq")[:, :l_steps, :]

        t_seq = times[:, 1 : 1 + l_steps].unsqueeze(-1)
        s_seq = locations[:, 1 : 1 + l_steps, :]
        t_prev_seq = times[:, :l_steps].unsqueeze(-1)
        n_idx = torch.arange(l_steps, device=device)
        mask = (n_idx.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()

        spatial_nll = self._spatial_sequence_nll_fn(
            z_seq=z_seq,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        )

        nll_matrix = temporal_nll + spatial_nll
        nll_masked = nll_matrix * mask
        total_nll = nll_masked.sum(dim=1)
        n_events = mask.sum(dim=1)
        base_mean_nll = total_nll.sum() / n_events.sum().clamp(min=1)

        temporal_energy_reg_val = None
        if state_regularization_terms is not None:
            temporal_energy_reg_val = state_regularization_terms.get("temporal_energy_reg")
        if temporal_energy_reg_val is None:
            temporal_energy_reg_val = state_ctx.payload.get("temporal_energy_reg", 0.0)

        temporal_energy_reg_t = self._as_like(temporal_energy_reg_val, base_mean_nll)
        spatial_reg_t = self._as_like(self._spatial_regularization_fn(), base_mean_nll)
        regularization_total = temporal_energy_reg_t + spatial_reg_t
        mean_nll = base_mean_nll + regularization_total

        return {
            "loss": mean_nll,
            "nll": mean_nll,
            "nll_per_event": total_nll / n_events.clamp(min=1),
            "total_events": n_events.sum(),
            "base_mean_nll": base_mean_nll,
            "nll_matrix": nll_matrix,
            "mask": mask,
            "temporal_nll_matrix": temporal_nll,
            "spatial_nll_matrix": spatial_nll,
            "temporal_energy_reg": temporal_energy_reg_t,
            "spatial_reg": spatial_reg_t,
            "regularization_total": regularization_total,
        }

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
        del x_field_at_events, marks
        if device is None:
            device = times.device
        return self._compute(
            times=times,
            locations=locations,
            lengths=lengths,
            state_ctx=state,
            state_regularization_terms=state_regularization_terms,
            device=device,
        )

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
