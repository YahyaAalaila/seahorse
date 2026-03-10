"""
EventModel for sequence-coupled faithful Neural STPP spatial decoders.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from ..abstractions import EventModel


SpatialSeqNLLFn = Callable[..., Tensor]
SpatialRegFn = Callable[[], Any]


class NeuralSTPPSequenceEventModel(EventModel):
    """
    EventModel for JumpCNFSpatial / SelfAttentiveCNFSpatial over backbone states.

    Output keeps event-wise tensors explicit for future objectives:
      - ``nll_matrix``
      - ``mask``
      - ``temporal_nll_matrix``
      - ``spatial_nll_matrix``
      - explicit regularization terms
    """

    def __init__(
        self,
        *,
        spatial_sequence_nll_fn: SpatialSeqNLLFn,
        spatial_regularization_fn: SpatialRegFn,
    ):
        super().__init__()
        self._spatial_sequence_nll_fn = spatial_sequence_nll_fn
        self._spatial_regularization_fn = spatial_regularization_fn

    @staticmethod
    def _as_like(x: Any, ref: Tensor) -> Tensor:
        if isinstance(x, Tensor):
            if x.device == ref.device and x.dtype == ref.dtype:
                return x
            return x.to(device=ref.device, dtype=ref.dtype)
        return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _get_state_term(state: Dict[str, Any], key: str) -> Tensor:
        val = state.get(key)
        if val is None:
            raise ValueError(f"NeuralSTPPSequenceEventModel requires state['{key}'].")
        return val

    def _compute(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: Dict[str, Any],
        device,
    ) -> Dict[str, Tensor]:
        B = times.shape[0]
        max_len = int(lengths.max().item())
        if max_len < 2:
            zeros_per_seq = torch.zeros(B, device=device)
            zero_scalar = torch.tensor(0.0, device=device)
            empty_l = 0
            return {
                "nll": zero_scalar,
                "nll_per_event": zeros_per_seq,
                "total_events": zero_scalar,
                "base_mean_nll": zero_scalar,
                "nll_matrix": torch.zeros(B, empty_l, device=device),
                "mask": torch.zeros(B, empty_l, device=device),
                "temporal_nll_matrix": torch.zeros(B, empty_l, device=device),
                "spatial_nll_matrix": torch.zeros(B, empty_l, device=device),
                "temporal_energy_reg": zero_scalar,
                "spatial_reg": zero_scalar,
                "regularization_total": zero_scalar,
            }

        L = max_len - 1
        temporal_nll = self._get_state_term(state, "temporal_nll_matrix")[:, :L]
        z_seq = self._get_state_term(state, "z_seq")[:, :L, :]
        temporal_energy_reg = self._get_state_term(state, "temporal_energy_reg")

        t_seq = times[:, 1 : 1 + L].unsqueeze(-1)     # (B, L, 1)
        s_seq = locations[:, 1 : 1 + L, :]            # (B, L, d)
        t_prev_seq = times[:, :L].unsqueeze(-1)       # (B, L, 1)
        n_idx = torch.arange(L, device=device)
        mask = (n_idx.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()  # (B, L)

        spatial_nll = self._spatial_sequence_nll_fn(
            z_seq=z_seq,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        )  # (B, L)

        nll_matrix = temporal_nll + spatial_nll
        nll_masked = nll_matrix * mask
        total_nll = nll_masked.sum(dim=1)      # (B,)
        n_events = mask.sum(dim=1)             # (B,)
        base_mean_nll = total_nll.sum() / n_events.sum().clamp(min=1)

        temporal_energy_reg_t = self._as_like(temporal_energy_reg, base_mean_nll)
        spatial_reg_t = self._as_like(self._spatial_regularization_fn(), base_mean_nll)
        regularization_total = temporal_energy_reg_t + spatial_reg_t
        mean_nll = base_mean_nll + regularization_total

        return {
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
        del x_field_at_events, marks
        if device is None:
            device = times.device
        return self._compute(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
            device=device,
        )

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
        return self.nll(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
            x_field_at_events=x_field_at_events,
            marks=marks,
            device=device,
        )
