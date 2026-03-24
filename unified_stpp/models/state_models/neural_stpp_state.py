"""StateModel for NeuralSTPP — owns JumpOdeIntensityProcess."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel


class NeuralSTPPStateModel(StateModel):
    """NeuralSTPP state model.  Owns JumpOdeIntensityProcess."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        spatial_dim: int,
        **backbone_extra,
    ):
        super().__init__()
        from ..temporal_models.jump_ode_intensity import JumpOdeIntensityProcess

        self.temporal_core = JumpOdeIntensityProcess(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            **backbone_extra,
        )

    @property
    def capabilities(self) -> StateCapabilities:
        return StateCapabilities(
            has_query_state=True,
            has_sequence_states=True,
            has_regularization_terms=True,
            state_kind="process_backbone",
        )

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
        del marks, x_event, x_field_at_events

        bsz = times.shape[0]
        h_dim = self.temporal_core._init_state.shape[0]
        max_len = int(lengths.max().item())
        if max_len < 2:
            h_init_b = (
                self.temporal_core._init_state.detach()
                .unsqueeze(0).expand(bsz, -1).contiguous()
            )
            t_zero = torch.zeros(bsz, 1, device=times.device, dtype=times.dtype)
            _tc = self.temporal_core

            def _h_at_query_empty(t_q_tensor: Tensor) -> Tuple[Tensor, Tensor]:
                h_q, _, _ = _tc._integrate(h_init_b[:1], t_zero[:1], t_q_tensor)
                return h_q, _tc.intensity_at(h_q)

            return StateContext(
                payload={
                    "temporal_nll_matrix": torch.zeros(bsz, 0, device=times.device),
                    "z_seq": torch.zeros(bsz, 0, h_dim, device=times.device),
                    "temporal_energy_reg": torch.tensor(0.0, device=times.device),
                    "times": times,
                    "locations": locations,
                    "lengths": lengths,
                    "h_init": self.temporal_core._init_state.detach(),
                    "h_post_final": h_init_b,
                    "t_last_event": t_zero,
                    "_h_at_query": _h_at_query_empty,
                }
            )

        events = torch.cat([times.unsqueeze(-1), locations], dim=-1)
        temporal_nll, h_seq_pre, energy_reg, h_final = self.temporal_core.sequence_nll_and_states(
            events, lengths
        )
        t_last = times[torch.arange(bsz), (lengths - 1).clamp(min=0)].unsqueeze(-1)  # (B, 1)
        _tc, _hf, _tl = self.temporal_core, h_final.detach(), t_last.detach()

        def _h_at_query(t_q_tensor: Tensor) -> Tuple[Tensor, Tensor]:
            h_q, _, _ = _tc._integrate(_hf[:1], _tl[:1], t_q_tensor)
            return h_q, _tc.intensity_at(h_q)

        return StateContext(
            payload={
                "temporal_nll_matrix": temporal_nll,
                "z_seq": h_seq_pre,
                "temporal_energy_reg": energy_reg,
                "times": times,
                "locations": locations,
                "lengths": lengths,
                "h_init": self.temporal_core._init_state.detach(),
                "h_post_final": h_final,
                "t_last_event": t_last,
                "_h_at_query": _h_at_query,
            }
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
        del times, locations, lengths, x_field_at_events
        return state_ctx

    def regularization_terms(
        self,
        state_ctx: StateContext,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        del times, locations, lengths, marks
        energy = state_ctx.payload.get("temporal_energy_reg")
        if isinstance(energy, Tensor):
            return {"temporal_energy_reg": energy}
        if energy is not None:
            return {"temporal_energy_reg": torch.as_tensor(energy)}
        return {}
