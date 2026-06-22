"""Shared state model for the Neural STPP family."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from seahorse.data.transforms import transform_from_spec
from ..abstractions import StateCapabilities, StateContext, StateModel
from ..model_registry import register_state


def _repair_strictly_increasing_times(times: Tensor, lengths: Tensor) -> Tensor:
    """Preserve strict monotonicity after model-side time reconstruction."""
    if times.ndim != 2:
        raise ValueError(f"expected times shape (B, T); got {tuple(times.shape)}")

    repaired = times.clone()
    inf = None
    eps = torch.finfo(repaired.dtype).eps
    for row in range(repaired.shape[0]):
        seq_len = int(lengths[row].item())
        if seq_len <= 1:
            continue
        seq = repaired[row, :seq_len]
        if bool(torch.all(seq[1:] > seq[:-1]).item()):
            continue
        for i in range(1, seq_len):
            if not bool((seq[i] > seq[i - 1]).item()):
                if hasattr(torch, "nextafter"):
                    if inf is None:
                        inf = torch.full((), float("inf"), device=seq.device, dtype=seq.dtype)
                    seq[i] = torch.nextafter(seq[i - 1], inf)
                else:  # pragma: no cover - torch.nextafter exists in supported runtimes
                    scale = torch.maximum(seq[i - 1].abs(), torch.ones_like(seq[i - 1]))
                    seq[i] = seq[i - 1] + eps * scale
        if not bool(torch.all(seq[1:] > seq[:-1]).item()):
            raw_dt = seq[1:] - seq[:-1]
            bad = raw_dt <= 0
            first_bad = int(torch.nonzero(bad, as_tuple=False)[0].item() + 1)
            raise RuntimeError(
                "Neural STPP state raw-time reconstruction failed to preserve strict monotonicity; "
                f"batch_index={row}, event_index={first_bad}, bad_count={int(bad.sum().item())}, "
                f"min_raw_dt={float(raw_dt.min().item()):.6e}"
            )
    return repaired


@register_state("neural_stpp")
class NeuralSTPPStateModel(StateModel):
    """Neural STPP state model with local raw-time reconstruction."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        spatial_dim: int,
        tpp_hidden_dims=None,
        tpp_cond: bool = True,
        tpp_style: str = "gru",
        tpp_actfn: str = "softplus",
        share_hidden: bool = True,
        solver: str = "dopri5",
        atol: float = 1e-4,
        rtol: float = 1e-4,
        use_adjoint: bool = False,
        energy_regularization: float = 1e-4,
        normalize_time_inputs: bool = False,
        normalize_space_inputs: bool = False,
        time_mean: float = 0.0,
        time_std: float = 1.0,
        input_transform: Optional[dict] = None,
        **backbone_extra,
    ):
        super().__init__()
        del backbone_extra
        from ..temporal_models.neural_point_process import NeuralPointProcess, _normalize_hidden_dims

        temporal_hidden_dims = _normalize_hidden_dims(tpp_hidden_dims, fallback=hidden_dim)
        temporal_hidden_dim = int(temporal_hidden_dims[0])
        temporal_hdim = temporal_hidden_dim // 2
        separate = 1 if share_hidden else 2

        self.normalize_time_inputs = bool(normalize_time_inputs)
        self.normalize_space_inputs = bool(normalize_space_inputs)
        self.time_mean = float(time_mean)
        self.time_std = float(time_std) if abs(float(time_std)) > 1e-12 else 1.0
        self._input_transform_spec = dict(input_transform or {})
        self._input_transform = transform_from_spec(self._input_transform_spec)
        self.temporal_hdim = int(temporal_hdim)
        self.spatial_aux_dim = int(max(0, temporal_hidden_dim - temporal_hdim))
        self.temporal_hidden_dim = temporal_hidden_dim

        self.temporal_core = NeuralPointProcess(
            cond_dim=spatial_dim,
            hidden_dims=temporal_hidden_dims,
            cond=tpp_cond,
            style=tpp_style,
            actfn=tpp_actfn,
            hdim=temporal_hdim,
            separate=separate,
            tol=max(float(atol), float(rtol)),
            otreg_strength=float(energy_regularization),
            method=solver,
            use_adjoint=use_adjoint,
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
        h_dim = self.temporal_core.hidden_dim
        max_len = int(lengths.max().item())
        if max_len < 1:
            h_init_b = (
                self.temporal_core._init_state.detach()
                .unsqueeze(0).expand(bsz, -1).contiguous()
            )
            t_zero = torch.zeros(bsz, 1, device=times.device, dtype=times.dtype)
            _tc = self.temporal_core

            def _h_at_query_empty_raw(t_q_tensor: Tensor) -> Tuple[Tensor, Tensor]:
                h_q, _, _ = _tc.integrate_hidden(h_init_b[:1], t_zero[:1], t_q_tensor)
                return h_q, _tc.get_intensity(h_q)

            return StateContext(
                payload={
                    "temporal_nll_matrix": torch.zeros(bsz, 0, device=times.device),
                    "z_seq": torch.zeros(bsz, 0, h_dim, device=times.device),
                    "temporal_hidden_seq": torch.zeros(bsz, 0, h_dim, device=times.device),
                    "temporal_intensity_hidden_seq": torch.zeros(
                        bsz, 0, self.temporal_hdim, device=times.device
                    ),
                    "spatial_aux_seq": torch.zeros(
                        bsz, 0, self.spatial_aux_dim, device=times.device
                    ),
                    "temporal_energy_reg": torch.tensor(0.0, device=times.device),
                    "times": times,
                    "locations": locations,
                    "times_raw": times,
                    "locations_norm": locations,
                    "lengths": lengths,
                    "t0_raw": t_zero,
                    "h_init": self.temporal_core._init_state.detach(),
                    "h_post_final": h_init_b,
                    "t_last_event_raw": t_zero,
                    "temporal_hdim": self.temporal_hdim,
                    "spatial_aux_dim": self.spatial_aux_dim,
                    "normalize_time_inputs": self.normalize_time_inputs,
                    "normalize_space_inputs": self.normalize_space_inputs,
                    "time_mean": self.time_mean,
                    "time_std": self.time_std,
                    "input_transform": self._input_transform_spec,
                    "_h_at_query_raw": _h_at_query_empty_raw,
                }
            )

        if self._input_transform is not None:
            times_raw = self._input_transform.inverse_times(times.to(torch.float64), lengths).to(times.dtype)
            locations_norm = self._input_transform.forward_locations(locations, lengths)
        elif self.normalize_time_inputs:
            times_raw = (
                times.to(torch.float64) * float(self.time_std) + float(self.time_mean)
            ).to(times.dtype)
            locations_norm = locations
        else:
            times_raw = times
            locations_norm = locations
        times_raw = _repair_strictly_increasing_times(times_raw[:, :max_len], lengths)
        event_mask = (
            torch.arange(max_len, device=times.device).unsqueeze(0) < lengths.unsqueeze(1)
        )
        temporal_nll, h_seq_pre, energy_reg, h_final = self.temporal_core.sequence_nll_and_states(
            times_raw,
            locations_norm[:, :max_len, :],
            event_mask,
            t0=torch.zeros(bsz, device=times.device, dtype=times.dtype),
            t1=None,
        )
        h_seq_pre = h_seq_pre[:, :max_len, :]
        temporal_nll = temporal_nll[:, :max_len]
        t_last_raw = times_raw[
            torch.arange(bsz, device=times.device),
            (lengths - 1).clamp(min=0),
        ].unsqueeze(-1)
        _tc, _hf, _tl = self.temporal_core, h_final.detach(), t_last_raw.detach()

        def _h_at_query_raw(t_q_tensor: Tensor) -> Tuple[Tensor, Tensor]:
            h_q, _, _ = _tc.integrate_hidden(_hf[:1], _tl[:1], t_q_tensor)
            return h_q, _tc.get_intensity(h_q)

        temporal_intensity_hidden_seq = h_seq_pre[..., : self.temporal_hdim]
        spatial_aux_seq = h_seq_pre[..., -self.spatial_aux_dim :] if self.spatial_aux_dim > 0 else h_seq_pre.new_zeros(
            h_seq_pre.shape[0], h_seq_pre.shape[1], 0
        )

        return StateContext(
            payload={
                "temporal_nll_matrix": temporal_nll,
                "z_seq": h_seq_pre,
                "temporal_hidden_seq": h_seq_pre,
                "temporal_intensity_hidden_seq": temporal_intensity_hidden_seq,
                "spatial_aux_seq": spatial_aux_seq,
                "temporal_energy_reg": energy_reg,
                "times": times,
                "locations": locations,
                "times_raw": times_raw,
                "locations_norm": locations_norm,
                "lengths": lengths,
                "t0_raw": torch.zeros(bsz, 1, device=times.device, dtype=times.dtype),
                "h_init": self.temporal_core._init_state.detach(),
                "h_post_final": h_final,
                "t_last_event_raw": t_last_raw,
                "temporal_hdim": self.temporal_hdim,
                "spatial_aux_dim": self.spatial_aux_dim,
                "normalize_time_inputs": self.normalize_time_inputs,
                "normalize_space_inputs": self.normalize_space_inputs,
                "time_mean": self.time_mean,
                "time_std": self.time_std,
                "input_transform": self._input_transform_spec,
                "_h_at_query_raw": _h_at_query_raw,
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
