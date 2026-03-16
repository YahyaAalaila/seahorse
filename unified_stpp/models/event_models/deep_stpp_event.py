"""EventModel wrapper for DeepSTPP decoder parameterization and likelihood terms."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext


DecodeFn = Callable[[Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]
TemporalLogFn = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
SpatialLogFn = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
BackgroundFn = Callable[[], Optional[Tensor]]


def _sliding_history_windows_with_times(
    times: Tensor, locations: Tensor, l_steps: int, seq_len: int
) -> Tensor:
    """Build history windows in DeepSTPP layout [times | flattened locations]."""
    bsz, _, d = locations.shape
    first_t = times[:, :1].expand(-1, seq_len - 1)
    padded_t = torch.cat([first_t, times[:, : l_steps + 1]], dim=1)
    first_s = locations[:, :1, :].expand(-1, seq_len - 1, -1)
    padded_s = torch.cat([first_s, locations[:, : l_steps + 1, :]], dim=1)

    t_windows = torch.stack([padded_t[:, n : n + seq_len] for n in range(l_steps)], dim=1)
    s_windows = torch.stack([padded_s[:, n : n + seq_len, :] for n in range(l_steps)], dim=1)
    return torch.cat([t_windows, s_windows.reshape(bsz, l_steps, seq_len * d)], dim=-1)


class DeepSTPPEventModel(EventModel):
    """Coarse DeepSTPP event model."""

    def __init__(
        self,
        *,
        decode_fn: DecodeFn,
        temporal_log_fn: TemporalLogFn,
        spatial_log_fn: SpatialLogFn,
        background_fn: BackgroundFn,
        seq_len: int,
        num_points: int,
        spatial_dim: int,
        expose_decoded_params: bool = False,
    ):
        super().__init__()
        self._decode_fn = decode_fn
        self._temporal_log_fn = temporal_log_fn
        self._spatial_log_fn = spatial_log_fn
        self._background_fn = background_fn
        self.seq_len = int(seq_len)
        self.num_points = int(num_points)
        self.spatial_dim = int(spatial_dim)
        self.expose_decoded_params = bool(expose_decoded_params)

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
    def _get_state_term(state_ctx: StateContext, key: str) -> Tensor:
        val = state_ctx.payload.get(key)
        if val is None:
            raise ValueError(f"DeepSTPPEventModel requires state['{key}'].")
        if not isinstance(val, Tensor):
            raise TypeError(f"DeepSTPPEventModel expects tensor for state['{key}'].")
        return val

    def _compute(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state_ctx: StateContext,
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
                "sll": zero_scalar,
                "tll": zero_scalar,
                "nll_matrix": torch.zeros(bsz, empty_l, device=device),
                "sll_matrix": torch.zeros(bsz, empty_l, device=device),
                "tll_matrix": torch.zeros(bsz, empty_l, device=device),
                "mask": torch.zeros(bsz, empty_l, device=device),
            }

        l_steps = max_len - 1
        z_all = self._get_state_term(state_ctx, "z")
        z_cond = z_all[:, :l_steps, :]

        t_target = times[:, 1 : 1 + l_steps].unsqueeze(-1)
        s_target = locations[:, 1 : 1 + l_steps, :]
        t_prev = times[:, :l_steps].unsqueeze(-1)

        n_idx = torch.arange(l_steps, device=device)
        mask = (n_idx.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()

        hist_windows = _sliding_history_windows_with_times(
            times, locations, l_steps, self.seq_len
        )
        x_field_flat = hist_windows.reshape(bsz * l_steps, -1)

        h = z_cond.shape[-1]
        d = s_target.shape[-1]
        z_flat = z_cond.reshape(bsz * l_steps, h)
        t_flat = t_target.reshape(bsz * l_steps, 1)
        s_flat = s_target.reshape(bsz * l_steps, d)
        t_prev_flat = t_prev.reshape(bsz * l_steps, 1)

        w_i, b_i, _sigma, inv_var = self._decode_fn(z_flat)

        t_hist = x_field_flat[:, : self.seq_len]
        s_hist = x_field_flat[:, self.seq_len :].reshape(bsz * l_steps, self.seq_len, d)
        tn_ti_h = (t_prev_flat - t_hist).clamp(min=0.0)
        tn_ti_bg = torch.zeros(bsz * l_steps, self.num_points, device=device)
        tn_ti = torch.cat([tn_ti_h, tn_ti_bg], dim=-1)
        dt = (t_flat - t_prev_flat).clamp(min=1e-6).reshape(bsz * l_steps)
        t_ti = (tn_ti + dt.unsqueeze(-1)).clamp(min=1e-6)

        background = self._background_fn()
        if background is not None:
            bg = background.unsqueeze(0).expand(bsz * l_steps, -1, -1)
            centers = torch.cat([s_hist, bg], dim=1)
        else:
            centers = s_hist
        s_diff = s_flat.unsqueeze(1) - centers

        tll_flat = self._temporal_log_fn(w_i, b_i, tn_ti, t_ti)
        sll_flat = self._spatial_log_fn(w_i, b_i, t_ti, s_diff, inv_var)
        tll_matrix = tll_flat.reshape(bsz, l_steps)
        sll_matrix = sll_flat.reshape(bsz, l_steps)
        nll_matrix = -(tll_matrix + sll_matrix)

        nll_masked = nll_matrix * mask
        total_nll = nll_masked.sum(dim=1)
        n_events = mask.sum(dim=1)
        n_events_total = n_events.sum().clamp(min=1)
        mean_nll = total_nll.sum() / n_events_total
        sll = (sll_matrix * mask).sum() / n_events_total
        tll = (tll_matrix * mask).sum() / n_events_total

        out = {
            "loss": mean_nll,
            "nll": mean_nll,
            "nll_per_event": total_nll / n_events.clamp(min=1),
            "total_events": n_events.sum(),
            "sll": sll,
            "tll": tll,
            "nll_matrix": nll_matrix,
            "sll_matrix": sll_matrix,
            "tll_matrix": tll_matrix,
            "mask": mask,
        }
        kl_loss = state_ctx.payload.get("kl_loss")
        if isinstance(kl_loss, Tensor):
            out["kl_loss"] = kl_loss
        if self.expose_decoded_params:
            out["w_i"] = w_i.reshape(bsz, l_steps, -1)
            out["b_i"] = b_i.reshape(bsz, l_steps, -1)
            out["inv_var"] = inv_var.reshape(
                bsz, l_steps, inv_var.shape[1], inv_var.shape[2]
            )
        return out

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
        del state_regularization_terms, x_field_at_events, marks
        if device is None:
            device = times.device
        return self._compute(
            times=times,
            locations=locations,
            lengths=lengths,
            state_ctx=state,
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
