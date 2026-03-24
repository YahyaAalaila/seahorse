"""StateModel for SMASH conditioning and Batch2toModel semantics."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel
from ..history_encoders.transformer_st import TransformerST


class SMASHStateModel(StateModel):
    """SMASH state encoder over unified_stpp batch tensors."""

    def __init__(
        self,
        *,
        transformer: TransformerST,
        loc_dim: int,
        num_types: int,
        log_normalization: bool = False,
        minmax_normalize_time: bool = True,
        mark_shift: int = 1,
    ):
        super().__init__()
        self.transformer = transformer
        self.loc_dim = int(loc_dim)
        self.num_types = int(num_types)
        self.log_normalization = bool(log_normalization)
        self.minmax_normalize_time = bool(minmax_normalize_time)
        self.mark_shift = int(mark_shift)

    @property
    def capabilities(self) -> StateCapabilities:
        return StateCapabilities(
            has_query_state=True,
            has_sequence_states=True,
            has_regularization_terms=False,
            state_kind="latent_static",
        )

    @staticmethod
    def _valid_mask(lengths: Tensor, max_len: int) -> Tensor:
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx < lengths.unsqueeze(1)

    def _build_event_time(self, times: Tensor, lengths: Tensor) -> Tensor:
        """Build SMASH event_time from unified absolute times."""
        max_len = times.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)

        dt = torch.zeros_like(times)
        dt[:, :1] = times[:, :1]
        if max_len > 1:
            dt[:, 1:] = times[:, 1:] - times[:, :-1]

        if self.log_normalization:
            dt = torch.log(dt.clamp(min=1e-4))

        if self.minmax_normalize_time:
            valid_vals = dt[valid_mask]
            if valid_vals.numel() > 0:
                vmin = valid_vals.min()
                vmax = valid_vals.max()
                scale = (vmax - vmin).clamp(min=1e-8)
                dt = (dt - vmin) / scale

        dt = torch.where(valid_mask, dt, torch.zeros_like(dt))
        return dt

    def _build_event_loc(
        self,
        *,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor],
    ) -> Tensor:
        max_len = locations.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)

        if self.loc_dim == 2:
            return torch.where(
                valid_mask.unsqueeze(-1),
                locations,
                torch.zeros_like(locations),
            )

        if marks is None:
            raise ValueError("SMASH marked mode requires marks in batch.")

        marks_shifted = marks.long()
        if self.mark_shift != 0:
            marks_shifted = marks_shifted + self.mark_shift
        marks_shifted = torch.where(valid_mask, marks_shifted, torch.zeros_like(marks_shifted))

        event_loc = torch.cat([marks_shifted.unsqueeze(-1).to(locations.dtype), locations], dim=-1)
        return torch.where(
            valid_mask.unsqueeze(-1),
            event_loc,
            torch.zeros_like(event_loc),
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
        del x_event, x_field_at_events

        event_time_origin = times
        event_time = self._build_event_time(times, lengths)
        event_loc = self._build_event_loc(
            locations=locations,
            lengths=lengths,
            marks=marks,
        )

        enc_out, non_pad_mask = self.transformer(
            event_loc=event_loc,
            event_time=event_time_origin,
            lengths=lengths,
        )

        cond_rows = []
        time_rows = []
        loc_rows = []
        cond_last_rows = []
        for b in range(times.shape[0]):
            L = int(lengths[b].item())
            if L <= 0:
                continue
            cond_last_rows.append(enc_out[b, L - 1 : L, :])
            if L <= 1:
                continue
            cond_rows.append(enc_out[b, : L - 1, :])
            time_rows.append(event_time[b, 1:L])
            loc_rows.append(event_loc[b, 1:L, :])

        cond_dim = enc_out.shape[-1]
        if cond_rows:
            cond_flat = torch.cat(cond_rows, dim=0).unsqueeze(1)
            event_time_non_mask = torch.cat(time_rows, dim=0).reshape(-1, 1, 1)
            event_loc_non_mask = torch.cat(loc_rows, dim=0).reshape(-1, 1, self.loc_dim)
        else:
            cond_flat = torch.zeros(0, 1, cond_dim, device=times.device, dtype=times.dtype)
            event_time_non_mask = torch.zeros(0, 1, 1, device=times.device, dtype=times.dtype)
            event_loc_non_mask = torch.zeros(
                0,
                1,
                self.loc_dim,
                device=times.device,
                dtype=times.dtype,
            )

        if cond_last_rows:
            cond_last = torch.cat(cond_last_rows, dim=0).unsqueeze(1)
        else:
            cond_last = torch.zeros(
                times.shape[0],
                1,
                cond_dim,
                device=times.device,
                dtype=times.dtype,
            )

        img_flat = torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1)

        mark_targets = None
        if self.loc_dim == 3 and event_loc_non_mask.numel() > 0:
            mark_targets = event_loc_non_mask[:, :, 0].long()

        total_events = torch.tensor(
            float(event_time_non_mask.shape[0]),
            device=times.device,
            dtype=times.dtype,
        )

        return StateContext(
            payload={
                "smash_img": img_flat,
                "smash_cond": cond_flat,
                "smash_cond_last": cond_last,
                "smash_total_events": total_events,
                "smash_mark_targets": mark_targets,
                "event_time_origin": event_time_origin,
                "event_time": event_time,
                "event_loc": event_loc,
                "enc_out": enc_out,
                "non_pad_mask": non_pad_mask,
                "lengths": lengths,
                "times": times,
                "locations": locations,
                "marks": marks,
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
