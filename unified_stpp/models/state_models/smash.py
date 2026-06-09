"""StateModel for SMASH conditioning and Batch2toModel semantics."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel
from ..model_registry import register_state
from ..history_encoders.smash_transformer import SMASHTransformerST


@register_state("smash")
class SMASHStateModel(StateModel):
    """SMASH state encoder over unified_stpp batch tensors."""

    def __init__(
        self,
        *,
        transformer: SMASHTransformerST,
        loc_dim: int,
        num_types: int,
        log_normalization: bool = False,
        minmax_normalize_time: bool = True,
        minmax_normalize_loc: bool = True,
        mark_shift: int = 1,
        input_time_normalized: bool = False,
        input_space_normalized: bool = False,
        input_time_mean: float = 0.0,
        input_time_std: float = 1.0,
        input_loc_mean: tuple[float, ...] = (0.0, 0.0),
        input_loc_std: tuple[float, ...] = (1.0, 1.0),
        token_time_min_raw: float = 0.0,
        token_time_range_raw: float = 1.0,
        token_time_min_log: float = 0.0,
        token_time_range_log: float = 1.0,
        token_loc_min: tuple[float, ...] = (0.0, 0.0),
        token_loc_range: tuple[float, ...] = (1.0, 1.0),
    ):
        super().__init__()
        self.transformer = transformer
        self.loc_dim = int(loc_dim)
        self.num_types = int(num_types)
        self.spatial_dim = self.loc_dim - 1 if self.num_types > 1 else self.loc_dim
        self.log_normalization = bool(log_normalization)
        self.minmax_normalize_time = bool(minmax_normalize_time)
        self.minmax_normalize_loc = bool(minmax_normalize_loc)
        self.mark_shift = int(mark_shift)
        self.input_time_normalized = bool(input_time_normalized)
        self.input_space_normalized = bool(input_space_normalized)

        self.register_buffer(
            "input_time_mean",
            torch.tensor(float(input_time_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "input_time_std",
            torch.tensor(float(input_time_std), dtype=torch.float32),
        )
        self.register_buffer(
            "input_loc_mean",
            torch.tensor(input_loc_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "input_loc_std",
            torch.tensor(input_loc_std, dtype=torch.float32),
        )
        self.register_buffer(
            "token_time_min_raw",
            torch.tensor(float(token_time_min_raw), dtype=torch.float32),
        )
        self.register_buffer(
            "token_time_range_raw",
            torch.tensor(float(token_time_range_raw), dtype=torch.float32),
        )
        self.register_buffer(
            "token_time_min_log",
            torch.tensor(float(token_time_min_log), dtype=torch.float32),
        )
        self.register_buffer(
            "token_time_range_log",
            torch.tensor(float(token_time_range_log), dtype=torch.float32),
        )
        self.register_buffer(
            "token_loc_min",
            torch.tensor(token_loc_min, dtype=torch.float32),
        )
        self.register_buffer(
            "token_loc_range",
            torch.tensor(token_loc_range, dtype=torch.float32),
        )

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

    def _recover_raw_inputs(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        max_len = times.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)

        raw_times = times
        if self.input_time_normalized:
            raw_times = times * self.input_time_std.to(device=times.device, dtype=times.dtype)
            raw_times = raw_times + self.input_time_mean.to(device=times.device, dtype=times.dtype)
        raw_times = torch.where(valid_mask, raw_times, torch.zeros_like(raw_times))

        raw_locations = locations
        if self.input_space_normalized:
            loc_std = self.input_loc_std.to(device=locations.device, dtype=locations.dtype).view(1, 1, -1)
            loc_mean = self.input_loc_mean.to(device=locations.device, dtype=locations.dtype).view(1, 1, -1)
            raw_locations = locations * loc_std + loc_mean
        raw_locations = torch.where(
            valid_mask.unsqueeze(-1),
            raw_locations,
            torch.zeros_like(raw_locations),
        )
        return raw_times, raw_locations, valid_mask

    def _build_event_time(self, raw_times: Tensor, valid_mask: Tensor) -> Tensor:
        """Build SMASH event_time from raw absolute times."""
        dt = torch.zeros_like(raw_times)
        dt[:, :1] = raw_times[:, :1]
        max_len = raw_times.shape[1]
        if max_len > 1:
            dt[:, 1:] = raw_times[:, 1:] - raw_times[:, :-1]

        time_min = self.token_time_min_raw
        time_range = self.token_time_range_raw
        if self.log_normalization:
            dt = torch.log(dt.clamp(min=1e-4))
            time_min = self.token_time_min_log
            time_range = self.token_time_range_log

        if self.minmax_normalize_time:
            time_min = time_min.to(device=dt.device, dtype=dt.dtype)
            time_range = time_range.to(device=dt.device, dtype=dt.dtype).clamp(min=1e-8)
            dt = (dt - time_min) / time_range

        dt = torch.where(valid_mask, dt, torch.zeros_like(dt))
        return dt

    def _build_event_loc(
        self,
        *,
        raw_locations: Tensor,
        valid_mask: Tensor,
        marks: Optional[Tensor],
    ) -> Tensor:
        spatial = raw_locations
        if self.minmax_normalize_loc:
            loc_min = self.token_loc_min.to(device=raw_locations.device, dtype=raw_locations.dtype).view(1, 1, -1)
            loc_range = self.token_loc_range.to(device=raw_locations.device, dtype=raw_locations.dtype).view(1, 1, -1)
            spatial = (raw_locations - loc_min) / loc_range.clamp(min=1e-8)

        if self.loc_dim == 2:
            return torch.where(
                valid_mask.unsqueeze(-1),
                spatial,
                torch.zeros_like(spatial),
            )

        if marks is None:
            raise ValueError("SMASH marked mode requires marks in batch.")

        marks_shifted = marks.long()
        if self.mark_shift != 0:
            marks_shifted = marks_shifted + self.mark_shift
        marks_shifted = torch.where(valid_mask, marks_shifted, torch.zeros_like(marks_shifted))

        event_loc = torch.cat([marks_shifted.unsqueeze(-1).to(spatial.dtype), spatial], dim=-1)
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

        raw_times, raw_locations, valid_mask = self._recover_raw_inputs(
            times=times,
            locations=locations,
            lengths=lengths,
        )

        event_time_origin = raw_times
        event_time = self._build_event_time(raw_times, valid_mask)
        event_loc = self._build_event_loc(
            raw_locations=raw_locations,
            valid_mask=valid_mask,
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
                "times": raw_times,
                "locations": raw_locations,
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
