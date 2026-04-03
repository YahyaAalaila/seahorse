"""StateModel for upstream-faithful Diffusion STPP conditioning (unmarked only).

Follows the same Batch2toModel flattening semantics as SMASHStateModel but:
- supports only the unmarked case (loc_dim = 2)
- uses ``diff_`` prefixed payload keys to avoid collision with other models
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel
from ..model_registry import register_state
from ..history_encoders.dstpp_transformer import DSTPPTransformerST


@register_state("diffusion_stpp")
class DiffusionStateModel(StateModel):
    """Diffusion STPP state encoder over unified_stpp batch tensors.

    Builds the conditioning tensors required by DiffusionEventModel from the
    standard unified batch (times, locations, lengths) via the same valid-
    position flattening as SMASHStateModel (Batch2toModel semantics):

        - For each sequence b with length L:
          * conditioning for event i+1 = enc_out[b, i]    (i = 0 .. L-2)
          * target time  for event i+1 = delta_time[b, i+1]
          * target loc   for event i+1 = locations[b, i+1]

    Payload keys (prefixed ``diff_`` to avoid collision):
        diff_img         : (N_flat, 1, 1+spatial_dim)  — [delta_time, *loc]
        diff_cond        : (N_flat, 1, 3*d_model)      — per-position conditioning
        diff_cond_last   : (B,      1, 3*d_model)      — last-position conditioning
        diff_total_events: scalar float tensor          — N_flat
    """

    def __init__(
        self,
        *,
        transformer: DSTPPTransformerST,
        spatial_dim: int = 2,
        minmax_normalize_time: bool = True,
        minmax_normalize_loc: bool = True,
        input_normalized: bool = False,
        input_time_mean: float = 0.0,
        input_time_std: float = 1.0,
        input_loc_mean: tuple[float, ...] = (0.0, 0.0),
        input_loc_std: tuple[float, ...] = (1.0, 1.0),
        token_delta_t_min: float = 0.0,
        token_delta_t_range: float = 1.0,
        token_loc_min: tuple[float, ...] = (0.0, 0.0),
        token_loc_range: tuple[float, ...] = (1.0, 1.0),
    ):
        super().__init__()
        self.transformer = transformer
        self.spatial_dim = int(spatial_dim)
        self.minmax_normalize_time = bool(minmax_normalize_time)
        self.minmax_normalize_loc = bool(minmax_normalize_loc)
        self.register_buffer(
            "input_normalized_flag",
            torch.tensor(1.0 if input_normalized else 0.0, dtype=torch.float32),
        )
        self.register_buffer(
            "input_time_mean",
            torch.tensor(float(input_time_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "input_time_std",
            torch.tensor(max(float(input_time_std), 1e-8), dtype=torch.float32),
        )
        self.register_buffer(
            "input_loc_mean",
            torch.as_tensor(input_loc_mean, dtype=torch.float32).reshape(self.spatial_dim),
        )
        self.register_buffer(
            "input_loc_std",
            torch.as_tensor(input_loc_std, dtype=torch.float32).reshape(self.spatial_dim).clamp(min=1e-8),
        )
        self.register_buffer(
            "token_delta_t_min",
            torch.tensor(float(token_delta_t_min), dtype=torch.float32),
        )
        self.register_buffer(
            "token_delta_t_range",
            torch.tensor(max(float(token_delta_t_range), 1e-8), dtype=torch.float32),
        )
        self.register_buffer(
            "token_loc_min",
            torch.as_tensor(token_loc_min, dtype=torch.float32).reshape(self.spatial_dim),
        )
        self.register_buffer(
            "token_loc_range",
            torch.as_tensor(token_loc_range, dtype=torch.float32).reshape(self.spatial_dim).clamp(min=1e-8),
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

    def _input_is_normalized(self) -> bool:
        return bool(self.input_normalized_flag.item() > 0.5)

    def _restore_raw_times(self, times: Tensor) -> Tensor:
        if not self._input_is_normalized():
            return times
        mean = self.input_time_mean.to(device=times.device, dtype=times.dtype)
        std = self.input_time_std.to(device=times.device, dtype=times.dtype).clamp(min=1e-8)
        return times * std + mean

    def _restore_raw_locations(self, locations: Tensor) -> Tensor:
        if not self._input_is_normalized():
            return locations
        mean = self.input_loc_mean.to(device=locations.device, dtype=locations.dtype)
        std = self.input_loc_std.to(device=locations.device, dtype=locations.dtype).clamp(min=1e-8)
        return locations * std + mean

    def _build_token_locations(self, locations_raw: Tensor, valid_mask: Tensor) -> Tensor:
        if self.minmax_normalize_loc:
            loc_min = self.token_loc_min.to(device=locations_raw.device, dtype=locations_raw.dtype)
            loc_range = self.token_loc_range.to(
                device=locations_raw.device, dtype=locations_raw.dtype
            ).clamp(min=1e-8)
            loc = ((locations_raw - loc_min) / loc_range).clamp_(0.0, 1.0)
        else:
            loc = locations_raw
        return torch.where(valid_mask.unsqueeze(-1), loc, torch.zeros_like(loc))

    def _build_delta_time(self, times_raw: Tensor, lengths: Tensor) -> Tensor:
        """Convert raw absolute times to diffusion token delta-times."""
        max_len = times_raw.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)

        dt = torch.zeros_like(times_raw)
        if max_len > 1:
            dt[:, 1:] = times_raw[:, 1:] - times_raw[:, :-1]

        if self.minmax_normalize_time:
            vmin = self.token_delta_t_min.to(device=dt.device, dtype=dt.dtype)
            vrange = self.token_delta_t_range.to(device=dt.device, dtype=dt.dtype).clamp(min=1e-8)
            dt = ((dt - vmin) / vrange).clamp_(0.0, 1.0)

        return torch.where(valid_mask, dt, torch.zeros_like(dt))

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

        max_len = locations.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)
        times_raw = self._restore_raw_times(times)
        locations_raw = self._restore_raw_locations(locations)
        event_time_origin = torch.where(valid_mask, times_raw, torch.zeros_like(times_raw))
        event_loc_encoder = self._build_token_locations(locations_raw, valid_mask)
        event_time = self._build_delta_time(times_raw, lengths)
        event_loc = self._build_token_locations(locations_raw, valid_mask)

        enc_out, non_pad_mask = self.transformer(
            event_loc=event_loc_encoder,
            event_time=event_time_origin,
            lengths=lengths,
        )

        cond_dim = enc_out.shape[-1]  # 3 * d_model for unmarked TransformerST

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

        if cond_rows:
            cond_flat = torch.cat(cond_rows, dim=0).unsqueeze(1)            # (N, 1, cond_dim)
            dt_flat = torch.cat(time_rows, dim=0).reshape(-1, 1, 1)         # (N, 1, 1)
            loc_flat = torch.cat(loc_rows, dim=0).reshape(-1, 1, self.spatial_dim)  # (N, 1, d)
        else:
            cond_flat = torch.zeros(0, 1, cond_dim, device=times.device, dtype=times.dtype)
            dt_flat = torch.zeros(0, 1, 1, device=times.device, dtype=times.dtype)
            loc_flat = torch.zeros(0, 1, self.spatial_dim, device=times.device, dtype=times.dtype)

        if cond_last_rows:
            cond_last = torch.cat(cond_last_rows, dim=0).unsqueeze(1)
        else:
            cond_last = torch.zeros(times.shape[0], 1, cond_dim, device=times.device, dtype=times.dtype)

        img_flat = torch.cat((dt_flat, loc_flat), dim=-1)  # (N, 1, 1+spatial_dim)

        total_events = torch.tensor(
            float(img_flat.shape[0]),
            device=times.device,
            dtype=times.dtype,
        )

        return StateContext(
            payload={
                "diff_img": img_flat,
                "diff_cond": cond_flat,
                "diff_cond_last": cond_last,
                "diff_total_events": total_events,
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
