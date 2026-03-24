"""StateModel for Diffusion STPP conditioning (unmarked only).

Follows the same Batch2toModel flattening semantics as SMASHStateModel but:
- supports only the unmarked case (loc_dim = 2)
- uses ``diff_`` prefixed payload keys to avoid collision with other models
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel
from ..history_encoders.transformer_st import TransformerST


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
        transformer: TransformerST,
        spatial_dim: int = 2,
        minmax_normalize_time: bool = True,
    ):
        super().__init__()
        self.transformer = transformer
        self.spatial_dim = int(spatial_dim)
        self.minmax_normalize_time = bool(minmax_normalize_time)

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

    def _build_delta_time(self, times: Tensor, lengths: Tensor) -> Tensor:
        """Convert absolute times to inter-event delta times with optional minmax normalisation."""
        max_len = times.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)

        dt = torch.zeros_like(times)
        dt[:, :1] = times[:, :1]
        if max_len > 1:
            dt[:, 1:] = times[:, 1:] - times[:, :-1]

        if self.minmax_normalize_time:
            valid_vals = dt[valid_mask]
            if valid_vals.numel() > 0:
                vmin = valid_vals.min()
                vmax = valid_vals.max()
                dt = (dt - vmin) / (vmax - vmin).clamp(min=1e-8)

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

        event_time_origin = times
        event_time = self._build_delta_time(times, lengths)

        # Mask padding positions to zero in locations
        max_len = locations.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)
        event_loc = torch.where(
            valid_mask.unsqueeze(-1),
            locations,
            torch.zeros_like(locations),
        )

        enc_out, non_pad_mask = self.transformer(
            event_loc=event_loc,
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

        # img = [delta_time, location] — the diffusion target token
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
