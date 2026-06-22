"""Thin state model for AutoSTPP preprocessing."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from seahorse.data.transforms import PaperAffineTransformArtifact, transform_from_spec
from ..abstractions import StateCapabilities, StateContext, StateModel
from ..model_registry import register_state


@register_state("auto_stpp")
class AutoSTPPStateModel(StateModel):
    """Restore raw inputs, build paper MinMax tensors, and package history."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        lookback: int = 20,
        lookahead: int = 1,
        input_normalized: bool = False,
        input_time_mean: float = 0.0,
        input_time_std: float = 1.0,
        input_loc_mean: tuple[float, ...] = (0.0, 0.0),
        input_loc_std: tuple[float, ...] = (1.0, 1.0),
        paper_dt_min: float = 0.0,
        paper_dt_range: float = 1.0,
        paper_loc_min: tuple[float, ...] = (0.0, 0.0),
        paper_loc_range: tuple[float, ...] = (1.0, 1.0),
        input_transform: Optional[dict] = None,
        **_,
    ):
        super().__init__()
        if int(spatial_dim) != 2:
            raise ValueError(
                f"AutoSTPP state model requires spatial_dim=2, got {spatial_dim}."
            )
        self.spatial_dim = int(spatial_dim)
        self.lookback = int(lookback)
        self.lookahead = int(lookahead)
        self._input_transform_spec = dict(input_transform or {})
        self._input_transform = transform_from_spec(self._input_transform_spec)
        if isinstance(self._input_transform, PaperAffineTransformArtifact):
            input_normalized = bool(self._input_transform.input_normalized)
            input_time_mean = float(self._input_transform.input_time_mean)
            input_time_std = float(self._input_transform.input_time_std)
            input_loc_mean = tuple(self._input_transform.input_loc_mean)
            input_loc_std = tuple(self._input_transform.input_loc_std)
            paper_dt_min = float(self._input_transform.paper_dt_min)
            paper_dt_range = float(self._input_transform.paper_dt_range)
            paper_loc_min = tuple(self._input_transform.paper_loc_min)
            paper_loc_range = tuple(self._input_transform.paper_loc_range)

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
            torch.as_tensor(input_loc_std, dtype=torch.float32)
            .reshape(self.spatial_dim)
            .clamp(min=1e-8),
        )
        self.register_buffer(
            "paper_dt_min",
            torch.tensor(float(paper_dt_min), dtype=torch.float32),
        )
        self.register_buffer(
            "paper_dt_range",
            torch.tensor(max(float(paper_dt_range), 1e-8), dtype=torch.float32),
        )
        self.register_buffer(
            "paper_loc_min",
            torch.as_tensor(paper_loc_min, dtype=torch.float32).reshape(self.spatial_dim),
        )
        self.register_buffer(
            "paper_loc_range",
            torch.as_tensor(paper_loc_range, dtype=torch.float32)
            .reshape(self.spatial_dim)
            .clamp(min=1e-8),
        )

    @property
    def capabilities(self) -> StateCapabilities:
        return StateCapabilities(
            has_query_state=True,
            has_sequence_states=True,
            has_regularization_terms=False,
            state_kind="history_passthrough",
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

    def _raw_delta_times(self, times_raw: Tensor, lengths: Tensor) -> Tensor:
        max_len = times_raw.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)
        delta_t = torch.zeros_like(times_raw)
        if max_len > 0:
            delta_t[:, 0] = times_raw[:, 0]
        if max_len > 1:
            delta_t[:, 1:] = times_raw[:, 1:] - times_raw[:, :-1]
        return torch.where(valid_mask, delta_t, torch.zeros_like(delta_t))

    def _scale_paper_locations(self, locations_raw: Tensor, valid_mask: Tensor) -> Tensor:
        loc_min = self.paper_loc_min.to(device=locations_raw.device, dtype=locations_raw.dtype)
        loc_range = self.paper_loc_range.to(
            device=locations_raw.device,
            dtype=locations_raw.dtype,
        ).clamp(min=1e-8)
        scaled = (locations_raw - loc_min) / loc_range
        return torch.where(valid_mask.unsqueeze(-1), scaled, torch.zeros_like(scaled))

    def _scale_paper_delta_t(self, delta_t_raw: Tensor, valid_mask: Tensor) -> Tensor:
        dt_min = self.paper_dt_min.to(device=delta_t_raw.device, dtype=delta_t_raw.dtype)
        dt_range = self.paper_dt_range.to(
            device=delta_t_raw.device,
            dtype=delta_t_raw.dtype,
        ).clamp(min=1e-8)
        scaled = (delta_t_raw - dt_min) / dt_range
        return torch.where(valid_mask, scaled, torch.zeros_like(scaled))

    def _build_training_windows(
        self,
        history_scaled: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        history_windows: list[Tensor] = []
        target_windows: list[Tensor] = []
        seq_ids: list[Tensor] = []
        counts = torch.zeros(lengths.shape[0], dtype=torch.long, device=lengths.device)

        for b in range(lengths.shape[0]):
            n_events = int(lengths[b].item())
            n_windows = max(0, n_events - self.lookback - self.lookahead + 1)
            counts[b] = n_windows
            if n_windows == 0:
                continue
            seq = history_scaled[b, :n_events]
            history_windows.append(
                torch.stack(
                    [seq[i : i + self.lookback] for i in range(n_windows)],
                    dim=0,
                )
            )
            target_windows.append(
                torch.stack(
                    [
                        seq[i + self.lookback : i + self.lookback + self.lookahead]
                        for i in range(n_windows)
                    ],
                    dim=0,
                )
            )
            seq_ids.append(
                torch.full((n_windows,), b, device=lengths.device, dtype=torch.long)
            )

        if not history_windows:
            empty_hist = history_scaled.new_zeros(0, self.lookback, 1 + self.spatial_dim)
            empty_tgt = history_scaled.new_zeros(0, self.lookahead, 1 + self.spatial_dim)
            empty_ids = torch.zeros(0, dtype=torch.long, device=lengths.device)
            return empty_hist, empty_tgt, empty_ids, counts

        return (
            torch.cat(history_windows, dim=0),
            torch.cat(target_windows, dim=0),
            torch.cat(seq_ids, dim=0),
            counts,
        )

    def _sampling_payload_common(self) -> dict[str, Tensor]:
        # Sampling-efficiency helper: exact AutoSTPP intensity queries only need
        # the raw/scaled history plus normalization metadata, not training windows.
        return {
            "input_normalized_flag": self.input_normalized_flag,
            "input_time_mean": self.input_time_mean,
            "input_time_std": self.input_time_std,
            "input_loc_mean": self.input_loc_mean,
            "input_loc_std": self.input_loc_std,
            "paper_dt_min": self.paper_dt_min,
            "paper_dt_range": self.paper_dt_range,
            "paper_loc_min": self.paper_loc_min,
            "paper_loc_range": self.paper_loc_range,
            "input_transform": self._input_transform_spec,
        }

    def encode_sampling_history(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor] = None,
        x_event: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
    ) -> StateContext:
        """Build only the query-side AutoSTPP history needed for sampling.

        This skips paper training-window construction and keeps just the raw and
        paper-scaled histories used by ``event_model.intensity(...)``.
        """
        del marks, x_event, x_field_at_events
        max_len = times.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)
        times_raw = self._restore_raw_times(times)
        locations_raw = self._restore_raw_locations(locations)
        delta_t_raw = self._raw_delta_times(times_raw, lengths)
        locations_paper = self._scale_paper_locations(locations_raw, valid_mask)
        delta_t_paper = self._scale_paper_delta_t(delta_t_raw, valid_mask)
        history_scaled = torch.cat([locations_paper, delta_t_paper.unsqueeze(-1)], dim=-1)

        payload = self._sampling_payload_common()
        payload.update(
            {
                "paper_history_scaled": history_scaled,
                "times_raw": times_raw,
                "locations_raw": locations_raw,
                "lengths": lengths,
            }
        )
        return StateContext(payload=payload)

    def append_sampling_event(
        self,
        state_ctx: StateContext,
        *,
        event_time_raw: Tensor,
        event_location_raw: Tensor,
    ) -> StateContext:
        """Append one sampled raw event to the cached AutoSTPP query history.

        Sampling-efficiency helper:
        - avoids rebuilding paper training windows
        - updates only the raw and paper-scaled history tensors needed for
          future intensity queries
        """
        times_raw = state_ctx.payload.get("times_raw")
        history_scaled = state_ctx.payload.get("paper_history_scaled")
        locations_raw = state_ctx.payload.get("locations_raw")
        lengths = state_ctx.payload.get("lengths")
        if times_raw is None or history_scaled is None or locations_raw is None or lengths is None:
            raise NotImplementedError(
                "append_sampling_event requires a query-only sampling state."
            )

        if times_raw.shape[0] != 1:
            raise NotImplementedError(
                "AutoSTPP sampling-state append currently supports batch size 1 only."
            )

        device = times_raw.device
        evt_t = event_time_raw.reshape(1, 1).to(device=device, dtype=times_raw.dtype)
        evt_s = event_location_raw.reshape(1, self.spatial_dim).to(
            device=device,
            dtype=locations_raw.dtype,
        )

        n_events = int(lengths.reshape(-1)[0].item())
        last_time_raw = times_raw[:, n_events - 1 : n_events] if n_events > 0 else torch.zeros_like(evt_t)
        evt_dt_raw = evt_t - last_time_raw

        paper_loc_min = self.paper_loc_min.to(device=device, dtype=evt_s.dtype)
        paper_loc_range = self.paper_loc_range.to(device=device, dtype=evt_s.dtype).clamp(min=1e-8)
        paper_dt_min = self.paper_dt_min.to(device=device, dtype=evt_t.dtype)
        paper_dt_range = self.paper_dt_range.to(device=device, dtype=evt_t.dtype).clamp(min=1e-8)

        evt_loc_scaled = (evt_s - paper_loc_min) / paper_loc_range
        evt_dt_scaled = (evt_dt_raw - paper_dt_min) / paper_dt_range
        evt_scaled = torch.cat([evt_loc_scaled, evt_dt_scaled], dim=-1).unsqueeze(1)

        next_times_raw = torch.cat([times_raw[:, :n_events], evt_t], dim=1)
        next_locations_raw = torch.cat([locations_raw[:, :n_events, :], evt_s.unsqueeze(1)], dim=1)
        next_history_scaled = torch.cat([history_scaled[:, :n_events, :], evt_scaled], dim=1)
        next_lengths = lengths.to(device=device).clone()
        next_lengths[0] = n_events + 1

        payload = self._sampling_payload_common()
        payload.update(
            {
                "paper_history_scaled": next_history_scaled,
                "times_raw": next_times_raw,
                "locations_raw": next_locations_raw,
                "lengths": next_lengths,
            }
        )
        return StateContext(payload=payload)

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
        max_len = times.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)
        times_raw = self._restore_raw_times(times)
        locations_raw = self._restore_raw_locations(locations)
        delta_t_raw = self._raw_delta_times(times_raw, lengths)
        locations_paper = self._scale_paper_locations(locations_raw, valid_mask)
        delta_t_paper = self._scale_paper_delta_t(delta_t_raw, valid_mask)
        history_scaled = torch.cat([locations_paper, delta_t_paper.unsqueeze(-1)], dim=-1)

        paper_st_x, paper_st_y, paper_seq_ids, paper_window_counts = self._build_training_windows(
            history_scaled,
            lengths,
        )

        return StateContext(
            payload={
                "paper_st_x": paper_st_x,
                "paper_st_y": paper_st_y,
                "paper_seq_ids": paper_seq_ids,
                "paper_window_counts": paper_window_counts,
                "paper_history_scaled": history_scaled,
                "times_raw": times_raw,
                "locations_raw": locations_raw,
                "lengths": lengths,
                "input_normalized_flag": self.input_normalized_flag,
                "input_time_mean": self.input_time_mean,
                "input_time_std": self.input_time_std,
                "input_loc_mean": self.input_loc_mean,
                "input_loc_std": self.input_loc_std,
                "paper_dt_min": self.paper_dt_min,
                "paper_dt_range": self.paper_dt_range,
                "paper_loc_min": self.paper_loc_min,
                "paper_loc_range": self.paper_loc_range,
                "input_transform": self._input_transform_spec,
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
