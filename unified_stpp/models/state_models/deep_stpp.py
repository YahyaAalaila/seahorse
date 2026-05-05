"""StateModel for DeepSTPP preprocessing and encoding."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from unified_stpp.data.transforms import PaperAffineTransformArtifact, transform_from_spec
from ..abstractions import StateCapabilities, StateContext, StateModel
from ..model_registry import register_state


@register_state("deep_stpp")
class DeepSTPPStateModel(StateModel):
    """Build paper windows and encode them with the DeepSTPP transformer."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        spatial_dim: int,
        event_cov_dim: int = 0,
        seq_len: int = 20,
        lookahead: int = 1,
        enc_num_heads: int = 2,
        enc_num_layers: int = 3,
        enc_dropout: float = 0.0,
        vae: bool = False,
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
        **enc_extra,
    ):
        super().__init__()
        del event_cov_dim, enc_extra

        from ..history_encoders import DeepSTPPTransformerEncoder

        self.encoder = DeepSTPPTransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=enc_num_heads,
            num_layers=enc_num_layers,
            dropout=enc_dropout,
            seq_len=seq_len,
        )
        self.hidden_dim = int(hidden_dim)
        self.spatial_dim = int(spatial_dim)
        self.seq_len = int(seq_len)
        self.lookahead = int(lookahead)
        self.vae = bool(vae)
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
            has_regularization_terms=True,
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
        xyt_scaled: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        history_windows: list[Tensor] = []
        target_windows: list[Tensor] = []
        seq_ids: list[Tensor] = []
        counts = torch.zeros(lengths.shape[0], dtype=torch.long, device=lengths.device)

        for b in range(lengths.shape[0]):
            n_events = int(lengths[b].item())
            n_windows = max(0, n_events - self.seq_len - self.lookahead + 1)
            counts[b] = n_windows
            if n_windows == 0:
                continue
            seq = xyt_scaled[b, :n_events]
            history_windows.append(
                torch.stack(
                    [seq[i : i + self.seq_len] for i in range(n_windows)],
                    dim=0,
                )
            )
            target_windows.append(
                torch.stack(
                    [
                        seq[i + self.seq_len : i + self.seq_len + self.lookahead]
                        for i in range(n_windows)
                    ],
                    dim=0,
                )
            )
            seq_ids.append(
                torch.full((n_windows,), b, device=lengths.device, dtype=torch.long)
            )

        if not history_windows:
            empty_hist = xyt_scaled.new_zeros(0, self.seq_len, 1 + self.spatial_dim)
            empty_tgt = xyt_scaled.new_zeros(0, self.lookahead, 1 + self.spatial_dim)
            empty_ids = torch.zeros(0, dtype=torch.long, device=lengths.device)
            return empty_hist, empty_tgt, empty_ids, counts

        return (
            torch.cat(history_windows, dim=0),
            torch.cat(target_windows, dim=0),
            torch.cat(seq_ids, dim=0),
            counts,
        )

    def _build_query_windows(
        self,
        xyt_scaled: Tensor,
        times_raw: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch = lengths.shape[0]
        query_windows = xyt_scaled.new_zeros(batch, self.seq_len, 1 + self.spatial_dim)
        last_times = times_raw.new_zeros(batch, 1)

        for b in range(batch):
            n_events = int(lengths[b].item())
            if n_events <= 0:
                continue
            seq = xyt_scaled[b, :n_events]
            last_times[b, 0] = times_raw[b, n_events - 1]
            if n_events >= self.seq_len:
                query_windows[b] = seq[n_events - self.seq_len : n_events]
                continue
            pad = seq[:1].expand(self.seq_len - n_events, -1)
            query_windows[b] = torch.cat([pad, seq], dim=0)

        return query_windows, last_times

    def _encode_windows(
        self,
        windows: Tensor,
        *,
        sample_latent: bool,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        mu, var = self.encoder(windows)
        if windows.shape[0] == 0:
            empty = windows.new_zeros(0, self.hidden_dim)
            zero = windows.new_tensor(0.0)
            if sample_latent:
                return empty, empty, empty, zero
            return empty, None, None, None
        if not sample_latent:
            return mu, None, None, None

        if self.training:
            z = mu + torch.randn_like(mu) * torch.sqrt(var)
        else:
            z = mu
        kl = 0.5 * (var + mu.pow(2) - torch.log(var.clamp(min=1e-8)) - 1.0).sum(-1).mean()
        return z, mu, var, kl

    def _sampling_payload_common(self) -> dict[str, Tensor]:
        # Sampling-efficiency helper: query-only intensity paths reuse the same
        # normalization and paper-space statistics as the full training encode.
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
        """Build only the query-side DeepSTPP state needed for sampling.

        This skips paper training-window construction and training-window latent
        encoding, which makes repeated ``lambda(t, s | H)`` queries much cheaper
        during post-hoc thinning-based sampling.
        """
        del marks, x_event, x_field_at_events

        max_len = locations.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)
        times_raw = self._restore_raw_times(times)
        locations_raw = self._restore_raw_locations(locations)
        delta_t_raw = self._raw_delta_times(times_raw, lengths)
        locations_paper = self._scale_paper_locations(locations_raw, valid_mask)
        delta_t_paper = self._scale_paper_delta_t(delta_t_raw, valid_mask)
        xyt_scaled = torch.cat([locations_paper, delta_t_paper.unsqueeze(-1)], dim=-1)
        query_st_x, query_last_time_raw = self._build_query_windows(
            xyt_scaled,
            times_raw,
            lengths,
        )
        query_z, _, _, _ = self._encode_windows(
            query_st_x,
            sample_latent=self.vae,
        )

        payload = self._sampling_payload_common()
        payload.update(
            {
                "query_z": query_z,
                "query_st_x": query_st_x,
                "query_last_time_raw": query_last_time_raw,
                "query_history_count": lengths.clamp(min=0, max=self.seq_len),
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
        """Incrementally update the cached DeepSTPP query window for sampling.

        Sampling-efficiency helper:
        - shifts the cached query window by one event
        - appends the new paper-scaled event
        - re-encodes only that single query window
        """
        query_st_x = state_ctx.payload.get("query_st_x")
        query_last_time_raw = state_ctx.payload.get("query_last_time_raw")
        query_history_count = state_ctx.payload.get("query_history_count")
        if query_st_x is None or query_last_time_raw is None or query_history_count is None:
            raise NotImplementedError(
                "append_sampling_event requires a query-only sampling state."
            )

        if query_st_x.shape[0] != 1:
            raise NotImplementedError(
                "DeepSTPP sampling-state append currently supports batch size 1 only."
            )

        evt_t = event_time_raw.reshape(1, 1).to(
            device=query_st_x.device,
            dtype=query_last_time_raw.dtype,
        )
        evt_s = event_location_raw.reshape(1, self.spatial_dim).to(
            device=query_st_x.device,
            dtype=query_st_x.dtype,
        )

        paper_loc_min = self.paper_loc_min.to(device=query_st_x.device, dtype=query_st_x.dtype)
        paper_loc_range = self.paper_loc_range.to(device=query_st_x.device, dtype=query_st_x.dtype).clamp(min=1e-8)
        paper_dt_min = self.paper_dt_min.to(device=query_st_x.device, dtype=query_last_time_raw.dtype)
        paper_dt_range = self.paper_dt_range.to(device=query_st_x.device, dtype=query_last_time_raw.dtype).clamp(min=1e-8)

        evt_loc_scaled = (evt_s - paper_loc_min) / paper_loc_range
        evt_dt_raw = evt_t - query_last_time_raw.to(device=query_st_x.device, dtype=evt_t.dtype)
        evt_dt_scaled = (evt_dt_raw - paper_dt_min) / paper_dt_range
        evt_scaled = torch.cat([evt_loc_scaled, evt_dt_scaled], dim=-1).unsqueeze(1)

        next_query_st_x = torch.cat([query_st_x[:, 1:, :], evt_scaled], dim=1)
        next_query_z, _, _, _ = self._encode_windows(
            next_query_st_x,
            sample_latent=self.vae,
        )

        payload = self._sampling_payload_common()
        payload.update(
            {
                "query_z": next_query_z,
                "query_st_x": next_query_st_x,
                "query_last_time_raw": evt_t,
                "query_history_count": (
                    query_history_count.to(device=query_st_x.device) + 1
                ).clamp(max=self.seq_len),
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

        max_len = locations.shape[1]
        valid_mask = self._valid_mask(lengths, max_len)
        times_raw = self._restore_raw_times(times)
        locations_raw = self._restore_raw_locations(locations)
        delta_t_raw = self._raw_delta_times(times_raw, lengths)
        locations_paper = self._scale_paper_locations(locations_raw, valid_mask)
        delta_t_paper = self._scale_paper_delta_t(delta_t_raw, valid_mask)
        xyt_scaled = torch.cat([locations_paper, delta_t_paper.unsqueeze(-1)], dim=-1)

        paper_st_x, paper_st_y, paper_seq_ids, paper_window_counts = self._build_training_windows(
            xyt_scaled,
            lengths,
        )
        query_st_x, query_last_time_raw = self._build_query_windows(
            xyt_scaled,
            times_raw,
            lengths,
        )

        z, qm, qv, kl_loss = self._encode_windows(
            paper_st_x,
            sample_latent=self.vae,
        )
        query_z, _, _, _ = self._encode_windows(
            query_st_x,
            sample_latent=self.vae,
        )

        return StateContext(
            payload={
                "z": z,
                "qm": qm,
                "qv": qv,
                "kl_loss": kl_loss,
                "paper_st_x": paper_st_x,
                "paper_st_y": paper_st_y,
                "paper_seq_ids": paper_seq_ids,
                "paper_window_counts": paper_window_counts,
                "query_z": query_z,
                "query_st_x": query_st_x,
                "query_last_time_raw": query_last_time_raw,
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
            },
            kl_loss=kl_loss,
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
        kl_loss = state_ctx.payload.get("kl_loss")
        if isinstance(kl_loss, Tensor):
            return {"kl_loss": kl_loss}
        if kl_loss is not None:
            return {"kl_loss": torch.as_tensor(kl_loss)}
        return {}
