"""Independent, paper-based GLIDE history encoder.

This module intentionally does not import or copy AONE-NLP/GLIDE.  See
``docs/models/GLIDE_PROVENANCE.md`` for the source mapping and limitations.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..abstractions import StateCapabilities, StateContext, StateModel
from ..model_registry import register_state


class _CausalGraphLayer(nn.Module):
    """Small directed causal message-passing layer without a PyG dependency."""

    def __init__(self, dim: int):
        super().__init__()
        self.self_linear = nn.Linear(dim, dim)
        self.message_linear = nn.Linear(dim, dim)
        self.edge_linear = nn.Linear(3, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, nodes: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # Nodes represent one sequence. Every edge has source < destination.
        out = self.self_linear(nodes)
        if edge_index.numel():
            src, dst = edge_index
            message = self.message_linear(nodes[src]) + self.edge_linear(edge_attr)
            aggregate = torch.zeros_like(out)
            aggregate.index_add_(0, dst, message)
            degree = torch.zeros(nodes.shape[0], 1, device=nodes.device, dtype=nodes.dtype)
            degree.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=nodes.device, dtype=nodes.dtype))
            out = out + aggregate / degree.clamp_min(1.0)
        return self.norm(nodes + torch.nn.functional.gelu(out))


@register_state("glide")
class GLIDEStateModel(StateModel):
    """Dual temporal/graph conditioning used by GLIDE.

    Tokens for target event ``i+1`` are conditioned on the graph state at event
    ``i``.  All token statistics are fitted on the training split by
    :class:`GLIDEConfig`; no validation/test coordinates enter them.
    """

    def __init__(
        self,
        *,
        spatial_dim: int = 2,
        d_model: int = 64,
        temporal_windows: Sequence[int] = (64, 32, 8, 2),
        spatial_windows: Sequence[int] = (64, 32, 8, 2),
        spatial_grid_sizes: Sequence[float] = (.2, .1, .02, .005),
        graph_layers: int = 2,
        graph_time_scales: Sequence[int] = (0, 0, 0),
        graph_neighbors: Sequence[int] = (0, 0, 0),
        num_time_harmonics: int = 8,
        input_normalized: bool = False,
        input_time_mean: float = 0.0,
        input_time_std: float = 1.0,
        input_loc_mean: Sequence[float] = (0.0, 0.0),
        input_loc_std: Sequence[float] = (1.0, 1.0),
        abs_time_min: float = 0.0,
        abs_time_range: float = 1.0,
        delta_mean: float = 0.0,
        delta_std: float = 1.0,
        delta_min: float = 0.0,
        delta_range: float = 1.0,
        loc_min: Sequence[float] = (0.0, 0.0),
        loc_range: Sequence[float] = (1.0, 1.0),
        loc_delta_mean: Sequence[float] = (0.0, 0.0),
        loc_delta_std: Sequence[float] = (1.0, 1.0),
    ):
        super().__init__()
        if spatial_dim != 2:
            raise ValueError("GLIDE's published implementation is two-dimensional and unmarked.")
        self.spatial_dim, self.d_model = int(spatial_dim), int(d_model)
        self.temporal_windows = tuple(int(w) for w in temporal_windows)
        self.spatial_windows = tuple(int(w) for w in spatial_windows)
        self.spatial_grid_sizes = tuple(float(v) for v in spatial_grid_sizes)
        self.graph_time_scales = tuple(int(v) for v in graph_time_scales)
        self.graph_neighbors = tuple(int(v) for v in graph_neighbors)
        self.num_time_harmonics = int(num_time_harmonics)
        if len(self.spatial_windows) != len(self.spatial_grid_sizes):
            raise ValueError("GLIDE spatial_windows and spatial_grid_sizes must have equal length.")
        if len(self.graph_time_scales) != len(self.graph_neighbors):
            raise ValueError("GLIDE graph_time_scales and graph_neighbors must have equal length.")
        # [absolute time, standardized delta, location, standardized location delta,
        #  temporal summaries, spatial summaries].
        feature_dim = (2 + 2 * spatial_dim + 2 * self.num_time_harmonics
                       + 3 * len(self.temporal_windows) + 3 * spatial_dim * len(self.spatial_windows))
        self.node_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=4 * d_model,
                                       dropout=0.0, activation="gelu", batch_first=True, norm_first=True),
            num_layers=2,
        )
        self.graph_layers = nn.ModuleList(_CausalGraphLayer(d_model) for _ in range(int(graph_layers)))
        self.spatial_fourier_frequency = nn.Parameter(torch.randn(16, spatial_dim))
        self.spatial_fourier_projection = nn.Linear(32, d_model)
        self.fusion = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.register_buffer("input_normalized_flag", torch.tensor(float(input_normalized)))
        for name, value in {
            "input_time_mean": input_time_mean, "input_time_std": max(float(input_time_std), 1e-8),
            "abs_time_min": abs_time_min, "abs_time_range": max(float(abs_time_range), 1e-8),
            "delta_mean": delta_mean, "delta_std": max(float(delta_std), 1e-8),
            "delta_min": delta_min, "delta_range": max(float(delta_range), 1e-8),
        }.items():
            self.register_buffer(name, torch.tensor(float(value), dtype=torch.float32))
        for name, value in {
            "input_loc_mean": input_loc_mean, "input_loc_std": input_loc_std,
            "loc_min": loc_min, "loc_range": loc_range,
            "loc_delta_mean": loc_delta_mean, "loc_delta_std": loc_delta_std,
        }.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).reshape(spatial_dim).clamp_min(1e-8) if name.endswith("std") or name.endswith("range") else torch.as_tensor(value, dtype=torch.float32).reshape(spatial_dim))

    @property
    def capabilities(self) -> StateCapabilities:
        return StateCapabilities(has_query_state=False, has_sequence_states=True, state_kind="history_passthrough")

    def _raw(self, times: Tensor, locations: Tensor) -> tuple[Tensor, Tensor]:
        if self.input_normalized_flag.item() <= .5:
            return times, locations
        return (times * self.input_time_std.to(times) + self.input_time_mean.to(times),
                locations * self.input_loc_std.to(locations) + self.input_loc_mean.to(locations))

    @staticmethod
    def _summary(values: Tensor, i: int, width: int) -> Tensor:
        part = values[max(0, i - width + 1): i + 1]
        mean = part.mean(dim=0)
        std = part.std(dim=0, unbiased=False)
        trend = (part[-1] - part[0]) / max(part.shape[0], 1)
        return torch.cat((mean.reshape(-1), std.reshape(-1), trend.reshape(-1)))

    def _features(self, time: Tensor, loc: Tensor) -> Tensor:
        n = time.shape[0]
        dt = torch.zeros_like(time)
        dloc = torch.zeros_like(loc)
        if n > 1:
            dt[1:] = time[1:] - time[:-1]
            dloc[1:] = loc[1:] - loc[:-1]
        t_abs = (time - self.abs_time_min.to(time)) / self.abs_time_range.to(time)
        t_delta = ((dt - self.delta_mean.to(time)) / self.delta_std.to(time)).clamp(-5., 5.)
        loc_norm = (loc - self.loc_min.to(loc)) / self.loc_range.to(loc)
        loc_delta = ((dloc - self.loc_delta_mean.to(loc)) / self.loc_delta_std.to(loc)).clamp(-5., 5.)
        # The author preprocessing explicitly resets the first location-difference
        # feature after z-scoring; retaining that convention avoids a spurious first node.
        loc_delta[0] = 0.
        rows = []
        for i in range(n):
            harmonic = 2 * torch.pi * t_abs[i] * torch.arange(1, self.num_time_harmonics + 1, device=time.device, dtype=time.dtype)
            parts = [t_abs[i:i+1], t_delta[i:i+1], loc_norm[i], loc_delta[i], harmonic.sin(), harmonic.cos()]
            parts += [self._summary(t_delta.unsqueeze(-1), i, w) for w in self.temporal_windows]
            for w, grid in zip(self.spatial_windows, self.spatial_grid_sizes):
                snapped = torch.round(loc_norm / grid) * grid if grid > 0 else loc_norm
                parts.append(self._summary(snapped, i, w))
            rows.append(torch.cat(parts))
        return torch.stack(rows), dt, loc_norm

    def _graph(self, times_abs_norm: Tensor, loc: Tensor) -> tuple[Tensor, Tensor]:
        edges, attrs = [], []
        n = times_abs_norm.shape[0]
        distance_matrix = torch.cdist(loc, loc)
        # Consecutive causal edges are always present, matching the author preprocessing.
        for dst in range(1, n):
            src = dst - 1
            # The released preprocessing divides by each *whole sequence* span.
            # That makes an old history representation change when a future suffix
            # is appended.  Fixed train-fitted token coordinates make these causal.
            d_t = (times_abs_norm[dst] - times_abs_norm[src]).abs()
            d_s = distance_matrix[dst, src]
            edges.append((src, dst))
            attrs.append(torch.stack((d_t * torch.sigmoid(-5 * d_t), d_s * torch.exp(-distance_matrix[dst, src]) * 1.2, d_t.new_zeros(()))))
        # Same causal, scale-indexed nearest-neighbour rule as preprocess.py.
        for scale_idx, (scale, k_base) in enumerate(zip(self.graph_time_scales, self.graph_neighbors)):
            if scale <= 0 or k_base <= 0:
                continue
            for dst in range(1, n):
                candidates = torch.arange(max(0, dst - scale), dst, device=loc.device)
                if not candidates.numel():
                    continue
                k = min(int(k_base), int(candidates.numel()))
                ranked = distance_matrix[dst, candidates].topk(k, largest=False).indices
                for idx in ranked:
                    src = int(candidates[idx].item())
                    d_t = (times_abs_norm[dst] - times_abs_norm[src]).abs()
                    d_s = distance_matrix[dst, src]
                    edges.append((src, dst))
                    attrs.append(torch.stack((d_t * torch.sigmoid(-5 * d_t), d_s * torch.exp(-distance_matrix[dst, src]) * 1.2,
                                              d_t.new_tensor(float((scale_idx + 1) / len(self.graph_time_scales))))))
        if not edges:
            return (torch.empty(2, 0, dtype=torch.long, device=times_abs_norm.device),
                    torch.empty(0, 3, dtype=times_abs_norm.dtype, device=times_abs_norm.device))
        return torch.tensor(edges, device=times_abs_norm.device, dtype=torch.long).t(), torch.stack(attrs)

    def encode_history(self, *, times: Tensor, locations: Tensor, lengths: Tensor,
                       marks: Optional[Tensor] = None, x_event: Optional[Tensor] = None,
                       x_field_at_events: Optional[Tensor] = None) -> StateContext:
        del marks, x_event, x_field_at_events
        if locations.shape[-1] != self.spatial_dim:
            raise ValueError("GLIDE location dimension does not match its configured spatial_dim.")
        raw_time, raw_loc = self._raw(times, locations)
        B, T = times.shape
        condition = times.new_zeros(B, max(T - 1, 0), 3 * self.d_model)
        target = times.new_zeros(B, max(T - 1, 0), 1 + self.spatial_dim)
        mask = torch.zeros(B, max(T - 1, 0), dtype=torch.bool, device=times.device)
        last = times.new_zeros(B, 1, 3 * self.d_model)
        for b in range(B):
            n = int(lengths[b].item())
            if n <= 0:
                continue
            feat, dt, loc_norm = self._features(raw_time[b, :n], raw_loc[b, :n])
            projected = self.node_projection(feat)
            causal = torch.triu(torch.ones(n, n, device=times.device, dtype=torch.bool), diagonal=1)
            temporal = self.temporal_encoder(projected.unsqueeze(0), mask=causal).squeeze(0)
            time_norm = (raw_time[b, :n] - self.abs_time_min.to(raw_time)) / self.abs_time_range.to(raw_time)
            edge_index, edge_attr = self._graph(time_norm, loc_norm)
            fourier = torch.cat(((2 * torch.pi * loc_norm @ self.spatial_fourier_frequency.t()).sin(),
                                 (2 * torch.pi * loc_norm @ self.spatial_fourier_frequency.t()).cos()), dim=-1)
            spatial = projected + self.spatial_fourier_projection(fourier)
            for layer in self.graph_layers:
                spatial = layer(spatial, edge_index, edge_attr)
            fused = self.fusion(torch.cat((temporal, spatial), dim=-1))
            all_cond = torch.cat((temporal, spatial, fused), dim=-1)
            last[b, 0] = all_cond[-1]
            if n > 1:
                m = n - 1
                condition[b, :m] = all_cond[:m]
                # Upstream min--max normalization does not clip held-out values;
                # preserving that allows a raw test event outside train support.
                target[b, :m, 0] = (dt[1:] - self.delta_min.to(dt)) / self.delta_range.to(dt)
                target[b, :m, 1:] = loc_norm[1:]
                mask[b, :m] = True
        return StateContext(payload={
            "glide_tokens": target, "glide_cond": condition, "glide_mask": mask,
            "glide_cond_last": last, "glide_total_events": mask.sum().to(dtype=times.dtype),
            "glide_token_delta_min": self.delta_min, "glide_token_delta_range": self.delta_range,
            "glide_token_loc_min": self.loc_min, "glide_token_loc_range": self.loc_range,
        })

    def sequence_states(self, state_ctx: StateContext, **kwargs) -> StateContext:
        del kwargs
        return state_ctx
