"""Independent DNSK+Barrier implementation (Dong et al., ICLR 2023).

Equations 3--8 and Algorithm 1 of arXiv:2211.11179 define the signed separable
history/displacement kernel and detached moving log barrier. No author source
was available to port. Quadrature and positivity are explicit diagnostics;
a grid check does not prove positivity between grid nodes.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from seahorse.data.transforms import transform_from_spec

from ..abstractions import EventCapabilities, EventModel, StateContext
from ..model_registry import register_event


class _Basis(nn.Module):
    """Independent two-hidden-layer Softplus networks with signed linear outputs."""

    def __init__(self, input_dim: int, rank: int, width: int):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, width), nn.Softplus(),
                          nn.Linear(width, width), nn.Softplus(), nn.Linear(width, 1))
            for _ in range(rank)
        ])

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([net(x) for net in self.nets], dim=-1)


@register_event("dnsk")
class DNSKEventModel(EventModel):
    """Signed low-rank intensity with a finite-rectangle compensator.

    Temporal interpolation and its antiderivative use the same piecewise-linear
    basis. Spatial midpoint nodes live in the displacement ball, then are clipped
    to the observation rectangle separately for every history event (Eq. 8).
    """

    def __init__(self, *, temporal_rank=2, spatial_rank=2, width=64, mu=1.0,
                 tau_max=0.25, radius=1.0, temporal_nodes=65, spatial_nodes=24,
                 barrier_time_nodes=8, barrier_space_nodes=8, barrier_weight=1.0,
                 barrier_decay=1.001, barrier_epsilon=1e-4,
                 space_min=(0.0, 0.0), space_max=(1.0, 1.0), t0=0.0,
                 query_chunk_size=256, **kwargs):
        super().__init__()
        if kwargs:
            raise ValueError(f"Unknown DNSK options: {sorted(kwargs)}")
        if min(temporal_rank, spatial_rank, width, spatial_nodes, barrier_time_nodes,
               barrier_space_nodes, query_chunk_size) < 1 or temporal_nodes < 2:
            raise ValueError("DNSK ranks, widths and quadrature sizes must be positive.")
        if min(mu, tau_max, radius, barrier_epsilon) <= 0 or barrier_decay < 1:
            raise ValueError("Invalid DNSK rate, influence support or barrier schedule.")
        self.temporal_rank, self.spatial_rank = temporal_rank, spatial_rank
        self.tau_max, self.radius, self.t0 = float(tau_max), float(radius), float(t0)
        self.barrier_time_nodes, self.barrier_space_nodes = barrier_time_nodes, barrier_space_nodes
        self.barrier_epsilon = float(barrier_epsilon)
        self.barrier_decay = float(barrier_decay)
        self.query_chunk_size = int(query_chunk_size)
        self.psi = _Basis(1, temporal_rank, width)
        self.phi = _Basis(1, temporal_rank, width)
        self.u = _Basis(2, spatial_rank, width)
        self.v = _Basis(2, spatial_rank, width)
        self.alpha = nn.Parameter(torch.empty(temporal_rank, spatial_rank))
        nn.init.normal_(self.alpha, std=1e-3)
        self.log_mu = nn.Parameter(torch.tensor(math.log(mu), dtype=torch.float32))
        self.register_buffer("barrier_weight", torch.tensor(float(barrier_weight)))
        self.register_buffer("optimizer_updates", torch.tensor(0, dtype=torch.long))
        lo, hi = torch.tensor(space_min, dtype=torch.float32), torch.tensor(space_max, dtype=torch.float32)
        if lo.shape != (2,) or hi.shape != (2,) or not torch.all(hi > lo):
            raise ValueError("DNSK requires a nonempty two-dimensional observation rectangle.")
        self.register_buffer("space_min", lo)
        self.register_buffer("space_max", hi)
        self.register_buffer("temporal_grid", torch.linspace(0, tau_max, temporal_nodes))
        axis = (torch.arange(spatial_nodes, dtype=torch.float32) + 0.5) * (2 * radius / spatial_nodes) - radius
        offsets = torch.cartesian_prod(axis, axis)
        self.register_buffer("spatial_offsets", offsets)
        self.register_buffer("spatial_offset_weights",
                             (offsets.square().sum(-1) <= radius**2).float() * (2 * radius / spatial_nodes)**2)

    @property
    def capabilities(self):
        return EventCapabilities(
            training_objective="nll_plus_log_barrier", metric_key="nll",
            objective_description="DNSK MLE with Algorithm 1 moving log barrier",
            objective_includes_regularization=True, nll_kind="exact",
            nll_path="quadrature",
            nll_description="conditional next-event NLL with finite-domain numerical quadrature",
            supports_raw_reporting=True, has_intensity=True, exposes_eventwise_terms=True,
        )

    def advance_barrier_schedule(self):
        """Call once after each actual optimizer update (never during evaluation)."""
        self.barrier_weight.div_(self.barrier_decay)
        self.optimizer_updates.add_(1)

    def _tables(self) -> tuple[Tensor, Tensor, Tensor]:
        values = self.phi(self.temporal_grid[:, None])
        step = self.temporal_grid[1] - self.temporal_grid[0]
        areas = (values[:-1] + values[1:]) * (step / 2)
        integral = torch.cat([areas.new_zeros(1, areas.shape[-1]), areas.cumsum(0)], dim=0)
        spatial = self.v(self.spatial_offsets)
        return values, integral, spatial

    def temporal_basis(self, dt: Tensor, tables=None, *, integral=False) -> Tensor:
        values, cumulative, _ = self._tables() if tables is None else tables
        step = self.temporal_grid[1] - self.temporal_grid[0]
        clipped = dt.clamp(0, self.tau_max)
        left = (clipped / step).floor().long().clamp(max=values.shape[0] - 2)
        delta = clipped - self.temporal_grid[left]
        slope = (values[left + 1] - values[left]) / step
        if integral:
            return cumulative[left] + delta[..., None] * values[left] + delta[..., None].square() * slope / 2
        result = values[left] + delta[..., None] * slope
        return result * ((dt >= 0) & (dt <= self.tau_max))[..., None]

    def spatial_integrals(self, history_locations: Tensor, tables=None) -> Tensor:
        """Integrate each signed spatial displacement basis over the actual rectangle."""
        _, _, spatial = self._tables() if tables is None else tables
        points = history_locations[..., None, :] + self.spatial_offsets
        inside = ((points >= self.space_min) & (points <= self.space_max)).all(-1)
        weights = inside.to(spatial.dtype) * self.spatial_offset_weights
        return torch.einsum("...q,qr->...r", weights, spatial)

    def kernel(self, source_times: Tensor, source_locations: Tensor,
               query_times: Tensor, query_locations: Tensor, tables=None) -> Tensor:
        """Eq. 4, with broadcasting over source/query axes supplied by the caller."""
        dt = query_times - source_times
        ds = query_locations - source_locations
        temporal = self.psi(source_times[..., None]) * self.temporal_basis(dt, tables)
        spatial = self.u(source_locations) * self.v(ds)
        result = torch.einsum("...l,lr,...r->...", temporal, self.alpha, spatial)
        return result * ((dt > 0) & (dt <= self.tau_max) & (ds.square().sum(-1) <= self.radius**2))

    def _intensity(self, history_times, history_locations, lengths, query_times, query_locations, tables=None):
        """Batched B×Q queries with strict-history causality; no positivity clipping."""
        tables = self._tables() if tables is None else tables
        valid = torch.arange(history_times.shape[1], device=history_times.device)[None] < lengths[:, None]
        chunks = []
        for start in range(0, query_times.shape[1], self.query_chunk_size):
            qt, qs = query_times[:, start:start+self.query_chunk_size], query_locations[:, start:start+self.query_chunk_size]
            pairs = self.kernel(history_times[:, None, :], history_locations[:, None, :, :],
                                qt[:, :, None], qs[:, :, None, :], tables)
            chunks.append(self.log_mu.exp() + (pairs * valid[:, None, :]).sum(-1))
        return torch.cat(chunks, dim=1) if chunks else history_times.new_empty(history_times.shape[0], 0)

    def interval_compensator(self, history_times, history_locations, lengths, start, end, tables=None):
        """Eq. 8 on B×Q intervals; includes the finite-domain boundary correction."""
        tables = self._tables() if tables is None else tables
        valid = torch.arange(history_times.shape[1], device=history_times.device)[None] < lengths[:, None]
        temporal = self.temporal_basis(end[:, :, None] - history_times[:, None, :], tables, integral=True)
        temporal = temporal - self.temporal_basis(start[:, :, None] - history_times[:, None, :], tables, integral=True)
        history_t = self.psi(history_times[..., None])
        history_s = self.u(history_locations) * self.spatial_integrals(history_locations, tables)
        influence = torch.einsum("bqhl,bhl,lr,bhr->bqh", temporal, history_t, self.alpha, history_s)
        # The clipped antiderivative is zero before a historical event occurs.
        return (end-start) * self.log_mu.exp() * (self.space_max-self.space_min).prod() + (influence * valid[:, None, :]).sum(-1)

    def _barrier(self, times, locs, lengths, tables):
        batch_size = times.shape[0]
        final = times.gather(1, (lengths - 1).clamp(min=0)[:, None]).squeeze(1)
        fractions = (torch.arange(self.barrier_time_nodes, device=times.device, dtype=times.dtype) + 0.5) / self.barrier_time_nodes
        grid_times = self.t0 + (final-self.t0)[:, None] * fractions
        axis = (torch.arange(self.barrier_space_nodes, device=times.device, dtype=times.dtype) + 0.5) / self.barrier_space_nodes
        points = self.space_min + torch.cartesian_prod(axis, axis) * (self.space_max-self.space_min)
        query_times = grid_times[:, :, None].expand(-1, -1, len(points)).reshape(batch_size, -1)
        query_locs = points[None, None].expand(batch_size, self.barrier_time_nodes, -1, -1).reshape(batch_size, -1, 2)
        intensity = self._intensity(times, locs, lengths, query_times, query_locs, tables)
        # Algorithm 1 treats b as fixed during the parameter update.
        bound = intensity.amin(dim=1, keepdim=True).detach() - self.barrier_epsilon
        penalty = -torch.log(intensity - bound).mean(dim=1)
        return penalty, intensity.amin()

    def training_loss(self, *, times, locations, lengths, state: StateContext,
                      state_regularization_terms=None, x_field_at_events=None, marks=None, device=None):
        del times, locations, state_regularization_terms, x_field_at_events, marks, device
        times, locs = state.payload["times"], state.payload["locations"]
        batch_size, n = times.shape
        mask = torch.arange(n, device=times.device)[None] < lengths[:, None]
        if not bool(mask.any()):
            zero = self.log_mu * 0
            return {"loss": zero, "nll": zero, "total_events": zero,
                    "nll_matrix": times.new_zeros(batch_size, n), "mask": mask,
                    "nll_per_event": times.new_zeros(batch_size),
                    "eventwise_target_indices": torch.full((batch_size, n), -1, dtype=torch.long, device=times.device)}
        if bool(((times < self.t0) & mask).any()):
            raise ValueError("DNSK events precede the declared observation origin.")
        if bool((((locs < self.space_min) | (locs > self.space_max)).any(-1) & mask).any()):
            raise ValueError("DNSK event lies outside the declared observation rectangle.")
        if n > 1 and bool(((times[:, 1:] <= times[:, :-1]) & mask[:, 1:]).any()):
            raise ValueError("DNSK requires strictly increasing event times.")
        tables = self._tables()
        intensity = self._intensity(times, locs, lengths, times, locs, tables)
        if bool(((intensity[mask] <= 0) | ~torch.isfinite(intensity[mask])).any()):
            raise ValueError("DNSK signed intensity became non-positive at a scored event; no clipping is applied.")
        previous = torch.cat([times.new_full((batch_size, 1), self.t0), times[:, :-1]], dim=1)
        integral = self.interval_compensator(times, locs, lengths, previous, times, tables)
        nll_matrix = torch.where(mask, -torch.log(intensity.clamp_min(torch.finfo(times.dtype).tiny)) + integral, 0.0)
        nll = nll_matrix.sum() / mask.sum()
        per_seq = nll_matrix.sum(1) / lengths.clamp(min=1)
        penalty, grid_min = self._barrier(times, locs, lengths, tables)
        if not self.training and float(grid_min.detach()) <= 0:
            raise ValueError("DNSK signed intensity failed the evaluation positivity grid.")
        active = lengths > 0
        barrier = penalty[active].mean()
        extra = self.raw_reporting_metrics(state=state, nll=nll, total_events=mask.sum())
        extra.update({"dnsk_barrier": float(barrier.detach()), "dnsk_grid_min_intensity": float(grid_min.detach()),
                      "dnsk_barrier_weight": float(self.barrier_weight), "dnsk_temporal_nodes": float(len(self.temporal_grid)),
                      "dnsk_spatial_nodes": float(round(len(self.spatial_offsets)**0.5))})
        indices = torch.arange(n, device=times.device)[None].expand(batch_size, -1).masked_fill(~mask, -1)
        return {"loss": nll + self.barrier_weight * barrier / lengths[active].float().mean(),
                "nll": nll, "total_events": mask.sum(), "nll_matrix": nll_matrix,
                "mask": mask, "nll_per_event": per_seq, "eventwise_target_indices": indices,
                "extra_metrics": extra}

    def intensity(self, *, state, query_times, query_locations, query_lengths=None, **kwargs):
        del query_lengths, kwargs
        qtimes, qlocs = query_times, query_locations
        if qtimes.ndim == 1:
            qtimes, qlocs = qtimes[None], qlocs[None]
        transform = transform_from_spec(state.payload.get("input_transform"))
        if transform is not None:
            qtimes = transform.forward_times(qtimes)
            qlocs = transform.forward_locations(qlocs)
        values = self._intensity(state.payload["times"], state.payload["locations"],
                                 state.payload["lengths"], qtimes, qlocs)
        inside = ((qlocs >= self.space_min) & (qlocs <= self.space_max)).all(-1)
        if bool(((values[inside] <= 0) | ~torch.isfinite(values[inside])).any()):
            raise ValueError("DNSK query encountered non-positive signed intensity.")
        if transform is not None:
            values = values * torch.exp(-transform.reporting_correction(ref=values))
        return torch.where(inside, values, 0.0)
