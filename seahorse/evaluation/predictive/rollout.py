"""Predictive rollout mechanics for sample-based model comparison."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from seahorse.data.transforms import transform_from_spec
from seahorse.evaluation.intensity import paper_output_scale_factor
from seahorse.evaluation.context import GenerativeRollouts
from seahorse.evaluation.runtime import (
    FrameWindow,
    LoadedRun,
    append_events,
    cap_history,
    copy_history,
    derive_seed,
    history_until_time,
    seed_everything,
)
from seahorse.runner.runner import STPPRunner


SUPPORTED_PRESETS = {"smash", "diffusion_stpp", "deep_stpp", "auto_stpp"}
AUTOREGRESSIVE_ROLLOUT_DEPTH = 10


def is_exact_preset(preset: str) -> bool:
    return preset in {"deep_stpp", "auto_stpp"}


def _spatial_bounds_from_seqs(
    seqs: list[dict[str, np.ndarray]],
    pad: float = 0.08,
) -> tuple[float, float, float, float]:
    loc_chunks = [
        np.asarray(s["locations"], dtype=np.float32)
        for s in seqs
        if np.asarray(s["locations"], dtype=np.float32).size > 0
    ]
    if not loc_chunks:
        return (0.0, 1.0, 0.0, 1.0)
    all_locs = np.concatenate(loc_chunks, axis=0).astype(np.float32)
    lo = all_locs.min(axis=0)
    hi = all_locs.max(axis=0)
    span = np.maximum(hi - lo, 1e-4)
    return (
        float(lo[0] - pad * span[0]),
        float(hi[0] + pad * span[0]),
        float(lo[1] - pad * span[1]),
        float(hi[1] + pad * span[1]),
    )


@dataclass(frozen=True)
class ExactProposalConfig:
    mode: str = "coarse"
    time_bins: int = 8
    spatial_bins: int = 8
    safety: float = 2.0


@dataclass
class ExactProposalCache:
    time_edges: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    cell_bounds: np.ndarray
    cell_probs: np.ndarray
    total_rates: np.ndarray
    cell_area: float
    safety: float


def collect_window_events(
    times_list: list[np.ndarray],
    locs_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not times_list:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    counts = np.asarray([arr.shape[0] for arr in times_list], dtype=np.int32)
    if int(counts.sum()) == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            counts,
        )
    times = np.concatenate(times_list, axis=0).astype(np.float32, copy=False)
    locs = np.concatenate(locs_list, axis=0).astype(np.float32, copy=False)
    return times, locs, counts


def normalize_history_for_runner(
    history: dict[str, np.ndarray],
    norm_stats: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(history["times"], dtype=np.float32)
    locs = np.asarray(history["locations"], dtype=np.float32)
    if not bool(norm_stats.get("normalize", False)):
        return times, locs
    t_mean = float(norm_stats.get("time_mean", 0.0))
    t_std = max(float(norm_stats.get("time_std", 1.0)), 1e-8)
    loc_mean = np.asarray(norm_stats.get("loc_mean", [0.0, 0.0]), dtype=np.float32)
    loc_std = np.maximum(
        np.asarray(norm_stats.get("loc_std", [1.0, 1.0]), dtype=np.float32),
        1e-8,
    )
    return (times - t_mean) / t_std, (locs - loc_mean) / loc_std


def build_state_from_history(
    runner: STPPRunner,
    history: dict[str, np.ndarray],
    device: torch.device,
):
    times_norm, locs_norm = normalize_history_for_runner(history, runner.norm_stats)
    times_t = torch.tensor(times_norm, dtype=torch.float32, device=device).unsqueeze(0)
    locs_t = torch.tensor(locs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    lengths_t = torch.tensor([len(times_norm)], dtype=torch.long, device=device)
    with torch.no_grad():
        return runner.model.state_model.encode_sampling_history(
            times=times_t,
            locations=locs_t,
            lengths=lengths_t,
        )


def append_sampling_state(
    runner: STPPRunner,
    state_ctx,
    *,
    history: dict[str, np.ndarray],
    event_time_raw: float,
    event_location_raw: np.ndarray,
    device: torch.device,
):
    state_model = runner.model.state_model
    evt_t = torch.tensor([event_time_raw], dtype=torch.float32, device=device)
    evt_s = torch.tensor(event_location_raw, dtype=torch.float32, device=device).reshape(1, -1)
    try:
        return state_model.append_sampling_event(
            state_ctx,
            event_time_raw=evt_t,
            event_location_raw=evt_s,
        )
    except NotImplementedError:
        return build_state_from_history(runner, history, device)


def _flatten_sample_tensor(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim == 3:
        return samples.reshape(-1, samples.shape[-1])
    if samples.ndim == 2:
        return samples
    raise ValueError(f"Unexpected sample tensor shape: {tuple(samples.shape)}")


def _denormalize_spatial_samples(runner: STPPRunner, sample_tokens: np.ndarray) -> np.ndarray:
    state_model = runner.model.state_model
    spatial_dim = int(getattr(state_model, "spatial_dim", 2))
    spatial_tokens = sample_tokens[:, -spatial_dim:]
    loc_min = state_model.token_loc_min.detach().cpu().numpy().astype(np.float32)
    loc_range = state_model.token_loc_range.detach().cpu().numpy().astype(np.float32)
    return spatial_tokens * loc_range + loc_min


def _denormalize_time_samples(runner: STPPRunner, sample_tokens: np.ndarray) -> np.ndarray:
    preset = runner.config.model.preset
    state_model = runner.model.state_model
    token_t = sample_tokens[:, 0].astype(np.float32)
    if preset == "diffusion_stpp":
        t_min = float(state_model.token_delta_t_min.detach().cpu().item())
        t_range = float(state_model.token_delta_t_range.detach().cpu().item())
        return token_t * t_range + t_min
    if preset == "smash":
        if bool(getattr(state_model, "minmax_normalize_time", False)):
            if bool(getattr(state_model, "log_normalization", False)):
                t_min = float(state_model.token_time_min_log.detach().cpu().item())
                t_range = float(state_model.token_time_range_log.detach().cpu().item())
                return np.exp(token_t * t_range + t_min).astype(np.float32)
            t_min = float(state_model.token_time_min_raw.detach().cpu().item())
            t_range = float(state_model.token_time_range_raw.detach().cpu().item())
            return (token_t * t_range + t_min).astype(np.float32)
        if bool(getattr(state_model, "log_normalization", False)):
            return np.exp(token_t).astype(np.float32)
        return token_t
    raise SystemExit(f"Unsupported preset for temporal denormalization: {preset}")


def sample_next_event_smash(
    runner: STPPRunner,
    history: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[float, np.ndarray]:
    state = build_state_from_history(runner, history, device)
    event_model = runner.model.event_model
    total_steps = int(event_model.score_matching.sampling_timesteps)
    with torch.enable_grad():
        out = event_model.sample_native(
            state=state,
            step=total_steps,
            is_last=True,
            n_samples=1,
            batch_size=1,
            device=device,
        )
    flat = _flatten_sample_tensor(out["samples"])[:1].detach().cpu().numpy()
    delta_t = float(_denormalize_time_samples(runner, flat)[0])
    loc = _denormalize_spatial_samples(runner, flat)[0]
    next_time = float(history["times"][-1] + max(delta_t, 0.0))
    return next_time, np.asarray(loc, dtype=np.float32)


def sample_next_events_smash_batch(
    runner: STPPRunner,
    history: dict[str, np.ndarray],
    n_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    state = build_state_from_history(runner, history, device)
    event_model = runner.model.event_model
    total_steps = int(event_model.score_matching.sampling_timesteps)
    with torch.enable_grad():
        out = event_model.sample_native(
            state=state,
            step=total_steps,
            is_last=True,
            n_samples=int(n_samples),
            batch_size=1,
            device=device,
        )
    flat = _flatten_sample_tensor(out["samples"])[: int(n_samples)].detach().cpu().numpy()
    delta_t = _denormalize_time_samples(runner, flat)
    locs = _denormalize_spatial_samples(runner, flat).astype(np.float32, copy=False)
    next_times = (
        float(history["times"][-1]) + np.maximum(delta_t.astype(np.float32, copy=False), 0.0)
    ).astype(np.float32, copy=False)
    return next_times, locs


def sample_next_event_diffusion(
    runner: STPPRunner,
    history: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[float, np.ndarray]:
    state = build_state_from_history(runner, history, device)
    out = runner.model.event_model.sample_native(
        state=state,
        batch_size=1,
        device=device,
    )
    flat = _flatten_sample_tensor(out["samples"])[:1].detach().cpu().numpy()
    delta_t = float(_denormalize_time_samples(runner, flat)[0])
    loc = _denormalize_spatial_samples(runner, flat)[0]
    next_time = float(history["times"][-1] + max(delta_t, 0.0))
    return next_time, np.asarray(loc, dtype=np.float32)


def sample_next_events_diffusion_batch(
    runner: STPPRunner,
    history: dict[str, np.ndarray],
    n_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    state = build_state_from_history(runner, history, device)
    out = runner.model.event_model.sample_native(
        state=state,
        batch_size=int(n_samples),
        device=device,
    )
    flat = _flatten_sample_tensor(out["samples"])[: int(n_samples)].detach().cpu().numpy()
    delta_t = _denormalize_time_samples(runner, flat)
    locs = _denormalize_spatial_samples(runner, flat).astype(np.float32, copy=False)
    next_times = (
        float(history["times"][-1]) + np.maximum(delta_t.astype(np.float32, copy=False), 0.0)
    ).astype(np.float32, copy=False)
    return next_times, locs


def _rollout_one_sequence_native(
    runner: STPPRunner,
    history: dict[str, np.ndarray],
    target_length: int,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one autoregressive rollout of target_length events."""
    preset = runner.config.model.preset
    h = copy_history(history)
    out_times: list[float] = []
    out_locs: list[np.ndarray] = []

    for _ in range(target_length):
        if preset == "smash":
            t_next, s_next = sample_next_event_smash(runner, h, device)
        else:
            t_next, s_next = sample_next_event_diffusion(runner, h, device)
        out_times.append(t_next)
        out_locs.append(s_next)
        h = append_events(
            h,
            np.asarray([t_next], dtype=np.float32),
            np.asarray(s_next, dtype=np.float32).reshape(1, 2),
        )

    return (
        np.asarray(out_times, dtype=np.float32),
        np.asarray(out_locs, dtype=np.float32).reshape(-1, 2),
    )


def _rollout_one_sequence_thinning(
    runner: STPPRunner,
    history: dict[str, np.ndarray],
    target_length: int,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    horizon: float,
    device: torch.device,
    initial_state_ctx=None,
    initial_proposal_cache=None,
) -> tuple[np.ndarray, np.ndarray]:
    t_start = float(history["times"][-1]) if history["times"].size > 0 else 0.0
    t_end = t_start + horizon
    exact_proposal = ExactProposalConfig(mode="coarse")

    out_t, out_s, _, _, _ = rollout_window_thinning(
        runner,
        history,
        window_start=t_start,
        window_end=t_end,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        lambda_bar=10.0,
        max_events=target_length,
        adaptive=True,
        device=device,
        initial_state_ctx=initial_state_ctx,
        initial_proposal_cache=initial_proposal_cache,
        exact_proposal=exact_proposal,
    )
    return out_t, out_s


def compute_generative_rollouts(
    runner: STPPRunner,
    test_seqs: list[dict[str, np.ndarray]],
    *,
    k: int,
    device: torch.device,
    seed: int = 1,
    n_context_events: int = 50,
) -> GenerativeRollouts:
    """Draw K fixed-depth autoregressive rollouts for each test sequence."""
    n_context_events = int(n_context_events)
    if n_context_events < 1:
        raise ValueError("n_context_events must be >= 1")
    np.random.seed(seed)
    caps = runner.model.event_model.capabilities
    use_native = caps.has_native_sampler
    method = "native" if use_native else "thinning"

    xmin, xmax, ymin, ymax = _spatial_bounds_from_seqs(test_seqs)
    iet_chunks = [
        np.diff(np.asarray(s["times"], dtype=np.float32))
        for s in test_seqs
        if np.asarray(s["times"], dtype=np.float32).shape[0] > 1
    ]
    median_iet = 1.0
    if iet_chunks:
        finite_iets = np.concatenate(iet_chunks)
        finite_iets = finite_iets[np.isfinite(finite_iets) & (finite_iets > 0.0)]
        if finite_iets.size > 0:
            median_iet = float(np.median(finite_iets))

    rollout_times: list[list[np.ndarray]] = []
    rollout_locs: list[list[np.ndarray]] = []
    true_times: list[np.ndarray] = []
    true_locs: list[np.ndarray] = []
    context_lengths: list[int] = []

    runner.model.eval()
    with torch.no_grad():
        for seq in test_seqs:
            times = np.asarray(seq["times"], dtype=np.float32)
            locs = np.asarray(seq["locations"], dtype=np.float32)
            n = times.shape[0]
            true_times.append(times)
            true_locs.append(locs)

            cond_len = min(n_context_events, n)
            context_lengths.append(int(cond_len))
            if cond_len == 0:
                rollout_times.append(
                    [np.zeros((0,), dtype=np.float32) for _ in range(k)]
                )
                rollout_locs.append(
                    [np.zeros((0, 2), dtype=np.float32) for _ in range(k)]
                )
                continue

            history = {
                "times": times[:cond_len].copy(),
                "locations": locs[:cond_len].copy(),
            }
            target_len = AUTOREGRESSIVE_ROLLOUT_DEPTH

            seq_rollout_t: list[np.ndarray] = []
            seq_rollout_s: list[np.ndarray] = []

            shared_state_ctx = None
            shared_proposal_cache = None
            horizon = float(target_len) * median_iet * 3.0
            if not use_native and target_len > 0:
                t_start = float(history["times"][-1]) if history["times"].size > 0 else 0.0
                shared_state_ctx = build_state_from_history(runner, history, device)
                intensity_fn = build_exact_intensity_fn(runner, shared_state_ctx, device)
                shared_proposal_cache, _ = _build_exact_proposal_cache(
                    intensity_fn,
                    t_start=t_start,
                    t_max=t_start + horizon,
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    config=ExactProposalConfig(mode="coarse"),
                    device=device,
                )

            for _ in range(k):
                if use_native:
                    t_out, s_out = _rollout_one_sequence_native(
                        runner, history, target_len, device=device
                    )
                else:
                    t_out, s_out = _rollout_one_sequence_thinning(
                        runner,
                        history,
                        target_len,
                        xmin=xmin,
                        xmax=xmax,
                        ymin=ymin,
                        ymax=ymax,
                        horizon=horizon,
                        device=device,
                        initial_state_ctx=shared_state_ctx,
                        initial_proposal_cache=shared_proposal_cache,
                    )
                seq_rollout_t.append(t_out)
                seq_rollout_s.append(s_out)

            rollout_times.append(seq_rollout_t)
            rollout_locs.append(seq_rollout_s)

    return GenerativeRollouts(
        rollout_times=rollout_times,
        rollout_locs=rollout_locs,
        true_times=true_times,
        true_locs=true_locs,
        context_lengths=context_lengths,
        method=method,
    )


def build_exact_intensity_fn(
    runner: STPPRunner,
    state,
    device: torch.device,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    norm_stats = runner.norm_stats
    normalize = bool(norm_stats.get("normalize", False))
    t_mean = float(norm_stats.get("time_mean", 0.0))
    t_std = max(float(norm_stats.get("time_std", 1.0)), 1e-8)
    loc_mean = np.asarray(norm_stats.get("loc_mean", [0.0, 0.0]), dtype=np.float32)
    loc_std = np.maximum(
        np.asarray(norm_stats.get("loc_std", [1.0, 1.0]), dtype=np.float32),
        1e-8,
    )
    payload = getattr(state, "payload", {})
    transform_spec = payload.get("input_transform")
    transform = transform_from_spec(transform_spec if isinstance(transform_spec, dict) else None)
    event_model = runner.model.event_model
    requires_fixed_time_batches = callable(getattr(event_model, "fixed_time_query_terms", None))

    def _transform_queries_if_needed(
        qt: torch.Tensor,
        qs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if transform is None:
            return qt, qs
        # Direct-history families such as NSMPP store transformed event vectors in
        # the state but do not own query-side transform routing inside intensity().
        # In that case, apply the transform here once.
        if "event_vectors" in payload and "paper_dt_range" not in payload:
            lengths = qt.new_full((qt.shape[0],), 1, dtype=torch.long)
            return (
                transform.forward_times(qt, lengths),
                transform.forward_locations(qs, lengths),
            )
        return qt, qs

    def _intensity_reporting_scale(values: torch.Tensor) -> torch.Tensor:
        if transform is not None and bool(getattr(transform, "supports_raw_reporting", False)):
            correction = transform.reporting_correction(ref=values)
            return torch.exp(correction).clamp(min=1e-8)
        scale_factor = paper_output_scale_factor(runner.model)
        if scale_factor is None:
            return values.new_ones(())
        return values.new_tensor(max(float(scale_factor), 1e-8))

    def _call_event_intensity(qt: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
        if not requires_fixed_time_batches:
            return event_model.intensity(
                state=state,
                query_times=qt,
                query_locations=qs,
                device=device,
            )
        q_flat = qt.reshape(-1)
        if q_flat.numel() == 0:
            return torch.zeros(0, device=device, dtype=torch.float32)
        if torch.allclose(q_flat, q_flat[0].expand_as(q_flat), atol=1e-7, rtol=1e-6):
            return event_model.intensity(
                state=state,
                query_times=qt,
                query_locations=qs,
                device=device,
            )

        values = torch.empty(q_flat.shape[0], device=device, dtype=torch.float32)
        unique_times, inverse = torch.unique(q_flat.detach(), sorted=False, return_inverse=True)
        for group_idx in range(int(unique_times.numel())):
            idx = torch.nonzero(inverse == group_idx, as_tuple=False).reshape(-1)
            group_values = event_model.intensity(
                state=state,
                query_times=qt.index_select(0, idx),
                query_locations=qs.index_select(0, idx),
                device=device,
            )
            values.index_copy_(0, idx, group_values.reshape(-1).to(dtype=torch.float32))
        return values

    def intensity_fn(query_times_raw: torch.Tensor, query_locations_raw: torch.Tensor) -> torch.Tensor:
        qt = query_times_raw.unsqueeze(-1) if query_times_raw.ndim == 1 else query_times_raw
        if query_locations_raw.ndim != 2:
            raise ValueError("query_locations_raw must have shape (B, d).")
        qt = qt.to(device=device, dtype=torch.float32)
        qs = query_locations_raw.to(device=device, dtype=torch.float32)
        if normalize:
            qt = (qt - t_mean) / t_std
            qs = (qs - torch.as_tensor(loc_mean, device=device, dtype=torch.float32)) / torch.as_tensor(
                loc_std,
                device=device,
                dtype=torch.float32,
            )
        qt, qs = _transform_queries_if_needed(qt, qs)
        with torch.no_grad():
            values = _call_event_intensity(qt, qs)
        values = values.to(dtype=torch.float32)
        values = values / _intensity_reporting_scale(values)
        return values.clamp(min=0.0)

    return intensity_fn


def _refresh_exact_proposal_interval(cache: ExactProposalCache, t_idx: int) -> None:
    flat_bounds = cache.cell_bounds[t_idx].reshape(-1)
    masses = np.maximum(flat_bounds, 0.0) * float(cache.cell_area)
    total = float(masses.sum())
    cache.total_rates[t_idx] = total
    if total <= 1e-12:
        cache.cell_probs[t_idx] = np.zeros_like(masses, dtype=np.float32)
        return
    cache.cell_probs[t_idx] = (masses / total).astype(np.float32)


def _build_exact_proposal_cache(
    intensity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    t_start: float,
    t_max: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    config: ExactProposalConfig,
    device: torch.device,
) -> tuple[ExactProposalCache, dict[str, int]]:
    time_edges = np.linspace(float(t_start), float(t_max), int(config.time_bins) + 1, dtype=np.float32)
    x_edges = np.linspace(float(xmin), float(xmax), int(config.spatial_bins) + 1, dtype=np.float32)
    y_edges = np.linspace(float(ymin), float(ymax), int(config.spatial_bins) + 1, dtype=np.float32)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    centers = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    cell_bounds = np.zeros(
        (int(config.time_bins), int(config.spatial_bins), int(config.spatial_bins)),
        dtype=np.float32,
    )
    n_spatial = centers.shape[0]
    eps = max(float(t_max - t_start) * 1e-6, 1e-6)

    # Collect all (time_bins * 3) time samples in one list. Query each time
    # slice separately because NeuralSTPP-style event models build ODE terms for
    # one fixed query time per call.
    all_t_vals: list[float] = []
    for t_idx in range(int(config.time_bins)):
        lo = float(time_edges[t_idx])
        hi = float(time_edges[t_idx + 1])
        mid = 0.5 * (lo + hi)
        all_t_vals.extend([lo, mid, max(lo, hi - eps)])

    n_t_samples = len(all_t_vals)  # time_bins * 3

    query_locs = torch.as_tensor(centers, dtype=torch.float32, device=device)

    vals_by_time: list[np.ndarray] = []
    with torch.no_grad():
        for t_val in all_t_vals:
            qt = torch.full((n_spatial,), float(t_val), dtype=torch.float32, device=device)
            vals = intensity_fn(qt, query_locs).detach().cpu().numpy().astype(np.float32).reshape(-1)
            vals_by_time.append(vals)
    all_vals = np.concatenate(vals_by_time, axis=0)

    # Reshape to (time_bins, 3, n_spatial) and take per-cell max over the 3 time samples
    vals_by_bin = all_vals.reshape(int(config.time_bins), 3, n_spatial)
    max_vals_by_bin = vals_by_bin.max(axis=1)  # (time_bins, n_spatial)

    for t_idx in range(int(config.time_bins)):
        cell_bounds[t_idx] = np.maximum(
            max_vals_by_bin[t_idx].reshape(int(config.spatial_bins), int(config.spatial_bins)) * float(config.safety),
            1e-8,
        )

    point_count = n_t_samples * n_spatial
    batch_count = n_t_samples
    cell_area = (
        max(float(xmax) - float(xmin), 1e-8)
        / max(int(config.spatial_bins), 1)
        * max(float(ymax) - float(ymin), 1e-8)
        / max(int(config.spatial_bins), 1)
    )
    cache = ExactProposalCache(
        time_edges=time_edges,
        x_edges=x_edges,
        y_edges=y_edges,
        cell_bounds=cell_bounds,
        cell_probs=np.zeros((int(config.time_bins), centers.shape[0]), dtype=np.float32),
        total_rates=np.zeros((int(config.time_bins),), dtype=np.float32),
        cell_area=float(cell_area),
        safety=float(config.safety),
    )
    for t_idx in range(int(config.time_bins)):
        _refresh_exact_proposal_interval(cache, t_idx)
    return cache, {
        "proposal_cache_builds": 1,
        "proposal_cache_query_batches": int(batch_count),
        "proposal_cache_query_points": int(point_count),
    }


def _thinning_k_chains_batched(
    intensity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    k: int,
    t_start: float,
    t_max: float,
    proposal_cache: ExactProposalCache,
    device: torch.device,
    max_rounds: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw K independent next-event samples via batched thinning.

    All K chains share the same conditioning state and proposal_cache — they are
    fully independent samples of the next event after t_start.  Instead of running
    K sequential single-point intensity calls, this function evaluates all active
    chains in one batched intensity call per round.  Expected rounds ≈ 2 at
    safety=2.0, so cost is O(rounds × batch) instead of O(K × proposals_per_sample).

    Returns
    -------
    result_t : (K,) float32 — sampled event times; t_max means no event in window.
    result_s : (K, 2) float32 — sampled event locations (random within domain when no event).
    """
    k = int(k)
    chain_t = np.full(k, float(t_start), dtype=np.float64)
    active = np.ones(k, dtype=bool)
    result_t = np.full(k, float(t_max), dtype=np.float32)
    result_s = np.zeros((k, 2), dtype=np.float32)
    n_x = int(proposal_cache.x_edges.shape[0] - 1)

    for _ in range(max_rounds):
        # Deactivate chains that have reached or passed the window end
        active &= chain_t < float(t_max)
        n_active = int(active.sum())
        if n_active == 0:
            break

        active_idx = np.where(active)[0]
        t_curr = chain_t[active_idx].astype(np.float32)

        # Determine time bin for each active chain
        t_bin = np.clip(
            np.searchsorted(proposal_cache.time_edges, t_curr, side="right") - 1,
            0,
            proposal_cache.time_edges.shape[0] - 2,
        ).astype(np.int32)
        interval_ends = proposal_cache.time_edges[t_bin + 1]   # (n_active,) float32
        total_rates = proposal_cache.total_rates[t_bin]         # (n_active,) float32

        # Zero-rate bins: advance chain to interval end, no intensity call needed
        zero_rate = total_rates <= 1e-12
        if zero_rate.any():
            chain_t[active_idx[zero_rate]] = interval_ends[zero_rate].astype(np.float64)

        has_rate = ~zero_rate
        if not has_rate.any():
            continue

        hr_idx = active_idx[has_rate]
        hr_rates = total_rates[has_rate].astype(np.float64)
        hr_t_curr = chain_t[hr_idx]
        hr_interval_ends = interval_ends[has_rate].astype(np.float64)

        # Sample next proposal time from Exp(total_rate)
        dt = np.random.exponential(1.0 / hr_rates)
        t_prop = (hr_t_curr + dt).astype(np.float32)

        # Proposals that overshoot the interval: advance chain_t, try next interval
        past = t_prop >= hr_interval_ends.astype(np.float32)
        if past.any():
            chain_t[hr_idx[past]] = hr_interval_ends[past]

        in_interval = ~past
        if not in_interval.any():
            continue

        vi_idx = hr_idx[in_interval]
        vi_t_prop = t_prop[in_interval]                           # (n_valid,) float32
        vi_t_bin = t_bin[has_rate][in_interval].astype(np.int32)  # (n_valid,)
        n_valid = int(in_interval.sum())

        # Sample spatial cells proportionally to cell probability mass
        flat_cell = np.array([
            int(np.random.choice(
                proposal_cache.cell_probs.shape[1],
                p=proposal_cache.cell_probs[ti],
            ))
            for ti in vi_t_bin
        ], dtype=np.int32)
        y_cell = flat_cell // n_x
        x_cell = flat_cell % n_x

        x_lo = proposal_cache.x_edges[x_cell]
        x_hi = proposal_cache.x_edges[x_cell + 1]
        y_lo = proposal_cache.y_edges[y_cell]
        y_hi = proposal_cache.y_edges[y_cell + 1]
        s_prop = np.column_stack([
            np.random.uniform(x_lo, x_hi),
            np.random.uniform(y_lo, y_hi),
        ]).astype(np.float32)

        # One batched intensity evaluation for all n_valid proposals
        qt = torch.as_tensor(vi_t_prop, dtype=torch.float32, device=device)
        qs = torch.as_tensor(s_prop, dtype=torch.float32, device=device)
        with torch.no_grad():
            lams = (
                intensity_fn(qt, qs).detach().cpu().numpy()
                .astype(np.float32).reshape(-1)
            )
        lams = np.maximum(lams, 0.0)

        # Adaptive bound updates for exceeded cells
        local_bounds = proposal_cache.cell_bounds[vi_t_bin, y_cell, x_cell].copy()
        exceeded = lams > local_bounds
        if exceeded.any():
            for ei in np.where(exceeded)[0]:
                tb, yc, xc = int(vi_t_bin[ei]), int(y_cell[ei]), int(x_cell[ei])
                new_bound = max(
                    float(lams[ei]) * max(float(proposal_cache.safety), 1.1),
                    float(proposal_cache.cell_bounds[tb, yc, xc]) * 2.0,
                    1e-8,
                )
                proposal_cache.cell_bounds[tb, yc, xc] = new_bound
                _refresh_exact_proposal_interval(proposal_cache, tb)
            local_bounds = proposal_cache.cell_bounds[vi_t_bin, y_cell, x_cell]

        # Vectorised accept/reject
        accept_p = lams / np.maximum(local_bounds, 1e-8)
        accepted = np.random.uniform(size=n_valid) < accept_p

        # Record accepted samples and deactivate those chains
        acc_idx = vi_idx[accepted]
        result_t[acc_idx] = vi_t_prop[accepted]
        result_s[acc_idx] = s_prop[accepted]
        active[acc_idx] = False

        # Rejected chains advance their current time to the rejection point
        rej_idx = vi_idx[~accepted]
        chain_t[rej_idx] = vi_t_prop[~accepted].astype(np.float64)

    return result_t, result_s


def rollout_window_native_sampler(
    runner: STPPRunner,
    next_event_fn: Callable[[STPPRunner, dict[str, np.ndarray], torch.device], tuple[float, np.ndarray]],
    initial_history: dict[str, np.ndarray],
    *,
    window_start: float,
    window_end: float,
    max_events: int,
    bridge_retries: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    history = copy_history(initial_history)
    sampled_times: list[float] = []
    sampled_locs: list[np.ndarray] = []
    bridge_rejects = 0
    bridge_exhausted = False

    def _append_one(t_evt: float, s_evt: np.ndarray) -> None:
        nonlocal history
        history = append_events(
            history,
            np.asarray([t_evt], dtype=np.float32),
            np.asarray(s_evt, dtype=np.float32).reshape(1, 2),
        )
        sampled_times.append(float(t_evt))
        sampled_locs.append(np.asarray(s_evt, dtype=np.float32).reshape(2))

    current_time = float(history["times"][-1])
    if current_time < float(window_start):
        accepted = False
        for _ in range(max(1, int(bridge_retries))):
            t_evt, s_evt = next_event_fn(runner, history, device)
            if t_evt + 1e-8 >= float(window_start):
                accepted = True
                if t_evt < float(window_end):
                    _append_one(t_evt, s_evt)
                break
            bridge_rejects += 1
        if not accepted:
            bridge_exhausted = True
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                history,
                {"bridge_rejects": bridge_rejects, "bridge_exhausted": bridge_exhausted},
            )
        if sampled_times and sampled_times[-1] >= float(window_end):
            return (
                np.asarray(sampled_times, dtype=np.float32),
                np.asarray(sampled_locs, dtype=np.float32),
                history,
                {"bridge_rejects": bridge_rejects, "bridge_exhausted": bridge_exhausted},
            )

    while len(sampled_times) < int(max_events):
        if sampled_times:
            current_time = float(sampled_times[-1])
        if current_time >= float(window_end):
            break
        t_evt, s_evt = next_event_fn(runner, history, device)
        if t_evt < float(window_end):
            _append_one(t_evt, s_evt)
            continue
        break

    return (
        np.asarray(sampled_times, dtype=np.float32),
        np.asarray(sampled_locs, dtype=np.float32).reshape(-1, 2),
        history,
        {"bridge_rejects": bridge_rejects, "bridge_exhausted": bridge_exhausted},
    )


def _sample_next_events_native_batch(
    loaded: LoadedRun,
    history: dict[str, np.ndarray],
    *,
    n_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if loaded.preset == "smash":
        return sample_next_events_smash_batch(loaded.runner, history, int(n_samples), device)
    if loaded.preset == "diffusion_stpp":
        return sample_next_events_diffusion_batch(loaded.runner, history, int(n_samples), device)
    raise ValueError(f"Unsupported native preset for batched sampling: {loaded.preset}")


def _native_next_event_fn(
    loaded: LoadedRun,
) -> Callable[[STPPRunner, dict[str, np.ndarray], torch.device], tuple[float, np.ndarray]]:
    if loaded.preset == "smash":
        return sample_next_event_smash
    if loaded.preset == "diffusion_stpp":
        return sample_next_event_diffusion
    raise ValueError(f"Unsupported native preset: {loaded.preset}")


def _teacher_forced_native_batched_rollouts(
    loaded: LoadedRun,
    *,
    base_history: dict[str, np.ndarray],
    window_start: float,
    window_end: float,
    n_rollouts: int,
    max_events: int,
    bridge_retries: int,
    device: torch.device,
    base_seed: int,
    frame_idx: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    if n_rollouts <= 0:
        return [], [], {
            "bridge_rejects": 0,
            "bridge_exhausted": False,
            "native_first_batch_calls": 0,
            "native_bridge_redraw_batches": 0,
            "native_serial_fallback_rollouts": 0,
            "native_bridge_exhausted_rollouts": 0,
        }

    next_event_fn = _native_next_event_fn(loaded)
    rollout_histories = [copy_history(base_history) for _ in range(int(n_rollouts))]
    rollout_times: list[list[float]] = [[] for _ in range(int(n_rollouts))]
    rollout_locs: list[list[np.ndarray]] = [[] for _ in range(int(n_rollouts))]
    current_time = float(base_history["times"][-1])
    needs_bridge = current_time < float(window_start)
    remaining = np.arange(int(n_rollouts), dtype=np.int64)
    bridge_rejects = 0
    native_first_batch_calls = 0
    native_bridge_redraw_batches = 0
    native_serial_fallback_rollouts = 0
    attempt = 0

    while remaining.size > 0:
        if needs_bridge and attempt >= max(1, int(bridge_retries)):
            break
        seed_everything(derive_seed(base_seed, loaded.label, frame_idx, "native_batch", attempt))
        native_first_batch_calls += 1
        if attempt > 0:
            native_bridge_redraw_batches += 1
        batch_times, batch_locs = _sample_next_events_native_batch(
            loaded,
            base_history,
            n_samples=int(remaining.size),
            device=device,
        )
        accepted = batch_times + 1e-8 >= float(window_start) if needs_bridge else np.ones(batch_times.shape[0], dtype=bool)
        if needs_bridge:
            bridge_rejects += int((~accepted).sum())
        accepted_indices = remaining[accepted]
        accepted_times = batch_times[accepted]
        accepted_locs = batch_locs[accepted]
        for local_pos, rollout_idx in enumerate(accepted_indices):
            t_evt = float(accepted_times[local_pos])
            s_evt = np.asarray(accepted_locs[local_pos], dtype=np.float32).reshape(2)
            if t_evt < float(window_end):
                rollout_times[int(rollout_idx)].append(t_evt)
                rollout_locs[int(rollout_idx)].append(s_evt)
                rollout_histories[int(rollout_idx)] = append_events(
                    rollout_histories[int(rollout_idx)],
                    np.asarray([t_evt], dtype=np.float32),
                    s_evt.reshape(1, 2),
                )
        if not needs_bridge:
            remaining = np.zeros((0,), dtype=np.int64)
            break
        remaining = remaining[~accepted]
        attempt += 1

    bridge_exhausted_rollouts = int(remaining.size)
    if max_events > 1:
        for rollout_idx, history_after_first in enumerate(rollout_histories):
            if len(rollout_times[rollout_idx]) != 1:
                continue
            native_serial_fallback_rollouts += 1
            seed_everything(derive_seed(base_seed, loaded.label, frame_idx, rollout_idx, "native_tail"))
            extra_times, extra_locs, _extra_history, extra_meta = rollout_window_native_sampler(
                loaded.runner,
                next_event_fn,
                history_after_first,
                window_start=window_start,
                window_end=window_end,
                max_events=max(int(max_events) - 1, 0),
                bridge_retries=bridge_retries,
                device=device,
            )
            bridge_rejects += int(extra_meta.get("bridge_rejects", 0))
            if extra_times.size > 0:
                rollout_times[rollout_idx].extend(extra_times.astype(np.float32, copy=False).tolist())
                rollout_locs[rollout_idx].extend(np.asarray(extra_locs, dtype=np.float32).reshape(-1, 2))

    return (
        [np.asarray(times, dtype=np.float32).reshape(-1) for times in rollout_times],
        [np.asarray(locs, dtype=np.float32).reshape(-1, 2) for locs in rollout_locs],
        {
            "bridge_rejects": int(bridge_rejects),
            "bridge_exhausted": bool(bridge_exhausted_rollouts > 0),
            "native_first_batch_calls": int(native_first_batch_calls),
            "native_bridge_redraw_batches": int(native_bridge_redraw_batches),
            "native_serial_fallback_rollouts": int(native_serial_fallback_rollouts),
            "native_bridge_exhausted_rollouts": int(bridge_exhausted_rollouts),
        },
    )


def _safe_thinning_one_event_uniform(
    intensity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    t_start: float,
    t_max: float,
    s_min: torch.Tensor,
    s_max: torch.Tensor,
    lambda_bar: float,
    adaptive: bool,
) -> tuple[float | None, np.ndarray | None, float, int, int]:
    device = s_min.device
    d = int(s_min.shape[0])
    vol = float((s_max - s_min).prod().item())
    t_current = float(t_start)
    lam_bar = max(float(lambda_bar), 1e-6)
    n_proposals = 0
    n_bound_updates = 0
    while True:
        rate = max(lam_bar * vol, 1e-6)
        dt = torch.distributions.Exponential(torch.tensor(rate, dtype=torch.float32, device=device)).sample()
        t_proposal = t_current + float(dt.item())
        if t_proposal >= float(t_max):
            return None, None, lam_bar, n_proposals, n_bound_updates
        u_s = torch.rand(d, device=device)
        s_proposal = s_min + u_s * (s_max - s_min)
        query_t = torch.tensor([[t_proposal]], dtype=torch.float32, device=device)
        query_s = s_proposal.unsqueeze(0)
        with torch.no_grad():
            lam = float(intensity_fn(query_t, query_s).reshape(-1)[0].item())
        lam = max(lam, 0.0)
        n_proposals += 1
        if not math.isfinite(lam):
            raise RuntimeError("Safe thinning encountered a non-finite intensity value.")
        if lam > lam_bar:
            lam_bar = max(lam * 1.5, lam_bar * 2.0, 1e-6)
            n_bound_updates += 1
            continue
        if adaptive:
            lam_bar = max(lam_bar, lam * 1.5, 1e-6)
        if float(torch.rand((), device=device).item()) < (lam / max(lam_bar, 1e-8)):
            return t_proposal, s_proposal.detach().cpu().numpy().astype(np.float32), lam_bar, n_proposals, n_bound_updates
        t_current = t_proposal


def _safe_thinning_one_event_coarse(
    intensity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    t_start: float,
    t_max: float,
    proposal_cache: ExactProposalCache,
    device: torch.device,
) -> tuple[float | None, np.ndarray | None, int, int]:
    t_current = float(t_start)
    n_proposals = 0
    n_bound_updates = 0
    n_x = int(proposal_cache.x_edges.shape[0] - 1)
    while t_current < float(t_max):
        t_idx = int(
            np.clip(
                np.searchsorted(proposal_cache.time_edges, t_current, side="right") - 1,
                0,
                proposal_cache.time_edges.shape[0] - 2,
            )
        )
        interval_end = min(float(proposal_cache.time_edges[t_idx + 1]), float(t_max))
        total_rate = float(proposal_cache.total_rates[t_idx])
        if total_rate <= 1e-12:
            t_current = interval_end
            continue
        dt = torch.distributions.Exponential(torch.tensor(total_rate, dtype=torch.float32, device=device)).sample()
        t_proposal = t_current + float(dt.item())
        if t_proposal >= interval_end:
            t_current = interval_end
            continue
        probs = proposal_cache.cell_probs[t_idx]
        if not np.isfinite(probs).all() or float(probs.sum()) <= 0.0:
            t_current = interval_end
            continue
        flat_idx = int(np.random.choice(probs.shape[0], p=probs))
        y_idx = flat_idx // n_x
        x_idx = flat_idx % n_x
        x_lo = float(proposal_cache.x_edges[x_idx])
        x_hi = float(proposal_cache.x_edges[x_idx + 1])
        y_lo = float(proposal_cache.y_edges[y_idx])
        y_hi = float(proposal_cache.y_edges[y_idx + 1])
        s_proposal_np = np.asarray([np.random.uniform(x_lo, x_hi), np.random.uniform(y_lo, y_hi)], dtype=np.float32)
        query_t = torch.tensor([[t_proposal]], dtype=torch.float32, device=device)
        query_s = torch.as_tensor(s_proposal_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            lam = float(intensity_fn(query_t, query_s).reshape(-1)[0].item())
        lam = max(lam, 0.0)
        n_proposals += 1
        if not math.isfinite(lam):
            raise RuntimeError("Coarse exact thinning encountered a non-finite intensity value.")
        local_bound = float(proposal_cache.cell_bounds[t_idx, y_idx, x_idx])
        if lam > local_bound:
            proposal_cache.cell_bounds[t_idx, y_idx, x_idx] = max(
                lam * max(float(proposal_cache.safety), 1.1),
                local_bound * 2.0,
                1e-8,
            )
            _refresh_exact_proposal_interval(proposal_cache, t_idx)
            n_bound_updates += 1
            continue
        if np.random.uniform() < (lam / max(local_bound, 1e-8)):
            return t_proposal, s_proposal_np, n_proposals, n_bound_updates
        t_current = t_proposal
    return None, None, n_proposals, n_bound_updates


def rollout_window_thinning(
    runner: STPPRunner,
    initial_history: dict[str, np.ndarray],
    *,
    window_start: float,
    window_end: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    lambda_bar: float,
    max_events: int,
    adaptive: bool,
    device: torch.device,
    initial_state_ctx=None,
    initial_proposal_cache: ExactProposalCache | None = None,
    exact_proposal: ExactProposalConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any], Any]:
    history = copy_history(initial_history)
    state_ctx = initial_state_ctx if initial_state_ctx is not None else build_state_from_history(runner, history, device)
    sampled_times: list[float] = []
    sampled_locs: list[np.ndarray] = []
    s_min = torch.tensor([xmin, ymin], dtype=torch.float32, device=device)
    s_max = torch.tensor([xmax, ymax], dtype=torch.float32, device=device)
    current_t = float(window_start)
    lambda_bar_current = float(lambda_bar)
    n_attempts = 0
    n_proposals_total = 0
    n_bound_updates_total = 0
    proposal_cache_builds_total = 0
    proposal_cache_query_batches_total = 0
    proposal_cache_query_points_total = 0
    proposal_cache = initial_proposal_cache
    for _ in range(int(max_events)):
        intensity_fn = build_exact_intensity_fn(runner, state_ctx, device)
        if exact_proposal is not None and exact_proposal.mode == "coarse":
            if proposal_cache is None:
                proposal_cache, cache_meta = _build_exact_proposal_cache(
                    intensity_fn,
                    t_start=current_t,
                    t_max=float(window_end),
                    xmin=float(xmin),
                    xmax=float(xmax),
                    ymin=float(ymin),
                    ymax=float(ymax),
                    config=exact_proposal,
                    device=device,
                )
                proposal_cache_builds_total += int(cache_meta["proposal_cache_builds"])
                proposal_cache_query_batches_total += int(cache_meta["proposal_cache_query_batches"])
                proposal_cache_query_points_total += int(cache_meta["proposal_cache_query_points"])
            t_evt, s_evt, n_props, n_updates = _safe_thinning_one_event_coarse(
                intensity_fn,
                t_start=current_t,
                t_max=float(window_end),
                proposal_cache=proposal_cache,
                device=device,
            )
        else:
            t_evt, s_evt, lambda_bar_current, n_props, n_updates = _safe_thinning_one_event_uniform(
                intensity_fn,
                t_start=current_t,
                t_max=float(window_end),
                s_min=s_min,
                s_max=s_max,
                lambda_bar=lambda_bar_current,
                adaptive=adaptive,
            )
        n_attempts += 1
        n_proposals_total += int(n_props)
        n_bound_updates_total += int(n_updates)
        if t_evt is None or s_evt is None or t_evt >= float(window_end):
            break
        sampled_times.append(t_evt)
        sampled_locs.append(s_evt)
        history = append_events(
            history,
            np.asarray([t_evt], dtype=np.float32),
            np.asarray(s_evt, dtype=np.float32).reshape(1, 2),
        )
        state_ctx = append_sampling_state(
            runner,
            state_ctx,
            history=history,
            event_time_raw=t_evt,
            event_location_raw=s_evt,
            device=device,
        )
        proposal_cache = None
        current_t = max(t_evt, current_t + 1e-6)
    return (
        np.asarray(sampled_times, dtype=np.float32),
        np.asarray(sampled_locs, dtype=np.float32).reshape(-1, 2),
        history,
        {
            "bridge_rejects": 0,
            "bridge_exhausted": False,
            "thinning_attempts": n_attempts,
            "thinning_proposals": n_proposals_total,
            "thinning_bound_updates": n_bound_updates_total,
            "thinning_final_lambda_bar": float(lambda_bar_current),
            "proposal_cache_builds": int(proposal_cache_builds_total),
            "proposal_cache_query_batches": int(proposal_cache_query_batches_total),
            "proposal_cache_query_points": int(proposal_cache_query_points_total),
        },
        state_ctx,
    )


def _sample_one_rollout(
    loaded: LoadedRun,
    history: dict[str, np.ndarray],
    *,
    window: FrameWindow,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    lambda_bar: float,
    max_events_per_window: int,
    bridge_retries: int,
    adaptive_thinning: bool,
    device: torch.device,
    initial_state_ctx=None,
    initial_proposal_cache: ExactProposalCache | None = None,
    exact_proposal: ExactProposalConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any], Any]:
    if loaded.preset == "smash":
        sample_times, sample_locs, next_history, meta = rollout_window_native_sampler(
            loaded.runner,
            sample_next_event_smash,
            history,
            window_start=window.start,
            window_end=window.end,
            max_events=max_events_per_window,
            bridge_retries=bridge_retries,
            device=device,
        )
        return sample_times, sample_locs, next_history, meta, None
    if loaded.preset == "diffusion_stpp":
        sample_times, sample_locs, next_history, meta = rollout_window_native_sampler(
            loaded.runner,
            sample_next_event_diffusion,
            history,
            window_start=window.start,
            window_end=window.end,
            max_events=max_events_per_window,
            bridge_retries=bridge_retries,
            device=device,
        )
        return sample_times, sample_locs, next_history, meta, None
    return rollout_window_thinning(
        loaded.runner,
        history,
        window_start=window.start,
        window_end=window.end,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        lambda_bar=lambda_bar,
        max_events=max_events_per_window,
        adaptive=adaptive_thinning,
        device=device,
        initial_state_ctx=initial_state_ctx,
        initial_proposal_cache=initial_proposal_cache,
        exact_proposal=exact_proposal,
    )


def evaluate_frame_rollouts(
    loaded: LoadedRun,
    *,
    seq: dict[str, np.ndarray],
    window: FrameWindow,
    histories_for_rollout: list[dict[str, np.ndarray]],
    states_for_rollout: list[Any],
    proposal_caches_for_rollout: list[ExactProposalCache | None],
    history_overlay: np.ndarray | None,
    history_length: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    lambda_bar: float,
    max_events_per_window: int,
    bridge_retries: int,
    adaptive_thinning: bool,
    n_rollouts: int,
    device: torch.device,
    carry_updates: bool,
    exact_proposal: ExactProposalConfig,
    shared_exact_proposal_meta: dict[str, int] | None = None,
    batch_teacher_forced_native_first: bool = False,
    base_seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]], list[Any]]:
    t0_frame = time.perf_counter()
    model_is_exact = is_exact_preset(loaded.preset)
    rollout_times: list[np.ndarray] = []
    rollout_locs: list[np.ndarray] = []
    updated_histories: list[dict[str, np.ndarray]] = []
    updated_states: list[Any] = []
    bridge_rejects_total = 0
    bridge_exhausted_any = False
    thinning_attempts_total = 0
    thinning_proposals_total = 0
    thinning_bound_updates_total = 0
    thinning_final_lambda_bar_values: list[float] = []
    proposal_cache_builds_total = int(0 if shared_exact_proposal_meta is None else shared_exact_proposal_meta.get("proposal_cache_builds", 0))
    proposal_cache_query_batches_total = int(0 if shared_exact_proposal_meta is None else shared_exact_proposal_meta.get("proposal_cache_query_batches", 0))
    proposal_cache_query_points_total = int(0 if shared_exact_proposal_meta is None else shared_exact_proposal_meta.get("proposal_cache_query_points", 0))
    native_first_batch_calls = 0
    native_bridge_redraw_batches = 0
    native_serial_fallback_rollouts = 0
    native_bridge_exhausted_rollouts = 0

    if batch_teacher_forced_native_first and not model_is_exact:
        batched_times, batched_locs, native_meta = _teacher_forced_native_batched_rollouts(
            loaded,
            base_history=histories_for_rollout[0],
            window_start=float(window.start),
            window_end=float(window.end),
            n_rollouts=int(n_rollouts),
            max_events=int(max_events_per_window),
            bridge_retries=int(bridge_retries),
            device=device,
            base_seed=base_seed,
            frame_idx=int(window.index),
        )
        rollout_times.extend(batched_times)
        rollout_locs.extend(batched_locs)
        bridge_rejects_total += int(native_meta.get("bridge_rejects", 0))
        bridge_exhausted_any = bridge_exhausted_any or bool(native_meta.get("bridge_exhausted", False))
        native_first_batch_calls = int(native_meta.get("native_first_batch_calls", 0))
        native_bridge_redraw_batches = int(native_meta.get("native_bridge_redraw_batches", 0))
        native_serial_fallback_rollouts = int(native_meta.get("native_serial_fallback_rollouts", 0))
        native_bridge_exhausted_rollouts = int(native_meta.get("native_bridge_exhausted_rollouts", 0))
    else:
        for rollout_idx, (hist, state_ctx, proposal_cache) in enumerate(
            zip(histories_for_rollout, states_for_rollout, proposal_caches_for_rollout),
            start=1,
        ):
            seed_everything(derive_seed(base_seed, loaded.label, window.index, rollout_idx, "rollout"))
            working_history = cap_history(hist, int(history_length))
            sample_times, sample_locs, next_history, rollout_meta, next_state_ctx = _sample_one_rollout(
                loaded,
                working_history,
                window=window,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                lambda_bar=float(lambda_bar),
                max_events_per_window=int(max_events_per_window),
                bridge_retries=int(bridge_retries),
                adaptive_thinning=bool(adaptive_thinning),
                device=device,
                initial_state_ctx=state_ctx,
                initial_proposal_cache=proposal_cache,
                exact_proposal=exact_proposal,
            )
            rollout_times.append(sample_times)
            rollout_locs.append(sample_locs)
            bridge_rejects_total += int(rollout_meta.get("bridge_rejects", 0))
            bridge_exhausted_any = bridge_exhausted_any or bool(rollout_meta.get("bridge_exhausted", False))
            thinning_attempts_total += int(rollout_meta.get("thinning_attempts", 0))
            thinning_proposals_total += int(rollout_meta.get("thinning_proposals", 0))
            thinning_bound_updates_total += int(rollout_meta.get("thinning_bound_updates", 0))
            proposal_cache_builds_total += int(rollout_meta.get("proposal_cache_builds", 0))
            proposal_cache_query_batches_total += int(rollout_meta.get("proposal_cache_query_batches", 0))
            proposal_cache_query_points_total += int(rollout_meta.get("proposal_cache_query_points", 0))
            if "thinning_final_lambda_bar" in rollout_meta:
                thinning_final_lambda_bar_values.append(float(rollout_meta["thinning_final_lambda_bar"]))
            if carry_updates:
                capped_next_history = cap_history(next_history, int(history_length))
                updated_histories.append(capped_next_history)
                if model_is_exact:
                    if int(history_length) > 0 and capped_next_history["times"].shape[0] != next_history["times"].shape[0]:
                        updated_states.append(build_state_from_history(loaded.runner, capped_next_history, device))
                    else:
                        updated_states.append(next_state_ctx)
                else:
                    updated_states.append(None)

    pooled_times, pooled_locs, per_rollout_counts = collect_window_events(rollout_times, rollout_locs)
    mean_events_per_rollout = float(per_rollout_counts.mean()) if per_rollout_counts.size else 0.0
    true_mask = (seq["times"] > float(window.start)) & (seq["times"] < float(window.end))
    true_window_times = np.asarray(seq["times"][true_mask], dtype=np.float32)
    true_window_locs = np.asarray(seq["locations"][true_mask], dtype=np.float32)
    elapsed_sec = float(time.perf_counter() - t0_frame)
    meta = {
        "frame_index": int(window.index),
        "window_start": float(window.start),
        "window_end": float(window.end),
        "history_locs": None if history_overlay is None else np.asarray(history_overlay, dtype=np.float32),
        "pooled_event_times": pooled_times,
        "pooled_event_locs": pooled_locs,
        "rollout_event_counts": per_rollout_counts,
        "mean_events_per_rollout": mean_events_per_rollout,
        "true_window_times": true_window_times,
        "true_window_locs": true_window_locs,
        "true_window_count": int(true_window_locs.shape[0]),
        "bridge_rejects": int(bridge_rejects_total),
        "bridge_exhausted": bool(bridge_exhausted_any),
        "thinning_attempts": int(thinning_attempts_total),
        "thinning_proposals": int(thinning_proposals_total),
        "thinning_bound_updates": int(thinning_bound_updates_total),
        "thinning_final_lambda_bar": float(max(thinning_final_lambda_bar_values) if thinning_final_lambda_bar_values else float(lambda_bar)),
        "proposal_cache_builds": int(proposal_cache_builds_total),
        "proposal_cache_query_batches": int(proposal_cache_query_batches_total),
        "proposal_cache_query_points": int(proposal_cache_query_points_total),
        "native_first_batch_calls": int(native_first_batch_calls),
        "native_bridge_redraw_batches": int(native_bridge_redraw_batches),
        "native_serial_fallback_rollouts": int(native_serial_fallback_rollouts),
        "native_bridge_exhausted_rollouts": int(native_bridge_exhausted_rollouts),
        "elapsed_sec": elapsed_sec,
    }
    return meta, updated_histories, updated_states


def evaluate_teacher_forced_frame(
    loaded: LoadedRun,
    *,
    seq: dict[str, np.ndarray],
    window: FrameWindow,
    history_length: int,
    n_rollouts: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    lambda_bar: float,
    max_events_per_window: int,
    bridge_retries: int,
    adaptive_thinning: bool,
    exact_proposal: ExactProposalConfig,
    device: torch.device,
    base_seed: int,
) -> dict[str, Any]:
    base_history = history_until_time(seq, t_cutoff=window.start, history_length=int(history_length))
    histories_for_rollout = [base_history] * int(n_rollouts)
    shared_exact_proposal_meta: dict[str, int] | None = None
    if is_exact_preset(loaded.preset):
        base_state_ctx = build_state_from_history(loaded.runner, base_history, device)
        states_for_rollout = [base_state_ctx] * int(n_rollouts)
        if exact_proposal.mode == "coarse":
            seed_everything(derive_seed(base_seed, loaded.label, window.index, "proposal_cache"))
            base_intensity_fn = build_exact_intensity_fn(loaded.runner, base_state_ctx, device)
            base_proposal_cache, shared_exact_proposal_meta = _build_exact_proposal_cache(
                base_intensity_fn,
                t_start=float(window.start),
                t_max=float(window.end),
                xmin=float(xmin),
                xmax=float(xmax),
                ymin=float(ymin),
                ymax=float(ymax),
                config=exact_proposal,
                device=device,
            )
            proposal_caches_for_rollout = [base_proposal_cache] * int(n_rollouts)
        else:
            proposal_caches_for_rollout = [None] * int(n_rollouts)
    else:
        states_for_rollout = [None] * int(n_rollouts)
        proposal_caches_for_rollout = [None] * int(n_rollouts)

    meta, _, _ = evaluate_frame_rollouts(
        loaded,
        seq=seq,
        window=window,
        histories_for_rollout=histories_for_rollout,
        states_for_rollout=states_for_rollout,
        proposal_caches_for_rollout=proposal_caches_for_rollout,
        history_overlay=base_history["locations"],
        history_length=int(history_length),
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        lambda_bar=float(lambda_bar),
        max_events_per_window=int(max_events_per_window),
        bridge_retries=int(bridge_retries),
        adaptive_thinning=bool(adaptive_thinning),
        n_rollouts=int(n_rollouts),
        device=device,
        carry_updates=False,
        exact_proposal=exact_proposal,
        shared_exact_proposal_meta=shared_exact_proposal_meta,
        batch_teacher_forced_native_first=bool(not is_exact_preset(loaded.preset)),
        base_seed=base_seed,
    )
    return meta


def evaluate_free_running_model(
    loaded: LoadedRun,
    *,
    initial_history: dict[str, np.ndarray],
    seq: dict[str, np.ndarray],
    schedule: list[FrameWindow],
    history_length: int,
    n_rollouts: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    lambda_bar: float,
    max_events_per_window: int,
    bridge_retries: int,
    adaptive_thinning: bool,
    exact_proposal: ExactProposalConfig,
    device: torch.device,
    base_seed: int,
) -> list[dict[str, Any]]:
    rollout_histories = [cap_history(initial_history, int(history_length)) for _ in range(int(n_rollouts))]
    if is_exact_preset(loaded.preset):
        base_state_ctx = build_state_from_history(loaded.runner, rollout_histories[0], device)
        rollout_states = [base_state_ctx] * int(n_rollouts)
    else:
        rollout_states = [None] * int(n_rollouts)
    rollout_proposal_caches: list[ExactProposalCache | None] = [None] * int(n_rollouts)
    frame_meta: list[dict[str, Any]] = []
    for window in schedule:
        meta, rollout_histories, rollout_states = evaluate_frame_rollouts(
            loaded,
            seq=seq,
            window=window,
            histories_for_rollout=rollout_histories,
            states_for_rollout=rollout_states,
            proposal_caches_for_rollout=rollout_proposal_caches,
            history_overlay=None,
            history_length=int(history_length),
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            lambda_bar=float(lambda_bar),
            max_events_per_window=int(max_events_per_window),
            bridge_retries=int(bridge_retries),
            adaptive_thinning=bool(adaptive_thinning),
            n_rollouts=int(n_rollouts),
            device=device,
            carry_updates=True,
            exact_proposal=exact_proposal,
            base_seed=base_seed,
        )
        frame_meta.append(meta)
    return frame_meta
