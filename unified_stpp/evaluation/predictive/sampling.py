"""Predictive next-event sampling helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from unified_stpp.runner.runner import STPPRunner

from ..artifacts import PredictiveSamples
from .rollout import (
    ExactProposalConfig,
    _build_exact_proposal_cache,
    _thinning_k_chains_batched,
    build_exact_intensity_fn,
    build_state_from_history,
    sample_next_events_diffusion_batch,
    sample_next_events_smash_batch,
)


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


def _coerce_sequence_arrays(seq: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(seq["times"], dtype=np.float32),
        np.asarray(seq["locations"], dtype=np.float32),
    )


def _adaptive_initial_horizon(test_seqs: list[dict[str, np.ndarray]]) -> float:
    delta_chunks = []
    for seq in test_seqs:
        times, _ = _coerce_sequence_arrays(seq)
        if times.shape[0] > 1:
            delta_chunks.append(np.diff(times))
    if not delta_chunks:
        return 1.0
    median_iet = float(np.median(np.concatenate(delta_chunks, axis=0)))
    return max(median_iet * 4.0, 1e-3)


def _thinning_next_events_adaptive_batch(
    runner: STPPRunner,
    history: dict[str, np.ndarray],
    k: int,
    *,
    initial_horizon: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    device: torch.device,
    exact_time_bins: int = 8,
    exact_spatial_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw K next-event samples via adaptive batched thinning."""
    if k <= 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.bool_),
        )

    t_start = float(history["times"][-1]) if history["times"].size > 0 else 0.0
    exact_proposal = ExactProposalConfig(
        mode="coarse",
        time_bins=int(exact_time_bins),
        spatial_bins=int(exact_spatial_bins),
    )

    state_ctx = build_state_from_history(runner, history, device)
    intensity_fn = build_exact_intensity_fn(runner, state_ctx, device)
    result_t = np.full((k,), np.nan, dtype=np.float32)
    result_s = np.full((k, 2), np.nan, dtype=np.float32)
    success = np.zeros((k,), dtype=np.bool_)
    remaining = np.arange(k, dtype=np.int64)
    window_start = float(t_start)
    horizon = float(initial_horizon)

    for _ in range(9):
        if remaining.size == 0:
            break
        t_end = window_start + float(horizon)
        proposal_cache, _ = _build_exact_proposal_cache(
            intensity_fn,
            t_start=window_start,
            t_max=t_end,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            config=exact_proposal,
            device=device,
        )
        sample_t, sample_s = _thinning_k_chains_batched(
            intensity_fn,
            k=int(remaining.size),
            t_start=window_start,
            t_max=t_end,
            proposal_cache=proposal_cache,
            device=device,
        )
        accepted = sample_t < (t_end - 1e-6)
        if np.any(accepted):
            accepted_ids = remaining[accepted]
            result_t[accepted_ids] = sample_t[accepted]
            result_s[accepted_ids] = sample_s[accepted]
            success[accepted_ids] = True
        remaining = remaining[~accepted]
        window_start = t_end
        horizon *= 2.0

    return result_t, result_s, success


def compute_predictive_samples(
    runner: STPPRunner,
    test_seqs: list[dict[str, np.ndarray]],
    *,
    k: int,
    device: torch.device,
    seed: int = 0,
    exact_time_bins: int = 8,
    exact_spatial_bins: int = 8,
) -> PredictiveSamples:
    """Compute K next-event samples for every held-out next-event context."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    preset = runner.config.model.preset
    caps = runner.model.event_model.capabilities
    use_native = caps.has_native_sampler
    sampling_backend = (
        "native_next_event_sampler" if use_native else "exact_intensity_thinning"
    )

    xmin, xmax, ymin, ymax = _spatial_bounds_from_seqs(test_seqs)
    initial_horizon = _adaptive_initial_horizon(test_seqs)
    loc_dim = 2

    all_next_times: list[np.ndarray] = []
    all_next_locs: list[np.ndarray] = []
    all_true_next_t: list[float] = []
    all_true_next_s: list[np.ndarray] = []
    all_hist_end_t: list[float] = []
    all_seq_idx: list[int] = []
    all_target_event_idx: list[int] = []
    all_history_length: list[int] = []
    all_is_last_context: list[bool] = []
    all_sampling_succeeded: list[bool] = []

    runner.model.eval()
    with torch.no_grad():
        for seq_i, seq in enumerate(test_seqs):
            times, locs = _coerce_sequence_arrays(seq)
            n = times.shape[0]
            if locs.ndim == 2 and locs.shape[1] > 0:
                loc_dim = int(locs.shape[1])
            for i in range(1, n):
                history = {
                    "times": times[:i].copy(),
                    "locations": locs[:i].copy(),
                }
                true_next_t = float(times[i])
                true_next_s = locs[i].copy()
                hist_end_t = float(times[i - 1])

                if use_native:
                    if preset == "smash":
                        s_t, s_s = sample_next_events_smash_batch(
                            runner, history, k, device
                        )
                    else:
                        s_t, s_s = sample_next_events_diffusion_batch(
                            runner, history, k, device
                        )
                    sampling_succeeded = bool(
                        np.isfinite(s_t).all() and np.isfinite(s_s).all()
                    )
                else:
                    s_t, s_s, success_mask = _thinning_next_events_adaptive_batch(
                        runner,
                        history,
                        k,
                        initial_horizon=initial_horizon,
                        xmin=xmin,
                        xmax=xmax,
                        ymin=ymin,
                        ymax=ymax,
                        device=device,
                        exact_time_bins=exact_time_bins,
                        exact_spatial_bins=exact_spatial_bins,
                    )
                    sampling_succeeded = bool(success_mask.all())

                if not sampling_succeeded:
                    s_t = np.full((k,), np.nan, dtype=np.float32)
                    s_s = np.full((k, loc_dim), np.nan, dtype=np.float32)

                all_next_times.append(s_t)
                all_next_locs.append(s_s)
                all_true_next_t.append(true_next_t)
                all_true_next_s.append(true_next_s)
                all_hist_end_t.append(hist_end_t)
                all_seq_idx.append(seq_i)
                all_target_event_idx.append(i)
                all_history_length.append(i)
                all_is_last_context.append(i == (n - 1))
                all_sampling_succeeded.append(sampling_succeeded)

    if all_next_times:
        next_times = np.stack(all_next_times, axis=0).astype(np.float32, copy=False)
        next_locs = np.stack(all_next_locs, axis=0).astype(np.float32, copy=False)
    else:
        next_times = np.zeros((0, int(k)), dtype=np.float32)
        next_locs = np.zeros((0, int(k), loc_dim), dtype=np.float32)
    return PredictiveSamples(
        next_times=next_times,
        next_locs=next_locs,
        true_next_times=np.asarray(all_true_next_t, dtype=np.float32),
        true_next_locs=np.asarray(all_true_next_s, dtype=np.float32),
        history_end_times=np.asarray(all_hist_end_t, dtype=np.float32),
        sequence_index=np.asarray(all_seq_idx, dtype=np.int64),
        target_event_index=np.asarray(all_target_event_idx, dtype=np.int64),
        history_length=np.asarray(all_history_length, dtype=np.int64),
        is_last_context=np.asarray(all_is_last_context, dtype=np.bool_),
        sampling_succeeded=np.asarray(all_sampling_succeeded, dtype=np.bool_),
        sampling_backend=sampling_backend,
    )
