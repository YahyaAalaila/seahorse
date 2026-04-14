"""Low-level helpers for computing EvalContext cached artifacts.

These functions are internal to the evaluation package.  External code should
use EvalContext's @cached_property attributes, not call these directly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from unified_stpp.runner.runner import STPPRunner

from .artifacts import PredictiveSamples
from .common import append_events, cap_history, copy_history, resolve_spatial_bounds
from .context import GenerativeRollouts, GroundTruth, IntensityGrid
from .predictive_sampling import (
    ExactProposalConfig,
    _build_exact_proposal_cache,
    build_exact_intensity_fn,
    build_state_from_history,
    normalize_history_for_runner,
    sample_next_events_diffusion_batch,
    sample_next_events_smash_batch,
)


# ---------------------------------------------------------------------------
# Predictive samples (K next-event samples per test event)
# ---------------------------------------------------------------------------


def _spatial_bounds_from_seqs(
    seqs: list[dict[str, np.ndarray]],
    pad: float = 0.08,
) -> tuple[float, float, float, float]:
    all_locs = np.concatenate([s["locations"] for s in seqs], axis=0).astype(np.float32)
    lo = all_locs.min(axis=0)
    hi = all_locs.max(axis=0)
    span = np.maximum(hi - lo, 1e-4)
    return (
        float(lo[0] - pad * span[0]),
        float(hi[0] + pad * span[0]),
        float(lo[1] - pad * span[1]),
        float(hi[1] + pad * span[1]),
    )


def _thinning_next_events_batch(
    runner: STPPRunner,
    history: dict[str, np.ndarray],
    k: int,
    *,
    horizon: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    device: torch.device,
    exact_time_bins: int = 12,
    exact_spatial_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw K next-event samples via serial thinning.  Returns (times, locs).

    Builds the history state and proposal cache **once** and reuses them
    across all K samples.  This reduces cost from O(K × encode + K × cache_build)
    to O(1 × encode + 1 × cache_build + K × thinning_proposals).
    """
    from .predictive_sampling import rollout_window_thinning

    t_start = float(history["times"][-1]) if history["times"].size > 0 else 0.0
    t_end = t_start + float(horizon)
    exact_proposal = ExactProposalConfig(
        mode="coarse",
        time_bins=int(exact_time_bins),
        spatial_bins=int(exact_spatial_bins),
    )

    # --- Build state and proposal cache once ---
    state_ctx = build_state_from_history(runner, history, device)
    intensity_fn = build_exact_intensity_fn(runner, state_ctx, device)
    proposal_cache, _ = _build_exact_proposal_cache(
        intensity_fn,
        t_start=t_start,
        t_max=t_end,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        config=exact_proposal,
        device=device,
    )

    sampled_t: list[float] = []
    sampled_s: list[np.ndarray] = []

    for _ in range(k):
        # Pass pre-built state and cache — rollout_window_thinning will not
        # re-encode the history or re-build the cache.
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
            max_events=1,
            adaptive=True,
            device=device,
            initial_state_ctx=state_ctx,
            initial_proposal_cache=proposal_cache,
            exact_proposal=exact_proposal,
        )
        if out_t.size > 0:
            sampled_t.append(float(out_t[0]))
            sampled_s.append(out_s[0])
        else:
            # No event sampled in window — use t_end with a random location
            rng_loc = np.random.uniform(
                [xmin, ymin], [xmax, ymax], size=2
            ).astype(np.float32)
            sampled_t.append(t_end)
            sampled_s.append(rng_loc)

    return (
        np.asarray(sampled_t, dtype=np.float32),
        np.asarray(sampled_s, dtype=np.float32),
    )


def compute_predictive_samples(
    runner: STPPRunner,
    test_seqs: list[dict[str, np.ndarray]],
    *,
    k: int,
    device: torch.device,
    seed: int = 0,
    exact_time_bins: int = 12,
    exact_spatial_bins: int = 12,
) -> PredictiveSamples:
    """Compute K next-event samples for every test event (teacher-forced history).

    For each event i ≥ 1 in each test sequence, we condition on events 0..i-1
    and draw K samples of the next event (time, location).
    """
    np.random.seed(seed)
    preset = runner.config.model.preset
    caps = runner.model.event_model.capabilities
    use_native = caps.has_native_sampler
    method = "native" if use_native else "thinning"

    xmin, xmax, ymin, ymax = _spatial_bounds_from_seqs(test_seqs)
    median_iet = np.median(
        np.concatenate(
            [
                np.diff(s["times"].astype(np.float32))
                for s in test_seqs
                if s["times"].shape[0] > 1
            ]
        )
    )
    horizon = max(float(median_iet) * 4.0, 1e-3)

    all_next_times: list[np.ndarray] = []   # each shape (K,)
    all_next_locs: list[np.ndarray] = []    # each shape (K, 2)
    all_true_next_t: list[float] = []
    all_true_next_s: list[np.ndarray] = []
    all_hist_end_t: list[float] = []
    all_seq_idx: list[int] = []

    runner.model.eval()
    with torch.no_grad():
        for seq_i, seq in enumerate(test_seqs):
            times = np.asarray(seq["times"], dtype=np.float32)
            locs = np.asarray(seq["locations"], dtype=np.float32)
            n = times.shape[0]
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
                else:
                    s_t, s_s = _thinning_next_events_batch(
                        runner,
                        history,
                        k,
                        horizon=horizon,
                        xmin=xmin,
                        xmax=xmax,
                        ymin=ymin,
                        ymax=ymax,
                        device=device,
                        exact_time_bins=exact_time_bins,
                        exact_spatial_bins=exact_spatial_bins,
                    )

                all_next_times.append(s_t)
                all_next_locs.append(s_s)
                all_true_next_t.append(true_next_t)
                all_true_next_s.append(true_next_s)
                all_hist_end_t.append(hist_end_t)
                all_seq_idx.append(seq_i)

    return PredictiveSamples(
        next_times=np.stack(all_next_times, axis=0),       # (N, K)
        next_locs=np.stack(all_next_locs, axis=0),         # (N, K, 2)
        true_next_times=np.asarray(all_true_next_t, dtype=np.float32),
        true_next_locs=np.asarray(all_true_next_s, dtype=np.float32),
        history_end_times=np.asarray(all_hist_end_t, dtype=np.float32),
        seq_indices=np.asarray(all_seq_idx, dtype=np.int64),
        method=method,
    )


# ---------------------------------------------------------------------------
# Generative rollouts (K full-sequence rollouts per test sequence)
# ---------------------------------------------------------------------------


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
            t_next, s_next = _sample_one_smash(runner, h, device)
        else:
            t_next, s_next = _sample_one_diffusion(runner, h, device)
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


def _sample_one_smash(runner, history, device) -> tuple[float, np.ndarray]:
    from .predictive_sampling import sample_next_event_smash
    return sample_next_event_smash(runner, history, device)


def _sample_one_diffusion(runner, history, device) -> tuple[float, np.ndarray]:
    from .predictive_sampling import sample_next_event_diffusion
    return sample_next_event_diffusion(runner, history, device)


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
    from .predictive_sampling import rollout_window_thinning

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
) -> GenerativeRollouts:
    """Draw K autoregressive rollouts for each test sequence."""
    np.random.seed(seed)
    caps = runner.model.event_model.capabilities
    use_native = caps.has_native_sampler
    method = "native" if use_native else "thinning"

    xmin, xmax, ymin, ymax = _spatial_bounds_from_seqs(test_seqs)
    median_iet = float(
        np.median(
            np.concatenate(
                [
                    np.diff(s["times"].astype(np.float32))
                    for s in test_seqs
                    if s["times"].shape[0] > 1
                ]
            )
        )
    )

    rollout_times: list[list[np.ndarray]] = []
    rollout_locs: list[list[np.ndarray]] = []
    true_times: list[np.ndarray] = []
    true_locs: list[np.ndarray] = []

    runner.model.eval()
    with torch.no_grad():
        for seq in test_seqs:
            times = np.asarray(seq["times"], dtype=np.float32)
            locs = np.asarray(seq["locations"], dtype=np.float32)
            n = times.shape[0]
            true_times.append(times)
            true_locs.append(locs)

            # Condition on the first half of the sequence; roll out the rest
            cond_len = max(1, n // 2)
            history = {
                "times": times[:cond_len].copy(),
                "locations": locs[:cond_len].copy(),
            }
            target_len = n - cond_len

            seq_rollout_t: list[np.ndarray] = []
            seq_rollout_s: list[np.ndarray] = []

            # For thinning models, build the conditioning state and initial
            # proposal cache once and share across all K rollouts.  The
            # proposal cache is valid for the first event of each rollout;
            # rollout_window_thinning clears it after each generated event.
            shared_state_ctx = None
            shared_proposal_cache = None
            if not use_native and target_len > 0:
                horizon = float(target_len) * median_iet * 3.0
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
        method=method,
    )


# ---------------------------------------------------------------------------
# Per-sequence NLL computation
# ---------------------------------------------------------------------------


def _build_single_seq_batch(
    seq: dict[str, np.ndarray],
    norm_stats: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Turn one raw sequence into a single-item padded batch for model.eval_forward."""
    times = np.asarray(seq["times"], dtype=np.float32)
    locs = np.asarray(seq["locations"], dtype=np.float32)
    normalize = bool(norm_stats.get("normalize", False))
    if normalize:
        t_mean = float(norm_stats.get("time_mean", 0.0))
        t_std = max(float(norm_stats.get("time_std", 1.0)), 1e-8)
        loc_mean = np.asarray(norm_stats.get("loc_mean", [0.0, 0.0]), dtype=np.float32)
        loc_std = np.maximum(
            np.asarray(norm_stats.get("loc_std", [1.0, 1.0]), dtype=np.float32), 1e-8
        )
        times = (times - t_mean) / t_std
        locs = (locs - loc_mean) / loc_std
    n = times.shape[0]
    return {
        "times": torch.tensor(times, dtype=torch.float32, device=device).unsqueeze(0),
        "locations": torch.tensor(locs, dtype=torch.float32, device=device).unsqueeze(0),
        "lengths": torch.tensor([n], dtype=torch.long, device=device),
    }


def compute_seq_nlls(
    runner: STPPRunner,
    seqs: list[dict[str, np.ndarray]],
    *,
    device: torch.device,
) -> np.ndarray:
    """Compute per-sequence mean NLL for each sequence.

    Returns a float32 array of shape (len(seqs),) where each entry is the
    mean NLL per event for that sequence.  NaN for SMASH (nll_kind="none").
    """
    caps = runner.model.event_model.capabilities
    if caps.nll_kind == "none":
        return np.full(len(seqs), float("nan"), dtype=np.float32)

    norm_stats = runner.norm_stats
    nlls: list[float] = []
    runner.model.eval()
    with torch.no_grad():
        for seq in seqs:
            batch = _build_single_seq_batch(seq, norm_stats, device)
            # eval_forward dispatches to training_loss for exact models,
            # or to a separate approx path for DSTPP.
            fwd = runner.model.eval_forward(
                times=batch["times"],
                locations=batch["locations"],
                lengths=batch["lengths"],
            )
            result = runner.model.compute_loss(fwd)
            nll_val = float(result.nll.item())
            nlls.append(nll_val)

    return np.asarray(nlls, dtype=np.float32)


# ---------------------------------------------------------------------------
# Intensity grid computation
# ---------------------------------------------------------------------------

_DEFAULT_GRID_SPEC: dict[str, Any] = {
    "x_range": [0.0, 1.0],
    "y_range": [0.0, 1.0],
    "x_resolution": 50,
    "y_resolution": 50,
    "t_resolution": 100,
}


def compute_intensity_grid(
    runner: STPPRunner,
    test_seqs: list[dict[str, np.ndarray]],
    *,
    grid_spec: dict[str, Any],
    ground_truth: GroundTruth | None,
    generative_rollouts: GenerativeRollouts | None,
    device: torch.device,
) -> IntensityGrid:
    """Compute a spatiotemporal intensity surface on the shared grid.

    Uses direct intensity() queries for models with has_intensity=True,
    otherwise falls back to 3D Gaussian KDE from generative rollouts.

    The surface is averaged over the first test sequence (or a representative
    window) as a diagnostic.  For per-sequence grids, callers should invoke
    the grid metrics directly with the runner.
    """
    spec = dict(_DEFAULT_GRID_SPEC)
    spec.update(grid_spec)

    caps = runner.model.event_model.capabilities
    x_range = spec.get("x_range", [0.0, 1.0])
    y_range = spec.get("y_range", [0.0, 1.0])
    xs = np.linspace(float(x_range[0]), float(x_range[1]), int(spec["x_resolution"]), dtype=np.float32)
    ys = np.linspace(float(y_range[0]), float(y_range[1]), int(spec["y_resolution"]), dtype=np.float32)

    # Use the first test sequence's time range for the temporal axis
    if test_seqs:
        t0 = float(test_seqs[0]["times"][0])
        t1 = float(test_seqs[0]["times"][-1])
    else:
        t0, t1 = 0.0, 1.0
    ts = np.linspace(t0, t1, int(spec["t_resolution"]), dtype=np.float32)

    if caps.has_intensity:
        lambda_hat = _direct_intensity_grid(runner, test_seqs, xs, ys, ts, device)
        method = "direct"
    elif generative_rollouts is not None:
        lambda_hat = _kde_intensity_grid(generative_rollouts, xs, ys, ts)
        method = "kde"
    else:
        # Should not happen if EvalContext gating is correct
        lambda_hat = np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)
        method = "unavailable"

    lambda_true = None
    if ground_truth is not None and ground_truth.intensity_grid is not None:
        lambda_true = np.asarray(ground_truth.intensity_grid, dtype=np.float32)

    return IntensityGrid(
        lambda_hat=lambda_hat,
        lambda_true=lambda_true,
        xs=xs,
        ys=ys,
        ts=ts,
        method=method,
    )


def _direct_intensity_grid(
    runner: STPPRunner,
    test_seqs: list[dict[str, np.ndarray]],
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    device: torch.device,
    chunk_size: int = 512,
) -> np.ndarray:
    """Evaluate intensity on the grid using the first test sequence as conditioning."""
    from unified_stpp.evaluation.intensity import eval_intensity

    if not test_seqs:
        return np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)

    seq = test_seqs[0]
    norm_stats = runner.norm_stats
    normalize = bool(norm_stats.get("normalize", False))
    t_mean = float(norm_stats.get("time_mean", 0.0)) if normalize else 0.0
    t_std = max(float(norm_stats.get("time_std", 1.0)), 1e-8) if normalize else 1.0
    loc_mean = (
        np.asarray(norm_stats.get("loc_mean", [0.0, 0.0]), dtype=np.float32)
        if normalize
        else np.zeros(2, dtype=np.float32)
    )
    loc_std = (
        np.maximum(
            np.asarray(norm_stats.get("loc_std", [1.0, 1.0]), dtype=np.float32), 1e-8
        )
        if normalize
        else np.ones(2, dtype=np.float32)
    )

    result = np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    s_grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)  # (G, 2)

    runner.model.eval()
    with torch.no_grad():
        for ti, t_query in enumerate(ts):
            # Build history up to t_query
            times = np.asarray(seq["times"], dtype=np.float32)
            hist_mask = times <= float(t_query)
            if not hist_mask.any():
                continue
            locs_full = np.asarray(seq["locations"], dtype=np.float32)
            h = {
                "times": times[hist_mask].copy(),
                "locations": locs_full[hist_mask].copy(),
            }
            intensity_vals = eval_intensity(
                runner.model,
                t_query=float(t_query),
                s_grid=s_grid,
                history_times=h["times"],
                history_locs=h["locations"],
                t_bias=t_mean,
                t_scale=t_std,
                s_bias=loc_mean,
                s_scale=loc_std,
                device=device,
                correct_for_normalization=normalize,
            )
            result[ti] = intensity_vals.reshape(len(xs), len(ys))

    return result


def _kde_intensity_grid(
    rollouts: GenerativeRollouts,
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:
    """Estimate intensity on the grid from generative rollout events via 3D KDE."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)

    # Pool all rollout events into a 3D point cloud (t, x, y)
    points: list[np.ndarray] = []
    for seq_rollouts_t, seq_rollouts_s in zip(rollouts.rollout_times, rollouts.rollout_locs):
        for r_t, r_s in zip(seq_rollouts_t, seq_rollouts_s):
            if r_t.size > 0 and r_s.shape[0] > 0:
                pts = np.column_stack([r_t, r_s[:, 0], r_s[:, 1]])
                points.append(pts)

    if not points:
        return np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)

    all_pts = np.concatenate(points, axis=0).astype(np.float64)  # (M, 3)
    n_pts = all_pts.shape[0]

    # Scott bandwidth with floor
    d = 3
    scott_bw = n_pts ** (-1.0 / (d + 4))
    t_range = float(ts[-1] - ts[0]) if len(ts) > 1 else 1.0
    x_range = float(xs[-1] - xs[0]) if len(xs) > 1 else 1.0
    y_range = float(ys[-1] - ys[0]) if len(ys) > 1 else 1.0
    bw_floor = 0.02
    bw = max(scott_bw, bw_floor)

    # Normalise to [0, 1] before KDE
    pts_norm = all_pts.copy()
    pts_norm[:, 0] = (all_pts[:, 0] - float(ts[0])) / max(t_range, 1e-8)
    pts_norm[:, 1] = (all_pts[:, 1] - float(xs[0])) / max(x_range, 1e-8)
    pts_norm[:, 2] = (all_pts[:, 2] - float(ys[0])) / max(y_range, 1e-8)

    try:
        kde = gaussian_kde(pts_norm.T, bw_method=bw)
    except Exception:
        return np.zeros((len(ts), len(xs), len(ys)), dtype=np.float32)

    tt, xx, yy = np.meshgrid(
        (ts - float(ts[0])) / max(t_range, 1e-8),
        (xs - float(xs[0])) / max(x_range, 1e-8),
        (ys - float(ys[0])) / max(y_range, 1e-8),
        indexing="ij",
    )
    grid_pts = np.stack([tt.ravel(), xx.ravel(), yy.ravel()], axis=0)  # (3, T*X*Y)
    density = kde(grid_pts).reshape(len(ts), len(xs), len(ys)).astype(np.float32)

    # Scale density to approximate intensity (events per unit volume per rollout)
    total_rollouts = sum(len(r) for r in rollouts.rollout_times)
    scale = float(n_pts) / max(float(total_rollouts), 1.0)
    return density * scale
