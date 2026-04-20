"""Likelihood-oriented evaluation helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from unified_stpp.runner.runner import STPPRunner

from .predictive.sampling import compute_predictive_samples


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
    """Compute per-sequence mean NLL for each sequence."""
    from unified_stpp.runner.results import resolve_loss_result_reporting

    caps = runner.model.event_model.capabilities
    if caps.nll_kind == "none":
        return np.full(len(seqs), float("nan"), dtype=np.float32)

    norm_stats = runner.norm_stats
    nlls: list[float] = []
    runner.model.eval()
    with torch.no_grad():
        for seq in seqs:
            batch = _build_single_seq_batch(seq, norm_stats, device)
            fwd = runner.model.eval_forward(
                times=batch["times"],
                locations=batch["locations"],
                lengths=batch["lengths"],
            )
            result = runner.model.compute_loss(fwd)
            nll_val, _temporal, _spatial, _extra, _space = resolve_loss_result_reporting(
                result,
                requested_space=runner.config.training.test_nll_space,
            )
            nlls.append(nll_val)

    return np.asarray(nlls, dtype=np.float32)


def compute_next_event_test_nll(
    runner: STPPRunner,
    seqs: list[dict[str, np.ndarray]],
    *,
    device: torch.device,
    predictive_samples: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute benchmark-facing held-out next-event test NLL over all valid prefixes."""
    caps = runner.model.event_model.capabilities
    if caps.nll_kind == "exact":
        return _compute_exact_next_event_test_nll(runner, seqs, device=device)
    if caps.nll_kind != "approx":
        return {
            "mean_nll": float("nan"),
            "method": "unavailable",
            "kind": caps.nll_kind,
            "report_space": "native",
            "description": "held-out next-event NLL unavailable for this family",
            "footnote": caps.nll_footnote,
            "per_context_nll": np.zeros((0,), dtype=np.float32),
            "n_contexts": 0,
            "n_scored_contexts": 0,
            "n_missing_contexts": 0,
            "sampling_backend": None,
        }
    k = int(
        predictive_samples
        if predictive_samples is not None
        else getattr(runner.config.training, "predictive_test_nll_samples", 128)
    )
    return _compute_sampled_next_event_test_nll(
        runner,
        seqs,
        device=device,
        predictive_samples=k,
        seed=runner.config.data.seed if seed is None else int(seed),
    )


def _compute_exact_next_event_test_nll(
    runner: STPPRunner,
    seqs: list[dict[str, np.ndarray]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    from unified_stpp.runner.results import resolve_loss_result_reporting

    requested_space = runner.config.training.test_nll_space
    report_space = "native"
    method = ""
    per_context_chunks: list[np.ndarray] = []

    runner.model.eval()
    with torch.no_grad():
        for seq in seqs:
            seq_len = int(np.asarray(seq["times"]).shape[0])
            if seq_len < 2:
                continue
            batch = _build_single_seq_batch(seq, runner.norm_stats, device)
            output = runner.model.eval_forward(
                times=batch["times"],
                locations=batch["locations"],
                lengths=batch["lengths"],
            )
            result = runner.model.compute_loss(output)
            reported_nll, _temporal, _spatial, _extra, report_space = (
                resolve_loss_result_reporting(
                    result,
                    requested_space=requested_space,
                )
            )
            correction = float(reported_nll - float(result.nll))
            per_context = _extract_eventwise_next_event_nlls(
                output,
                seq_len=seq_len,
                correction=correction,
            )
            if per_context is not None:
                method = method or "exact_next_event_from_eventwise_terms"
            else:
                per_context = _prefix_difference_next_event_nlls(
                    runner,
                    seq,
                    device=device,
                )
                method = "exact_next_event_from_prefix_differences"
            per_context_chunks.append(per_context.astype(np.float32, copy=False))

    per_context_nll = (
        np.concatenate(per_context_chunks, axis=0).astype(np.float32, copy=False)
        if per_context_chunks
        else np.zeros((0,), dtype=np.float32)
    )
    description = "exact held-out next-event NLL/event over teacher-forced test prefixes"
    if report_space == "raw":
        description += " (raw/original data space)"
    return {
        "mean_nll": float(np.nanmean(per_context_nll)) if per_context_nll.size else float("nan"),
        "method": method or "exact_next_event_from_eventwise_terms",
        "kind": "exact",
        "report_space": report_space,
        "description": description,
        "footnote": "",
        "per_context_nll": per_context_nll,
        "n_contexts": int(per_context_nll.size),
        "n_scored_contexts": int(np.isfinite(per_context_nll).sum()),
        "n_missing_contexts": int((~np.isfinite(per_context_nll)).sum()),
        "sampling_backend": None,
    }


def _extract_eventwise_next_event_nlls(
    output: dict[str, Any],
    *,
    seq_len: int,
    correction: float,
) -> np.ndarray | None:
    nll_matrix = output.get("nll_matrix")
    mask = output.get("mask")
    if not isinstance(nll_matrix, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return None
    if nll_matrix.ndim != 2 or mask.ndim != 2 or nll_matrix.shape[0] != 1 or mask.shape[0] != 1:
        return None

    native = nll_matrix[0].detach().cpu().numpy().astype(np.float64, copy=False)
    valid = mask[0].detach().cpu().numpy() > 0
    target_count = max(seq_len - 1, 0)
    if target_count == 0:
        return np.zeros((0,), dtype=np.float64)

    next_event_mask = output.get("next_event_mask")
    if isinstance(next_event_mask, torch.Tensor):
        if (
            next_event_mask.ndim == 2
            and next_event_mask.shape == nll_matrix.shape
            and next_event_mask.shape[0] == 1
        ):
            candidate = native[next_event_mask[0].detach().cpu().numpy() > 0]
            if candidate.shape[0] == target_count:
                return candidate + float(correction)

    if native.shape[0] >= seq_len and valid.shape[0] >= seq_len:
        candidate = native[1:seq_len][valid[1:seq_len]]
        if candidate.shape[0] == target_count:
            return candidate + float(correction)

    candidate = native[valid]
    if candidate.shape[0] == target_count:
        return candidate + float(correction)

    if native.shape[0] >= target_count and valid.shape[0] >= target_count:
        candidate = native[:target_count][valid[:target_count]]
        if candidate.shape[0] == target_count:
            return candidate + float(correction)

    return None


def _prefix_difference_next_event_nlls(
    runner: STPPRunner,
    seq: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> np.ndarray:
    times = np.asarray(seq["times"], dtype=np.float32)
    n_prefixes = int(times.shape[0])
    if n_prefixes < 2:
        return np.zeros((0,), dtype=np.float64)

    prefix_totals: list[np.ndarray] = []
    for prefix_lengths in _prefix_length_chunks(
        n_prefixes=n_prefixes,
        token_budget=_prefix_chunk_token_budget(device),
    ):
        chunk_totals = _batched_prefix_total_nlls(
            runner,
            seq,
            prefix_lengths=prefix_lengths,
            device=device,
        )
        if chunk_totals is None:
            return _prefix_difference_next_event_nlls_unbatched(
                runner,
                seq,
                device=device,
            )
        prefix_totals.append(chunk_totals)

    if not prefix_totals:
        return np.zeros((0,), dtype=np.float64)
    all_prefix_totals = np.concatenate(prefix_totals, axis=0)
    if all_prefix_totals.shape[0] != n_prefixes:
        return _prefix_difference_next_event_nlls_unbatched(
            runner,
            seq,
            device=device,
        )
    return np.diff(all_prefix_totals.astype(np.float64, copy=False))


def _prefix_difference_next_event_nlls_unbatched(
    runner: STPPRunner,
    seq: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> np.ndarray:
    prefix_totals: list[float] = []
    times = np.asarray(seq["times"], dtype=np.float32)
    locs = np.asarray(seq["locations"], dtype=np.float32)
    for prefix_len in range(1, int(times.shape[0]) + 1):
        prefix = {
            "times": times[:prefix_len],
            "locations": locs[:prefix_len],
        }
        prefix_totals.append(
            _sequence_total_reported_nll(runner, prefix, device=device)
        )
    if len(prefix_totals) < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.diff(np.asarray(prefix_totals, dtype=np.float64))


def _prefix_chunk_token_budget(device: torch.device) -> int:
    return 16384 if device.type == "cuda" else 4096


def _prefix_length_chunks(
    *,
    n_prefixes: int,
    token_budget: int,
) -> list[np.ndarray]:
    if n_prefixes <= 0:
        return []
    budget = max(1, int(token_budget))
    chunks: list[np.ndarray] = []
    start = 1
    while start <= n_prefixes:
        end = start
        while end < n_prefixes:
            candidate_end = end + 1
            candidate_size = candidate_end - start + 1
            if candidate_size * candidate_end > budget:
                break
            end = candidate_end
        chunks.append(np.arange(start, end + 1, dtype=np.int64))
        start = end + 1
    return chunks


def _build_multi_prefix_batch(
    seq: dict[str, np.ndarray],
    *,
    prefix_lengths: np.ndarray,
    norm_stats: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    times = np.asarray(seq["times"], dtype=np.float32)
    locs = np.asarray(seq["locations"], dtype=np.float32)
    normalize = bool(norm_stats.get("normalize", False))
    if normalize:
        t_mean = float(norm_stats.get("time_mean", 0.0))
        t_std = max(float(norm_stats.get("time_std", 1.0)), 1e-8)
        loc_mean = np.asarray(norm_stats.get("loc_mean", [0.0, 0.0]), dtype=np.float32)
        loc_std = np.maximum(
            np.asarray(norm_stats.get("loc_std", [1.0, 1.0]), dtype=np.float32),
            1e-8,
        )
        times = (times - t_mean) / t_std
        locs = (locs - loc_mean) / loc_std

    lengths = np.asarray(prefix_lengths, dtype=np.int64).reshape(-1)
    batch_size = int(lengths.shape[0])
    max_len = int(lengths.max()) if lengths.size else 0
    spatial_dim = int(locs.shape[-1]) if locs.ndim == 2 else 0

    padded_times = np.zeros((batch_size, max_len), dtype=np.float32)
    padded_locs = np.zeros((batch_size, max_len, spatial_dim), dtype=np.float32)
    for row, prefix_len in enumerate(lengths.tolist()):
        padded_times[row, :prefix_len] = times[:prefix_len]
        padded_locs[row, :prefix_len] = locs[:prefix_len]

    return {
        "times": torch.tensor(padded_times, dtype=torch.float32, device=device),
        "locations": torch.tensor(padded_locs, dtype=torch.float32, device=device),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
    }


def _batched_prefix_total_nlls(
    runner: STPPRunner,
    seq: dict[str, np.ndarray],
    *,
    prefix_lengths: np.ndarray,
    device: torch.device,
) -> np.ndarray | None:
    from unified_stpp.runner.results import resolve_loss_result_reporting

    batch = _build_multi_prefix_batch(
        seq,
        prefix_lengths=prefix_lengths,
        norm_stats=runner.norm_stats,
        device=device,
    )
    output = runner.model.eval_forward(
        times=batch["times"],
        locations=batch["locations"],
        lengths=batch["lengths"],
    )
    nll_per_event = output.get("nll_per_event")
    mask = output.get("mask")
    if not isinstance(nll_per_event, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return None
    if mask.ndim != 2 or mask.shape[0] != int(prefix_lengths.shape[0]):
        return None

    per_prefix_mean = nll_per_event.detach().reshape(-1)
    if per_prefix_mean.numel() != int(prefix_lengths.shape[0]):
        return None
    event_counts = mask.sum(dim=1).detach().reshape(-1)
    if event_counts.numel() != int(prefix_lengths.shape[0]):
        return None

    result = runner.model.compute_loss(output)
    reported_nll, _temporal, _spatial, _extra, _space = resolve_loss_result_reporting(
        result,
        requested_space=runner.config.training.test_nll_space,
    )
    correction = float(reported_nll - float(result.nll))
    corrected_means = per_prefix_mean.to(dtype=torch.float64) + correction
    totals = corrected_means * event_counts.to(dtype=torch.float64)
    return totals.cpu().numpy().astype(np.float64, copy=False)


def _sequence_total_reported_nll(
    runner: STPPRunner,
    seq: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> float:
    from unified_stpp.runner.results import resolve_loss_result_reporting

    batch = _build_single_seq_batch(seq, runner.norm_stats, device)
    output = runner.model.eval_forward(
        times=batch["times"],
        locations=batch["locations"],
        lengths=batch["lengths"],
    )
    result = runner.model.compute_loss(output)
    reported_nll, _temporal, _spatial, _extra, _space = resolve_loss_result_reporting(
        result,
        requested_space=runner.config.training.test_nll_space,
    )
    return float(reported_nll) * float(result.total_events)


def _compute_sampled_next_event_test_nll(
    runner: STPPRunner,
    seqs: list[dict[str, np.ndarray]],
    *,
    device: torch.device,
    predictive_samples: int,
    seed: int,
) -> dict[str, Any]:
    samples = compute_predictive_samples(
        runner,
        seqs,
        k=predictive_samples,
        device=device,
        seed=seed,
    )
    n_contexts = int(samples.true_next_times.shape[0])
    per_context_nll = np.full((n_contexts,), np.nan, dtype=np.float32)
    for idx in range(n_contexts):
        if not bool(samples.sampling_succeeded[idx]):
            continue
        per_context_nll[idx] = _joint_sample_kde_next_event_nll(
            next_times=samples.next_times[idx],
            next_locs=samples.next_locs[idx],
            true_next_time=float(samples.true_next_times[idx]),
            true_next_loc=np.asarray(samples.true_next_locs[idx], dtype=np.float32),
            history_end_time=float(samples.history_end_times[idx]),
        )

    return {
        "mean_nll": float(np.nanmean(per_context_nll)) if np.isfinite(per_context_nll).any() else float("nan"),
        "method": "approx_next_event_joint_sample_kde",
        "kind": "approx",
        "report_space": "raw",
        "description": (
            "approximate held-out next-event NLL/event over teacher-forced test prefixes "
            "from a joint sample-KDE predictive density"
        ),
        "footnote": "‡ approximate held-out next-event NLL from joint sample-KDE predictive density",
        "per_context_nll": per_context_nll,
        "n_contexts": n_contexts,
        "n_scored_contexts": int(np.isfinite(per_context_nll).sum()),
        "n_missing_contexts": int((~np.isfinite(per_context_nll)).sum()),
        "sampling_backend": samples.sampling_backend,
    }


def _joint_sample_kde_next_event_nll(
    *,
    next_times: np.ndarray,
    next_locs: np.ndarray,
    true_next_time: float,
    true_next_loc: np.ndarray,
    history_end_time: float,
) -> float:
    from scipy.stats import gaussian_kde

    sample_dt = np.asarray(next_times, dtype=np.float64) - float(history_end_time)
    sample_locs = np.asarray(next_locs, dtype=np.float64)
    valid = np.isfinite(sample_dt) & (sample_dt > 0.0)
    valid &= np.all(np.isfinite(sample_locs), axis=1)
    if int(valid.sum()) < max(8, sample_locs.shape[-1] + 3):
        return float("nan")

    log_dt = np.log(np.maximum(sample_dt[valid], 1e-8))
    pts = np.column_stack([log_dt, sample_locs[valid]])
    target_dt = max(float(true_next_time) - float(history_end_time), 1e-8)
    target = np.concatenate([[math.log(target_dt)], np.asarray(true_next_loc, dtype=np.float64)])

    try:
        kde = gaussian_kde(pts.T)
    except Exception:
        scale = np.maximum(np.std(pts, axis=0, ddof=0), 1e-6)
        jitter = np.random.default_rng(0).normal(scale=1e-4 * scale, size=pts.shape)
        kde = gaussian_kde((pts + jitter).T)

    density_log_dt = float(kde(target)[0])
    if not math.isfinite(density_log_dt) or density_log_dt <= 0.0:
        return float("nan")
    density_raw = density_log_dt / max(target_dt, 1e-8)
    return float(-math.log(max(density_raw, 1e-12)))
