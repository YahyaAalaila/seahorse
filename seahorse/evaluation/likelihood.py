"""Likelihood-oriented evaluation helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from seahorse.data.dataset import _strict_float32_times
from seahorse.runner.runner import STPPRunner

from .predictive.sampling import compute_predictive_samples
from .backends import is_statistical_runner


_NON_INCREASING_ACTIVE_EVENT_TIMES = (
    "Neural STPP temporal integrate_lambda received non-increasing active event times"
)


def _prepare_eval_sequence_arrays(
    seq: dict[str, np.ndarray],
    norm_stats: dict[str, Any],
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one evaluation sequence and preserve strict time ordering."""
    times = np.asarray(seq["times"], dtype=np.float64)
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
    times = _strict_float32_times(times, context=context)
    return times, locs


def _build_single_seq_batch(
    seq: dict[str, np.ndarray],
    norm_stats: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Turn one raw sequence into a single-item padded batch for model.eval_forward.

    Marks are carried through when the sequence has them, and are **never**
    rescaled -- they are categorical labels, not coordinates, so ``norm_stats``
    must not touch them. Omitting them here used to make every marked preset
    score its mark-free ground process instead of the joint model it declares:
    finite, plausible, and wrong by 29.5% on a fitted model. The gap is exactly
    zero on an untrained model, which is genuinely mark-exchangeable at
    initialisation, so no smoke test could catch it.
    """
    times, locs = _prepare_eval_sequence_arrays(
        seq,
        norm_stats,
        context="likelihood single-sequence eval batch",
    )
    n = times.shape[0]
    batch = {
        "times": torch.tensor(times, dtype=torch.float32, device=device).unsqueeze(0),
        "locations": torch.tensor(locs, dtype=torch.float32, device=device).unsqueeze(0),
        "lengths": torch.tensor([n], dtype=torch.long, device=device),
    }
    marks = seq.get("marks")
    if marks is not None:
        marks_arr = np.asarray(marks).reshape(-1)
        if marks_arr.shape[0] != n:
            raise ValueError(
                f"marks length {marks_arr.shape[0]} does not match {n} events in the "
                f"likelihood single-sequence eval batch."
            )
        batch["marks"] = torch.tensor(
            marks_arr.astype(np.int64), dtype=torch.long, device=device
        ).unsqueeze(0)
    return batch


def compute_seq_nlls(
    runner: STPPRunner,
    seqs: list[dict[str, np.ndarray]],
    *,
    device: torch.device,
) -> np.ndarray:
    """Compute per-sequence mean NLL for each sequence."""
    if is_statistical_runner(runner):
        from .statistical import compute_statistical_next_event_nll
        return np.asarray([
            compute_statistical_next_event_nll(runner, [seq])["mean_nll"]
            for seq in seqs
        ], dtype=np.float64)
    from seahorse.runner.results import resolve_loss_result_reporting

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
                marks=batch.get("marks"),
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
    """Compute held-out next-event NLL, not full observation-window likelihood.

    History-only targets are unscored, not zero-loss predictions. Comparisons
    across families must use the same eligible targets, since history requirements
    can differ. Observation-window endpoints are not needed for this statistic.

    Exact scores expose an aligned ``per_context_scored_mask``. Missing contexts
    have no score; ``n_unscored_contexts`` excludes caught evaluator exceptions.
    ``n_failed_contexts`` counts exceptions and scored NaNs. A scored infinity
    remains an infinite score and is counted in ``n_nonfinite_scored_contexts``.
    """
    if is_statistical_runner(runner):
        from .statistical import compute_statistical_next_event_nll
        return compute_statistical_next_event_nll(runner, seqs)
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
    from seahorse.runner.results import resolve_loss_result_reporting

    requested_space = runner.config.training.test_nll_space
    report_space = "native"
    method = ""
    per_context_chunks: list[np.ndarray] = []
    scored_chunks: list[np.ndarray] = []
    failed_chunks: list[np.ndarray] = []

    runner.model.eval()
    with torch.no_grad():
        for seq in seqs:
            seq_len = int(np.asarray(seq["times"]).shape[0])
            if seq_len < 2:
                continue
            failed = np.zeros(seq_len - 1, dtype=bool)
            try:
                batch = _build_single_seq_batch(seq, runner.norm_stats, device)
                output = runner.model.eval_forward(
                    times=batch["times"],
                    locations=batch["locations"],
                    lengths=batch["lengths"],
                    marks=batch.get("marks"),
                )
                result = runner.model.compute_loss(output)
                reported_nll, _temporal, _spatial, _extra, report_space = (
                    resolve_loss_result_reporting(
                        result,
                        requested_space=requested_space,
                    )
                )
                correction = _reporting_correction(result, reported_nll, report_space)
                terms = _extract_eventwise_next_event_terms(
                    output,
                    seq_len=seq_len,
                    correction=correction,
                )
                if terms is not None:
                    per_context, scored = terms
                    method = method or "exact_next_event_from_eventwise_terms"
                else:
                    per_context, scored = _prefix_difference_next_event_terms(
                        runner,
                        seq,
                        device=device,
                    )
                    method = "exact_next_event_from_prefix_differences"
            except RuntimeError as exc:
                if _NON_INCREASING_ACTIVE_EVENT_TIMES not in str(exc):
                    raise
                per_context = np.full((seq_len - 1,), np.nan, dtype=np.float32)
                scored = np.zeros(seq_len - 1, dtype=bool)
                failed[:] = True
            per_context_chunks.append(per_context.astype(np.float32, copy=False))
            scored_chunks.append(scored)
            failed_chunks.append(failed)

    per_context_nll = (
        np.concatenate(per_context_chunks, axis=0).astype(np.float32, copy=False)
        if per_context_chunks
        else np.zeros((0,), dtype=np.float32)
    )
    description = "exact held-out next-event NLL/event over teacher-forced test prefixes"
    scored_mask = np.concatenate(scored_chunks) if scored_chunks else np.zeros(0, dtype=bool)
    failed_mask = np.concatenate(failed_chunks) if failed_chunks else np.zeros(0, dtype=bool)
    scored_values = per_context_nll[scored_mask]
    if report_space == "raw":
        description += " (raw/original data space)"
    return {
        # Preserve scored NaNs (undefined evaluation) and +inf (zero density).
        "mean_nll": float(np.mean(scored_values)) if scored_values.size else float("nan"),
        "method": method or "exact_next_event_from_eventwise_terms",
        "kind": "exact",
        "report_space": report_space,
        "description": description,
        "footnote": "",
        "per_context_nll": per_context_nll,
        "per_context_scored_mask": scored_mask,
        "per_context_failure_mask": failed_mask | (scored_mask & np.isnan(per_context_nll)),
        "n_contexts": int(per_context_nll.size),
        "n_scored_contexts": int(scored_mask.sum()),
        "n_missing_contexts": int((~scored_mask).sum()),
        "n_unscored_contexts": int((~scored_mask & ~failed_mask).sum()),
        "n_failed_contexts": int(failed_mask.sum() + np.isnan(scored_values).sum()),
        "n_nonfinite_scored_contexts": int((~np.isfinite(scored_values)).sum()),
        "sampling_backend": None,
    }


def _reporting_correction(result: Any, reported_nll: float, report_space: str) -> float:
    """Use the declared Jacobian; subtracting nonfinite means cannot recover it."""
    if report_space != "raw":
        return 0.0
    extra = getattr(result, "extra_metrics", {}) or {}
    correction = float(extra.get("raw_reporting_correction", reported_nll - float(result.nll)))
    if not math.isfinite(correction):
        raise ValueError(
            "Raw-space eventwise reporting requires a finite raw_reporting_correction; "
            "it cannot be inferred by subtracting nonfinite aggregate NLLs."
        )
    return correction


def _extract_eventwise_next_event_nlls(
    output: dict[str, Any],
    *,
    seq_len: int,
    correction: float,
) -> np.ndarray | None:
    """Compatibility wrapper returning losses aligned to original targets 1..L-1."""
    terms = _extract_eventwise_next_event_terms(output, seq_len=seq_len, correction=correction)
    return None if terms is None else terms[0]


def _extract_eventwise_next_event_terms(
    output: dict[str, Any],
    *,
    seq_len: int,
    correction: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return aligned NLLs and scored-target mask, independently of loss finiteness.

    A packed matrix declares ``eventwise_target_indices`` with the same shape:
    zero-based original event indices at valid entries, and -1 at padding.
    Index zero (an unconditional first event) is excluded from next-event scoring.
    """
    nll_matrix = output.get("nll_matrix")
    mask = output.get("mask")
    if not isinstance(nll_matrix, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return None
    if nll_matrix.ndim != 2 or mask.ndim != 2 or nll_matrix.shape[0] != 1 or mask.shape[0] != 1:
        return None
    if nll_matrix.shape != mask.shape:
        raise ValueError("nll_matrix and mask must have the same shape.")

    native = nll_matrix[0].detach().cpu().numpy().astype(np.float64, copy=False)
    valid = mask[0].detach().cpu().numpy() > 0
    target_count = max(seq_len - 1, 0)
    aligned = np.full(target_count, np.nan, dtype=np.float64)
    scored = np.zeros(target_count, dtype=bool)

    target_indices = output.get("eventwise_target_indices")
    if target_indices is not None:
        if (
            not isinstance(target_indices, torch.Tensor)
            or target_indices.shape != nll_matrix.shape
            or target_indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("eventwise_target_indices must be an integer tensor matching nll_matrix.")
        indices = target_indices[0].detach().cpu().numpy()
        active = indices[valid]
        if (
            np.any(indices[~valid] != -1)
            or np.any(active < 0)
            or np.any(active >= seq_len)
            or np.any(np.diff(active) <= 0)
        ):
            raise ValueError(
                "eventwise_target_indices must be strictly increasing valid event indices "
                "with -1 only at padding."
            )
        selected = valid & (indices > 0)
        slots = indices[selected] - 1
        aligned[slots] = native[selected] + float(correction)
        scored[slots] = True
        return aligned, scored

    if target_count == 0:
        return aligned, scored

    next_event_mask = output.get("next_event_mask")
    selected = valid
    if isinstance(next_event_mask, torch.Tensor):
        if (
            next_event_mask.ndim == 2
            and next_event_mask.shape == nll_matrix.shape
            and next_event_mask.shape[0] == 1
        ):
            selected = valid & (next_event_mask[0].detach().cpu().numpy() > 0)
            candidate = native[selected]
            if candidate.shape[0] == target_count:
                return candidate + float(correction), np.ones(target_count, dtype=bool)

    if native.shape[0] >= seq_len and valid.shape[0] >= seq_len:
        scored = selected[1:seq_len]
        aligned[scored] = native[1:seq_len][scored] + float(correction)
        return aligned, scored

    candidate = native[valid]
    if candidate.shape[0] == target_count:
        return candidate + float(correction), np.ones(target_count, dtype=bool)

    if native.shape[0] >= target_count and valid.shape[0] >= target_count:
        candidate = native[:target_count][valid[:target_count]]
        if candidate.shape[0] == target_count:
            return candidate + float(correction), np.ones(target_count, dtype=bool)

    return None


def _prefix_difference_next_event_nlls(
    runner: STPPRunner,
    seq: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> np.ndarray:
    return _prefix_difference_next_event_terms(runner, seq, device=device)[0]


def _prefix_difference_next_event_terms(
    runner: STPPRunner,
    seq: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(seq["times"], dtype=np.float64).reshape(-1)
    n_prefixes = int(times.shape[0])
    if n_prefixes < 2:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)

    prefix_totals: list[np.ndarray] = []
    prefix_counts: list[np.ndarray] = []
    for prefix_lengths in _prefix_length_chunks(
        n_prefixes=n_prefixes,
        token_budget=_prefix_chunk_token_budget(device),
    ):
        chunk = _batched_prefix_totals_and_counts(
            runner,
            seq,
            prefix_lengths=prefix_lengths,
            device=device,
        )
        if chunk is None:
            return _prefix_difference_next_event_terms_unbatched(
                runner,
                seq,
                device=device,
            )
        prefix_totals.append(chunk[0])
        prefix_counts.append(chunk[1])

    if not prefix_totals:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)
    all_prefix_totals = np.concatenate(prefix_totals, axis=0)
    if all_prefix_totals.shape[0] != n_prefixes:
        return _prefix_difference_next_event_terms_unbatched(
            runner,
            seq,
            device=device,
        )
    return _difference_scored_prefix_totals(all_prefix_totals, np.concatenate(prefix_counts))


def _prefix_difference_next_event_nlls_unbatched(
    runner: STPPRunner,
    seq: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> np.ndarray:
    return _prefix_difference_next_event_terms_unbatched(runner, seq, device=device)[0]


def _prefix_difference_next_event_terms_unbatched(
    runner: STPPRunner,
    seq: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    prefix_totals: list[float] = []
    prefix_counts: list[float] = []
    # Prefixes remain raw here: the single-sequence batch builder normalizes once.
    times = np.asarray(seq["times"], dtype=np.float64)
    locs = np.asarray(seq["locations"], dtype=np.float32)
    seq_marks = seq.get("marks")
    marks_arr = (
        np.asarray(seq_marks).reshape(-1).astype(np.int64)
        if seq_marks is not None else None
    )
    for prefix_len in range(1, int(times.shape[0]) + 1):
        prefix = {
            "times": times[:prefix_len],
            "locations": locs[:prefix_len],
        }
        if marks_arr is not None:
            prefix["marks"] = marks_arr[:prefix_len]
        total, count = _sequence_total_and_count(runner, prefix, device=device)
        prefix_totals.append(total)
        prefix_counts.append(count)
    if len(prefix_totals) < 2:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)
    return _difference_scored_prefix_totals(
        np.asarray(prefix_totals, dtype=np.float64), np.asarray(prefix_counts, dtype=np.float64)
    )


def _difference_scored_prefix_totals(
    totals: np.ndarray, counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Difference additive prefix totals only when one scored event is appended.

    This fallback requires prefix-additive losses. Eventwise terms are preferable;
    count growth alone cannot establish additivity for a new model family.
    """
    increments = np.diff(counts)
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts != np.floor(counts))
        or np.any((increments != 0) & (increments != 1))
    ):
        raise ValueError("Prefix scored-event counts must grow by zero or one per appended event.")
    scored = increments == 1
    values = np.full(scored.shape, np.nan, dtype=np.float64)
    # inf-inf has no defined per-event decomposition; retain NaN as a scored failure.
    with np.errstate(invalid="ignore"):
        values[scored] = totals[1:][scored] - totals[:-1][scored]
    return values, scored


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
    times, locs = _prepare_eval_sequence_arrays(
        seq,
        norm_stats,
        context="likelihood multi-prefix eval batch",
    )

    lengths = np.asarray(prefix_lengths, dtype=np.int64).reshape(-1)
    batch_size = int(lengths.shape[0])
    max_len = int(lengths.max()) if lengths.size else 0
    spatial_dim = int(locs.shape[-1]) if locs.ndim == 2 else 0

    seq_marks = seq.get("marks")
    marks_arr = None
    if seq_marks is not None:
        marks_arr = np.asarray(seq_marks).reshape(-1).astype(np.int64)
        if marks_arr.shape[0] != times.shape[0]:
            raise ValueError(
                f"marks length {marks_arr.shape[0]} does not match "
                f"{times.shape[0]} events in the multi-prefix eval batch."
            )

    padded_times = np.zeros((batch_size, max_len), dtype=np.float32)
    padded_locs = np.zeros((batch_size, max_len, spatial_dim), dtype=np.float32)
    padded_marks = (
        np.zeros((batch_size, max_len), dtype=np.int64) if marks_arr is not None else None
    )
    for row, prefix_len in enumerate(lengths.tolist()):
        padded_times[row, :prefix_len] = times[:prefix_len]
        padded_locs[row, :prefix_len] = locs[:prefix_len]
        if padded_marks is not None:
            padded_marks[row, :prefix_len] = marks_arr[:prefix_len]

    batch = {
        "times": torch.tensor(padded_times, dtype=torch.float32, device=device),
        "locations": torch.tensor(padded_locs, dtype=torch.float32, device=device),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
    }
    if padded_marks is not None:
        batch["marks"] = torch.tensor(padded_marks, dtype=torch.long, device=device)
    return batch


def _batched_prefix_totals_and_counts(
    runner: STPPRunner,
    seq: dict[str, np.ndarray],
    *,
    prefix_lengths: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray] | None:
    from seahorse.runner.results import resolve_loss_result_reporting

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
        marks=batch.get("marks"),
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
    reported_nll, _temporal, _spatial, _extra, report_space = resolve_loss_result_reporting(
        result,
        requested_space=runner.config.training.test_nll_space,
    )
    correction = _reporting_correction(result, reported_nll, report_space)
    corrected_means = per_prefix_mean.to(dtype=torch.float64) + correction
    totals = torch.where(
        event_counts > 0,
        corrected_means * event_counts.to(dtype=torch.float64),
        torch.zeros_like(corrected_means),
    )
    return (
        totals.cpu().numpy().astype(np.float64, copy=False),
        event_counts.cpu().numpy().astype(np.float64, copy=False),
    )


def _sequence_total_and_count(
    runner: STPPRunner,
    seq: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[float, float]:
    from seahorse.runner.results import resolve_loss_result_reporting

    batch = _build_single_seq_batch(seq, runner.norm_stats, device)
    output = runner.model.eval_forward(
        times=batch["times"],
        locations=batch["locations"],
        lengths=batch["lengths"],
        marks=batch.get("marks"),
    )
    result = runner.model.compute_loss(output)
    reported_nll, _temporal, _spatial, _extra, _space = resolve_loss_result_reporting(
        result,
        requested_space=runner.config.training.test_nll_space,
    )
    count = float(result.total_events)
    return (float(reported_nll) * count if count else 0.0), count


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
