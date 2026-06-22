"""NLL-family metrics (M1–M5, M28).

Registers:
  nll                  — Raw-space NLL (catalog M1), exact / VB
  temporal_nll         — Temporal marginal NLL (catalog M2)
  spatial_nll          — Spatial conditional NLL (catalog M3)
  train_test_nll_gap   — NLL train/test gap (catalog M4)
  gt_nll_gap           — NLL gap to ground truth (catalog M5, synthetic only)
  context_sensitivity  — Context sensitivity curve (catalog M28)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from seahorse.evaluation.profiles import (
    GROUND_TRUTH,
    NLL_ARRAYS,
    PREDICTIVE_SAMPLES,
    TRAIN_DATA,
)
from seahorse.evaluation.registry import register_metric
from seahorse.evaluation.result import Metric, MetricResult

if TYPE_CHECKING:
    from seahorse.evaluation.context import EvalContext


# ---------------------------------------------------------------------------
# M1: Raw-space NLL
# ---------------------------------------------------------------------------


@register_metric
class NLL(Metric):
    """M1: Overall raw-space per-event NLL.

    Uses exact log-likelihood for AutoSTPP/DeepSTPP/NeuralSTPP,
    and the ELBO bound for DSTPP.  SMASH (nll_kind="none") is
    recorded as unavailable.
    """

    name = "nll"
    catalog_id = "M1"
    requires = frozenset()
    artifact_families = frozenset({NLL_ARRAYS})
    cost_class = "report"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        caps = ctx.available_capabilities
        if "nll_exact" not in caps and "nll_approx" not in caps:
            return MetricResult(
                value=None,
                available=False,
                reason="nll_kind is 'none' — model has no density",
            )
        nlls = ctx.seq_nlls  # (n_seqs,) mean NLL per event per sequence
        valid = nlls[np.isfinite(nlls)]
        if valid.size == 0:
            return MetricResult(value=None, available=False, reason="all NLL values are NaN")
        method = "exact" if "nll_exact" in caps else "vb"
        return MetricResult(
            value=float(valid.mean()),
            per_event=nlls.astype(np.float64),
            method=method,
        )


# ---------------------------------------------------------------------------
# M2: Temporal NLL
# ---------------------------------------------------------------------------


@register_metric
class TemporalNLL(Metric):
    """M2: Temporal marginal NLL: -1/N Σ log f*_T(t_i | H_{t_i}).

    For factorized models (DeepSTPP, NeuralSTPP): decoded directly.
    For AutoSTPP (joint): not easily separated — recorded as unavailable
    until a quadrature path is added.
    Sample-KDE fallbacks are intentionally split into
    ``temporal_nll_sample_kde`` so this metric never triggers predictive
    sampling implicitly.
    """

    name = "temporal_nll"
    catalog_id = "M2"
    requires = frozenset()
    artifact_families = frozenset({NLL_ARRAYS})
    cost_class = "report"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        # For factorized models with exact NLL, we can split temporal/spatial
        # if the model exposes event-wise terms.
        model_caps = ctx.runner.model.event_model.capabilities
        if model_caps.nll_kind == "exact" and model_caps.has_density:
            # Factorized model — try to compute temporal NLL directly.
            return self._compute_factorized(ctx)

        return MetricResult(
            value=None,
            available=False,
            reason=(
                "no exact temporal NLL path available for this model; "
                "sample-KDE fallback is explicit as temporal_nll_sample_kde "
                "under a predictive metric profile"
            ),
        )

    def _compute_factorized(self, ctx: "EvalContext") -> MetricResult:
        """Compute temporal NLL using the model's factorized temporal density."""
        import torch

        runner = ctx.runner
        device = ctx.device
        norm_stats = runner.norm_stats
        normalize = bool(norm_stats.get("normalize", False))
        t_mean = float(norm_stats.get("time_mean", 0.0)) if normalize else 0.0
        t_std = max(float(norm_stats.get("time_std", 1.0)), 1e-8) if normalize else 1.0

        per_seq_nll: list[float] = []
        runner.model.eval()
        with torch.no_grad():
            for seq in ctx.test_seqs:
                times = np.asarray(seq["times"], dtype=np.float32)
                locs = np.asarray(seq["locations"], dtype=np.float32)
                n = times.shape[0]
                if n < 2:
                    continue

                # Normalize times if needed
                t_norm = (times - t_mean) / t_std if normalize else times.copy()
                t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=device).unsqueeze(0)
                l_tensor = torch.tensor(locs, dtype=torch.float32, device=device).unsqueeze(0)
                if normalize:
                    loc_mean = np.asarray(norm_stats.get("loc_mean", [0.0, 0.0]), dtype=np.float32)
                    loc_std = np.maximum(np.asarray(norm_stats.get("loc_std", [1.0, 1.0]), dtype=np.float32), 1e-8)
                    l_tensor = (l_tensor - torch.tensor(loc_mean, device=device)) / torch.tensor(loc_std, device=device)

                lengths = torch.tensor([n], dtype=torch.long, device=device)
                fwd = runner.model.eval_forward(
                    times=t_tensor, locations=l_tensor, lengths=lengths
                )
                # Try to extract temporal_nll from forward output
                if hasattr(fwd, "temporal_nll") and fwd.temporal_nll is not None:
                    nll_val = float(fwd.temporal_nll.item())
                    # Jacobian: NLL in normalized space → raw space
                    if normalize:
                        nll_val -= math.log(max(t_std, 1e-8))
                    per_seq_nll.append(nll_val)

        if not per_seq_nll:
            return MetricResult(
                value=None,
                available=False,
                reason="model forward did not expose temporal_nll field",
            )
        arr = np.asarray(per_seq_nll, dtype=np.float64)
        return MetricResult(value=float(arr.mean()), per_event=arr, method="exact")

    def _compute_kde(self, ctx: "EvalContext") -> MetricResult:
        """Estimate temporal NLL from predictive samples via 1D KDE of IETs."""
        try:
            from scipy.stats import gaussian_kde
        except ImportError:
            return MetricResult(value=None, available=False, reason="scipy not available")

        samples = ctx.samples_predictive
        success_mask = np.asarray(
            getattr(samples, "sampling_succeeded", np.ones(samples.next_times.shape[0], dtype=np.bool_)),
            dtype=np.bool_,
        )
        # samples.next_times: (N, K) absolute times
        # true inter-event times: true_next_times - history_end_times
        true_iets = samples.true_next_times - samples.history_end_times  # (N,)
        per_event_nll: list[float] = []

        for i in range(samples.next_times.shape[0]):
            if not bool(success_mask[i]):
                per_event_nll.append(float("nan"))
                continue
            sample_iets = samples.next_times[i] - samples.history_end_times[i]  # (K,)
            sample_iets = np.maximum(sample_iets, 1e-8)
            if sample_iets.std() < 1e-10:
                per_event_nll.append(float("nan"))
                continue
            try:
                bw = max(sample_iets.std() * sample_iets.shape[0] ** (-0.2), 1e-6)
                kde = gaussian_kde(sample_iets, bw_method=bw / sample_iets.std())
                log_f = float(np.log(max(kde(max(float(true_iets[i]), 1e-8))[0], 1e-8)))
            except Exception:
                log_f = float("nan")
            per_event_nll.append(-log_f)

        arr = np.asarray(per_event_nll, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return MetricResult(value=None, available=False, reason="all KDE evaluations failed")
        return MetricResult(value=float(valid.mean()), per_event=arr, method="kde")


# ---------------------------------------------------------------------------
# M3: Spatial NLL
# ---------------------------------------------------------------------------


@register_metric
class SpatialNLL(Metric):
    """M3: Spatial conditional NLL: -1/N Σ log f*_S(s_i | t_i, H).

    Derived as nll − temporal_nll for joint models, or from model directly
    for factorized models.
    """

    name = "spatial_nll"
    catalog_id = "M3"
    requires = frozenset()
    artifact_families = frozenset({NLL_ARRAYS})
    cost_class = "report"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        model_caps = ctx.runner.model.event_model.capabilities
        if model_caps.nll_kind == "exact" and model_caps.has_density:
            return self._compute_factorized(ctx)
        return MetricResult(
            value=None,
            available=False,
            reason=(
                "no exact spatial NLL path available for this model; "
                "sample-KDE fallback is explicit as spatial_nll_sample_kde "
                "under a predictive metric profile"
            ),
        )

    def _compute_factorized(self, ctx: "EvalContext") -> MetricResult:
        import torch

        runner = ctx.runner
        device = ctx.device
        norm_stats = runner.norm_stats
        normalize = bool(norm_stats.get("normalize", False))
        loc_std = (
            np.maximum(np.asarray(norm_stats.get("loc_std", [1.0, 1.0]), dtype=np.float32), 1e-8)
            if normalize
            else np.ones(2, dtype=np.float32)
        )
        log_jac = float(np.sum(np.log(np.maximum(loc_std, 1e-8)))) if normalize else 0.0

        per_seq_nll: list[float] = []
        runner.model.eval()
        with torch.no_grad():
            for seq in ctx.test_seqs:
                times = np.asarray(seq["times"], dtype=np.float32)
                locs = np.asarray(seq["locations"], dtype=np.float32)
                n = times.shape[0]
                if n < 2:
                    continue
                t_norm = times.copy()
                l_norm = locs.copy()
                if normalize:
                    t_mean = float(norm_stats.get("time_mean", 0.0))
                    t_std = max(float(norm_stats.get("time_std", 1.0)), 1e-8)
                    loc_mean = np.asarray(norm_stats.get("loc_mean", [0.0, 0.0]), dtype=np.float32)
                    t_norm = (times - t_mean) / t_std
                    l_norm = (locs - loc_mean) / loc_std
                t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=device).unsqueeze(0)
                l_tensor = torch.tensor(l_norm, dtype=torch.float32, device=device).unsqueeze(0)
                lengths = torch.tensor([n], dtype=torch.long, device=device)
                fwd = runner.model.eval_forward(
                    times=t_tensor, locations=l_tensor, lengths=lengths
                )
                if hasattr(fwd, "spatial_nll") and fwd.spatial_nll is not None:
                    nll_val = float(fwd.spatial_nll.item()) - log_jac
                    per_seq_nll.append(nll_val)

        if not per_seq_nll:
            return MetricResult(
                value=None,
                available=False,
                reason="model forward did not expose spatial_nll field",
            )
        arr = np.asarray(per_seq_nll, dtype=np.float64)
        return MetricResult(value=float(arr.mean()), per_event=arr, method="exact")

    def _compute_kde(self, ctx: "EvalContext") -> MetricResult:
        try:
            from scipy.stats import gaussian_kde
        except ImportError:
            return MetricResult(value=None, available=False, reason="scipy not available")

        samples = ctx.samples_predictive
        success_mask = np.asarray(
            getattr(samples, "sampling_succeeded", np.ones(samples.next_locs.shape[0], dtype=np.bool_)),
            dtype=np.bool_,
        )
        true_locs = samples.true_next_locs  # (N, 2)
        per_event_nll: list[float] = []

        for i in range(samples.next_locs.shape[0]):
            if not bool(success_mask[i]):
                per_event_nll.append(float("nan"))
                continue
            s_samples = samples.next_locs[i]  # (K, 2)
            if s_samples.shape[0] < 5:
                per_event_nll.append(float("nan"))
                continue
            try:
                kde = gaussian_kde(s_samples.T)
                log_f = float(np.log(max(kde(true_locs[i])[0], 1e-8)))
            except Exception:
                log_f = float("nan")
            per_event_nll.append(-log_f)

        arr = np.asarray(per_event_nll, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return MetricResult(value=None, available=False, reason="all KDE evaluations failed")
        return MetricResult(value=float(valid.mean()), per_event=arr, method="kde")


@register_metric
class TemporalNLLSampleKDE(TemporalNLL):
    """Sample-KDE temporal NLL, explicitly backed by predictive samples."""

    name = "temporal_nll_sample_kde"
    catalog_id = "M2"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        return self._compute_kde(ctx)


@register_metric
class SpatialNLLSampleKDE(SpatialNLL):
    """Sample-KDE spatial NLL, explicitly backed by predictive samples."""

    name = "spatial_nll_sample_kde"
    catalog_id = "M3"
    requires = frozenset({"samples_predictive"})
    artifact_families = frozenset({PREDICTIVE_SAMPLES})
    cost_class = "sampling_heavy"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        return self._compute_kde(ctx)


# ---------------------------------------------------------------------------
# M4: NLL train/test gap
# ---------------------------------------------------------------------------


@register_metric
class TrainTestNLLGap(Metric):
    """M4: NLL gap between train and test sets — a proxy for overfitting."""

    name = "train_test_nll_gap"
    catalog_id = "M4"
    requires = frozenset({"train_data"})
    artifact_families = frozenset({NLL_ARRAYS, TRAIN_DATA})
    cost_class = "report"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        caps = ctx.available_capabilities
        if "nll_exact" not in caps and "nll_approx" not in caps:
            return MetricResult(
                value=None,
                available=False,
                reason="model has no density (nll_kind='none')",
            )
        from seahorse.evaluation.likelihood import compute_seq_nlls

        test_nlls = ctx.seq_nlls
        train_nlls = compute_seq_nlls(ctx.runner, ctx.train_data, device=ctx.device)

        test_mean = float(np.nanmean(test_nlls))
        train_mean = float(np.nanmean(train_nlls))
        gap = test_mean - train_mean

        method = "exact" if "nll_exact" in caps else "vb"
        return MetricResult(
            value=gap,
            method=method,
        )


# ---------------------------------------------------------------------------
# M5: NLL gap to ground truth (synthetic only)
# ---------------------------------------------------------------------------


@register_metric
class GTNLLGap(Metric):
    """M5: Gap between model NLL and the ground-truth process NLL (synthetic only)."""

    name = "gt_nll_gap"
    catalog_id = "M5"
    requires = frozenset({"ground_truth_params"})
    artifact_families = frozenset({NLL_ARRAYS, GROUND_TRUTH})
    cost_class = "report"

    def compute(self, ctx: "EvalContext") -> MetricResult:
        caps = ctx.available_capabilities
        if "nll_exact" not in caps and "nll_approx" not in caps:
            return MetricResult(
                value=None,
                available=False,
                reason="model has no density — cannot compute NLL gap",
            )

        gt_params = ctx.ground_truth.params if ctx.ground_truth else None
        if gt_params is None:
            return MetricResult(value=None, available=False, reason="no ground_truth params")

        # True NLL requires computing the true process log-likelihood on the
        # test events.  If gt_params contains pre-computed per-event log-likelihoods
        # (key "true_per_event_nll"), use those directly.
        if "true_per_event_nll" not in gt_params:
            return MetricResult(
                value=None,
                available=False,
                reason=(
                    "ground_truth.params does not contain 'true_per_event_nll'; "
                    "pre-compute true log-likelihood and include in GroundTruth.params"
                ),
            )

        true_nlls = np.asarray(gt_params["true_per_event_nll"], dtype=np.float64)
        model_nlls = ctx.seq_nlls.astype(np.float64)

        gap = float(np.nanmean(model_nlls) - np.nanmean(true_nlls))
        method = "exact" if "nll_exact" in caps else "vb"
        return MetricResult(value=gap, method=method)


# ---------------------------------------------------------------------------
# M28: Context sensitivity curve
# ---------------------------------------------------------------------------


@register_metric
class ContextSensitivity(Metric):
    """M28: NLL as a function of conditioning history length k.

    Evaluates NLL with the k most-recent events as history, for k in
    {0, 1, 5, 20, 100, full}.  Returns the result as a curve.
    """

    name = "context_sensitivity"
    catalog_id = "M28"
    requires = frozenset()
    artifact_families = frozenset({NLL_ARRAYS})
    cost_class = "repeated_nll"

    _K_VALUES = [0, 1, 5, 20, 100]

    def compute(self, ctx: "EvalContext") -> MetricResult:
        caps = ctx.available_capabilities
        if "nll_exact" not in caps and "nll_approx" not in caps:
            return MetricResult(
                value=None,
                available=False,
                reason="model has no density — cannot compute context sensitivity",
            )

        from seahorse.evaluation.likelihood import compute_seq_nlls

        curve: dict[str, float] = {}

        # Uncapped ("full") baseline
        full_nlls = ctx.seq_nlls
        curve["full"] = float(np.nanmean(full_nlls))

        # Capped variants
        from seahorse.evaluation.runtime import cap_history

        for k in self._K_VALUES:
            capped_seqs = [
                {
                    **seq,
                    "times": _cap_times(seq["times"], k),
                    "locations": _cap_locs(seq["locations"], k),
                }
                for seq in ctx.test_seqs
            ]
            nlls_k = compute_seq_nlls(ctx.runner, capped_seqs, device=ctx.device)
            curve[str(k)] = float(np.nanmean(nlls_k))

        method = "exact" if "nll_exact" in caps else "vb"
        return MetricResult(value=None, curve=curve, method=method)


def _cap_times(times: np.ndarray, k: int) -> np.ndarray:
    """Return the last k elements of times (all events if k==0 or len<=k)."""
    if k == 0:
        return np.zeros(0, dtype=times.dtype)
    return times[-k:] if times.shape[0] > k else times


def _cap_locs(locs: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return np.zeros((0, locs.shape[1] if locs.ndim > 1 else 2), dtype=locs.dtype)
    return locs[-k:] if locs.shape[0] > k else locs
