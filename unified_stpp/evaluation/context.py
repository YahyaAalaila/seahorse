"""EvalContext — lazy shared artifacts for post-fit metric computation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unified_stpp.runner.runner import STPPRunner
from unified_stpp.evaluation.artifacts import PredictiveSamples
from unified_stpp.evaluation.profiles import (
    GENERATIVE_ROLLOUTS,
    INTENSITY_GRID,
    MetricPlanError,
    PREDICTIVE_SAMPLES,
)


# ---------------------------------------------------------------------------
# Artifact dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GenerativeRollouts:
    """K full-sequence rollouts from the start of each test sequence.

    Attributes
    ----------
    rollout_times:  [seq_idx][k] = (n_events_k,) float32 — sampled event times.
    rollout_locs:   [seq_idx][k] = (n_events_k, 2) float32 — sampled locations.
    true_times:     [seq_idx] = (n_events,) float32 — ground-truth event times.
    true_locs:      [seq_idx] = (n_events, 2) float32 — ground-truth locations.
    method:         "thinning" | "native"
    """

    rollout_times: list[list[np.ndarray]]
    rollout_locs: list[list[np.ndarray]]
    true_times: list[np.ndarray]
    true_locs: list[np.ndarray]
    method: str


@dataclass
class IntensityGrid:
    """Spatiotemporal intensity surface on a shared grid.

    Attributes
    ----------
    lambda_hat:  (T, X, Y) float32 — model-predicted intensity.
    lambda_true: (T, X, Y) float32 | None — ground-truth intensity (synthetic only).
    xs:          (X,) float32 — x-axis grid coordinates.
    ys:          (Y,) float32 — y-axis grid coordinates.
    ts:          (T,) float32 — time-axis grid coordinates.
    method:      "direct" (intensity query) | "kde" (from samples).
    """

    lambda_hat: np.ndarray
    lambda_true: np.ndarray | None
    xs: np.ndarray
    ys: np.ndarray
    ts: np.ndarray
    method: str


@dataclass
class GroundTruth:
    """Ground-truth information from a synthetic generating process.

    Attributes
    ----------
    intensity_grid: Pre-computed true intensity on the shared spatiotemporal grid.
    params:         Dict of generating process parameters (e.g. for Hawkes processes).
    """

    intensity_grid: np.ndarray | None = None  # (T, X, Y) if available
    params: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# EvalContext
# ---------------------------------------------------------------------------

# Fixed vocabulary for capability/artifact strings used in Metric.requires.
_CAPABILITY_VOCAB = frozenset(
    {
        "nll_exact",
        "nll_approx",
        "intensity",
        "density",
        "samples_predictive",
        "samples_generative",
        "ground_truth_intensity",
        "ground_truth_params",
        "domain_mask",
        "train_data",
    }
)

# Default K budgets (can be overridden on EvalContext)
_K_PRED_DEFAULT = 32
_K_GEN_DEFAULT = 20
_ARTIFACT_MODES = frozenset({"load_or_compute", "load_only"})


class EvalContext:
    """Shared context passed to every Metric.compute() call.

    Expensive artifacts (samples, intensity grids) are computed lazily on first
    access and cached for reuse across metrics.  If no metric requests
    ``samples_predictive``, sampling never runs.

    Parameters
    ----------
    runner:        Loaded STPPRunner with model weights in eval mode.
    test_seqs:     List of test sequences (dicts with "times" and "locations" arrays).
    device:        Torch device for model inference.  Defaults to the model's device.
    ground_truth:  GroundTruth object for synthetic datasets (None for real data).
    domain_mask:   Boolean numpy array (X, Y) marking forbidden spatial regions.
    train_data:    Training sequences (needed for train/test gap metrics).
    k_pred:        Number of next-event samples per test event.
    k_gen:         Number of full-sequence rollouts per test sequence.
    grid_spec:     Dict with keys x_resolution, y_resolution, t_resolution and
                   x_range, y_range (each a [lo, hi] list).  Used for intensity grids.
    seed:          Base random seed for reproducible sampling.
    artifact_dir:  Optional root for persisted metric artifacts.
    artifact_mode: "load_or_compute" or "load_only" when artifact_dir is set.
    """

    def __init__(
        self,
        runner: STPPRunner,
        test_seqs: list[dict[str, np.ndarray]],
        *,
        device: torch.device | str | None = None,
        ground_truth: GroundTruth | None = None,
        domain_mask: np.ndarray | None = None,
        train_data: list[dict[str, np.ndarray]] | None = None,
        k_pred: int = _K_PRED_DEFAULT,
        k_gen: int = _K_GEN_DEFAULT,
        exact_time_bins: int = 8,
        exact_spatial_bins: int = 8,
        grid_spec: dict[str, Any] | None = None,
        seed: int = 0,
        planned_artifact_families: set[str] | frozenset[str] | None = None,
        artifact_dir: str | Path | None = None,
        artifact_mode: str = "load_or_compute",
    ) -> None:
        self.runner = runner
        self.test_seqs = test_seqs
        self.ground_truth = ground_truth
        self.domain_mask = domain_mask
        self.train_data = train_data
        self.k_pred = k_pred
        self.k_gen = k_gen
        self.exact_time_bins = int(exact_time_bins)
        self.exact_spatial_bins = int(exact_spatial_bins)
        self.grid_spec = grid_spec or {}
        self.seed = seed
        self.planned_artifact_families = frozenset(planned_artifact_families or ())
        self.artifact_dir = None if artifact_dir is None else Path(artifact_dir).resolve()
        self.artifact_mode = str(artifact_mode)
        if self.artifact_mode not in _ARTIFACT_MODES:
            raise ValueError(
                f"Unknown artifact_mode {artifact_mode!r}. "
                f"Expected one of {sorted(_ARTIFACT_MODES)}."
            )
        if self.artifact_dir is None and self.artifact_mode == "load_only":
            raise ValueError("artifact_mode='load_only' requires artifact_dir.")
        self.artifact_events: dict[str, dict[str, Any]] = {}

        if device is None:
            try:
                self.device = next(runner.model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.caps = runner.model.event_model.capabilities

    def _require_planned_artifact(self, family: str) -> None:
        if family not in self.planned_artifact_families:
            raise MetricPlanError(
                f"Heavy evaluation artifact {family!r} was not planned for this "
                "evaluation call. Select an explicit metric profile or pass "
                "allowed_artifact_families before running sampling-heavy metrics."
            )

    def ensure_artifacts(self, families: set[str] | frozenset[str]) -> None:
        """Materialize planned heavy artifacts before metric execution."""
        for family in sorted(families):
            if family == PREDICTIVE_SAMPLES:
                _ = self.samples_predictive
            elif family == GENERATIVE_ROLLOUTS:
                _ = self.samples_generative
            elif family == INTENSITY_GRID:
                _ = self.intensity_grid

    # ------------------------------------------------------------------
    # Capability resolution
    # ------------------------------------------------------------------

    @cached_property
    def available_capabilities(self) -> set[str]:
        """Compute the set of capability/artifact strings available in this context.

        Model capability mapping (EventCapabilities → vocabulary strings):
          nll_kind == "exact"                        → "nll_exact"
          nll_kind == "approx"                       → "nll_approx"
          has_intensity                              → "intensity"
          has_density                                → "density"
          has_intensity OR has_native_sampler        → "samples_predictive", "samples_generative"

        Context input mapping:
          ground_truth is not None                   → "ground_truth_intensity", "ground_truth_params"
          domain_mask is not None                    → "domain_mask"
          train_data is not None                     → "train_data"
        """
        caps: set[str] = set()

        # --- Model NLL capability ---
        if self.caps.nll_kind == "exact":
            caps.add("nll_exact")
        elif self.caps.nll_kind == "approx":
            caps.add("nll_approx")
        # nll_kind == "none" → neither added

        # --- Pointwise query capability ---
        if self.caps.has_intensity:
            caps.add("intensity")
        if self.caps.has_density:
            caps.add("density")

        # --- Sampling capability ---
        # Any model that can evaluate intensity (enabling thinning) or has a
        # native sampler can produce samples.  All five models in this codebase
        # satisfy at least one of these conditions.
        if self.caps.has_intensity or self.caps.has_native_sampler:
            caps.add("samples_predictive")
            caps.add("samples_generative")

        # --- Optional context inputs ---
        if self.ground_truth is not None:
            if self.ground_truth.intensity_grid is not None:
                caps.add("ground_truth_intensity")
            if self.ground_truth.params is not None:
                caps.add("ground_truth_params")
        if self.domain_mask is not None:
            caps.add("domain_mask")
        if self.train_data is not None:
            caps.add("train_data")

        return caps

    # ------------------------------------------------------------------
    # Lazy shared artifacts
    # ------------------------------------------------------------------

    @cached_property
    def samples_predictive(self) -> PredictiveSamples:
        """K next-event samples for every test event (teacher-forced history)."""
        self._require_planned_artifact(PREDICTIVE_SAMPLES)
        if self.artifact_dir is not None:
            return self._load_or_compute_predictive_samples()
        return self._compute_predictive_samples(memory_only=True)

    def _load_or_compute_predictive_samples(self) -> PredictiveSamples:
        from .artifacts import (
            build_predictive_samples_key,
            manifest_path_for_key,
            predictive_samples_payload_path,
        )
        from .bundle_io import (
            load_predictive_samples_artifact,
            write_predictive_samples_artifact,
        )

        assert self.artifact_dir is not None
        key = build_predictive_samples_key(
            self.runner,
            self.test_seqs,
            k=self.k_pred,
            seed=self.seed,
            device=str(self.device),
            exact_time_bins=self.exact_time_bins,
            exact_spatial_bins=self.exact_spatial_bins,
        )
        loaded = load_predictive_samples_artifact(self.artifact_dir, key)
        if loaded is not None:
            self.artifact_events[PREDICTIVE_SAMPLES] = {
                "status": "loaded_from_cache",
                "key": key.digest,
                "manifest_path": str(manifest_path_for_key(self.artifact_dir, key)),
                "payload_path": str(predictive_samples_payload_path(self.artifact_dir, key)),
            }
            return loaded
        if self.artifact_mode == "load_only":
            raise MetricPlanError(
                f"Missing predictive_samples artifact for key {key.digest} in "
                f"{self.artifact_dir} and artifact_mode='load_only'."
            )
        samples = self._compute_predictive_samples(memory_only=False)
        written = write_predictive_samples_artifact(self.artifact_dir, key, samples)
        self.artifact_events[PREDICTIVE_SAMPLES] = {
            "status": "computed_and_saved",
            "key": key.digest,
            "manifest_path": str(written["manifest"]),
            "payload_path": str(written["payload"]),
        }
        return samples

    def _compute_predictive_samples(self, *, memory_only: bool) -> PredictiveSamples:
        from .predictive.sampling import compute_predictive_samples

        samples = compute_predictive_samples(
            self.runner,
            self.test_seqs,
            k=self.k_pred,
            device=self.device,
            seed=self.seed,
            exact_time_bins=self.exact_time_bins,
            exact_spatial_bins=self.exact_spatial_bins,
        )
        if memory_only:
            self.artifact_events[PREDICTIVE_SAMPLES] = {
                "status": "computed_ephemeral",
                "key": None,
                "manifest_path": None,
                "payload_path": None,
            }
        return samples

    @cached_property
    def samples_generative(self) -> GenerativeRollouts:
        """K full-sequence rollouts for every test sequence."""
        from .predictive.rollout import compute_generative_rollouts

        self._require_planned_artifact(GENERATIVE_ROLLOUTS)
        return compute_generative_rollouts(
            self.runner,
            self.test_seqs,
            k=self.k_gen,
            device=self.device,
            seed=self.seed + 1,
        )

    @cached_property
    def intensity_grid(self) -> IntensityGrid:
        """Spatiotemporal intensity surface on the shared grid spec."""
        from .intensity import compute_intensity_grid

        self._require_planned_artifact(INTENSITY_GRID)
        generative_rollouts = None
        if not self.caps.has_intensity:
            self._require_planned_artifact(GENERATIVE_ROLLOUTS)
            generative_rollouts = self.samples_generative
        return compute_intensity_grid(
            self.runner,
            self.test_seqs,
            grid_spec=self.grid_spec,
            ground_truth=self.ground_truth,
            generative_rollouts=generative_rollouts,
            device=self.device,
        )

    @cached_property
    def seq_nlls(self) -> np.ndarray:
        """Per-sequence mean NLL values (one float per test sequence).

        For exact models this is the exact mean NLL/event.
        For approx models (DSTPP) this is the ELBO-based approximation.
        For SMASH (nll_kind="none") this is NaN for all sequences.
        """
        from .likelihood import compute_seq_nlls

        return compute_seq_nlls(self.runner, self.test_seqs, device=self.device)

    @cached_property
    def inter_event_times(self) -> np.ndarray:
        """Concatenated inter-event times across all test sequences.

        Returns a 1-D float32 array of length N_events where N_events is the
        total number of events across all test sequences.  The first event of
        each sequence has no predecessor, so it is excluded.
        """
        result: list[float] = []
        for seq in self.test_seqs:
            times = np.asarray(seq["times"], dtype=np.float32)
            if times.shape[0] > 1:
                result.extend((times[1:] - times[:-1]).tolist())
        return np.asarray(result, dtype=np.float32)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_test_events(self) -> int:
        """Total number of events across all test sequences."""
        return sum(np.asarray(s["times"]).shape[0] for s in self.test_seqs)

    @property
    def n_test_seqs(self) -> int:
        return len(self.test_seqs)

    @property
    def median_inter_event_time(self) -> float:
        """Median inter-event time across the test set (used to set rollout horizon)."""
        iets = self.inter_event_times
        if iets.size == 0:
            return 1.0
        return float(np.median(iets))
