"""
TuningConfig — HPO procedure configuration.

Owns the *procedure* of hyperparameter optimization (algorithm, scheduler,
budget, resources, reproducibility), not the *search space* (which is encoded
in the model/training YAML fields as {min/max}/list syntax).

Two orthogonal axes are modelled as separate fields:
  search_alg  — HOW to propose the next trial point ("random" | "bayesian")
  scheduler   — WHEN to early-stop poor trials ("asha" | "none")

Search-space YAML syntax (parsed by TuningConfig.parse_config):
  scalar                        → fixed (not tuned)
  [a, b, c]                     → choice([a, b, c])          (non-empty)
  {min: x, max: y}              → uniform / loguniform / randint (inferred)
  {min: x, max: y, scale: log}  → loguniform (explicit; requires min > 0)

Log-scale auto-inference: only ``lr`` and ``learning_rate`` and
``weight_decay`` are auto-inferred as log-scale. All other continuous params
require ``scale: log`` to use loguniform sampling.

Ray imports are deferred inside builder methods so this module can be imported
without Ray installed.
"""
from __future__ import annotations

from typing import Any
import warnings

from pydantic import BaseModel, field_validator, model_validator

# ---------------------------------------------------------------------------
# Search-space parsing helpers (module-private)
# ---------------------------------------------------------------------------

# Only names that are unambiguously positive and span multiple orders of magnitude.
# Ambiguous param names (alpha, beta, lambda, …) must use explicit scale: log.
_LOG_SCALE_SAFE = frozenset({"lr", "learning_rate", "weight_decay"})


def _infer_log_scale(name: str) -> bool:
    """Return True only for unambiguously log-scale hyperparameter names."""
    return name.lower() in _LOG_SCALE_SAFE


def _classify_value(
    key: str, value: Any, search_space: dict, fixed: dict, prefix: str
) -> None:
    """Recursively classify one config leaf as tunable or fixed.

    Populates *search_space* with ``(kind, *args)`` tuples and *fixed* with
    scalar values.  Raises ``ValueError`` for malformed search-space specs.
    """
    full_key = f"{prefix}.{key}" if prefix else key

    if isinstance(value, dict):
        if "min" in value and "max" in value:
            lo, hi = value["min"], value["max"]
            if lo >= hi:
                raise ValueError(
                    f"Search space '{full_key}': min={lo} must be < max={hi}"
                )
            explicit_scale = value.get("scale")
            scale = explicit_scale or ("log" if _infer_log_scale(key) else "linear")
            if scale in ("log", "ln"):
                if lo <= 0:
                    raise ValueError(
                        f"Search space '{full_key}': loguniform requires min > 0,"
                        f" got min={lo}"
                    )
                search_space[full_key] = ("loguniform", float(lo), float(hi))
            elif isinstance(lo, int) and isinstance(hi, int) and not explicit_scale:
                # Both int and no explicit scale → discrete integer range
                search_space[full_key] = ("randint", lo, hi + 1)
            else:
                search_space[full_key] = ("uniform", float(lo), float(hi))
        else:
            # Nested dict — recurse
            for k, v in value.items():
                _classify_value(k, v, search_space, fixed, full_key)
    elif isinstance(value, list):
        if not value:
            raise ValueError(
                f"Search space '{full_key}': choice list must not be empty"
            )
        search_space[full_key] = ("choice", value)
    else:
        fixed[full_key] = value


class TuningConfig(BaseModel):
    """Configuration for a Ray Tune HPO run."""

    # Trial budget
    n_trials: int = 50

    # Search algorithm — HOW to propose the next point
    search_alg: str = "random"
    """Proposal algorithm.
    ``"random"``   — Ray's default random/quasi-random sampler (no search_alg object).
    ``"bayesian"`` — BayesOptSearch (sequential; incompatible with scheduler="asha").
    """

    # Early-stopping scheduler — WHEN to kill poor trials
    scheduler: str = "asha"
    """Early-stopping policy.
    ``"asha"`` — ASHAScheduler (population-based; requires multiple concurrent trials).
    ``"none"`` — no early stopping.
    """

    # Objective
    metric: str = "val_objective"
    mode: str = "min"
    """Optimization direction: ``"min"`` or ``"max"``."""

    # Per-trial resources
    n_cpus_per_trial: int = 1
    n_gpus_per_trial: float = 0.0

    # Reproducibility
    seed: int = 0
    """RNG seed passed to ``tune.TuneConfig(seed=...)``."""

    # Robustness / concurrency
    fail_fast: bool = False
    """Stop all trials on the first failure."""
    max_concurrent_trials: int = 1
    """Maximum number of trials running simultaneously.
    ``1`` = fully sequential.  Values > 1 require a Ray cluster or sufficient
    local resources; ASHA also benefits from higher concurrency.
    """

    # Verbosity
    verbose: int = 1
    """Ray Tune verbosity level (0–3)."""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def from_sources(
        cls,
        yaml_tuning: Optional["TuningConfig | dict[str, Any]"] = None,
        *,
        cli_values: Optional[dict[str, Any]] = None,
    ) -> "TuningConfig":
        """Build tuning config from YAML defaults plus explicit CLI values."""
        if isinstance(yaml_tuning, cls):
            merged = yaml_tuning.model_dump()
        else:
            merged = dict(yaml_tuning or {})
        if cli_values:
            merged.update(cli_values)
        return cls(**merged)

    @model_validator(mode="after")
    def check_asha_bayesian_incompatibility(self) -> "TuningConfig":
        """ASHA requires population-based evaluation and is incompatible with
        the sequential BayesOptSearch sampler."""
        if self.search_alg == "bayesian" and self.scheduler == "asha":
            raise ValueError(
                "search_alg='bayesian' is sequential and incompatible with "
                "scheduler='asha' (population-based). "
                "Use scheduler='none' with bayesian, or search_alg='random' with asha."
            )
        return self

    @field_validator("metric", mode="before")
    @classmethod
    def canonicalize_metric(cls, value: object) -> str:
        if value is None:
            return "val_objective"
        metric = str(value).strip()
        if metric == "val_nll":
            warnings.warn(
                "TuningConfig.metric='val_nll' is deprecated; use 'val_objective'.",
                UserWarning,
                stacklevel=2,
            )
            return "val_objective"
        return metric

    @field_validator("seed", mode="before")
    @classmethod
    def require_explicit_seed(cls, value: object) -> int:
        if value is None:
            return 0
        return int(value)

    # ------------------------------------------------------------------
    # Search-space parsing — no Ray dependency
    # ------------------------------------------------------------------

    @staticmethod
    def parse_config(config_dict: dict) -> tuple[dict, dict]:
        """Split a raw config dict into (search_space_specs, fixed_values).

        ``search_space`` maps dotted keys to spec tuples::

            ("choice",     [v1, v2, ...])
            ("uniform",    lo, hi)
            ("loguniform", lo, hi)       # requires lo > 0
            ("randint",    lo, hi_excl)  # both int, no explicit scale

        ``fixed`` maps dotted keys to scalar values.

        Raises ``ValueError`` for malformed specs: ``min >= max``,
        non-positive loguniform bounds, or empty choice lists.
        """
        search_space: dict = {}
        fixed: dict = {}
        for k, v in config_dict.items():
            _classify_value(k, v, search_space, fixed, "")
        return search_space, fixed

    # ------------------------------------------------------------------
    # Builders — Ray imports are deferred
    # ------------------------------------------------------------------

    def build_ray_config(self, config_dict: dict) -> tuple[dict, dict]:
        """Parse *config_dict* and return ``(ray_space, fixed)`` ready for ``tune.run``.

        ``ray_space`` maps dotted keys to ``ray.tune`` distribution objects.
        ``fixed`` maps dotted keys to scalar values.

        Raises ``ImportError`` if Ray is not installed.
        """
        from ray import tune  # type: ignore[import]

        search_space, fixed = self.parse_config(config_dict)
        ray_space: dict = {}
        for key, spec in search_space.items():
            kind = spec[0]
            if kind == "choice":
                ray_space[key] = tune.choice(spec[1])
            elif kind == "uniform":
                ray_space[key] = tune.uniform(spec[1], spec[2])
            elif kind == "loguniform":
                ray_space[key] = tune.loguniform(spec[1], spec[2])
            elif kind == "randint":
                ray_space[key] = tune.randint(spec[1], spec[2])
        return ray_space, fixed

    def build_scheduler(self) -> Any | None:
        """Return an ``ASHAScheduler`` for ``scheduler='asha'``, else ``None``."""
        if self.scheduler == "asha":
            from ray.tune.schedulers import ASHAScheduler
            return ASHAScheduler()
        return None  # "none"

    def build_search_alg(self) -> Any | None:
        """Return a ``BayesOptSearch`` for ``search_alg='bayesian'``, else ``None``."""
        if self.search_alg == "bayesian":
            from ray.tune.search.bayesopt import BayesOptSearch
            return BayesOptSearch()
        return None  # "random" → Ray default

    @property
    def resources_per_trial(self) -> dict:
        """Resource dict consumed by ``tune.run(resources_per_trial=...)``."""
        return {"cpu": self.n_cpus_per_trial, "gpu": self.n_gpus_per_trial}
