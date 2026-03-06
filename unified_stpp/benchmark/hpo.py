"""
YAML-as-search-space parser and Ray Tune wrapper for HPO.

YAML value interpretation:
  scalar          → fixed (not tuned)
  [a, b, c]       → tune.choice([a, b, c])
  {min: x, max: y}       → tune.uniform / tune.loguniform (inferred by param name)
  {min: x, max: y, scale: log}  → tune.loguniform (explicit)
  {min: i, max: j}  (both int)  → tune.randint(i, j+1)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Parameter names that trigger log-scale by default
_LOG_SCALE_KEYWORDS = ("lr", "learning_rate", "weight_decay", "alpha", "beta", "lambda")


def _is_log_param(name: str) -> bool:
    return any(k in name.lower() for k in _LOG_SCALE_KEYWORDS)


def _parse_value(key: str, value: Any, search_space: dict, fixed: dict, prefix: str):
    """Recursively classify one config leaf as tunable or fixed."""
    full_key = f"{prefix}.{key}" if prefix else key

    if isinstance(value, dict):
        # Check for search-space shorthand {min, max} or {min, max, scale}
        if "min" in value and "max" in value:
            lo, hi = value["min"], value["max"]
            scale = value.get("scale", "log" if _is_log_param(key) else "linear")
            if isinstance(lo, int) and isinstance(hi, int) and scale not in ("log", "ln"):
                search_space[full_key] = ("randint", lo, hi + 1)
            elif scale == "log":
                search_space[full_key] = ("loguniform", float(lo), float(hi))
            else:
                search_space[full_key] = ("uniform", float(lo), float(hi))
        else:
            # Nested dict — recurse
            for k, v in value.items():
                _parse_value(k, v, search_space, fixed, full_key)
    elif isinstance(value, list):
        search_space[full_key] = ("choice", value)
    else:
        fixed[full_key] = value


def parse_tunable_config(config_dict: dict) -> tuple[dict, dict]:
    """Split a (possibly nested) config dict into Ray Tune search space + fixed dict.

    Returns
    -------
    search_space : dict mapping dotted keys to ``(kind, *args)`` tuples.
    fixed        : dict mapping dotted keys to scalar values.

    The ``search_space`` dict can be converted to Ray Tune primitives with
    :func:`_to_ray_space`.
    """
    search_space: dict[str, Any] = {}
    fixed: dict[str, Any] = {}
    for k, v in config_dict.items():
        _parse_value(k, v, search_space, fixed, "")
    return search_space, fixed


def _to_ray_space(search_space: dict) -> dict:
    """Convert parsed search_space tuples to ``ray.tune`` primitives."""
    from ray import tune  # type: ignore[import]

    ray_space = {}
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
    return ray_space


def _unflatten(flat: dict) -> dict:
    """Convert ``{"a.b.c": v}`` to ``{"a": {"b": {"c": v}}}``."""
    out: dict = {}
    for key, val in flat.items():
        parts = key.split(".")
        d = out
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return out


def run_hpo(
    config_dict: dict,
    train_seqs: list[dict],
    val_seqs: list[dict],
    n_trials: int = 50,
    algorithm: str = "asha",
    metric: str = "val_nll",
    n_cpus_per_trial: int = 1,
    n_gpus_per_trial: float = 0.0,
):
    """Run HPO over the tunable parameters in *config_dict*.

    Parameters
    ----------
    config_dict:      Raw (nested) config dict — values may be scalars, lists
                      (discrete choice) or ``{min, max}`` dicts (continuous range).
                      Pass ``STPPConfig.model_dump(mode="json")`` when starting
                      from a validated config with no search-space entries.
    train_seqs/val_seqs: Data sequences for each trial.
    n_trials:         Maximum number of trials.
    algorithm:        ``"asha"`` (default) | ``"bayesian"`` | ``"grid"``.
    metric:           Metric to minimise (logged by ``STPPRunner.fit``).
    n_cpus/gpus_per_trial: Resources per Ray trial.

    Returns
    -------
    ``STPPConfig`` for the best trial (all tunable params resolved to scalars).

    Raises ``ImportError`` if Ray is not installed.
    If no tunable parameters are found, returns an ``STPPConfig`` built from
    *config_dict* unchanged.
    """
    try:
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
    except ImportError as exc:
        raise ImportError(
            "HPO requires Ray Tune: pip install 'unified-stpp[hpo]'"
        ) from exc

    from unified_stpp.runner import STPPRunner

    search_space_raw, fixed = parse_tunable_config(config_dict)

    if not search_space_raw:
        from unified_stpp.config import STPPConfig
        return STPPConfig(**config_dict)  # nothing to tune

    ray_space = _to_ray_space(search_space_raw)

    def _trial_fn(trial_config: dict):
        # Merge trial params (flat) back into config
        merged_flat = dict(fixed)
        merged_flat.update(trial_config)
        merged_nested = _unflatten(merged_flat)

        from unified_stpp.config import STPPConfig

        trial_cfg = STPPConfig(**merged_nested)
        runner = STPPRunner(trial_cfg)
        result = runner.fit(train_seqs, val_seqs, dataset_id="hpo_trial")
        tune.report({metric: result.val_nll})

    if algorithm == "asha":
        # metric/mode must not be set on the scheduler when also passed to tune.run()
        scheduler = ASHAScheduler()
        search_alg = None
    elif algorithm == "bayesian":
        from ray.tune.search.bayesopt import BayesOptSearch  # type: ignore[import]
        scheduler = None
        search_alg = BayesOptSearch()
    else:
        scheduler = None
        search_alg = None

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    analysis = tune.run(
        _trial_fn,
        config=ray_space,
        num_samples=n_trials,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={"cpu": n_cpus_per_trial, "gpu": n_gpus_per_trial},
        metric=metric,
        mode="min",
        verbose=1,
    )

    best_flat = dict(fixed)
    best_flat.update(analysis.best_config)
    best_nested = _unflatten(best_flat)

    from unified_stpp.config import STPPConfig

    return STPPConfig(**best_nested)
