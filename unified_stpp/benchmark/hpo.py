"""
Ray Tune wrapper for HPO.

Search-space YAML syntax is documented in
:meth:`unified_stpp.config.tuning.TuningConfig.parse_config`.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unified_stpp.config.tuning import TuningConfig


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


def _mem_debug_enabled() -> bool:
    return os.environ.get("UNIFIED_STPP_HPO_MEM_DEBUG", "").lower() not in {
        "",
        "0",
        "false",
        "no",
    }


def _current_rss_mb() -> float:
    rss_kb = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        text=True,
    ).strip()
    return float(rss_kb) / 1024.0


def _peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _estimate_size_bytes(obj, *, _seen: "set[int] | None" = None) -> int:
    """Recursively estimate the in-memory footprint of a Python object graph."""
    import sys as _sys

    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)

    size = _sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(
            _estimate_size_bytes(k, _seen=_seen) + _estimate_size_bytes(v, _seen=_seen)
            for k, v in obj.items()
        )
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_estimate_size_bytes(v, _seen=_seen) for v in obj)
    elif hasattr(obj, "nbytes"):
        try:
            size += int(obj.nbytes)
        except (TypeError, ValueError):
            pass
    return size


def _emit_mem_event(stage: str, **payload) -> None:
    if not _mem_debug_enabled():
        return
    event = {
        "kind": "hpo_mem",
        "stage": stage,
        "pid": os.getpid(),
        "rss_mb": round(_current_rss_mb(), 2),
        "peak_rss_mb": round(_peak_rss_mb(), 2),
    }
    event.update(payload)
    print(json.dumps(event, sort_keys=True), flush=True)


def run_hpo(
    config_dict: dict,
    tuning: "TuningConfig",
    *,
    train_seqs: "list[dict] | None" = None,
    val_seqs: "list[dict] | None" = None,
    train_path: "str | os.PathLike[str] | None" = None,
    val_path: "str | os.PathLike[str] | None" = None,
    return_analysis: bool = False,
):
    """Run HPO over the tunable parameters in *config_dict*.

    Parameters
    ----------
    config_dict:      Raw (nested) config dict — values may be scalars, lists
                      (discrete choice) or ``{min, max}`` dicts (continuous range).
                      Pass ``STPPConfig.model_dump(mode="json")`` when starting
                      from a validated config with no search-space entries.
    train_seqs/val_seqs: Optional pre-loaded data sequences for each trial.
                         When omitted, trials load splits from ``train_path`` /
                         ``val_path`` inside the trial process.
    train_path/val_path: Optional dataset paths for path-based trial loading.
    tuning:           :class:`TuningConfig` controlling the HPO procedure
                      (algorithm, scheduler, budget, resources, etc.).

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
    except ImportError as exc:
        raise ImportError(
            "HPO requires Ray Tune: pip install 'unified-stpp[hpo]'"
        ) from exc

    from unified_stpp.runner import STPPRunner
    from unified_stpp.training.data_module import STPPDataModule
    from unified_stpp.utils import load_jsonl

    if train_seqs is None or val_seqs is None:
        if train_path is None or val_path is None:
            raise ValueError(
                "run_hpo requires either pre-loaded train_seqs/val_seqs or "
                "train_path/val_path."
            )
        train_path = str(Path(train_path).expanduser().resolve())
        val_path = str(Path(val_path).expanduser().resolve())
    else:
        _emit_mem_event(
            "driver_after_load",
            train_estimated_mb=round(_estimate_size_bytes(train_seqs) / (1024.0 * 1024.0), 2),
            val_estimated_mb=round(_estimate_size_bytes(val_seqs) / (1024.0 * 1024.0), 2),
            train_len=len(train_seqs),
            val_len=len(val_seqs),
        )

    ray_space, fixed = tuning.build_ray_config(config_dict)

    if not ray_space:
        from unified_stpp.config import STPPConfig
        return STPPConfig(**config_dict)  # nothing to tune

    metric = tuning.metric

    def _trial_fn(trial_config: dict):
        merged_flat = dict(fixed)
        merged_flat.update(trial_config)
        merged_nested = _unflatten(merged_flat)

        from unified_stpp.config import STPPConfig

        rss_trial_start_mb = _current_rss_mb()
        _emit_mem_event(
            "trial_start",
            input_mode="captured" if train_seqs is not None else "path",
        )
        if train_seqs is not None and val_seqs is not None:
            trial_train = train_seqs
            trial_val = val_seqs
            train_estimated_mb = _estimate_size_bytes(trial_train) / (1024.0 * 1024.0)
            val_estimated_mb = _estimate_size_bytes(trial_val) / (1024.0 * 1024.0)
        else:
            assert train_path is not None and val_path is not None
            trial_train = load_jsonl(train_path)
            trial_val = load_jsonl(val_path)
            train_estimated_mb = _estimate_size_bytes(trial_train) / (1024.0 * 1024.0)
            val_estimated_mb = _estimate_size_bytes(trial_val) / (1024.0 * 1024.0)
            _emit_mem_event(
                "trial_after_split_load",
                train_estimated_mb=round(train_estimated_mb, 2),
                val_estimated_mb=round(val_estimated_mb, 2),
                train_len=len(trial_train),
                val_len=len(trial_val),
            )

        trial_cfg = STPPConfig(**merged_nested)
        bundle = trial_cfg.build_data_bundle(trial_train, trial_val, None)
        dm = STPPDataModule(
            bundle,
            batch_size=trial_cfg.data.batch_size,
            num_workers=trial_cfg.data.num_workers,
            seed=trial_cfg.data.seed,
        )
        rss_after_dataset_mb = _current_rss_mb()
        _emit_mem_event(
            "trial_after_dataset_build",
            train_estimated_mb=round(train_estimated_mb, 2),
            val_estimated_mb=round(val_estimated_mb, 2),
            train_dataset_len=len(dm.train_dataset),
        )
        runner = STPPRunner(trial_cfg)
        result = runner.fit(
            trial_train,
            trial_val,
            dataset_id="hpo_trial",
            data_module=dm,
        )
        tune.report({
            metric: result.val_objective,
            "mem_rss_trial_start_mb": round(rss_trial_start_mb, 2),
            "mem_rss_after_dataset_mb": round(rss_after_dataset_mb, 2),
            "mem_peak_rss_mb": round(_peak_rss_mb(), 2),
            "mem_train_estimated_mb": round(train_estimated_mb, 2),
            "mem_val_estimated_mb": round(val_estimated_mb, 2),
        })

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    analysis = tune.run(
        _trial_fn,
        config=ray_space,
        num_samples=tuning.n_trials,
        scheduler=tuning.build_scheduler(),
        search_alg=tuning.build_search_alg(),
        resources_per_trial=tuning.resources_per_trial,
        metric=metric,
        mode=tuning.mode,
        verbose=tuning.verbose,
        fail_fast=tuning.fail_fast,
        max_concurrent_trials=tuning.max_concurrent_trials,
        **({"seed": tuning.seed} if tuning.seed is not None else {}),
    )

    best_flat = dict(fixed)
    best_flat.update(analysis.best_config)
    best_nested = _unflatten(best_flat)

    from unified_stpp.config import STPPConfig

    best_config = STPPConfig(**best_nested)
    if return_analysis:
        return best_config, analysis
    return best_config
