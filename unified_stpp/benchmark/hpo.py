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
import csv
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def extract_trial_history(analysis) -> list[dict[str, Any]]:
    """Return a compact, JSON-serializable summary of Ray Tune trials."""
    records: list[dict[str, Any]] = []
    for trial in getattr(analysis, "trials", []) or []:
        last_result = dict(getattr(trial, "last_result", {}) or {})
        config = dict(getattr(trial, "config", {}) or {})
        records.append(
            {
                "trial_id": getattr(trial, "trial_id", None),
                "status": str(getattr(trial, "status", "")),
                "config": config,
                "last_result": {
                    key: value
                    for key, value in last_result.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                },
            }
        )
    return records


def _git(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def extract_analysis_metadata(analysis) -> dict[str, Any]:
    """Extract stable, JSON-serializable Ray Tune metadata when available."""
    if analysis is None:
        return {}

    best_result = {
        key: value
        for key, value in dict(getattr(analysis, "best_result", {}) or {}).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    best_trial = getattr(analysis, "best_trial", None)
    best_trial_status = None
    if best_trial is not None:
        status = getattr(best_trial, "status", None)
        best_trial_status = None if status is None else str(status)

    return {
        "ray_experiment_path": (
            getattr(analysis, "experiment_path", None)
            or getattr(analysis, "_experiment_path", None)
        ),
        "best_trial_id": None if best_trial is None else getattr(best_trial, "trial_id", None),
        "best_trial_status": best_trial_status,
        "best_trial_logdir": getattr(analysis, "best_logdir", None),
        "best_result": best_result or None,
    }


def build_hpo_manifest(
    *,
    source: str,
    preset: str,
    dataset_id: str,
    data_source_fingerprint: str | None,
    tuning,
    best_config_path: Path | None,
    trials_json_path: Path | None = None,
    trials_csv_path: Path | None = None,
    analysis=None,
    argv: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a durable HPO manifest for later provenance joins."""
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "preset": preset,
        "dataset_id": dataset_id,
        "data_source_fingerprint": data_source_fingerprint,
        "objective_metric": getattr(tuning, "metric", None),
        "tuning": (
            tuning.model_dump(mode="json")
            if hasattr(tuning, "model_dump")
            else tuning
        ),
        "argv": argv,
        "git_sha": _git(["git", "rev-parse", "HEAD"]),
        "git_branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git(["git", "status", "--porcelain"])),
        "container_image_tag": os.environ.get("UNIFIED_STPP_IMAGE_TAG"),
        "best_config_path": (
            None if best_config_path is None else str(best_config_path.resolve())
        ),
        "best_config_sha256": (
            None
            if best_config_path is None or not best_config_path.exists()
            else _sha256_file(best_config_path)
        ),
        "trials_json_path": (
            None
            if trials_json_path is None or not trials_json_path.exists()
            else str(trials_json_path.resolve())
        ),
        "trials_csv_path": (
            None
            if trials_csv_path is None or not trials_csv_path.exists()
            else str(trials_csv_path.resolve())
        ),
    }
    manifest.update(extract_analysis_metadata(analysis))
    if extra:
        manifest.update(extra)
    return manifest


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_trial_history(analysis, *, json_path: Path, csv_path: Path | None = None) -> None:
    """Persist Ray Tune trial history in JSON and optional CSV form."""
    records = extract_trial_history(analysis)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    if csv_path is None:
        return
    metric_keys = sorted(
        {
            key
            for record in records
            for key in (record.get("last_result", {}) or {}).keys()
        }
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial_id", "status", *metric_keys],
        )
        writer.writeheader()
        for record in records:
            row = {
                "trial_id": record.get("trial_id"),
                "status": record.get("status"),
            }
            row.update(record.get("last_result", {}) or {})
            writer.writerow(row)


def run_hpo(
    config_dict: dict,
    tuning: "TuningConfig",
    *,
    train_seqs: "list[dict] | None" = None,
    val_seqs: "list[dict] | None" = None,
    train_path: "str | os.PathLike[str] | None" = None,
    val_path: "str | os.PathLike[str] | None" = None,
    dataset_id: str = "hpo_trial",
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
        best_config = STPPConfig(**config_dict)
        if return_analysis:
            return best_config, None
        return best_config  # nothing to tune

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
            dataset_id=dataset_id,
            data_module=dm,
        )
        tune.report({
            "val_objective": result.val_objective,
            "val_metric_key": result.val_metric_key,
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
        seed=tuning.seed,
    )

    best_flat = dict(fixed)
    best_flat.update(analysis.best_config)
    best_nested = _unflatten(best_flat)

    from unified_stpp.config import STPPConfig

    best_config = STPPConfig(**best_nested)
    if return_analysis:
        return best_config, analysis
    return best_config
