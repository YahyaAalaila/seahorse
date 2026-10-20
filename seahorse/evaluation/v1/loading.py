"""Checkpoint loading shim for paper-era (``unified_stpp``) runs.

The paper-era checkpoints were written by the ``unified_stpp`` package. The
current ``seahorse`` package has the same module layout and parameter names,
but some event models gained *derived, non-learned* buffers after the paper
(e.g. the NLL fix ``29458c6`` added ``spatial_cell_area`` and
``support_t0_value`` to NSMPP). Those buffers are recomputed deterministically
in ``__init__`` from the run's resolved config, so they are safe to take from
the freshly built model instead of the checkpoint.

``load_runner_compat`` therefore loads with ``strict=False`` and then *fails
loudly* unless every missing key is on the explicit whitelist below and no key
is unexpected. No model code is modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Buffers that are pure functions of the resolved config (recomputed in
# ``__init__``) and were added to the module after the paper-era checkpoints
# were written. Anything else missing is a real incompatibility.
DERIVED_BUFFER_WHITELIST: frozenset[str] = frozenset(
    {
        "event_model.spatial_cell_area",
        "event_model.support_t0_value",
    }
)


class CheckpointCompatError(RuntimeError):
    """Raised when a checkpoint cannot be loaded without changing learned state."""


def _check_keys(missing: Iterable[str], unexpected: Iterable[str]) -> list[str]:
    missing = sorted(missing)
    unexpected = sorted(unexpected)
    bad_missing = [k for k in missing if k not in DERIVED_BUFFER_WHITELIST]
    if bad_missing or unexpected:
        raise CheckpointCompatError(
            "checkpoint/model mismatch beyond derived buffers: "
            f"missing={bad_missing} unexpected={unexpected}"
        )
    return missing


def load_runner_compat(run_dir: str | Path, *, selection: str | None = None, train_path: str | None = None):
    """Restore an ``STPPRunner`` from a paper-era run directory.

    Returns ``(runner, info)`` where ``info`` records which derived buffers were
    taken from the freshly built model (empty for a fully strict load).
    """
    from seahorse.config import STPPConfig
    from seahorse.runner.artifacts import load_norm_stats, load_state_dict
    from seahorse.runner.runner import STPPRunner
    from seahorse.training.lightning_module import STPPLightningModule

    p = Path(run_dir).resolve()
    cfg_path = p / "resolved_config.yaml"
    if not cfg_path.exists():
        cfg_path = p / "config.yaml"
    config = STPPConfig.from_yaml(cfg_path, sanitize=False)
    transform_source = "resolved_config"
    if cfg_path.name != "resolved_config.yaml":
        transform_source = _refit_transform_if_needed(config, train_path)
    runner = STPPRunner(config)
    model = config.model.build_model()
    sel = selection or config.training.checkpoint_select
    state = load_state_dict(p, selection=sel)
    result = model.load_state_dict(state, strict=False)
    rebuilt = _check_keys(result.missing_keys, result.unexpected_keys)
    runner._lightning_module = STPPLightningModule(model=model, tc=config.training)
    runner._run_dir = p
    ns = load_norm_stats(p)
    norm_stats_source = "run_result.json"
    if ns is None:
        # Checkpoint-only run (killed before run_result.json was written). The
        # runner-level normalisation is fully determined by the config when
        # data.normalize is False (identity); otherwise refuse.
        if bool(getattr(config.data, "normalize", False)):
            raise CheckpointCompatError(
                "no run_result.json (norm_stats) and data.normalize=True: cannot recover normalisation"
            )
        ns = {"normalize": False, "coordinate_space": "raw", "time_mean": 0.0, "time_std": 1.0,
              "loc_mean": [0.0, 0.0], "loc_std": [1.0, 1.0]}
        norm_stats_source = "identity_from_config(data.normalize=False)"
    runner._norm_stats = ns
    model.eval()
    info = {
        "run_dir": str(p),
        "config_path": str(cfg_path),
        "checkpoint_selection": sel,
        "derived_buffers_rebuilt": rebuilt,
        "strict_load": not rebuilt,
        "norm_stats_source": norm_stats_source,
        "input_transform_source": transform_source,
    }
    return runner, info


def _refit_transform_if_needed(config, train_path):
    """Reconstruct the data-fitted model overrides of a run killed before it
    wrote ``resolved_config.yaml``.

    ``config.yaml`` holds the *requested* config. At fit time the runner adds
    ``ConfigRegistry.data_init_overrides(preset, dm)`` (data-dependent init plus
    the fitted coordinate transform, e.g. the Neural-STPP spatial z-score)
    computed from the training split. We call that same function on the same
    training sequences, which reproduces the overrides deterministically
    (verified against completed runs, RUNBOOK §4). Refuses when the training
    data is unknown.
    """
    from types import SimpleNamespace

    from seahorse.models.configs import ConfigRegistry
    from seahorse.utils import deep_update, load_jsonl

    m = config.model
    extra = m.model_extra if m.model_extra is not None else {}
    if extra.get("input_transform"):
        return "config.yaml"
    if bool(getattr(config.data, "normalize", False)) or getattr(config.data, "protocol", "raw") != "raw":
        raise CheckpointCompatError("cannot refit data overrides for non-raw/normalized protocol")
    path = train_path or getattr(config.data, "train_path", None)
    if not path:
        raise CheckpointCompatError("no resolved_config.yaml and no training data to refit data overrides")
    seqs = load_jsonl(str(path))
    dm = SimpleNamespace(train_dataset=SimpleNamespace(sequences=seqs, coordinate_space="raw"))
    overrides = ConfigRegistry.data_init_overrides(m.preset, dm)
    if not overrides:
        return "none_needed"
    deep_update(extra, overrides)
    return f"refit_from_train:{path}"
