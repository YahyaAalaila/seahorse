"""
Artifact utilities for STPPRunner fit() runs and benchmark runs.

Responsible for:
  - Generating unique run IDs
  - Managing the ``latest`` symlink
  - Writing per-run artifact files (resolved_config.yaml, run_result.json, artifacts.json)
  - Generating unique benchmark output directory names
  - Writing benchmark-level metadata (bench_meta.json)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

if TYPE_CHECKING:
    from unified_stpp.config.benchmark import BenchmarkConfig
    from unified_stpp.config import STPPConfig
    from unified_stpp.runner.results import RunResult


def _yaml_safe_obj(value):
    """Convert run artifact payloads to YAML-safe JSON-like values."""
    return json.loads(json.dumps(value, default=str))


def make_run_id() -> str:
    """Return a unique run ID: ``YYYYMMDD_HHMMSS_{8-char-git-hash}``."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        h = hashlib.md5(os.urandom(8)).hexdigest()[:8]
    return f"{ts}_{h}"


def make_bench_run_id() -> str:
    """Return a unique benchmark run ID: ``bench_YYYYMMDD_HHMMSS_{8-char-git-hash}``."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        h = hashlib.md5(os.urandom(8)).hexdigest()[:8]
    return f"bench_{ts}_{h}"


def write_bench_meta(
    out_dir: Path,
    bench_id: str,
    argv: list[str],
    splits_dir: str,
    datasets: list[str],
    presets: list[str],
    benchmark_config: "BenchmarkConfig",
    overrides: list[str],
    hpo_configs_dir: Optional[str],
    data_manifest: Optional[dict] = None,
    hpo_provenance: Optional[dict] = None,
    cell_index_path: Optional[str] = None,
) -> None:
    """Write ``bench_meta.json`` to *out_dir* before any training begins.

    Captures full experiment identity: CLI invocation, git state, dataset/preset/seed
    configuration, and environment info for cluster-grade reproducibility.
    """
    import platform
    import socket
    import sys

    import torch

    def _git(cmd: list[str]) -> Optional[str]:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    sha_full  = _git(["git", "rev-parse", "HEAD"])
    sha_short = sha_full[:8] if sha_full else bench_id.split("_")[-1]
    branch    = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty     = bool(_git(["git", "status", "--porcelain"]))

    meta = {
        "bench_id":        bench_id,
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "argv":            argv,
        "git_sha":         sha_full,
        "git_sha_short":   sha_short,
        "git_branch":      branch,
        "git_dirty":       dirty,
        "splits_dir":      splits_dir,
        "datasets":        datasets,
        "presets":         presets,
        "seeds":           benchmark_config.seeds,
        "normalize":       benchmark_config.normalize,
        "n_workers":       benchmark_config.n_workers,
        "backend":         benchmark_config.backend,
        "run_hpo":         benchmark_config.run_hpo,
        "tune_dataset":    benchmark_config.tune_dataset,
        "primary_metric":  benchmark_config.primary_metric,
        "benchmark_config": benchmark_config.model_dump(mode="json"),
        "benchmark_policy": {
            "protocol": benchmark_config.protocol,
            "normalize": benchmark_config.normalize,
            "checkpoint_select": benchmark_config.checkpoint_select,
            "test_nll_space": benchmark_config.test_nll_space,
            "predictive_test_nll_samples": benchmark_config.predictive_test_nll_samples,
            "test_nll_definition": "held_out_next_event_over_teacher_forced_test_prefixes",
            "allow_mixed_hpo_provenance": benchmark_config.allow_mixed_hpo_provenance,
        },
        "overrides":       overrides,
        "hpo_configs_dir": hpo_configs_dir,
        "data_manifest": data_manifest,
        "hpo_provenance": hpo_provenance,
        "cell_index_path": cell_index_path,
        "python_version":  sys.version.split()[0],
        "torch_version":   torch.__version__,
        "platform":        f"{platform.system().lower()}-{platform.machine()}",
        "hostname":        socket.gethostname(),
        "out_dir":         str(out_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bench_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    rerun_path = out_dir / "rerun.sh"
    with open(rerun_path, "w") as f:
        f.write("#!/bin/bash\n" + sys.executable + " -m unified_stpp " + " ".join(argv) + "\n")
    rerun_path.chmod(0o755)


def update_latest_symlink(run_dir: Path) -> None:
    """Point ``{run_dir.parent}/latest`` at *run_dir* (relative symlink).

    Silently skips on Windows or if permission is denied.
    """
    latest = run_dir.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except (OSError, NotImplementedError):
        pass


def save_run_artifacts(run_dir: Path, result: "RunResult", cfg: "STPPConfig") -> None:
    """Write post-training artifact files into *run_dir*.

    Files written:
    - ``resolved_config.yaml`` — full config after Pydantic validation
    - ``run_result.json``      — all metrics, norm_stats, effective_config
    - ``artifacts.json``       — manifest mapping artifact roles to relative paths
    """
    resolved = result.effective_config if result.effective_config else cfg.model_dump(mode="json")
    resolved = _yaml_safe_obj(resolved)
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(resolved, f, default_flow_style=False, sort_keys=False)

    result.to_json(run_dir / "run_result.json")

    manifest = {
        "run_id": run_dir.name,
        "preset": cfg.model.preset,
        "config": "config.yaml",
        "resolved_config": "resolved_config.yaml",
        "run_result": "run_result.json",
        "metrics": "metrics.csv",
        "checkpoint_best": "checkpoints/best.ckpt",
        "checkpoint_last": "checkpoints/last.ckpt",
    }
    with open(run_dir / "artifacts.json", "w") as f:
        json.dump(manifest, f, indent=2)


def checkpoint_file(run_dir: Path, selection: str = "best") -> Path:
    """Return the preferred Lightning checkpoint file under *run_dir*."""
    choice = str(selection).strip().lower()
    if choice not in {"best", "last"}:
        raise ValueError(
            f"Unknown checkpoint selection {selection!r}. Expected 'best' or 'last'."
        )
    return run_dir / "checkpoints" / f"{choice}.ckpt"


def load_state_dict(run_dir: Path, selection: str = "best") -> dict:
    """Load model weights from *run_dir*, supporting two formats.

    Preference order:
    1. ``checkpoints/{selection}.ckpt`` — Lightning checkpoint written by ``fit()``.
       State-dict keys are prefixed with ``"model."``; the prefix is stripped.
    2. ``model.ckpt`` — plain ``state_dict`` written by ``runner.save()``.

    Raises ``FileNotFoundError`` if neither file exists.
    """
    import torch

    pl_ckpt = checkpoint_file(run_dir, selection=selection)
    if pl_ckpt.exists():
        ckpt = torch.load(pl_ckpt, map_location="cpu", weights_only=False)
        pl_state = ckpt["state_dict"]
        return {
            k[len("model."):]: v
            for k, v in pl_state.items()
            if k.startswith("model.")
        }

    plain_ckpt = run_dir / "model.ckpt"
    if plain_ckpt.exists():
        return torch.load(plain_ckpt, map_location="cpu", weights_only=False)

    raise FileNotFoundError(
        f"No checkpoint found in {run_dir}. "
        f"Expected 'checkpoints/{selection}.ckpt' or 'model.ckpt'."
    )


def load_norm_stats(run_dir: Path) -> "Optional[dict]":
    """Load ``norm_stats`` from ``run_result.json`` in *run_dir*, or ``None``."""
    result_json = run_dir / "run_result.json"
    if result_json.exists():
        with open(result_json) as f:
            return json.load(f).get("norm_stats")
    return None


def _extend_viz_manifest(run_dir: Path, viz_artifacts: dict) -> None:
    """Append surface visualization entries to the existing artifacts.json.

    Parameters
    ----------
    run_dir       : run directory that already contains ``artifacts.json``
    viz_artifacts : ``{artifact_name: Path}`` dict returned by the workflow
    """
    manifest_path = run_dir / "artifacts.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}

    for name, path in viz_artifacts.items():
        manifest[name] = str(Path(path).relative_to(run_dir))

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
