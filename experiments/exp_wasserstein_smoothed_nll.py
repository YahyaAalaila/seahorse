#!/usr/bin/env python3
"""
Optional STPP experiment: baseline NLL vs. smoothed NLL vs. OT distance.

Why this exists:
- Baseline event NLL can be overly sensitive to tiny spatial/temporal shifts.
- Smoothed NLL replaces only the point log-intensity term with a local mass
  integral over a radius/time window, while keeping the baseline remainder
  term (including compensator/other decoder terms) unchanged.
- Wasserstein/OT gives a distribution-level distance between observed events
  and simulated events from the fitted model.

This file is intentionally standalone and optional. It does not change core
training/model code and may be used as a local experiment script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

# Keep matplotlib-dependent synthetic imports non-interactive and writable.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from unified_stpp.data import collate_fn
from unified_stpp.data.regime_gated_hawkes import generate_regime_gated_hawkes_stpp
from unified_stpp.data.synthetic import (
    InhomogeneousPoissonSyntheticDataset,
    STHPDataset,
    generate_hawkes_stpp,
    generate_inhomogeneous_stpp,
    generate_marked_hawkes_stpp,
    generate_moving_hotspot_stpp,
    generate_pinwheel_hawkes_stpp,
)
from unified_stpp.models import IntensityEvaluator, thinning_sample
from unified_stpp.models.dynamics.identity import IdentityDynamics
from unified_stpp.registry import build_model
from unified_stpp.training import Trainer
from unified_stpp.training.data_module import STPPDataModule

try:
    import ot as pot_ot  # POT (optional)
except Exception:  # pragma: no cover - optional dependency
    pot_ot = None


EPS = 1e-12
KT_DEFAULT = 11
KS_DEFAULT = 64
KNOWN_PRESETS = ("neural_stpp", "deep_stpp", "dstpp")


@dataclass
class ModelSpec:
    preset: str
    hidden_dim: int
    spatial_dim: int
    event_cov_dim: int
    field_cov_dim: int
    n_marks: int
    overrides: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional STPP experiment: baseline NLL, smoothed NLL, and OT distance."
    )
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r", type=float, default=500.0)
    parser.add_argument("--tau", type=float, default=300.0)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--n_sims", type=int, default=20)
    parser.add_argument(
        "--ot_mode",
        type=str,
        choices=["hungarian", "sinkhorn"],
        default="hungarian",
    )
    parser.add_argument(
        "--ot_axes",
        type=str,
        choices=["joint", "space_only", "time_only"],
        default="joint",
    )
    parser.add_argument("--sinkhorn_reg", type=float, default=0.05)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--r_grid", type=str, default="linspace:0.1:2.0:10")
    parser.add_argument("--tau_grid", type=str, default="linspace:0.05:1.0:10")
    parser.add_argument("--sweep_only", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/exp_wasserstein_smoothed_nll",
    )
    return parser.parse_args()


def _parse_grid_spec(spec: str, name: str) -> np.ndarray:
    spec = str(spec).strip()
    if not spec:
        raise ValueError(f"--{name} must not be empty.")
    if spec.startswith("linspace:"):
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"--{name} linspace format must be 'linspace:start:end:n'. Got: {spec}"
            )
        start = float(parts[1])
        end = float(parts[2])
        n = int(parts[3])
        if n <= 0:
            raise ValueError(f"--{name} linspace n must be > 0.")
        vals = np.linspace(start, end, n, dtype=np.float64)
    else:
        try:
            vals = np.asarray(
                [float(x.strip()) for x in spec.split(",") if x.strip() != ""],
                dtype=np.float64,
            )
        except Exception as exc:
            raise ValueError(f"Could not parse --{name}: {spec}") from exc
    if vals.size == 0:
        raise ValueError(f"--{name} produced empty grid.")
    if np.any(~np.isfinite(vals)):
        raise ValueError(f"--{name} has non-finite values.")
    if np.any(vals <= 0.0):
        raise ValueError(f"--{name} values must be > 0.")
    vals = np.unique(vals)
    vals.sort()
    return vals


def _finite_mean_std(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr[mask])),
        "std": float(np.std(arr[mask])),
        "n": int(np.sum(mask)),
    }


def _json_metric_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    mean = d.get("mean", None)
    std = d.get("std", None)
    n = d.get("n", 0)
    return {
        "mean": (float(mean) if mean is not None and np.isfinite(mean) else None),
        "std": (float(std) if std is not None and np.isfinite(std) else None),
        "n": int(n) if n is not None else 0,
    }


def _git_commit_short() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def _parse_run_id(run_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse run_id pattern:
      <data_type>_<preset>_<YYYYMMDD>_<HHMMSS>_<hash>
    where data_type and preset may contain underscores.
    """
    m = re.match(r"^(?P<body>.+)_\d{8}_\d{6}_[0-9a-fA-F]{6,12}$", run_id)
    if not m:
        return None, None
    body = m.group("body")
    for preset in sorted(KNOWN_PRESETS, key=len, reverse=True):
        suffix = "_" + preset
        if body.endswith(suffix):
            data_type = body[: -len(suffix)]
            return (preset, data_type if data_type else None)
    return None, None


def _find_logs_dir(run_id: str) -> Optional[Path]:
    base = Path("logs") / run_id
    if not base.exists():
        return None
    versions = sorted(base.glob("version_*"))
    if versions:
        return versions[-1]
    return base


def _parse_ckpt_val(path: Path) -> Optional[float]:
    m = re.search(r"val_nll([0-9]+(?:\.[0-9]+)?)\.ckpt$", path.name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _select_checkpoint(run_id: str) -> Path:
    ckpt_dir = Path("checkpoints") / run_id
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under: {ckpt_dir}")

    scored: List[Tuple[float, Path]] = []
    for ck in ckpts:
        val = _parse_ckpt_val(ck)
        if val is not None:
            scored.append((val, ck))
    if scored:
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _candidate_config_paths(preset: Optional[str]) -> List[Path]:
    if not preset:
        return []
    names = [f"{preset}_lightning.yaml", f"{preset}.yaml"]
    out: List[Path] = []
    for n in names:
        out.append(Path("configs") / n)
        out.append(Path("unified_stpp") / "configs" / n)
    return out


def _default_settings() -> Dict[str, Any]:
    # Mirrors train.py defaults as closely as possible.
    return {
        "preset": "deep_stpp",
        "hidden_dim": 64,
        "spatial_dim": 2,
        "event_cov_dim": 0,
        "field_cov_dim": 0,
        "n_marks": 0,
        "data_type": "hawkes",
        "n_train": 200,
        "n_val": 50,
        "t_end": 5.0,
        "base_rate": 2.0,
        "data_seed": 42,
        "dataset_cache": None,
        "data_covariate_dim": None,
        "spatial_min": -5.0,
        "spatial_max": 5.0,
        "sthp_alpha": 0.5,
        "sthp_beta": 1.0,
        "sthp_mu": 1.0,
        "hotspot_sigma": 0.9,
        "hotspot_switch_frac": 0.5,
        "hotspot_switch_time": None,
        "hotspot_move_duration": None,
        "hotspot_t1_frac": 0.32,
        "hotspot_t2_frac": 0.46,
        "hotspot_jitter_radius": 0.55,
        "hotspot_jitter_f1": 0.8,
        "hotspot_jitter_f2": 1.25,
        "hotspot_amp0": 0.0,
        "hotspot_amp1": 0.55,
        "hotspot_amp_noise": 0.10,
        "hotspot_noise_knots": 16,
        "hotspot_weight": 3.0,
        "hotspot_interaction_weight": 1.2,
        "hotspot_tod_weight": 0.6,
        "rg_num_marks": 3,
        "rg_num_regimes": 3,
        "rg_env_dim": 4,
        "rg_sigma_reg": 0.15,
        "rg_switch_rate": 1.2,
        "rg_sigma_hotspot": 1.0,
        "rg_lambda_bg": 0.02,
        "rg_tau_t": 0.8,
        "rg_sigma_exc": 0.9,
        "rg_max_events_per_seq": 250,
        "pw_num_arms": 10,
        "pw_mu": 0.05,
        "pw_alpha": 0.6,
        "pw_omega": 10.0,
        "batch_size": 32,
        "config_path_used": None,
    }


def _apply_config(settings: Dict[str, Any], cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    if not isinstance(data_cfg, dict):
        data_cfg = {}

    settings["preset"] = model_cfg.get("preset", settings["preset"])
    settings["hidden_dim"] = int(model_cfg.get("hidden_dim", settings["hidden_dim"]))
    settings["spatial_dim"] = int(model_cfg.get("spatial_dim", settings["spatial_dim"]))
    settings["event_cov_dim"] = int(model_cfg.get("event_cov_dim", settings["event_cov_dim"]))
    settings["field_cov_dim"] = int(model_cfg.get("field_cov_dim", settings["field_cov_dim"]))
    settings["n_marks"] = int(model_cfg.get("n_marks", settings["n_marks"]))

    settings["batch_size"] = int(train_cfg.get("batch_size", settings["batch_size"]))

    settings["data_type"] = data_cfg.get("type", settings["data_type"])
    settings["n_train"] = int(data_cfg.get("n_train", settings["n_train"]))
    settings["n_val"] = int(data_cfg.get("n_val", settings["n_val"]))
    settings["t_end"] = float(
        data_cfg.get(
            "T",
            data_cfg.get("t_end", settings["t_end"]),
        )
    )
    settings["base_rate"] = float(data_cfg.get("base_rate", settings["base_rate"]))
    settings["data_seed"] = int(data_cfg.get("seed", settings["data_seed"]))
    settings["dataset_cache"] = data_cfg.get("dataset_cache", settings["dataset_cache"])
    settings["data_covariate_dim"] = data_cfg.get("covariate_dim", settings["data_covariate_dim"])
    settings["spatial_min"] = float(data_cfg.get("spatial_min", settings["spatial_min"]))
    settings["spatial_max"] = float(data_cfg.get("spatial_max", settings["spatial_max"]))

    settings["sthp_alpha"] = float(data_cfg.get("sthp_alpha", settings["sthp_alpha"]))
    settings["sthp_beta"] = float(data_cfg.get("sthp_beta", settings["sthp_beta"]))
    settings["sthp_mu"] = float(data_cfg.get("sthp_mu", settings["sthp_mu"]))

    settings["hotspot_sigma"] = float(data_cfg.get("hotspot_sigma", settings["hotspot_sigma"]))
    settings["hotspot_switch_frac"] = float(
        data_cfg.get("hotspot_switch_frac", settings["hotspot_switch_frac"])
    )
    settings["hotspot_switch_time"] = data_cfg.get(
        "hotspot_switch_time", settings["hotspot_switch_time"]
    )
    settings["hotspot_move_duration"] = data_cfg.get(
        "hotspot_move_duration", settings["hotspot_move_duration"]
    )
    settings["hotspot_t1_frac"] = float(data_cfg.get("hotspot_t1_frac", settings["hotspot_t1_frac"]))
    settings["hotspot_t2_frac"] = float(data_cfg.get("hotspot_t2_frac", settings["hotspot_t2_frac"]))
    settings["hotspot_jitter_radius"] = float(
        data_cfg.get("hotspot_jitter_radius", settings["hotspot_jitter_radius"])
    )
    settings["hotspot_jitter_f1"] = float(
        data_cfg.get("hotspot_jitter_f1", settings["hotspot_jitter_f1"])
    )
    settings["hotspot_jitter_f2"] = float(
        data_cfg.get("hotspot_jitter_f2", settings["hotspot_jitter_f2"])
    )
    settings["hotspot_amp0"] = float(data_cfg.get("hotspot_amp0", settings["hotspot_amp0"]))
    settings["hotspot_amp1"] = float(data_cfg.get("hotspot_amp1", settings["hotspot_amp1"]))
    settings["hotspot_amp_noise"] = float(
        data_cfg.get("hotspot_amp_noise", settings["hotspot_amp_noise"])
    )
    settings["hotspot_noise_knots"] = int(
        data_cfg.get("hotspot_noise_knots", settings["hotspot_noise_knots"])
    )
    settings["hotspot_weight"] = float(data_cfg.get("hotspot_weight", settings["hotspot_weight"]))
    settings["hotspot_interaction_weight"] = float(
        data_cfg.get("hotspot_interaction_weight", settings["hotspot_interaction_weight"])
    )
    settings["hotspot_tod_weight"] = float(
        data_cfg.get("hotspot_tod_weight", settings["hotspot_tod_weight"])
    )

    settings["rg_num_marks"] = int(data_cfg.get("rg_num_marks", settings["rg_num_marks"]))
    settings["rg_num_regimes"] = int(data_cfg.get("rg_num_regimes", settings["rg_num_regimes"]))
    settings["rg_env_dim"] = int(data_cfg.get("rg_env_dim", settings["rg_env_dim"]))
    settings["rg_sigma_reg"] = float(data_cfg.get("rg_sigma_reg", settings["rg_sigma_reg"]))
    settings["rg_switch_rate"] = float(data_cfg.get("rg_switch_rate", settings["rg_switch_rate"]))
    settings["rg_sigma_hotspot"] = float(
        data_cfg.get("rg_sigma_hotspot", settings["rg_sigma_hotspot"])
    )
    settings["rg_lambda_bg"] = float(data_cfg.get("rg_lambda_bg", settings["rg_lambda_bg"]))
    settings["rg_tau_t"] = float(data_cfg.get("rg_tau_t", settings["rg_tau_t"]))
    settings["rg_sigma_exc"] = float(data_cfg.get("rg_sigma_exc", settings["rg_sigma_exc"]))
    settings["rg_max_events_per_seq"] = int(
        data_cfg.get("rg_max_events_per_seq", settings["rg_max_events_per_seq"])
    )

    settings["pw_num_arms"] = int(data_cfg.get("pw_num_arms", settings["pw_num_arms"]))
    settings["pw_mu"] = float(data_cfg.get("pw_mu", settings["pw_mu"]))
    settings["pw_alpha"] = float(data_cfg.get("pw_alpha", settings["pw_alpha"]))
    settings["pw_omega"] = float(data_cfg.get("pw_omega", settings["pw_omega"]))

    settings["config_path_used"] = str(cfg_path)
    return settings


def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            out[k[6:]] = v
        else:
            out[k] = v
    return out


def _infer_preset_from_state_keys(keys: Sequence[str]) -> str:
    keyset = set(keys)
    if any(k.startswith("decoder.temporal.param_net.") for k in keyset):
        return "deep_stpp"
    if any(k.startswith("decoder.temporal.intensity_net.") for k in keyset):
        return "neural_stpp"
    if any("decoder.score_net" in k for k in keyset):
        return "dstpp"
    return "deep_stpp"


def _infer_model_spec(
    run_preset: Optional[str],
    state_dict_stripped: Dict[str, torch.Tensor],
) -> ModelSpec:
    keys = list(state_dict_stripped.keys())
    preset = run_preset or _infer_preset_from_state_keys(keys)

    if "encoder.embed.weight" not in state_dict_stripped:
        raise KeyError("Checkpoint missing encoder.embed.weight; cannot infer model shape.")

    enc_w = state_dict_stripped["encoder.embed.weight"]
    hidden_dim = int(enc_w.shape[0])
    encoder_input_dim = int(enc_w.shape[1])

    # field_cov_dim inference
    field_cov_dim = 0
    if "decoder.temporal.param_net.0.weight" in state_dict_stripped:
        field_cov_dim = int(state_dict_stripped["decoder.temporal.param_net.0.weight"].shape[1] - hidden_dim)
    elif "decoder.temporal.intensity_net.0.weight" in state_dict_stripped:
        field_cov_dim = int(
            state_dict_stripped["decoder.temporal.intensity_net.0.weight"].shape[1]
            - hidden_dim
            - 1
        )
    field_cov_dim = max(field_cov_dim, 0)

    # spatial_dim inference
    spatial_dim = 2
    temporal_components = None
    spatial_components = None

    if (
        "decoder.temporal.param_net.2.weight" in state_dict_stripped
        and "decoder.spatial.param_net.2.weight" in state_dict_stripped
    ):
        out_t = int(state_dict_stripped["decoder.temporal.param_net.2.weight"].shape[0])
        out_s = int(state_dict_stripped["decoder.spatial.param_net.2.weight"].shape[0])
        if out_t % 3 == 0:
            temporal_components = out_t // 3
            if temporal_components > 0 and out_s % temporal_components == 0:
                candidate = out_s // temporal_components - 2
                if candidate > 0:
                    spatial_dim = candidate
                    spatial_components = out_s // (spatial_dim + 2)
    elif "decoder.spatial.velocity.net.0.weight" in state_dict_stripped:
        inp = int(state_dict_stripped["decoder.spatial.velocity.net.0.weight"].shape[1])
        candidate = inp - 1 - hidden_dim - field_cov_dim
        if candidate > 0:
            spatial_dim = candidate

    # marks + event cov
    n_marks = 0
    mark_embed_dim = 0
    if "mark_embedding.embedding.weight" in state_dict_stripped:
        emb = state_dict_stripped["mark_embedding.embedding.weight"]
        n_marks = int(emb.shape[0])
        mark_embed_dim = int(emb.shape[1])

    effective_event_cov = encoder_input_dim - (1 + spatial_dim)
    if effective_event_cov < 0:
        effective_event_cov = 0
    event_cov_dim = max(0, effective_event_cov - mark_embed_dim)

    overrides: Dict[str, Any] = {}
    if temporal_components is not None or spatial_components is not None:
        overrides["decoder"] = {"temporal": {}, "spatial": {}}
        if temporal_components is not None:
            overrides["decoder"]["temporal"]["n_components"] = int(temporal_components)
        if spatial_components is not None:
            overrides["decoder"]["spatial"]["n_components"] = int(spatial_components)

    return ModelSpec(
        preset=preset,
        hidden_dim=hidden_dim,
        spatial_dim=spatial_dim,
        event_cov_dim=event_cov_dim,
        field_cov_dim=field_cov_dim,
        n_marks=n_marks,
        overrides=overrides,
    )


def _resolve_settings(
    run_id: str,
    model_spec: ModelSpec,
    run_data_type: Optional[str],
) -> Dict[str, Any]:
    settings = _default_settings()
    settings["preset"] = model_spec.preset
    if run_data_type:
        settings["data_type"] = run_data_type

    # Apply first matching config file for this preset.
    for cfg_path in _candidate_config_paths(model_spec.preset):
        if cfg_path.exists():
            cfg = _load_yaml(cfg_path)
            settings = _apply_config(settings, cfg, cfg_path)
            break

    # Final authority: model inferred from checkpoint
    settings["preset"] = model_spec.preset
    settings["hidden_dim"] = model_spec.hidden_dim
    settings["spatial_dim"] = model_spec.spatial_dim
    settings["event_cov_dim"] = model_spec.event_cov_dim
    settings["field_cov_dim"] = model_spec.field_cov_dim
    settings["n_marks"] = model_spec.n_marks
    if run_data_type:
        settings["data_type"] = run_data_type

    # If run_id includes data type and it differs from config, prefer run_id.
    parsed_preset, parsed_data = _parse_run_id(run_id)
    if parsed_data:
        settings["data_type"] = parsed_data
    if parsed_preset:
        settings["preset"] = parsed_preset

    return settings


def _build_model_from_checkpoint(
    ckpt_path: Path,
    model_spec: ModelSpec,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {ckpt_path}")

    stripped = _strip_state_dict_prefix(state_dict)
    model = build_model(
        config=model_spec.overrides.copy(),
        spatial_dim=model_spec.spatial_dim,
        hidden_dim=model_spec.hidden_dim,
        event_cov_dim=model_spec.event_cov_dim,
        field_cov_dim=model_spec.field_cov_dim,
        preset=model_spec.preset,
        n_marks=model_spec.n_marks,
    )
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}\n"
            f"Unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )
    model.eval()
    return model, ckpt


def _generate_sequences(settings: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data_type = settings["data_type"]
    n_train = int(settings["n_train"])
    n_val = int(settings["n_val"])
    total = n_train + n_val
    t_end = float(settings["t_end"])
    spatial_dim = int(settings["spatial_dim"])
    spatial_bounds = (float(settings["spatial_min"]), float(settings["spatial_max"]))
    data_seed = int(settings["data_seed"])
    field_cov_dim = int(settings["field_cov_dim"])
    data_cov_dim = settings["data_covariate_dim"]
    dataset_cache = settings.get("dataset_cache")

    default_data_cov_dim = max(1, field_cov_dim)
    if data_type == "moving_hotspot":
        default_data_cov_dim = 4
    elif data_type == "regime_gated_hawkes":
        default_data_cov_dim = int(settings["rg_num_regimes"]) + 2 + int(settings["rg_env_dim"])
    cov_dim = int(default_data_cov_dim if data_cov_dim is None else max(1, int(data_cov_dim)))

    all_seqs: Optional[List[Dict[str, Any]]] = None
    if dataset_cache and Path(dataset_cache).exists():
        with open(dataset_cache, "rb") as f:
            all_seqs = pickle.load(f)
        if len(all_seqs) < total:
            raise ValueError(
                f"Cached dataset has {len(all_seqs)} sequences but {total} are required."
            )

    if all_seqs is None:
        if data_type == "hawkes":
            all_seqs = generate_hawkes_stpp(
                n_sequences=total,
                T=t_end,
                spatial_bounds=spatial_bounds,
                spatial_dim=spatial_dim,
                seed=data_seed,
            )
        elif data_type == "inhomogeneous":
            all_seqs = generate_inhomogeneous_stpp(
                n_sequences=total,
                T=t_end,
                spatial_bounds=spatial_bounds,
                spatial_dim=spatial_dim,
                base_rate=float(settings["base_rate"]),
                covariate_dim=cov_dim,
                seed=data_seed,
            )
        elif data_type == "inhomogeneous_class":
            gen = InhomogeneousPoissonSyntheticDataset(
                spatial_dim=spatial_dim,
                base_rate=float(settings["base_rate"]),
                covariate_dim=cov_dim,
                seed=data_seed,
            )
            all_seqs = gen.generate_sequences(
                n_sequences=total,
                t_start=0.0,
                t_end=t_end,
            )
        elif data_type == "sthp_class":
            if spatial_dim != 2:
                raise ValueError("sthp_class currently supports only spatial_dim=2")
            gen = STHPDataset(
                s_mu=[0.0, 0.0],
                g0_cov=[[1.0, 0.0], [0.0, 1.0]],
                g2_cov=[[1.0, 0.0], [0.0, 1.0]],
                alpha=float(settings["sthp_alpha"]),
                beta=float(settings["sthp_beta"]),
                mu=float(settings["sthp_mu"]),
                seed=data_seed,
            )
            all_seqs = gen.generate_sequences(
                n_sequences=total,
                t_start=0.0,
                t_end=t_end,
            )
        elif data_type == "moving_hotspot":
            all_seqs = generate_moving_hotspot_stpp(
                n_sequences=total,
                T=t_end,
                spatial_bounds=spatial_bounds,
                spatial_dim=spatial_dim,
                base_rate=float(settings["base_rate"]),
                sigma=float(settings["hotspot_sigma"]),
                switch_frac=float(settings["hotspot_switch_frac"]),
                switch_time=settings["hotspot_switch_time"],
                move_duration=settings["hotspot_move_duration"],
                t1_frac=float(settings["hotspot_t1_frac"]),
                t2_frac=float(settings["hotspot_t2_frac"]),
                jitter_radius=float(settings["hotspot_jitter_radius"]),
                jitter_f1=float(settings["hotspot_jitter_f1"]),
                jitter_f2=float(settings["hotspot_jitter_f2"]),
                amp0=float(settings["hotspot_amp0"]),
                amp1=float(settings["hotspot_amp1"]),
                amp_noise=float(settings["hotspot_amp_noise"]),
                n_noise_knots=int(settings["hotspot_noise_knots"]),
                hotspot_weight=float(settings["hotspot_weight"]),
                interaction_weight=float(settings["hotspot_interaction_weight"]),
                tod_weight=float(settings["hotspot_tod_weight"]),
                covariate_dim=cov_dim,
                seed=data_seed,
            )
        elif data_type == "regime_gated_hawkes":
            all_seqs = generate_regime_gated_hawkes_stpp(
                n_sequences=total,
                T=t_end,
                spatial_bounds=spatial_bounds,
                n_marks=int(settings["rg_num_marks"]),
                n_regimes=int(settings["rg_num_regimes"]),
                env_dim=int(settings["rg_env_dim"]),
                sigma_reg=float(settings["rg_sigma_reg"]),
                switch_rate=float(settings["rg_switch_rate"]),
                sigma_hotspot=float(settings["rg_sigma_hotspot"]),
                lambda_bg=float(settings["rg_lambda_bg"]),
                tau_t=float(settings["rg_tau_t"]),
                sigma_exc=float(settings["rg_sigma_exc"]),
                max_events_per_seq=int(settings["rg_max_events_per_seq"]),
                seed=data_seed,
            )
        elif data_type == "marked_hawkes":
            all_seqs = generate_marked_hawkes_stpp(
                n_sequences=total,
                T=t_end,
                spatial_bounds=spatial_bounds,
                spatial_dim=spatial_dim,
                n_marks=max(int(settings["n_marks"]), 3),
                seed=data_seed,
            )
        elif data_type == "pinwheel":
            all_seqs = generate_pinwheel_hawkes_stpp(
                n_sequences=total,
                T=t_end,
                num_arms=int(settings["pw_num_arms"]),
                mu_per_arm=float(settings["pw_mu"]),
                alpha_offdiag=float(settings["pw_alpha"]),
                omega=float(settings["pw_omega"]),
                seed=data_seed,
            )
        else:
            raise ValueError(f"Unsupported data type in experiment script: {data_type}")

    if int(settings["field_cov_dim"]) == 0:
        for seq in all_seqs:
            seq.pop("field_covariates", None)

    train_seqs = all_seqs[:n_train]
    val_seqs = all_seqs[n_train : n_train + n_val]
    return train_seqs, val_seqs


def _build_data(
    train_seqs: List[Dict[str, Any]],
    val_seqs: List[Dict[str, Any]],
    batch_size: int,
) -> Tuple[STPPDataModule, Any, Any]:
    dm = STPPDataModule(
        train_seqs,
        val_seqs,
        batch_size=batch_size,
        num_workers=0,
    )
    dm.setup()
    return dm, dm._train_dataset, dm._val_dataset


def _stats_from_train_dataset(train_dataset: Any) -> Dict[str, np.ndarray]:
    return {
        "time_mean": np.array(float(train_dataset.time_mean), dtype=np.float64),
        "time_std": np.array(float(train_dataset.time_std), dtype=np.float64),
        "loc_mean": np.asarray(train_dataset.loc_mean, dtype=np.float64),
        "loc_std": np.asarray(train_dataset.loc_std, dtype=np.float64),
    }


def _to_norm_time(t_raw: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(t_raw, dtype=np.float64) - stats["time_mean"]) / stats["time_std"]


def _to_norm_loc(s_raw: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(s_raw, dtype=np.float64) - stats["loc_mean"]) / stats["loc_std"]


def _to_raw_time(t_norm: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(t_norm, dtype=np.float64) * stats["time_std"] + stats["time_mean"]


def _to_raw_loc(s_norm: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(s_norm, dtype=np.float64) * stats["loc_std"] + stats["loc_mean"]


def _build_encoder_x_event(
    model: torch.nn.Module,
    times: torch.Tensor,        # (1, N)
    locations: torch.Tensor,    # (1, N, d)
    marks: Optional[torch.Tensor],           # (1, N) or None
    x_event: Optional[torch.Tensor],         # (1, N, p) or None
) -> Optional[torch.Tensor]:
    # Reproduce UnifiedSTPP forward behavior, with zero-fill safety for
    # models that require covariate channels.
    x_event_hist = x_event
    if getattr(model, "mark_embedding", None) is not None:
        bsz, n = times.shape
        if marks is not None:
            x_mark = model.mark_embedding(marks.long())  # (1, N, embed_dim)
        else:
            x_mark = torch.zeros(
                bsz,
                n,
                model.mark_embedding.embed_dim,
                dtype=times.dtype,
                device=times.device,
            )
        if x_event_hist is None:
            x_event_hist = x_mark
        else:
            x_event_hist = torch.cat([x_event_hist, x_mark], dim=-1)

    expected_total = int(model.encoder.embed.in_features - (1 + locations.shape[-1]))
    if expected_total <= 0:
        return None
    if x_event_hist is None:
        return torch.zeros(
            times.shape[0],
            times.shape[1],
            expected_total,
            dtype=times.dtype,
            device=times.device,
        )
    current = int(x_event_hist.shape[-1])
    if current == expected_total:
        return x_event_hist
    if current > expected_total:
        return x_event_hist[..., :expected_total]
    pad = torch.zeros(
        times.shape[0],
        times.shape[1],
        expected_total - current,
        dtype=x_event_hist.dtype,
        device=x_event_hist.device,
    )
    return torch.cat([x_event_hist, pad], dim=-1)


def _positive_intensity_from_model_output(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    # If the evaluator emitted log-intensity, values tend to be <= 0.
    if np.all(vals <= 0.0):
        vals = np.exp(np.clip(vals, -50.0, 50.0))
    vals = np.maximum(vals, EPS)
    return vals


def _eval_intensity(
    evaluator: IntensityEvaluator,
    t_norm: np.ndarray,               # (M,)
    s_norm: np.ndarray,               # (M, d)
    x_field_const_norm: Optional[np.ndarray],  # (r,) or None
) -> np.ndarray:
    t_t = torch.as_tensor(t_norm, dtype=torch.float32).reshape(-1, 1)
    s_t = torch.as_tensor(s_norm, dtype=torch.float32)
    x_field_t = None
    if x_field_const_norm is not None:
        x_field_arr = np.asarray(x_field_const_norm, dtype=np.float32).reshape(1, -1)
        x_field_arr = np.repeat(x_field_arr, t_t.shape[0], axis=0)
        x_field_t = torch.as_tensor(x_field_arr, dtype=torch.float32)
    with torch.no_grad():
        lam = evaluator.intensity(t_t, s_t, x_field=x_field_t).detach().cpu().numpy()
    return _positive_intensity_from_model_output(lam)


def _disk_samples(radius: float, ks: int) -> np.ndarray:
    # Deterministic sunflower-like samples in a disk (2D).
    if ks <= 0:
        raise ValueError("Ks must be positive.")
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    out = np.zeros((ks, 2), dtype=np.float64)
    for m in range(ks):
        rho = radius * math.sqrt((m + 0.5) / ks)
        theta = 2.0 * math.pi * ((m * phi) % 1.0)
        out[m, 0] = rho * math.cos(theta)
        out[m, 1] = rho * math.sin(theta)
    return out


def _eventwise_terms(
    model: torch.nn.Module,
    seq_item: Dict[str, torch.Tensor],
) -> List[Dict[str, Any]]:
    """
    Per-event baseline terms for a single sequence, matching model forward logic.
    Returns one row per predicted event (index 1..N-1).
    """
    times = seq_item["times"].unsqueeze(0).float()
    locs = seq_item["locations"].unsqueeze(0).float()
    length = int(seq_item["length"])
    lengths = torch.tensor([length], dtype=torch.long)

    marks = seq_item.get("marks")
    if marks is not None:
        marks = marks.unsqueeze(0).long()
    x_event = seq_item.get("event_covariates")
    if x_event is not None:
        x_event = x_event.unsqueeze(0).float()
    x_field = seq_item.get("field_covariates")
    if x_field is not None:
        x_field = x_field.unsqueeze(0).float()

    x_event_enc = _build_encoder_x_event(model, times, locs, marks, x_event)
    events = torch.cat([times.unsqueeze(-1), locs], dim=-1)
    with torch.no_grad():
        _, all_states = model.encoder(events, lengths, x_event=x_event_enc)

    rows: List[Dict[str, Any]] = []
    if length < 2:
        return rows

    for n in range(length - 1):
        z_n = all_states[:, n, :]  # (1, h)
        t_prev = times[:, n : n + 1]  # (1,1)
        t_tgt = times[:, n + 1 : n + 2]  # (1,1)
        s_tgt = locs[:, n + 1, :]  # (1,d)

        x_field_dec = x_field[:, n, :] if x_field is not None else None  # (1,r) or None
        if isinstance(model.dynamics, IdentityDynamics):
            z_t = z_n
        else:
            dt = (t_tgt - t_prev).clamp(min=1e-6)
            x_field_dyn = x_field_dec.unsqueeze(1) if x_field_dec is not None else None
            z_t = model.dynamics(z_n, dt, x_field=x_field_dyn).squeeze(1)
            cached_lambda = getattr(model.dynamics, "_cached_Lambda", None)
            if cached_lambda is not None:
                temporal = getattr(model.decoder, "temporal", None)
                if temporal is not None and hasattr(temporal, "_precomputed_lambda"):
                    temporal._precomputed_lambda = cached_lambda[:, 0]

        with torch.no_grad():
            ground_nll = model.decoder.nll(
                z_t, t_tgt, s_tgt, t_prev, x_field=x_field_dec
            ).squeeze(0)
            total_nll = ground_nll
            mark_nll_val = torch.tensor(0.0, dtype=ground_nll.dtype)
            if getattr(model, "mark_decoder", None) is not None and marks is not None:
                k_tgt = marks[:, n + 1]
                mark_nll_val = model.mark_decoder.nll(
                    z_t, t_tgt, s_tgt, k_tgt, x_field=x_field_dec
                ).squeeze(0)
                total_nll = total_nll + mark_nll_val

        rows.append(
            {
                "step": n,
                "z_n": z_n.squeeze(0).detach().cpu(),
                "t_prev_norm": float(t_prev.item()),
                "t_target_norm": float(t_tgt.item()),
                "s_target_norm": s_tgt.squeeze(0).detach().cpu().numpy().astype(np.float64),
                "x_field_prev_norm": (
                    x_field_dec.squeeze(0).detach().cpu().numpy().astype(np.float64)
                    if x_field_dec is not None
                    else None
                ),
                "baseline_total_nll": float(total_nll.item()),
                "baseline_ground_nll": float(ground_nll.item()),
                "baseline_mark_nll": float(mark_nll_val.item()),
            }
        )
    return rows


def _local_smoothed_mass(
    evaluator: IntensityEvaluator,
    t_center_raw: float,
    s_center_raw: np.ndarray,   # (2,)
    t_prev_raw: float,
    t_horizon_raw: float,
    radius: float,
    tau: float,
    stats: Dict[str, np.ndarray],
    x_field_prev_norm: Optional[np.ndarray],
    kt: int = KT_DEFAULT,
    ks: int = KS_DEFAULT,
) -> Tuple[float, float]:
    t_lo = max(float(t_center_raw - tau), float(t_prev_raw + 1e-6), 0.0)
    t_hi = min(float(t_center_raw + tau), float(t_horizon_raw))
    if t_hi <= t_lo:
        return EPS, EPS

    t_grid = np.linspace(t_lo, t_hi, kt, dtype=np.float64)
    offsets = _disk_samples(radius, ks)  # (ks,2)
    area = math.pi * radius * radius

    jac = 1.0 / (float(stats["time_std"]) * float(np.prod(stats["loc_std"])))
    spatial_vals = np.zeros(kt, dtype=np.float64)

    for i, t_raw in enumerate(t_grid):
        s_raw = s_center_raw.reshape(1, 2) + offsets  # (ks,2)
        t_norm = _to_norm_time(np.full(ks, t_raw, dtype=np.float64), stats)
        s_norm = _to_norm_loc(s_raw, stats)
        lam_norm = _eval_intensity(
            evaluator=evaluator,
            t_norm=t_norm,
            s_norm=s_norm,
            x_field_const_norm=x_field_prev_norm,
        )
        spatial_vals[i] = (area / ks) * float(np.sum(lam_norm))

    if kt == 1:
        t_measure = (t_hi - t_lo)
        mass = jac * spatial_vals[0] * t_measure
    else:
        ones = np.ones_like(t_grid, dtype=np.float64)
        if hasattr(np, "trapezoid"):
            t_measure = float(np.trapezoid(ones, t_grid))
            mass = jac * float(np.trapezoid(spatial_vals, t_grid))
        else:  # NumPy < 2.0
            t_measure = float(np.trapz(ones, t_grid))
            mass = jac * float(np.trapz(spatial_vals, t_grid))

    # Volume from the same quadrature weight sum: |B| = Σ_k w_k.
    volume = area * max(t_measure, 0.0)
    return float(max(mass, EPS)), float(max(volume, EPS))


def compute_smoothed_nll(
    model: torch.nn.Module,
    val_dataset: Any,
    stats: Dict[str, np.ndarray],
    settings: Dict[str, Any],
    r: float,
    tau: float,
    kt: int = KT_DEFAULT,
    ks: int = KS_DEFAULT,
) -> Tuple[float, float, Dict[str, Any]]:
    all_event_terms_raw: List[float] = []
    all_event_terms_volcorr: List[float] = []
    baseline_from_terms: List[float] = []
    all_event_volumes: List[float] = []

    t_horizon_raw = float(settings["t_end"])
    for idx in range(len(val_dataset)):
        seq_item = val_dataset[idx]
        seq_raw = val_dataset.sequences[idx]

        rows = _eventwise_terms(model, seq_item)
        raw_times = np.asarray(seq_raw["times"], dtype=np.float64)
        raw_locs = np.asarray(seq_raw["locations"], dtype=np.float64)
        if raw_locs.ndim != 2 or raw_locs.shape[1] != 2:
            raise ValueError(
                f"Smoothed NLL expects 2D locations; got shape {raw_locs.shape} "
                f"for sequence {idx}."
            )

        for row in rows:
            n = int(row["step"])
            z_n = row["z_n"].unsqueeze(0)
            t_prev_norm = torch.tensor([[row["t_prev_norm"]]], dtype=torch.float32)
            evaluator = IntensityEvaluator(model, z=z_n, t_prev=t_prev_norm)

            # Point intensity at observed event
            lam_point = _eval_intensity(
                evaluator=evaluator,
                t_norm=np.array([row["t_target_norm"]], dtype=np.float64),
                s_norm=np.asarray(row["s_target_norm"], dtype=np.float64).reshape(1, -1),
                x_field_const_norm=row["x_field_prev_norm"],
            )[0]

            t_target_raw = float(raw_times[n + 1])
            s_target_raw = raw_locs[n + 1]
            t_prev_raw = float(raw_times[n])

            mass_local, volume_local = _local_smoothed_mass(
                evaluator=evaluator,
                t_center_raw=t_target_raw,
                s_center_raw=s_target_raw,
                t_prev_raw=t_prev_raw,
                t_horizon_raw=t_horizon_raw,
                radius=r,
                tau=tau,
                stats=stats,
                x_field_prev_norm=row["x_field_prev_norm"],
                kt=kt,
                ks=ks,
            )

            baseline_total = float(row["baseline_total_nll"])
            remainder = baseline_total + math.log(max(lam_point, EPS))
            smoothed_term_raw = remainder - math.log(max(mass_local, EPS))
            # Volume-corrected first term:
            # -log((1/|B|) * ∫_B λ) = -log(∫_B λ) + log(|B|)
            smoothed_term_volcorr = smoothed_term_raw + math.log(max(volume_local, EPS))

            baseline_from_terms.append(baseline_total)
            all_event_terms_raw.append(smoothed_term_raw)
            all_event_terms_volcorr.append(smoothed_term_volcorr)
            all_event_volumes.append(float(volume_local))

    if not all_event_terms_raw:
        raise RuntimeError("No evaluable events found in validation set for smoothed NLL.")

    smoothed_nll_raw = float(np.mean(all_event_terms_raw))
    smoothed_nll_volcorr = float(np.mean(all_event_terms_volcorr))
    baseline_from_terms_mean = float(np.mean(baseline_from_terms))
    vols = np.asarray(all_event_volumes, dtype=np.float64)
    nominal_volume = float((2.0 * tau) * (math.pi * r * r))
    clipped_count = int(np.sum(vols < (nominal_volume - 1e-12)))
    details = {
        "Kt": int(kt),
        "Ks": int(ks),
        "event_count": int(len(all_event_terms_raw)),
        "baseline_mean_from_event_terms": baseline_from_terms_mean,
        "volume_used": {
            "integration": "quadrature_weight_sum",
            "units": "raw_data_units",
            "nominal_analytic_volume": nominal_volume,
            "mean": float(np.mean(vols)),
            "std": float(np.std(vols)),
            "min": float(np.min(vols)),
            "max": float(np.max(vols)),
            "clipped_event_count": clipped_count,
        },
    }
    return smoothed_nll_raw, smoothed_nll_volcorr, details


def _points_from_raw_sequence_for_ot(seq_raw: Dict[str, Any]) -> np.ndarray:
    t = np.asarray(seq_raw["times"], dtype=np.float64)
    s = np.asarray(seq_raw["locations"], dtype=np.float64)
    if t.ndim != 1 or s.ndim != 2:
        raise ValueError("Unexpected sequence format for OT.")
    if len(t) >= 2:
        # Match conditional simulation that uses first event as history.
        t_use = t[1:]
        s_use = s[1:]
    else:
        t_use = t
        s_use = s
    if len(t_use) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate([t_use.reshape(-1, 1), s_use], axis=1)


def _choose_reference_val_seq(val_dataset: Any) -> int:
    for i, seq in enumerate(val_dataset.sequences):
        if len(seq.get("times", [])) >= 2:
            return i
    for i, seq in enumerate(val_dataset.sequences):
        if len(seq.get("times", [])) >= 1:
            return i
    raise RuntimeError("Validation set contains no events.")


def _simulate_exact_model_sample(
    model: torch.nn.Module,
    seq_item: Dict[str, torch.Tensor],
    settings: Dict[str, Any],
    stats: Dict[str, np.ndarray],
    max_events: int,
) -> np.ndarray:
    times = seq_item["times"]
    locs = seq_item["locations"]
    if int(seq_item["length"]) < 1:
        return np.zeros((0, 3), dtype=np.float64)

    hist_len = 1
    h_times = times[:hist_len].unsqueeze(0).float()
    h_locs = locs[:hist_len].unsqueeze(0).float()
    h_lens = torch.tensor([hist_len], dtype=torch.long)
    h_marks = None
    if seq_item.get("marks") is not None:
        h_marks = seq_item["marks"][:hist_len].unsqueeze(0).long()

    t_max_norm = float(_to_norm_time(np.array([settings["t_end"]], dtype=np.float64), stats)[0])
    with torch.no_grad():
        s_t, s_x, s_mask, _ = model.sample(
            history_times=h_times,
            history_locations=h_locs,
            history_lengths=h_lens,
            n_samples=int(max_events),
            t_max=t_max_norm,
            history_marks=h_marks,
        )

    mask = s_mask[0].detach().cpu().numpy().astype(bool)
    t_norm = s_t[0].detach().cpu().numpy()[mask]
    x_norm = s_x[0].detach().cpu().numpy()[mask]
    if t_norm.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    t_raw = _to_raw_time(t_norm, stats)
    x_raw = _to_raw_loc(x_norm, stats)
    pts = np.concatenate([t_raw.reshape(-1, 1), x_raw], axis=1)
    pts = pts[np.argsort(pts[:, 0])]
    return pts


def _simulate_thinning_fixed_history(
    model: torch.nn.Module,
    seq_item: Dict[str, torch.Tensor],
    settings: Dict[str, Any],
    stats: Dict[str, np.ndarray],
    max_events: int,
) -> np.ndarray:
    times = seq_item["times"].unsqueeze(0).float()
    locs = seq_item["locations"].unsqueeze(0).float()
    length = int(seq_item["length"])
    if length < 1:
        return np.zeros((0, 3), dtype=np.float64)
    lengths = torch.tensor([length], dtype=torch.long)

    marks = seq_item.get("marks")
    if marks is not None:
        marks = marks.unsqueeze(0).long()
    x_event = seq_item.get("event_covariates")
    if x_event is not None:
        x_event = x_event.unsqueeze(0).float()
    x_field = seq_item.get("field_covariates")
    if x_field is not None:
        x_field = x_field.unsqueeze(0).float()

    x_event_enc = _build_encoder_x_event(model, times, locs, marks, x_event)
    events = torch.cat([times.unsqueeze(-1), locs], dim=-1)
    with torch.no_grad():
        _, all_states = model.encoder(events, lengths, x_event=x_event_enc)
    z0 = all_states[:, 0, :]
    t0 = times[:, 0:1]

    evaluator = IntensityEvaluator(model, z=z0, t_prev=t0)
    x_field_const = None
    if x_field is not None and x_field.shape[1] > 0:
        x_field_const = x_field[:, 0, :].squeeze(0).detach().cpu().numpy().astype(np.float64)
    elif int(settings["field_cov_dim"]) > 0:
        x_field_const = np.zeros(int(settings["field_cov_dim"]), dtype=np.float64)

    loc_mean = stats["loc_mean"]
    loc_std = stats["loc_std"]
    smin_raw = np.array([settings["spatial_min"], settings["spatial_min"]], dtype=np.float64)
    smax_raw = np.array([settings["spatial_max"], settings["spatial_max"]], dtype=np.float64)
    smin_norm = _to_norm_loc(smin_raw, stats)
    smax_norm = _to_norm_loc(smax_raw, stats)

    t_max_norm = float(_to_norm_time(np.array([settings["t_end"]], dtype=np.float64), stats)[0])

    def intensity_fn(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x_field_t = None
        if x_field_const is not None:
            xf = np.repeat(x_field_const.reshape(1, -1), t.shape[0], axis=0)
            x_field_t = torch.as_tensor(xf, dtype=torch.float32)
        with torch.no_grad():
            lam = evaluator.intensity(t, s, x_field=x_field_t)
        lam_np = _positive_intensity_from_model_output(lam.detach().cpu().numpy())
        return torch.as_tensor(lam_np, dtype=torch.float32)

    with torch.no_grad():
        samp_t, samp_s, counts = thinning_sample(
            intensity_fn=intensity_fn,
            t_start=t0,
            t_max=t_max_norm,
            spatial_bounds=(
                torch.as_tensor(smin_norm, dtype=torch.float32),
                torch.as_tensor(smax_norm, dtype=torch.float32),
            ),
            lambda_bar=10.0,
            max_events=int(max_events),
            adaptive=True,
        )
    n = int(counts[0].item())
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    t_norm = samp_t[0, :n].detach().cpu().numpy()
    s_norm = samp_s[0, :n].detach().cpu().numpy()
    t_raw = _to_raw_time(t_norm, stats)
    s_raw = _to_raw_loc(s_norm, stats)
    pts = np.concatenate([t_raw.reshape(-1, 1), s_raw], axis=1)
    pts = pts[np.argsort(pts[:, 0])]
    return pts


def _simulate_bootstrap(
    obs_points: np.ndarray,
    rng: np.random.Generator,
    t_end: float,
) -> np.ndarray:
    if obs_points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    n_obs = int(obs_points.shape[0])
    n = int(max(1, rng.poisson(max(1, n_obs))))
    idx = rng.integers(0, n_obs, size=n)
    pts = obs_points[idx].copy()

    t_std = float(np.std(obs_points[:, 0])) if n_obs > 1 else 1.0
    s_std = float(np.std(obs_points[:, 1:])) if n_obs > 1 else 1.0
    t_j = max(1e-4, 0.05 * t_std)
    s_j = max(1e-4, 0.05 * s_std)

    pts[:, 0] = pts[:, 0] + rng.normal(0.0, t_j, size=n)
    pts[:, 1:] = pts[:, 1:] + rng.normal(0.0, s_j, size=(n, 2))
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(t_end))
    pts = pts[np.argsort(pts[:, 0])]
    return pts


def _equalize_cardinality(
    obs_points: np.ndarray,
    sim_points: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    n = obs_points.shape[0]
    m = sim_points.shape[0]
    if n == 0 or m == 0:
        return obs_points[:0], sim_points[:0], "empty_set"
    if n == m:
        return obs_points, sim_points, None
    if n > m:
        idx = rng.choice(n, size=m, replace=False)
        return obs_points[idx], sim_points, "subsample_observed"
    idx = rng.choice(m, size=n, replace=False)
    return obs_points, sim_points[idx], "subsample_simulated"


def _cost_matrix(
    obs_points: np.ndarray,
    sim_points: np.ndarray,
    p: int,
    normalize: bool,
    r: float,
    tau: float,
    ot_axes: str,
) -> np.ndarray:
    t1 = obs_points[:, 0:1]  # (N,1)
    t2 = sim_points[:, 0:1]  # (M,1)
    dt = np.abs(t1 - t2.T)   # (N,M)

    s1 = obs_points[:, 1:]   # (N,2)
    s2 = sim_points[:, 1:]   # (M,2)
    ds = np.linalg.norm(s1[:, None, :] - s2[None, :, :], axis=-1)

    if ot_axes == "joint":
        if normalize:
            d = np.sqrt((ds / max(r, EPS)) ** 2 + (dt / max(tau, EPS)) ** 2)
        else:
            d = np.sqrt(ds ** 2 + dt ** 2)
    elif ot_axes == "space_only":
        d = ds / max(r, EPS) if normalize else ds
    elif ot_axes == "time_only":
        d = dt / max(tau, EPS) if normalize else dt
    else:
        raise ValueError(f"Unsupported ot_axes: {ot_axes}")
    return np.power(d, p, dtype=np.float64)


def _ot_distance(
    cost: np.ndarray,
    mode: str,
    p: int,
    sinkhorn_reg: float,
) -> float:
    if cost.size == 0:
        return float("nan")
    n, m = cost.shape
    if n != m:
        raise ValueError("OT expects equal-size point sets after subsampling.")

    if mode == "hungarian":
        row_ind, col_ind = linear_sum_assignment(cost)
        mean_cost = float(cost[row_ind, col_ind].mean())
        return float(mean_cost ** (1.0 / p))

    if mode == "sinkhorn":
        if pot_ot is None:
            raise RuntimeError(
                "POT/ot is not installed. Install 'pot' or use --ot_mode hungarian."
            )
        a = np.full(n, 1.0 / n, dtype=np.float64)
        b = np.full(n, 1.0 / n, dtype=np.float64)
        sink = pot_ot.sinkhorn2(a, b, cost, reg=float(sinkhorn_reg))
        mean_cost = float(np.asarray(sink).reshape(-1)[0])
        return float(mean_cost ** (1.0 / p))

    raise ValueError(f"Unsupported ot_mode: {mode}")


def _assignment_error_metrics(
    obs_points: np.ndarray,
    sim_points: np.ndarray,
    row_ind: np.ndarray,
    col_ind: np.ndarray,
    r: float,
    tau: float,
    include_norm: bool,
) -> Dict[str, float]:
    obs_m = obs_points[row_ind]
    sim_m = sim_points[col_ind]
    d_space = np.linalg.norm(obs_m[:, 1:] - sim_m[:, 1:], axis=-1)
    d_time = np.abs(obs_m[:, 0] - sim_m[:, 0])

    out: Dict[str, float] = {
        "mae_space": float(np.mean(d_space)),
        "rmse_space": float(np.sqrt(np.mean(d_space ** 2))),
        "mae_time": float(np.mean(d_time)),
        "rmse_time": float(np.sqrt(np.mean(d_time ** 2))),
    }
    if include_norm:
        d_space_n = d_space / max(float(r), EPS)
        d_time_n = d_time / max(float(tau), EPS)
        out.update(
            {
                "mae_space_norm": float(np.mean(d_space_n)),
                "rmse_space_norm": float(np.sqrt(np.mean(d_space_n ** 2))),
                "mae_time_norm": float(np.mean(d_time_n)),
                "rmse_time_norm": float(np.sqrt(np.mean(d_time_n ** 2))),
            }
        )
    return out


def compute_ot_metrics(
    model: torch.nn.Module,
    val_dataset: Any,
    settings: Dict[str, Any],
    stats: Dict[str, np.ndarray],
    seed: int,
    r: float,
    tau: float,
    p: int,
    n_sims: int,
    mode: str,
    sinkhorn_reg: float,
    normalize: bool,
    ot_axes: str,
) -> Tuple[float, float, Dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(seed)
    ref_idx = _choose_reference_val_seq(val_dataset)
    seq_item = val_dataset[ref_idx]
    seq_raw = val_dataset.sequences[ref_idx]
    obs_points = _points_from_raw_sequence_for_ot(seq_raw)
    if obs_points.shape[0] == 0:
        raise RuntimeError("Reference validation sequence has zero OT-evaluable events.")

    dists: List[float] = []
    method_counts = {"exact": 0, "thinning": 0, "bootstrap": 0}
    subsample_counts = {"subsample_observed": 0, "subsample_simulated": 0, "empty_set": 0}
    assignment_source = "hungarian" if mode == "hungarian" else "hungarian_aux_for_diag"

    mae_space_vals: List[float] = []
    rmse_space_vals: List[float] = []
    mae_time_vals: List[float] = []
    rmse_time_vals: List[float] = []
    mae_space_norm_vals: List[float] = []
    rmse_space_norm_vals: List[float] = []
    mae_time_norm_vals: List[float] = []
    rmse_time_norm_vals: List[float] = []
    count_diffs: List[float] = []
    n_obs_per_sim: List[int] = []
    n_sim_per_sim: List[int] = []

    max_events = max(64, int(3 * obs_points.shape[0]) + 20)

    for k in range(n_sims):
        method = "exact"
        sim_points = np.zeros((0, 3), dtype=np.float64)

        # Attempt exact autoregressive sampler from fitted model.
        try:
            torch.manual_seed(seed + 1000 + k)
            sim_points = _simulate_exact_model_sample(
                model=model,
                seq_item=seq_item,
                settings=settings,
                stats=stats,
                max_events=max_events,
            )
            if sim_points.shape[0] == 0:
                raise RuntimeError("exact sampler produced empty sample")
        except Exception:
            method = "thinning"
            try:
                torch.manual_seed(seed + 2000 + k)
                sim_points = _simulate_thinning_fixed_history(
                    model=model,
                    seq_item=seq_item,
                    settings=settings,
                    stats=stats,
                    max_events=max_events,
                )
                if sim_points.shape[0] == 0:
                    raise RuntimeError("thinning sampler produced empty sample")
            except Exception:
                method = "bootstrap"
                sim_points = _simulate_bootstrap(
                    obs_points=obs_points,
                    rng=rng,
                    t_end=float(settings["t_end"]),
                )

        method_counts[method] += 1
        n_obs_k = int(obs_points.shape[0])
        n_sim_k = int(sim_points.shape[0])
        n_obs_per_sim.append(n_obs_k)
        n_sim_per_sim.append(n_sim_k)
        count_diffs.append(float(n_sim_k - n_obs_k))

        obs_eq, sim_eq, note = _equalize_cardinality(obs_points, sim_points, rng)
        if note is not None:
            subsample_counts[note] += 1
        if obs_eq.shape[0] == 0 or sim_eq.shape[0] == 0:
            dists.append(float("nan"))
            mae_space_vals.append(float("nan"))
            rmse_space_vals.append(float("nan"))
            mae_time_vals.append(float("nan"))
            rmse_time_vals.append(float("nan"))
            if normalize:
                mae_space_norm_vals.append(float("nan"))
                rmse_space_norm_vals.append(float("nan"))
                mae_time_norm_vals.append(float("nan"))
                rmse_time_norm_vals.append(float("nan"))
            continue

        c = _cost_matrix(
            obs_points=obs_eq,
            sim_points=sim_eq,
            p=p,
            normalize=normalize,
            r=r,
            tau=tau,
            ot_axes=ot_axes,
        )
        row_ind, col_ind = linear_sum_assignment(c)
        if mode == "hungarian":
            mean_cost = float(c[row_ind, col_ind].mean())
            d = float(mean_cost ** (1.0 / p))
        else:
            d = _ot_distance(c, mode=mode, p=p, sinkhorn_reg=sinkhorn_reg)

        diag = _assignment_error_metrics(
            obs_points=obs_eq,
            sim_points=sim_eq,
            row_ind=row_ind,
            col_ind=col_ind,
            r=float(r),
            tau=float(tau),
            include_norm=bool(normalize),
        )
        dists.append(float(d))
        mae_space_vals.append(diag["mae_space"])
        rmse_space_vals.append(diag["rmse_space"])
        mae_time_vals.append(diag["mae_time"])
        rmse_time_vals.append(diag["rmse_time"])
        if normalize:
            mae_space_norm_vals.append(diag["mae_space_norm"])
            rmse_space_norm_vals.append(diag["rmse_space_norm"])
            mae_time_norm_vals.append(diag["mae_time_norm"])
            rmse_time_norm_vals.append(diag["rmse_time_norm"])

    d_arr = np.asarray(dists, dtype=np.float64)
    finite = np.isfinite(d_arr)
    if not np.any(finite):
        raise RuntimeError("All OT simulation distances are non-finite.")
    mean = float(np.mean(d_arr[finite]))
    std = float(np.std(d_arr[finite]))

    diff_arr = np.asarray(count_diffs, dtype=np.float64)
    count_mae = float(np.mean(np.abs(diff_arr))) if diff_arr.size > 0 else float("nan")
    count_rmse = (
        float(np.sqrt(np.mean(diff_arr ** 2))) if diff_arr.size > 0 else float("nan")
    )

    details = {
        "reference_val_index": int(ref_idx),
        "reference_observed_count": int(obs_points.shape[0]),
        "ot_mode": mode,
        "ot_axes": ot_axes,
        "assignment_source_for_diagnostics": assignment_source,
        "method_counts": method_counts,
        "subsample_counts": subsample_counts,
        "finite_sims": int(np.sum(finite)),
        "ot": {"mean": mean, "std": std, "n": int(np.sum(finite))},
        "mae_space": _finite_mean_std(mae_space_vals),
        "rmse_space": _finite_mean_std(rmse_space_vals),
        "mae_time": _finite_mean_std(mae_time_vals),
        "rmse_time": _finite_mean_std(rmse_time_vals),
        "count_diff": [float(x) for x in diff_arr.tolist()],
        "n_obs_per_sim": [int(x) for x in n_obs_per_sim],
        "n_sim_per_sim": [int(x) for x in n_sim_per_sim],
        "count_mae": count_mae,
        "count_rmse": count_rmse,
    }
    if normalize:
        details["mae_space_norm"] = _finite_mean_std(mae_space_norm_vals)
        details["rmse_space_norm"] = _finite_mean_std(rmse_space_norm_vals)
        details["mae_time_norm"] = _finite_mean_std(mae_time_norm_vals)
        details["rmse_time_norm"] = _finite_mean_std(rmse_time_norm_vals)
    return mean, std, details, obs_points


def _plot_heatmap_if_available(
    arr: np.ndarray,
    r_values: np.ndarray,
    tau_values: np.ndarray,
    title: str,
    out_path: Path,
) -> Tuple[bool, Optional[str]]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return False, f"matplotlib unavailable: {exc}"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("r")
    ax.set_ylabel("tau")
    ax.set_title(title)

    xt = np.arange(len(r_values))
    yt = np.arange(len(tau_values))
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{v:.3g}" for v in r_values], rotation=45, ha="right")
    ax.set_yticklabels([f"{v:.3g}" for v in tau_values])

    finite = np.isfinite(arr)
    if np.any(finite):
        min_idx = np.unravel_index(np.nanargmin(arr), arr.shape)
        i_tau, i_r = int(min_idx[0]), int(min_idx[1])
        min_val = float(arr[i_tau, i_r])
        ax.scatter([i_r], [i_tau], marker="x", s=70, c="red")
        ax.annotate(
            f"min={min_val:.4g}\n(r={r_values[i_r]:.3g}, tau={tau_values[i_tau]:.3g})",
            (i_r, i_tau),
            xytext=(8, 8),
            textcoords="offset points",
            color="red",
            fontsize=9,
        )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("NLL")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True, None


def run_smoothed_sweep(
    model: torch.nn.Module,
    val_dataset: Any,
    stats: Dict[str, np.ndarray],
    settings: Dict[str, Any],
    run_id: str,
    r_values: np.ndarray,
    tau_values: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    n_tau = int(len(tau_values))
    n_r = int(len(r_values))
    raw_arr = np.zeros((n_tau, n_r), dtype=np.float64)
    vol_arr = np.zeros((n_tau, n_r), dtype=np.float64)

    for i_tau, tau in enumerate(tau_values):
        for i_r, r in enumerate(r_values):
            sm_raw, sm_vol, _ = compute_smoothed_nll(
                model=model,
                val_dataset=val_dataset,
                stats=stats,
                settings=settings,
                r=float(r),
                tau=float(tau),
                kt=KT_DEFAULT,
                ks=KS_DEFAULT,
            )
            raw_arr[i_tau, i_r] = sm_raw
            vol_arr[i_tau, i_r] = sm_vol

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_npy = output_dir / f"{run_id}_smoothed_nll_raw.npy"
    vol_npy = output_dir / f"{run_id}_smoothed_nll_volcorr.npy"
    np.save(raw_npy, raw_arr)
    np.save(vol_npy, vol_arr)

    heat_raw_path = output_dir / "heatmap_smoothed_nll_raw.png"
    heat_vol_path = output_dir / "heatmap_smoothed_nll_volcorr.png"
    ok_raw, msg_raw = _plot_heatmap_if_available(
        raw_arr,
        r_values=r_values,
        tau_values=tau_values,
        title=f"Smoothed NLL Raw | {run_id}",
        out_path=heat_raw_path,
    )
    ok_vol, msg_vol = _plot_heatmap_if_available(
        vol_arr,
        r_values=r_values,
        tau_values=tau_values,
        title=f"Smoothed NLL Vol-Corrected | {run_id}",
        out_path=heat_vol_path,
    )
    if not ok_raw or not ok_vol:
        msg = msg_raw if not ok_raw else msg_vol
        print(f"Sweep plotting skipped: {msg}")

    min_raw_idx = np.unravel_index(np.nanargmin(raw_arr), raw_arr.shape)
    min_vol_idx = np.unravel_index(np.nanargmin(vol_arr), vol_arr.shape)
    return {
        "r_values": [float(v) for v in r_values.tolist()],
        "tau_values": [float(v) for v in tau_values.tolist()],
        "shape": [n_tau, n_r],
        "smoothed_nll_raw": raw_arr.tolist(),
        "smoothed_nll_volcorr": vol_arr.tolist(),
        "npy_paths": {
            "smoothed_nll_raw": str(raw_npy),
            "smoothed_nll_volcorr": str(vol_npy),
        },
        "heatmap_paths": {
            "smoothed_nll_raw": str(heat_raw_path) if ok_raw else None,
            "smoothed_nll_volcorr": str(heat_vol_path) if ok_vol else None,
        },
        "minima": {
            "smoothed_nll_raw": {
                "value": float(raw_arr[min_raw_idx]),
                "tau": float(tau_values[min_raw_idx[0]]),
                "r": float(r_values[min_raw_idx[1]]),
            },
            "smoothed_nll_volcorr": {
                "value": float(vol_arr[min_vol_idx]),
                "tau": float(tau_values[min_vol_idx[0]]),
                "r": float(r_values[min_vol_idx[1]]),
            },
        },
    }


def _summary_print(
    run_id: str,
    ckpt_path: Path,
    baseline_nll: float,
    smoothed_nll_raw: float,
    smoothed_nll_volcorr: float,
    smoothed_info: Dict[str, Any],
    ot_mean: float,
    ot_std: float,
    ot_info: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    print("")
    print("Experiment Summary")
    print("------------------")
    print(f"run_id                : {run_id}")
    print(f"checkpoint            : {ckpt_path}")
    if np.isfinite(baseline_nll):
        print(f"baseline_nll          : {baseline_nll:.6f}")
    else:
        print("baseline_nll          : skipped")
    print(
        "smoothed_nll_raw      : "
        f"{smoothed_nll_raw:.6f}  (r={args.r}, tau={args.tau}, Kt={smoothed_info['Kt']}, Ks={smoothed_info['Ks']})"
    )
    print(
        "smoothed_nll_volcorr  : "
        f"{smoothed_nll_volcorr:.6f}"
    )
    vol = smoothed_info.get("volume_used", {})
    print(
        "smoothed_volume       : "
        f"mean={vol.get('mean', float('nan')):.6g}, "
        f"min={vol.get('min', float('nan')):.6g}, "
        f"max={vol.get('max', float('nan')):.6g}, "
        f"nominal={vol.get('nominal_analytic_volume', float('nan')):.6g}"
    )
    axes = ot_info.get("ot_axes", args.ot_axes)
    if np.isfinite(ot_mean):
        print(
            f"ot_distance_{axes:<9}: "
            f"{ot_mean:.6f} ± {ot_std:.6f}  "
            f"(mode={args.ot_mode}, p={args.p}, normalize={args.normalize}, n_sims={args.n_sims})"
        )
    else:
        print(
            f"ot_distance_{axes:<9}: skipped "
            f"(mode={args.ot_mode}, p={args.p}, normalize={args.normalize}, n_sims={args.n_sims})"
        )
    if "mae_space" in ot_info:
        ms = ot_info["mae_space"]
        rs = ot_info["rmse_space"]
        mt = ot_info["mae_time"]
        rt = ot_info["rmse_time"]
        print(
            "transport_space       : "
            f"MAE={ms['mean']:.6f}±{ms['std']:.6f}, "
            f"RMSE={rs['mean']:.6f}±{rs['std']:.6f}"
        )
        print(
            "transport_time        : "
            f"MAE={mt['mean']:.6f}±{mt['std']:.6f}, "
            f"RMSE={rt['mean']:.6f}±{rt['std']:.6f}"
        )
        if args.normalize and "mae_space_norm" in ot_info:
            msn = ot_info["mae_space_norm"]
            rsn = ot_info["rmse_space_norm"]
            mtn = ot_info["mae_time_norm"]
            rtn = ot_info["rmse_time_norm"]
            print(
                "transport_space_norm  : "
                f"MAE={msn['mean']:.6f}±{msn['std']:.6f}, "
                f"RMSE={rsn['mean']:.6f}±{rsn['std']:.6f}"
            )
            print(
                "transport_time_norm   : "
                f"MAE={mtn['mean']:.6f}±{mtn['std']:.6f}, "
                f"RMSE={rtn['mean']:.6f}±{rtn['std']:.6f}"
            )
        print(
            "count_mismatch        : "
            f"MAE={ot_info.get('count_mae', float('nan')):.6f}, "
            f"RMSE={ot_info.get('count_rmse', float('nan')):.6f}"
        )
    print(
        "ot_sim_methods        : "
        f"exact={ot_info['method_counts']['exact']}, "
        f"thinning={ot_info['method_counts']['thinning']}, "
        f"bootstrap={ot_info['method_counts']['bootstrap']}"
    )
    if "n_obs_per_sim" in ot_info and "n_sim_per_sim" in ot_info:
        n_obs = ot_info["n_obs_per_sim"]
        n_sim = ot_info["n_sim_per_sim"]
        if n_obs and n_sim:
            print(
                "count_samples         : "
                f"N_obs(first)={n_obs[0]}, "
                f"N_sim(mean)={np.mean(n_sim):.3f}"
            )
            diffs = ot_info.get("count_diff", [])
            if diffs:
                show = diffs[:8]
                suffix = " ..." if len(diffs) > len(show) else ""
                print(f"count_diff_per_sim    : {show}{suffix}")
    print("")


def main() -> None:
    args = parse_args()
    if args.sweep_only:
        args.sweep = True
    if args.r <= 0 or args.tau <= 0:
        raise ValueError("--r and --tau must be positive.")
    if args.p <= 0:
        raise ValueError("--p must be a positive integer.")
    if args.n_sims <= 0:
        raise ValueError("--n_sims must be positive.")
    if args.ot_mode == "sinkhorn" and pot_ot is None and not args.sweep_only:
        raise RuntimeError(
            "--ot_mode sinkhorn requested but POT (ot) is not installed. "
            "Install `pot` or use --ot_mode hungarian."
        )

    r_grid_vals = None
    tau_grid_vals = None
    if args.sweep:
        r_grid_vals = _parse_grid_spec(args.r_grid, "r_grid")
        tau_grid_vals = _parse_grid_spec(args.tau_grid, "tau_grid")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    run_preset, run_data_type = _parse_run_id(args.run_id)
    logs_dir = _find_logs_dir(args.run_id)
    ckpt_path = _select_checkpoint(args.run_id)

    ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state = ckpt_obj.get("state_dict", ckpt_obj)
    if not isinstance(raw_state, dict):
        raise RuntimeError(f"Unexpected checkpoint structure: {ckpt_path}")
    stripped_state = _strip_state_dict_prefix(raw_state)
    model_spec = _infer_model_spec(run_preset, stripped_state)

    model, ckpt = _build_model_from_checkpoint(ckpt_path, model_spec)
    settings = _resolve_settings(args.run_id, model_spec, run_data_type)
    if args.sweep and r_grid_vals is not None and tau_grid_vals is not None:
        r_rec_max = 0.5 * float(settings["spatial_max"] - settings["spatial_min"])
        tau_rec_max = 0.5 * float(settings["t_end"])
        if np.any(r_grid_vals > r_rec_max + 1e-12):
            print(
                "Warning: some r_grid values exceed half spatial range "
                f"({r_rec_max:.6g})."
            )
        if np.any(tau_grid_vals > tau_rec_max + 1e-12):
            print(
                "Warning: some tau_grid values exceed T/2 "
                f"({tau_rec_max:.6g})."
            )

    train_seqs, val_seqs = _generate_sequences(settings)
    dm, train_dataset, val_dataset = _build_data(
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        batch_size=int(settings["batch_size"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    baseline_nll = float("nan")
    if not args.sweep_only:
        # Baseline NLL: call existing evaluation function unchanged.
        eval_trainer = Trainer(
            model=model,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip=0.0,
            device="cpu",
        )
        baseline_metrics = eval_trainer.evaluate(val_loader)
        baseline_nll = float(baseline_metrics["nll"])

    stats = _stats_from_train_dataset(train_dataset)
    smoothed_nll_raw, smoothed_nll_volcorr, smoothed_info = compute_smoothed_nll(
        model=model,
        val_dataset=val_dataset,
        stats=stats,
        settings=settings,
        r=float(args.r),
        tau=float(args.tau),
        kt=KT_DEFAULT,
        ks=KS_DEFAULT,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = None
    if args.sweep and r_grid_vals is not None and tau_grid_vals is not None:
        sweep_results = run_smoothed_sweep(
            model=model,
            val_dataset=val_dataset,
            stats=stats,
            settings=settings,
            run_id=args.run_id,
            r_values=r_grid_vals,
            tau_values=tau_grid_vals,
            output_dir=out_dir,
        )

    ot_mean = float("nan")
    ot_std = float("nan")
    obs_points_ref = np.zeros((0, 3), dtype=np.float64)
    ot_info: Dict[str, Any] = {
        "ot_mode": args.ot_mode,
        "ot_axes": args.ot_axes,
        "reference_val_index": None,
        "method_counts": {"exact": 0, "thinning": 0, "bootstrap": 0},
        "subsample_counts": {"subsample_observed": 0, "subsample_simulated": 0, "empty_set": 0},
        "finite_sims": 0,
    }
    if not args.sweep_only:
        ot_mean, ot_std, ot_info, obs_points_ref = compute_ot_metrics(
            model=model,
            val_dataset=val_dataset,
            settings=settings,
            stats=stats,
            seed=int(args.seed),
            r=float(args.r),
            tau=float(args.tau),
            p=int(args.p),
            n_sims=int(args.n_sims),
            mode=args.ot_mode,
            sinkhorn_reg=float(args.sinkhorn_reg),
            normalize=bool(args.normalize),
            ot_axes=args.ot_axes,
        )

    # Tiny unit-style self-check.
    if not args.sweep_only:
        assert np.isfinite(baseline_nll), "baseline_nll is not finite"
    assert np.isfinite(smoothed_nll_raw), "smoothed_nll_raw is not finite"
    assert np.isfinite(smoothed_nll_volcorr), "smoothed_nll_volcorr is not finite"
    if not args.sweep_only:
        assert np.isfinite(ot_mean) and np.isfinite(ot_std), "OT summary is not finite"
        assert obs_points_ref.ndim == 2 and obs_points_ref.shape[1] == 3, "OT point shape mismatch"
    if sweep_results is not None:
        n_tau = len(sweep_results["tau_values"])
        n_r = len(sweep_results["r_values"])
        arr_raw = np.asarray(sweep_results["smoothed_nll_raw"], dtype=np.float64)
        arr_vol = np.asarray(sweep_results["smoothed_nll_volcorr"], dtype=np.float64)
        assert arr_raw.shape == (n_tau, n_r), "Sweep raw array shape mismatch"
        assert arr_vol.shape == (n_tau, n_r), "Sweep volcorr array shape mismatch"
        assert np.all(np.isfinite(arr_raw)), "Sweep raw has non-finite entries"
        assert np.all(np.isfinite(arr_vol)), "Sweep volcorr has non-finite entries"

    _summary_print(
        run_id=args.run_id,
        ckpt_path=ckpt_path,
        baseline_nll=baseline_nll,
        smoothed_nll_raw=smoothed_nll_raw,
        smoothed_nll_volcorr=smoothed_nll_volcorr,
        smoothed_info=smoothed_info,
        ot_mean=ot_mean,
        ot_std=ot_std,
        ot_info=ot_info,
        args=args,
    )
    if sweep_results is not None:
        print(
            "sweep_grid            : "
            f"tau={len(sweep_results['tau_values'])}, r={len(sweep_results['r_values'])}"
        )
        print(
            "sweep_min_raw         : "
            f"{sweep_results['minima']['smoothed_nll_raw']['value']:.6f} "
            f"at (tau={sweep_results['minima']['smoothed_nll_raw']['tau']:.6g}, "
            f"r={sweep_results['minima']['smoothed_nll_raw']['r']:.6g})"
        )
        print(
            "sweep_min_volcorr     : "
            f"{sweep_results['minima']['smoothed_nll_volcorr']['value']:.6f} "
            f"at (tau={sweep_results['minima']['smoothed_nll_volcorr']['tau']:.6g}, "
            f"r={sweep_results['minima']['smoothed_nll_volcorr']['r']:.6g})"
        )
        print(
            "sweep_heatmaps        : "
            f"raw={sweep_results['heatmap_paths']['smoothed_nll_raw']}, "
            f"volcorr={sweep_results['heatmap_paths']['smoothed_nll_volcorr']}"
        )

    out_path = out_dir / f"{args.run_id}.json"

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "git_commit": _git_commit_short(),
        "run_id": args.run_id,
        "checkpoint": str(ckpt_path),
        "logs_dir": str(logs_dir) if logs_dir is not None else None,
        "preset": settings["preset"],
        "data_type": settings["data_type"],
        "dataset_split": "val",
        "dataset_split_identifier": (
            f"val_n{settings['n_val']}_seed{settings['data_seed']}_type{settings['data_type']}"
        ),
        "reference_val_sequence_index": ot_info["reference_val_index"],
        "config_path_used": settings.get("config_path_used"),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else None,
    }

    payload: Dict[str, Any] = {
        "baseline_nll": (float(baseline_nll) if np.isfinite(baseline_nll) else None),
        "smoothed_nll_raw": smoothed_nll_raw,
        "smoothed_nll_volcorr": smoothed_nll_volcorr,
        "smoothed_config": {
            "r": float(args.r),
            "tau": float(args.tau),
            "Kt": int(smoothed_info["Kt"]),
            "Ks": int(smoothed_info["Ks"]),
            "event_count": int(smoothed_info["event_count"]),
            "volume_used": smoothed_info["volume_used"],
        },
        "ot_mean": (float(ot_mean) if np.isfinite(ot_mean) else None),
        "ot_std": (float(ot_std) if np.isfinite(ot_std) else None),
        "ot_config": {
            "p": int(args.p),
            "normalize": bool(args.normalize),
            "mode": args.ot_mode,
            "ot_axes": args.ot_axes,
            "n_sims": int(args.n_sims),
            "sinkhorn_reg": float(args.sinkhorn_reg),
            "method_counts": ot_info["method_counts"],
            "subsample_counts": ot_info["subsample_counts"],
        },
        "ot": _json_metric_dict(
            ot_info.get("ot", {"mean": ot_mean, "std": ot_std, "n": ot_info.get("finite_sims", 0)})
        ),
        "mae_space": _json_metric_dict(
            ot_info.get("mae_space", {"mean": float("nan"), "std": float("nan"), "n": 0})
        ),
        "rmse_space": _json_metric_dict(
            ot_info.get("rmse_space", {"mean": float("nan"), "std": float("nan"), "n": 0})
        ),
        "mae_time": _json_metric_dict(
            ot_info.get("mae_time", {"mean": float("nan"), "std": float("nan"), "n": 0})
        ),
        "rmse_time": _json_metric_dict(
            ot_info.get("rmse_time", {"mean": float("nan"), "std": float("nan"), "n": 0})
        ),
        "count_diff": ot_info.get("count_diff", []),
        "n_obs_per_sim": ot_info.get("n_obs_per_sim", []),
        "n_sim_per_sim": ot_info.get("n_sim_per_sim", []),
        "count_mae": (
            float(ot_info.get("count_mae"))
            if np.isfinite(ot_info.get("count_mae", float("nan")))
            else None
        ),
        "count_rmse": (
            float(ot_info.get("count_rmse"))
            if np.isfinite(ot_info.get("count_rmse", float("nan")))
            else None
        ),
        "meta": meta,
    }
    payload[f"ot_distance_{args.ot_axes}"] = {
        "mean": (float(ot_mean) if np.isfinite(ot_mean) else None),
        "std": (float(ot_std) if np.isfinite(ot_std) else None),
        "n": int(ot_info.get("finite_sims", 0)),
    }
    if args.normalize:
        payload["mae_space_norm"] = _json_metric_dict(
            ot_info.get("mae_space_norm", {"mean": float("nan"), "std": float("nan"), "n": 0})
        )
        payload["rmse_space_norm"] = _json_metric_dict(
            ot_info.get("rmse_space_norm", {"mean": float("nan"), "std": float("nan"), "n": 0})
        )
        payload["mae_time_norm"] = _json_metric_dict(
            ot_info.get("mae_time_norm", {"mean": float("nan"), "std": float("nan"), "n": 0})
        )
        payload["rmse_time_norm"] = _json_metric_dict(
            ot_info.get("rmse_time_norm", {"mean": float("nan"), "std": float("nan"), "n": 0})
        )
    if sweep_results is not None:
        payload["grids"] = {
            "r_values": sweep_results["r_values"],
            "tau_values": sweep_results["tau_values"],
        }
        payload["heatmaps"] = {
            "smoothed_nll_raw": {
                "array": sweep_results["smoothed_nll_raw"],
                "npy_path": sweep_results["npy_paths"]["smoothed_nll_raw"],
                "png_path": sweep_results["heatmap_paths"]["smoothed_nll_raw"],
                "min": sweep_results["minima"]["smoothed_nll_raw"],
            },
            "smoothed_nll_volcorr": {
                "array": sweep_results["smoothed_nll_volcorr"],
                "npy_path": sweep_results["npy_paths"]["smoothed_nll_volcorr"],
                "png_path": sweep_results["heatmap_paths"]["smoothed_nll_volcorr"],
                "min": sweep_results["minima"]["smoothed_nll_volcorr"],
            },
        }

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path}")
    print("Self-check: passed")


if __name__ == "__main__":
    main()
