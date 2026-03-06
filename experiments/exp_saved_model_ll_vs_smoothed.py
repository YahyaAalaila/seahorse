#!/usr/bin/env python3
"""
Saved-model experiment: held-out NLL vs. smoothed NLL (vol-corrected).

Goal
----
Load a trained checkpoint, reconstruct the corresponding held-out split using
the same pathway as the existing saved-model evaluator, and compare:
1) standard held-out NLL (baseline),
2) smoothed held-out NLL (raw and volume-corrected) on a fixed (r, tau) grid.

This script is standalone and does not modify existing code.

Notes on scale interpretability
-------------------------------
- Spatial and temporal domains are inferred from the run configuration:
  spatial domain [spatial_min, spatial_max] x [spatial_min, spatial_max],
  temporal domain [0, t_end].
- r and tau are interpreted in these native units.
- Grid ranges are validated and warnings are printed if they exceed half-range.
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
from torch.utils.data import DataLoader

# Keep matplotlib non-interactive and writable cache for headless runs.
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
from unified_stpp.models import IntensityEvaluator
from unified_stpp.models.dynamics.identity import IdentityDynamics
from unified_stpp.registry import build_model
from unified_stpp.training import Trainer
from unified_stpp.training.data_module import STPPDataModule


EPS = 1e-12
KNOWN_PRESETS = ("neural_stpp", "deep_stpp", "dstpp")
DEFAULT_R_GRID = "0.2,0.5,1.0"
DEFAULT_TAU_GRID = f"{5/60:.10f},{10/60:.10f},{20/60:.10f}"


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
        description="Evaluate saved STPP model: held-out NLL vs smoothed NLL."
    )
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r_grid", type=str, default=DEFAULT_R_GRID)
    parser.add_argument("--tau_grid", type=str, default=DEFAULT_TAU_GRID)
    parser.add_argument("--Kt", type=int, default=11)
    parser.add_argument("--Ks", type=int, default=128)
    parser.add_argument("--plot_times", type=str, default="2,5,8")
    parser.add_argument("--plot_grid_n", type=int, default=120)
    parser.add_argument("--ref_seq_idx", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/exp_saved_model_ll_vs_smoothed",
    )
    return parser.parse_args()


def _parse_float_grid(spec: str, name: str) -> np.ndarray:
    spec = str(spec).strip()
    if not spec:
        raise ValueError(f"--{name} must not be empty.")
    if spec.startswith("linspace:"):
        p = spec.split(":")
        if len(p) != 4:
            raise ValueError(f"--{name} linspace format must be linspace:a:b:n")
        a = float(p[1])
        b = float(p[2])
        n = int(p[3])
        if n <= 0:
            raise ValueError(f"--{name} n must be > 0.")
        vals = np.linspace(a, b, n, dtype=np.float64)
    else:
        vals = np.asarray([float(x.strip()) for x in spec.split(",") if x.strip()], dtype=np.float64)
    if vals.size == 0:
        raise ValueError(f"--{name} parsed to empty.")
    if np.any(~np.isfinite(vals)):
        raise ValueError(f"--{name} contains non-finite values.")
    if np.any(vals <= 0):
        raise ValueError(f"--{name} values must be > 0.")
    vals = np.unique(vals)
    vals.sort()
    return vals


def _parse_plot_times(spec: str) -> List[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError("--plot_times must contain at least one value.")
    return vals


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
    m = re.match(r"^(?P<body>.+)_\d{8}_\d{6}_[0-9a-fA-F]{6,12}$", run_id)
    if not m:
        return None, None
    body = m.group("body")
    for preset in sorted(KNOWN_PRESETS, key=len, reverse=True):
        suffix = "_" + preset
        if body.endswith(suffix):
            return preset, body[: -len(suffix)] if body[: -len(suffix)] else None
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
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

    scored: List[Tuple[float, Path]] = []
    for ck in ckpts:
        v = _parse_ckpt_val(ck)
        if v is not None:
            scored.append((v, ck))
    if scored:
        scored.sort(key=lambda x: x[0])
        return scored[0][1]
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        d = yaml.safe_load(f)
    return d if isinstance(d, dict) else {}


def _candidate_config_paths(preset: Optional[str]) -> List[Path]:
    if not preset:
        return []
    out = []
    for n in (f"{preset}_lightning.yaml", f"{preset}.yaml"):
        out.append(Path("configs") / n)
        out.append(Path("unified_stpp") / "configs" / n)
    return out


def _default_settings() -> Dict[str, Any]:
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
    m = cfg.get("model", {})
    t = cfg.get("training", {})
    d = cfg.get("data", {})
    if not isinstance(m, dict):
        m = {}
    if not isinstance(t, dict):
        t = {}
    if not isinstance(d, dict):
        d = {}

    settings["preset"] = m.get("preset", settings["preset"])
    settings["hidden_dim"] = int(m.get("hidden_dim", settings["hidden_dim"]))
    settings["spatial_dim"] = int(m.get("spatial_dim", settings["spatial_dim"]))
    settings["event_cov_dim"] = int(m.get("event_cov_dim", settings["event_cov_dim"]))
    settings["field_cov_dim"] = int(m.get("field_cov_dim", settings["field_cov_dim"]))
    settings["n_marks"] = int(m.get("n_marks", settings["n_marks"]))
    settings["batch_size"] = int(t.get("batch_size", settings["batch_size"]))

    settings["data_type"] = d.get("type", settings["data_type"])
    settings["n_train"] = int(d.get("n_train", settings["n_train"]))
    settings["n_val"] = int(d.get("n_val", settings["n_val"]))
    settings["t_end"] = float(d.get("T", d.get("t_end", settings["t_end"])))
    settings["base_rate"] = float(d.get("base_rate", settings["base_rate"]))
    settings["data_seed"] = int(d.get("seed", settings["data_seed"]))
    settings["dataset_cache"] = d.get("dataset_cache", settings["dataset_cache"])
    settings["data_covariate_dim"] = d.get("covariate_dim", settings["data_covariate_dim"])
    settings["spatial_min"] = float(d.get("spatial_min", settings["spatial_min"]))
    settings["spatial_max"] = float(d.get("spatial_max", settings["spatial_max"]))

    settings["sthp_alpha"] = float(d.get("sthp_alpha", settings["sthp_alpha"]))
    settings["sthp_beta"] = float(d.get("sthp_beta", settings["sthp_beta"]))
    settings["sthp_mu"] = float(d.get("sthp_mu", settings["sthp_mu"]))
    settings["hotspot_sigma"] = float(d.get("hotspot_sigma", settings["hotspot_sigma"]))
    settings["hotspot_switch_frac"] = float(d.get("hotspot_switch_frac", settings["hotspot_switch_frac"]))
    settings["hotspot_switch_time"] = d.get("hotspot_switch_time", settings["hotspot_switch_time"])
    settings["hotspot_move_duration"] = d.get("hotspot_move_duration", settings["hotspot_move_duration"])
    settings["hotspot_t1_frac"] = float(d.get("hotspot_t1_frac", settings["hotspot_t1_frac"]))
    settings["hotspot_t2_frac"] = float(d.get("hotspot_t2_frac", settings["hotspot_t2_frac"]))
    settings["hotspot_jitter_radius"] = float(d.get("hotspot_jitter_radius", settings["hotspot_jitter_radius"]))
    settings["hotspot_jitter_f1"] = float(d.get("hotspot_jitter_f1", settings["hotspot_jitter_f1"]))
    settings["hotspot_jitter_f2"] = float(d.get("hotspot_jitter_f2", settings["hotspot_jitter_f2"]))
    settings["hotspot_amp0"] = float(d.get("hotspot_amp0", settings["hotspot_amp0"]))
    settings["hotspot_amp1"] = float(d.get("hotspot_amp1", settings["hotspot_amp1"]))
    settings["hotspot_amp_noise"] = float(d.get("hotspot_amp_noise", settings["hotspot_amp_noise"]))
    settings["hotspot_noise_knots"] = int(d.get("hotspot_noise_knots", settings["hotspot_noise_knots"]))
    settings["hotspot_weight"] = float(d.get("hotspot_weight", settings["hotspot_weight"]))
    settings["hotspot_interaction_weight"] = float(d.get("hotspot_interaction_weight", settings["hotspot_interaction_weight"]))
    settings["hotspot_tod_weight"] = float(d.get("hotspot_tod_weight", settings["hotspot_tod_weight"]))

    settings["rg_num_marks"] = int(d.get("rg_num_marks", settings["rg_num_marks"]))
    settings["rg_num_regimes"] = int(d.get("rg_num_regimes", settings["rg_num_regimes"]))
    settings["rg_env_dim"] = int(d.get("rg_env_dim", settings["rg_env_dim"]))
    settings["rg_sigma_reg"] = float(d.get("rg_sigma_reg", settings["rg_sigma_reg"]))
    settings["rg_switch_rate"] = float(d.get("rg_switch_rate", settings["rg_switch_rate"]))
    settings["rg_sigma_hotspot"] = float(d.get("rg_sigma_hotspot", settings["rg_sigma_hotspot"]))
    settings["rg_lambda_bg"] = float(d.get("rg_lambda_bg", settings["rg_lambda_bg"]))
    settings["rg_tau_t"] = float(d.get("rg_tau_t", settings["rg_tau_t"]))
    settings["rg_sigma_exc"] = float(d.get("rg_sigma_exc", settings["rg_sigma_exc"]))
    settings["rg_max_events_per_seq"] = int(d.get("rg_max_events_per_seq", settings["rg_max_events_per_seq"]))

    settings["pw_num_arms"] = int(d.get("pw_num_arms", settings["pw_num_arms"]))
    settings["pw_mu"] = float(d.get("pw_mu", settings["pw_mu"]))
    settings["pw_alpha"] = float(d.get("pw_alpha", settings["pw_alpha"]))
    settings["pw_omega"] = float(d.get("pw_omega", settings["pw_omega"]))

    settings["config_path_used"] = str(cfg_path)
    return settings


def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        out[k[6:] if k.startswith("model.") else k] = v
    return out


def _infer_preset_from_state_keys(keys: Sequence[str]) -> str:
    ks = set(keys)
    if any(k.startswith("decoder.temporal.param_net.") for k in ks):
        return "deep_stpp"
    if any(k.startswith("decoder.temporal.intensity_net.") for k in ks):
        return "neural_stpp"
    if any("decoder.score_net" in k for k in ks):
        return "dstpp"
    return "deep_stpp"


def _infer_model_spec(run_preset: Optional[str], stripped_sd: Dict[str, torch.Tensor]) -> ModelSpec:
    keys = list(stripped_sd.keys())
    preset = run_preset or _infer_preset_from_state_keys(keys)
    if "encoder.embed.weight" not in stripped_sd:
        raise KeyError("Missing encoder.embed.weight in checkpoint.")

    enc_w = stripped_sd["encoder.embed.weight"]
    hidden_dim = int(enc_w.shape[0])
    enc_in = int(enc_w.shape[1])

    field_cov_dim = 0
    if "decoder.temporal.param_net.0.weight" in stripped_sd:
        field_cov_dim = int(stripped_sd["decoder.temporal.param_net.0.weight"].shape[1] - hidden_dim)
    elif "decoder.temporal.intensity_net.0.weight" in stripped_sd:
        field_cov_dim = int(stripped_sd["decoder.temporal.intensity_net.0.weight"].shape[1] - hidden_dim - 1)
    field_cov_dim = max(field_cov_dim, 0)

    spatial_dim = 2
    temporal_components = None
    spatial_components = None
    if (
        "decoder.temporal.param_net.2.weight" in stripped_sd
        and "decoder.spatial.param_net.2.weight" in stripped_sd
    ):
        out_t = int(stripped_sd["decoder.temporal.param_net.2.weight"].shape[0])
        out_s = int(stripped_sd["decoder.spatial.param_net.2.weight"].shape[0])
        if out_t % 3 == 0:
            temporal_components = out_t // 3
            if temporal_components > 0 and out_s % temporal_components == 0:
                cand = out_s // temporal_components - 2
                if cand > 0:
                    spatial_dim = cand
                    spatial_components = out_s // (spatial_dim + 2)
    elif "decoder.spatial.velocity.net.0.weight" in stripped_sd:
        inp = int(stripped_sd["decoder.spatial.velocity.net.0.weight"].shape[1])
        cand = inp - 1 - hidden_dim - field_cov_dim
        if cand > 0:
            spatial_dim = cand

    n_marks = 0
    mark_embed_dim = 0
    if "mark_embedding.embedding.weight" in stripped_sd:
        emb = stripped_sd["mark_embedding.embedding.weight"]
        n_marks = int(emb.shape[0])
        mark_embed_dim = int(emb.shape[1])

    effective_cov = enc_in - (1 + spatial_dim)
    if effective_cov < 0:
        effective_cov = 0
    event_cov_dim = max(0, effective_cov - mark_embed_dim)

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


def _resolve_settings(run_id: str, model_spec: ModelSpec, run_data_type: Optional[str]) -> Dict[str, Any]:
    settings = _default_settings()
    settings["preset"] = model_spec.preset
    if run_data_type:
        settings["data_type"] = run_data_type

    for p in _candidate_config_paths(model_spec.preset):
        if p.exists():
            settings = _apply_config(settings, _load_yaml(p), p)
            break

    settings["preset"] = model_spec.preset
    settings["hidden_dim"] = model_spec.hidden_dim
    settings["spatial_dim"] = model_spec.spatial_dim
    settings["event_cov_dim"] = model_spec.event_cov_dim
    settings["field_cov_dim"] = model_spec.field_cov_dim
    settings["n_marks"] = model_spec.n_marks
    if run_data_type:
        settings["data_type"] = run_data_type

    parsed_preset, parsed_data = _parse_run_id(run_id)
    if parsed_data:
        settings["data_type"] = parsed_data
    if parsed_preset:
        settings["preset"] = parsed_preset
    return settings


def _build_model_from_checkpoint(ckpt_path: Path, model_spec: ModelSpec) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ck.get("state_dict", ck)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")
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
            f"Missing: {missing[:8]}{'...' if len(missing) > 8 else ''}\n"
            f"Unexpected: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )
    model.eval()
    return model, ck


def _generate_sequences(settings: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data_type = settings["data_type"]
    n_train = int(settings["n_train"])
    n_val = int(settings["n_val"])
    total = n_train + n_val
    t_end = float(settings["t_end"])
    spatial_dim = int(settings["spatial_dim"])
    bounds = (float(settings["spatial_min"]), float(settings["spatial_max"]))
    data_seed = int(settings["data_seed"])
    field_cov_dim = int(settings["field_cov_dim"])
    data_cov_dim = settings["data_covariate_dim"]
    dataset_cache = settings.get("dataset_cache")

    default_cov = max(1, field_cov_dim)
    if data_type == "moving_hotspot":
        default_cov = 4
    elif data_type == "regime_gated_hawkes":
        default_cov = int(settings["rg_num_regimes"]) + 2 + int(settings["rg_env_dim"])
    cov_dim = int(default_cov if data_cov_dim is None else max(1, int(data_cov_dim)))

    all_seqs: Optional[List[Dict[str, Any]]] = None
    if dataset_cache and Path(dataset_cache).exists():
        with open(dataset_cache, "rb") as f:
            all_seqs = pickle.load(f)
        if len(all_seqs) < total:
            raise ValueError(f"Cached dataset has {len(all_seqs)} but needs {total}.")

    if all_seqs is None:
        if data_type == "hawkes":
            all_seqs = generate_hawkes_stpp(total, T=t_end, spatial_bounds=bounds, spatial_dim=spatial_dim, seed=data_seed)
        elif data_type == "inhomogeneous":
            all_seqs = generate_inhomogeneous_stpp(
                total,
                T=t_end,
                spatial_bounds=bounds,
                spatial_dim=spatial_dim,
                base_rate=float(settings["base_rate"]),
                covariate_dim=cov_dim,
                seed=data_seed,
            )
        elif data_type == "inhomogeneous_class":
            g = InhomogeneousPoissonSyntheticDataset(
                spatial_dim=spatial_dim,
                base_rate=float(settings["base_rate"]),
                covariate_dim=cov_dim,
                seed=data_seed,
            )
            all_seqs = g.generate_sequences(total, t_start=0.0, t_end=t_end)
        elif data_type == "sthp_class":
            if spatial_dim != 2:
                raise ValueError("sthp_class requires spatial_dim=2.")
            g = STHPDataset(
                s_mu=[0.0, 0.0],
                g0_cov=[[1.0, 0.0], [0.0, 1.0]],
                g2_cov=[[1.0, 0.0], [0.0, 1.0]],
                alpha=float(settings["sthp_alpha"]),
                beta=float(settings["sthp_beta"]),
                mu=float(settings["sthp_mu"]),
                seed=data_seed,
            )
            all_seqs = g.generate_sequences(total, t_start=0.0, t_end=t_end)
        elif data_type == "moving_hotspot":
            all_seqs = generate_moving_hotspot_stpp(
                total,
                T=t_end,
                spatial_bounds=bounds,
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
                total,
                T=t_end,
                spatial_bounds=bounds,
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
                total,
                T=t_end,
                spatial_bounds=bounds,
                spatial_dim=spatial_dim,
                n_marks=max(int(settings["n_marks"]), 3),
                seed=data_seed,
            )
        elif data_type == "pinwheel":
            all_seqs = generate_pinwheel_hawkes_stpp(
                total,
                T=t_end,
                num_arms=int(settings["pw_num_arms"]),
                mu_per_arm=float(settings["pw_mu"]),
                alpha_offdiag=float(settings["pw_alpha"]),
                omega=float(settings["pw_omega"]),
                seed=data_seed,
            )
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    if int(settings["field_cov_dim"]) == 0:
        for s in all_seqs:
            s.pop("field_covariates", None)

    return all_seqs[:n_train], all_seqs[n_train : n_train + n_val]


def _build_data(train_seqs: List[Dict[str, Any]], test_seqs: List[Dict[str, Any]], batch_size: int) -> Tuple[STPPDataModule, Any, Any]:
    dm = STPPDataModule(train_seqs, test_seqs, batch_size=batch_size, num_workers=0)
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


def _positive_intensity(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    # If a decoder returns log-intensity-like values, exponentiate defensively.
    if np.all(vals <= 0.0):
        vals = np.exp(np.clip(vals, -50.0, 50.0))
    return np.maximum(vals, EPS)


def _eval_intensity(
    evaluator: IntensityEvaluator,
    t_norm: np.ndarray,
    s_norm: np.ndarray,
    x_field_const_norm: Optional[np.ndarray],
) -> np.ndarray:
    t_t = torch.as_tensor(t_norm, dtype=torch.float32).reshape(-1, 1)
    s_t = torch.as_tensor(s_norm, dtype=torch.float32)
    x_t = None
    if x_field_const_norm is not None:
        xf = np.repeat(np.asarray(x_field_const_norm, dtype=np.float32).reshape(1, -1), t_t.shape[0], axis=0)
        x_t = torch.as_tensor(xf, dtype=torch.float32)
    with torch.no_grad():
        lam = evaluator.intensity(t_t, s_t, x_field=x_t).detach().cpu().numpy()
    return _positive_intensity(lam)


def _build_encoder_x_event(
    model: torch.nn.Module,
    times: torch.Tensor,
    locations: torch.Tensor,
    marks: Optional[torch.Tensor],
    x_event: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    x_hist = x_event
    if getattr(model, "mark_embedding", None) is not None:
        bsz, n = times.shape
        if marks is not None:
            x_mark = model.mark_embedding(marks.long())
        else:
            x_mark = torch.zeros(
                bsz,
                n,
                model.mark_embedding.embed_dim,
                dtype=times.dtype,
                device=times.device,
            )
        x_hist = x_mark if x_hist is None else torch.cat([x_hist, x_mark], dim=-1)

    expected = int(model.encoder.embed.in_features - (1 + locations.shape[-1]))
    if expected <= 0:
        return None
    if x_hist is None:
        return torch.zeros(times.shape[0], times.shape[1], expected, dtype=times.dtype, device=times.device)
    cur = int(x_hist.shape[-1])
    if cur == expected:
        return x_hist
    if cur > expected:
        return x_hist[..., :expected]
    pad = torch.zeros(times.shape[0], times.shape[1], expected - cur, dtype=x_hist.dtype, device=x_hist.device)
    return torch.cat([x_hist, pad], dim=-1)


def _eventwise_terms(model: torch.nn.Module, seq_item: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
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

    x_enc = _build_encoder_x_event(model, times, locs, marks, x_event)
    events = torch.cat([times.unsqueeze(-1), locs], dim=-1)
    with torch.no_grad():
        _, all_states = model.encoder(events, lengths, x_event=x_enc)

    rows: List[Dict[str, Any]] = []
    if length < 2:
        return rows

    for n in range(length - 1):
        z_n = all_states[:, n, :]
        t_prev = times[:, n : n + 1]
        t_tgt = times[:, n + 1 : n + 2]
        s_tgt = locs[:, n + 1, :]

        x_field_dec = x_field[:, n, :] if x_field is not None else None
        if isinstance(model.dynamics, IdentityDynamics):
            z_t = z_n
        else:
            dt = (t_tgt - t_prev).clamp(min=1e-6)
            x_dyn = x_field_dec.unsqueeze(1) if x_field_dec is not None else None
            z_t = model.dynamics(z_n, dt, x_field=x_dyn).squeeze(1)
            cached = getattr(model.dynamics, "_cached_Lambda", None)
            if cached is not None:
                temporal = getattr(model.decoder, "temporal", None)
                if temporal is not None and hasattr(temporal, "_precomputed_lambda"):
                    temporal._precomputed_lambda = cached[:, 0]

        with torch.no_grad():
            ground_nll = model.decoder.nll(z_t, t_tgt, s_tgt, t_prev, x_field=x_field_dec).squeeze(0)
            total_nll = ground_nll
            mark_nll_val = torch.tensor(0.0, dtype=ground_nll.dtype)
            if getattr(model, "mark_decoder", None) is not None and marks is not None:
                k_tgt = marks[:, n + 1]
                mark_nll_val = model.mark_decoder.nll(z_t, t_tgt, s_tgt, k_tgt, x_field=x_field_dec).squeeze(0)
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
            }
        )
    return rows


def _disk_samples(radius: float, Ks: int) -> np.ndarray:
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    pts = np.zeros((Ks, 2), dtype=np.float64)
    for i in range(Ks):
        rho = radius * math.sqrt((i + 0.5) / Ks)
        ang = 2.0 * math.pi * ((i * phi) % 1.0)
        pts[i, 0] = rho * math.cos(ang)
        pts[i, 1] = rho * math.sin(ang)
    return pts


def _local_smoothed_mass(
    evaluator: IntensityEvaluator,
    t_center_raw: float,
    s_center_raw: np.ndarray,
    t_prev_raw: float,
    t_horizon_raw: float,
    radius: float,
    tau: float,
    stats: Dict[str, np.ndarray],
    x_field_prev_norm: Optional[np.ndarray],
    Kt: int,
    Ks: int,
) -> Tuple[float, float]:
    t_lo = max(float(t_center_raw - tau), float(t_prev_raw + 1e-6), 0.0)
    t_hi = min(float(t_center_raw + tau), float(t_horizon_raw))
    if t_hi <= t_lo:
        return EPS, EPS

    t_grid = np.linspace(t_lo, t_hi, Kt, dtype=np.float64)
    offsets = _disk_samples(radius, Ks)
    area = math.pi * radius * radius
    jac = 1.0 / (float(stats["time_std"]) * float(np.prod(stats["loc_std"])))

    spatial_vals = np.zeros(Kt, dtype=np.float64)
    for i, t_raw in enumerate(t_grid):
        s_raw = s_center_raw.reshape(1, 2) + offsets
        t_norm = _to_norm_time(np.full(Ks, t_raw, dtype=np.float64), stats)
        s_norm = _to_norm_loc(s_raw, stats)
        lam = _eval_intensity(evaluator, t_norm, s_norm, x_field_prev_norm)
        spatial_vals[i] = (area / Ks) * float(np.sum(lam))

    if Kt == 1:
        t_measure = (t_hi - t_lo)
        mass = jac * spatial_vals[0] * t_measure
    else:
        ones = np.ones_like(t_grid)
        t_measure = float(np.trapezoid(ones, t_grid))
        mass = jac * float(np.trapezoid(spatial_vals, t_grid))

    volume = area * max(t_measure, 0.0)
    return float(max(mass, EPS)), float(max(volume, EPS))


def compute_smoothed_nll(
    model: torch.nn.Module,
    test_dataset: Any,
    stats: Dict[str, np.ndarray],
    settings: Dict[str, Any],
    r: float,
    tau: float,
    Kt: int,
    Ks: int,
) -> Tuple[float, float, Dict[str, Any]]:
    raw_terms: List[float] = []
    volcorr_terms: List[float] = []
    volumes: List[float] = []
    baseline_terms: List[float] = []

    t_horizon = float(settings["t_end"])
    for idx in range(len(test_dataset)):
        seq_item = test_dataset[idx]
        seq_raw = test_dataset.sequences[idx]
        raw_times = np.asarray(seq_raw["times"], dtype=np.float64)
        raw_locs = np.asarray(seq_raw["locations"], dtype=np.float64)
        if raw_locs.ndim != 2 or raw_locs.shape[1] != 2:
            raise ValueError(f"Expected 2D locations for smoothing; got {raw_locs.shape}.")

        rows = _eventwise_terms(model, seq_item)
        for row in rows:
            n = int(row["step"])
            z_n = row["z_n"].unsqueeze(0)
            t_prev_norm = torch.tensor([[row["t_prev_norm"]]], dtype=torch.float32)
            evaluator = IntensityEvaluator(model, z=z_n, t_prev=t_prev_norm)

            lam_point = _eval_intensity(
                evaluator,
                t_norm=np.array([row["t_target_norm"]], dtype=np.float64),
                s_norm=np.asarray(row["s_target_norm"], dtype=np.float64).reshape(1, -1),
                x_field_const_norm=row["x_field_prev_norm"],
            )[0]

            mass, vol = _local_smoothed_mass(
                evaluator=evaluator,
                t_center_raw=float(raw_times[n + 1]),
                s_center_raw=raw_locs[n + 1],
                t_prev_raw=float(raw_times[n]),
                t_horizon_raw=t_horizon,
                radius=r,
                tau=tau,
                stats=stats,
                x_field_prev_norm=row["x_field_prev_norm"],
                Kt=Kt,
                Ks=Ks,
            )

            baseline_total = float(row["baseline_total_nll"])
            remainder = baseline_total + math.log(max(lam_point, EPS))
            term_raw = remainder - math.log(max(mass, EPS))
            term_vol = term_raw + math.log(max(vol, EPS))
            raw_terms.append(term_raw)
            volcorr_terms.append(term_vol)
            volumes.append(vol)
            baseline_terms.append(baseline_total)

    if not raw_terms:
        raise RuntimeError("No held-out events available for smoothed NLL.")

    vols = np.asarray(volumes, dtype=np.float64)
    nominal = float((2.0 * tau) * (math.pi * r * r))
    details = {
        "event_count": int(len(raw_terms)),
        "baseline_mean_from_event_terms": float(np.mean(baseline_terms)),
        "volume_used": {
            "integration": "quadrature_weight_sum",
            "units": "native_model_units",
            "nominal_analytic_volume": nominal,
            "mean": float(np.mean(vols)),
            "std": float(np.std(vols)),
            "min": float(np.min(vols)),
            "max": float(np.max(vols)),
            "clipped_event_count": int(np.sum(vols < (nominal - 1e-12))),
        },
    }
    return float(np.mean(raw_terms)), float(np.mean(volcorr_terms)), details


def _build_evaluator_for_time(
    model: torch.nn.Module,
    seq_item: Dict[str, torch.Tensor],
    seq_raw: Dict[str, Any],
    t_raw: float,
) -> Tuple[Optional[IntensityEvaluator], Optional[np.ndarray]]:
    raw_times = np.asarray(seq_raw["times"], dtype=np.float64)
    if raw_times.size == 0:
        return None, None

    n_hist = int(np.sum(raw_times < t_raw))
    n_hist = max(1, min(n_hist, int(seq_item["length"])))

    h_times = seq_item["times"][:n_hist].unsqueeze(0).float()
    h_locs = seq_item["locations"][:n_hist].unsqueeze(0).float()
    h_lens = torch.tensor([n_hist], dtype=torch.long)

    h_marks = None
    if seq_item.get("marks") is not None:
        h_marks = seq_item["marks"][:n_hist].unsqueeze(0).long()
    h_x_event = None
    if seq_item.get("event_covariates") is not None:
        h_x_event = seq_item["event_covariates"][:n_hist].unsqueeze(0).float()
    h_x_field = None
    if seq_item.get("field_covariates") is not None:
        h_x_field = seq_item["field_covariates"][:n_hist].unsqueeze(0).float()

    x_enc = _build_encoder_x_event(model, h_times, h_locs, h_marks, h_x_event)
    events = torch.cat([h_times.unsqueeze(-1), h_locs], dim=-1)
    with torch.no_grad():
        _, all_states = model.encoder(events, h_lens, x_event=x_enc)
    z = all_states[:, n_hist - 1, :]
    t_prev = h_times[:, n_hist - 1 : n_hist]
    evaluator = IntensityEvaluator(model, z=z, t_prev=t_prev)

    x_field_prev = None
    if h_x_field is not None and h_x_field.shape[1] >= n_hist:
        x_field_prev = h_x_field[:, n_hist - 1, :].squeeze(0).detach().cpu().numpy().astype(np.float64)
    elif getattr(model, "decoder", None) is not None and hasattr(model.decoder, "temporal"):
        # If model expects field covariates but none exist, keep None (as training would).
        x_field_prev = None
    return evaluator, x_field_prev


def _plot_scatter_heldout(
    out_path: Path,
    heldout_sequences: List[Dict[str, Any]],
    spatial_min: float,
    spatial_max: float,
) -> None:
    import matplotlib.pyplot as plt

    ts = []
    xs = []
    ys = []
    for s in heldout_sequences:
        t = np.asarray(s["times"], dtype=np.float64)
        loc = np.asarray(s["locations"], dtype=np.float64)
        if t.size == 0:
            continue
        ts.append(t)
        xs.append(loc[:, 0])
        ys.append(loc[:, 1])

    fig, ax = plt.subplots(figsize=(8, 7))
    if ts:
        t = np.concatenate(ts)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        sc = ax.scatter(x, y, c=t, s=8, alpha=0.45, cmap="viridis")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("time (native units)")
    ax.set_xlim(spatial_min, spatial_max)
    ax.set_ylim(spatial_min, spatial_max)
    ax.set_xlabel("x (native space units)")
    ax.set_ylabel("y (native space units)")
    ax.set_title("Held-out Events Scatter")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_intensity_slices(
    out_path: Path,
    model: torch.nn.Module,
    test_dataset: Any,
    stats: Dict[str, np.ndarray],
    t_slices_raw: List[float],
    spatial_min: float,
    spatial_max: float,
    ref_seq_idx: int,
    grid_n: int,
) -> None:
    import matplotlib.pyplot as plt

    if len(test_dataset) == 0:
        raise RuntimeError("Held-out dataset is empty.")
    ref_idx = max(0, min(ref_seq_idx, len(test_dataset) - 1))
    seq_item = test_dataset[ref_idx]
    seq_raw = test_dataset.sequences[ref_idx]

    x = np.linspace(spatial_min, spatial_max, grid_n)
    y = np.linspace(spatial_min, spatial_max, grid_n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    s_raw = np.stack([xx.ravel(), yy.ravel()], axis=1)

    fig, axes = plt.subplots(1, len(t_slices_raw), figsize=(5 * len(t_slices_raw), 4.5))
    if len(t_slices_raw) == 1:
        axes = [axes]

    vmin = None
    vmax = None
    maps: List[np.ndarray] = []
    for t_raw in t_slices_raw:
        evaluator, x_field_prev = _build_evaluator_for_time(model, seq_item, seq_raw, t_raw=t_raw)
        if evaluator is None:
            lam_map = np.zeros_like(xx, dtype=np.float64)
        else:
            t_norm = _to_norm_time(np.full(s_raw.shape[0], t_raw, dtype=np.float64), stats)
            s_norm = _to_norm_loc(s_raw, stats)
            lam = _eval_intensity(evaluator, t_norm=t_norm, s_norm=s_norm, x_field_const_norm=x_field_prev)
            lam_map = lam.reshape(xx.shape)
        maps.append(lam_map)
        lvmin = float(np.min(lam_map))
        lvmax = float(np.max(lam_map))
        vmin = lvmin if vmin is None else min(vmin, lvmin)
        vmax = lvmax if vmax is None else max(vmax, lvmax)

    for i, (t_raw, lam_map) in enumerate(zip(t_slices_raw, maps)):
        im = axes[i].imshow(
            lam_map,
            origin="lower",
            extent=[spatial_min, spatial_max, spatial_min, spatial_max],
            aspect="equal",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        axes[i].set_title(f"fitted intensity @ t={t_raw:.3g}")
        axes[i].set_xlabel("x (native space units)")
        axes[i].set_ylabel("y (native space units)")
        cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label("intensity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_smoothed_grid(
    out_path: Path,
    r_vals: np.ndarray,
    tau_vals: np.ndarray,
    nll_volcorr: np.ndarray,
    native_time_label: str = "native time units",
    native_space_label: str = "native space units",
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    im = axes[0].imshow(nll_volcorr, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_xticks(np.arange(len(r_vals)))
    axes[0].set_yticks(np.arange(len(tau_vals)))
    axes[0].set_xticklabels([f"{r:.3g}" for r in r_vals])
    axes[0].set_yticklabels([f"{t:.3g}" for t in tau_vals])
    axes[0].set_xlabel(f"r ({native_space_label})")
    axes[0].set_ylabel(f"tau ({native_time_label})")
    axes[0].set_title("Smoothed NLL (vol-corrected)")
    cbar = fig.colorbar(im, ax=axes[0])
    cbar.set_label("NLL per event")
    idx = np.unravel_index(np.nanargmin(nll_volcorr), nll_volcorr.shape)
    axes[0].scatter([idx[1]], [idx[0]], marker="x", s=70, c="red")
    axes[0].annotate(f"min={nll_volcorr[idx]:.4g}", (idx[1], idx[0]), xytext=(8, 6), textcoords="offset points", color="red")

    for i, tau in enumerate(tau_vals):
        axes[1].plot(r_vals, nll_volcorr[i], marker="o", lw=2, label=f"tau={tau:.3g}")
    axes[1].set_xlabel(f"r ({native_space_label})")
    axes[1].set_ylabel("smoothed NLL (vol-corrected)")
    axes[1].set_title("Smoothed NLL vs r")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_smoothed_raw_heatmap(
    out_path: Path,
    r_vals: np.ndarray,
    tau_vals: np.ndarray,
    nll_raw: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.imshow(nll_raw, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(r_vals)))
    ax.set_yticks(np.arange(len(tau_vals)))
    ax.set_xticklabels([f"{r:.3g}" for r in r_vals])
    ax.set_yticklabels([f"{t:.3g}" for t in tau_vals])
    ax.set_xlabel("r (native space units)")
    ax.set_ylabel("tau (native time units)")
    ax.set_title("Smoothed NLL (raw)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("NLL per event")
    idx = np.unravel_index(np.nanargmin(nll_raw), nll_raw.shape)
    ax.scatter([idx[1]], [idx[0]], marker="x", s=70, c="red")
    ax.annotate(f"min={nll_raw[idx]:.4g}", (idx[1], idx[0]), xytext=(8, 6), textcoords="offset points", color="red")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    r_vals = _parse_float_grid(args.r_grid, "r_grid")
    tau_vals = _parse_float_grid(args.tau_grid, "tau_grid")
    t_slices = _parse_plot_times(args.plot_times)
    if args.Kt <= 2 or args.Ks <= 8:
        raise ValueError("Kt and Ks are too small for stable smoothed quadrature.")

    run_preset, run_data_type = _parse_run_id(args.run_id)
    logs_dir = _find_logs_dir(args.run_id)
    ckpt_path = _select_checkpoint(args.run_id)

    raw_ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw_ck.get("state_dict", raw_ck)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint structure: {ckpt_path}")
    model_spec = _infer_model_spec(run_preset, _strip_state_dict_prefix(state))
    settings = _resolve_settings(args.run_id, model_spec, run_data_type)
    model, ckpt = _build_model_from_checkpoint(ckpt_path, model_spec)

    spatial_min = float(settings["spatial_min"])
    spatial_max = float(settings["spatial_max"])
    t_end = float(settings["t_end"])
    spatial_range = spatial_max - spatial_min
    if spatial_range <= 0:
        raise ValueError(f"Invalid spatial range: [{spatial_min}, {spatial_max}]")
    if t_end <= 0:
        raise ValueError(f"Invalid t_end: {t_end}")

    if np.any(r_vals > 0.5 * spatial_range + 1e-12):
        print(
            f"Warning: some r values exceed half spatial range ({0.5 * spatial_range:.6g}). "
            "They are still evaluated in native units."
        )
    if np.any(tau_vals > 0.5 * t_end + 1e-12):
        print(
            f"Warning: some tau values exceed T/2 ({0.5 * t_end:.6g}). "
            "They are still evaluated in native units."
        )

    train_seqs, heldout_seqs = _generate_sequences(settings)
    dm, train_dataset, heldout_dataset = _build_data(
        train_seqs=train_seqs,
        test_seqs=heldout_seqs,
        batch_size=int(settings["batch_size"]),
    )
    _ = dm
    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    stats = _stats_from_train_dataset(train_dataset)

    evaluator = Trainer(model=model, lr=1e-3, weight_decay=0.0, grad_clip=0.0, device="cpu")
    heldout_metrics = evaluator.evaluate(heldout_loader)
    heldout_nll = float(heldout_metrics["nll"])

    n_tau = len(tau_vals)
    n_r = len(r_vals)
    sm_raw = np.zeros((n_tau, n_r), dtype=np.float64)
    sm_vol = np.zeros((n_tau, n_r), dtype=np.float64)
    volume_stats: Dict[str, Dict[str, Any]] = {}

    for i_tau, tau in enumerate(tau_vals):
        for j_r, r in enumerate(r_vals):
            raw_nll, vol_nll, details = compute_smoothed_nll(
                model=model,
                test_dataset=heldout_dataset,
                stats=stats,
                settings=settings,
                r=float(r),
                tau=float(tau),
                Kt=int(args.Kt),
                Ks=int(args.Ks),
            )
            sm_raw[i_tau, j_r] = raw_nll
            sm_vol[i_tau, j_r] = vol_nll
            volume_stats[f"tau={tau:.10g}|r={r:.10g}"] = details["volume_used"]

    out_dir = Path(args.output_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    p_scatter = out_dir / "heldout_events_scatter.png"
    p_intensity = out_dir / "intensity_slices_fitted.png"
    p_grid = out_dir / "smoothed_nll_volcorr_grid_and_rsweep.png"
    p_grid_raw = out_dir / "smoothed_nll_raw_heatmap.png"

    _plot_scatter_heldout(
        out_path=p_scatter,
        heldout_sequences=heldout_dataset.sequences,
        spatial_min=spatial_min,
        spatial_max=spatial_max,
    )
    t_slices_clamped = [min(max(float(t), 0.0), t_end) for t in t_slices]
    _plot_intensity_slices(
        out_path=p_intensity,
        model=model,
        test_dataset=heldout_dataset,
        stats=stats,
        t_slices_raw=t_slices_clamped,
        spatial_min=spatial_min,
        spatial_max=spatial_max,
        ref_seq_idx=int(args.ref_seq_idx),
        grid_n=int(args.plot_grid_n),
    )
    _plot_smoothed_grid(
        out_path=p_grid,
        r_vals=r_vals,
        tau_vals=tau_vals,
        nll_volcorr=sm_vol,
    )
    _plot_smoothed_raw_heatmap(
        out_path=p_grid_raw,
        r_vals=r_vals,
        tau_vals=tau_vals,
        nll_raw=sm_raw,
    )

    assert np.isfinite(heldout_nll), "Held-out baseline NLL not finite."
    assert np.all(np.isfinite(sm_raw)), "Raw smoothed NLL has non-finite values."
    assert np.all(np.isfinite(sm_vol)), "Vol-corrected smoothed NLL has non-finite values."
    assert sm_raw.shape == (n_tau, n_r) and sm_vol.shape == (n_tau, n_r), "Grid shape mismatch."
    assert p_scatter.exists() and p_intensity.exists() and p_grid.exists() and p_grid_raw.exists(), "Plot save failed."

    min_raw = np.unravel_index(np.nanargmin(sm_raw), sm_raw.shape)
    min_vol = np.unravel_index(np.nanargmin(sm_vol), sm_vol.shape)

    print("")
    print("Saved-Model NLL vs Smoothed NLL")
    print("-------------------------------")
    print(f"run_id: {args.run_id}")
    print(f"checkpoint: {ckpt_path}")
    print(f"held-out split used: validation split (n_sequences={len(heldout_dataset)})")
    print(
        f"inferred domain: x,y in [{spatial_min:.6g}, {spatial_max:.6g}], "
        f"t in [0, {t_end:.6g}]"
    )
    print("r and tau are interpreted in the above native units.")
    print(f"held-out baseline NLL (per event): {heldout_nll:.6f}")
    print("")
    print("Smoothed held-out NLL (per event)")
    print("rows=tau, cols=r")
    print("tau values:", [float(x) for x in tau_vals.tolist()])
    print("r values  :", [float(x) for x in r_vals.tolist()])
    print("vol-corrected:")
    print(np.array2string(sm_vol, precision=6, suppress_small=False))
    print("raw:")
    print(np.array2string(sm_raw, precision=6, suppress_small=False))
    print(
        f"min vol-corrected at tau={tau_vals[min_vol[0]]:.6g}, r={r_vals[min_vol[1]]:.6g}, "
        f"value={sm_vol[min_vol]:.6f}"
    )
    print(
        f"min raw at tau={tau_vals[min_raw[0]]:.6g}, r={r_vals[min_raw[1]]:.6g}, "
        f"value={sm_raw[min_raw]:.6f}"
    )
    print(f"artifacts: {out_dir}")

    summary = {
        "run_id": args.run_id,
        "seed": int(args.seed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_short(),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else None,
        "logs_dir": str(logs_dir) if logs_dir is not None else None,
        "domain_native_units": {
            "spatial_min": spatial_min,
            "spatial_max": spatial_max,
            "temporal_min": 0.0,
            "temporal_max": t_end,
            "spatial_range": spatial_range,
            "temporal_range": t_end,
        },
        "model_info": {
            "preset": settings["preset"],
            "data_type": settings["data_type"],
            "spatial_dim": int(settings["spatial_dim"]),
            "event_cov_dim": int(settings["event_cov_dim"]),
            "field_cov_dim": int(settings["field_cov_dim"]),
            "n_marks": int(settings["n_marks"]),
        },
        "heldout_split": {
            "name": "validation_as_test",
            "n_sequences": int(len(heldout_dataset)),
            "n_events": int(heldout_metrics.get("n_events", 0)),
        },
        "quadrature": {
            "Kt": int(args.Kt),
            "Ks": int(args.Ks),
            "smoothing_integration": "temporal_trapezoid + spatial_disk_mean",
        },
        "baseline_nll_per_event": heldout_nll,
        "smoothed_nll_raw": sm_raw.tolist(),
        "smoothed_nll_volcorr": sm_vol.tolist(),
        "grids": {
            "r_values": r_vals.tolist(),
            "tau_values": tau_vals.tolist(),
        },
        "minima": {
            "raw": {
                "tau": float(tau_vals[min_raw[0]]),
                "r": float(r_vals[min_raw[1]]),
                "value": float(sm_raw[min_raw]),
            },
            "volcorr": {
                "tau": float(tau_vals[min_vol[0]]),
                "r": float(r_vals[min_vol[1]]),
                "value": float(sm_vol[min_vol]),
            },
        },
        "volume_stats_by_grid": volume_stats,
        "artifacts": {
            "heldout_scatter": str(p_scatter),
            "intensity_slices": str(p_intensity),
            "smoothed_volcorr_grid_rsweep": str(p_grid),
            "smoothed_raw_heatmap": str(p_grid_raw),
        },
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("self-check: passed")


if __name__ == "__main__":
    main()

