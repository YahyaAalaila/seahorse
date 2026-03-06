#!/usr/bin/env python3
"""
Parity audit for intensity-grid computation.

Goal
----
Compare intensity grids from:
  (A) unified intensity pathway (IntensityEvaluator-driven),
  (B) DeepSTPP reference-style calc_lamb adapter,
  (C) AutoSTPP reference-style calc_lamb adapter,
on the same sequence, same history-index rule, and same (t, x, y) grid.

Important
---------
This script does NOT change model code. It is an analysis harness only.
The two "reference-style" adapters follow the index/scaling/batching structure
of the provided snippets, but they are implemented against this repo's model
APIs (which are unified variants, not the original external classes).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from unified_stpp.data.synthetic import STHPDataset
from unified_stpp.models import IntensityEvaluator
from unified_stpp.registry import build_model
from unified_stpp.training.data_module import STPPDataModule


EPS = 1e-12


@dataclass
class SequenceData:
    times_raw: np.ndarray   # (N,)
    locs_raw: np.ndarray    # (N,2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check intensity-grid parity between unified and reference-style adapters."
    )
    p.add_argument(
        "--results_json",
        type=str,
        required=True,
        help="Path to results.json from experiments/exp_repro_autostpp_synth_sthp.py",
    )
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--start_idx", type=int, default=2, help="Sequence index in selected split.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--t_nstep", type=int, default=21)
    p.add_argument("--x_nstep", type=int, default=31)
    p.add_argument("--y_nstep", type=int, default=31)
    p.add_argument("--round_time", action="store_true", default=False)
    p.add_argument("--use_unit_box_default", action="store_true", default=False)
    p.add_argument("--xmin", type=float, default=None)
    p.add_argument("--xmax", type=float, default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument(
        "--deep_tau_bias_mode",
        type=str,
        default="with_bias",
        choices=["with_bias", "no_bias"],
        help="Reference deep adapter tau scaling: (tau-bias)/scale or tau/scale.",
    )
    p.add_argument("--auto_trunc", action="store_true", default=False)
    p.add_argument("--auto_trunc_k", type=int, default=20)
    p.add_argument("--threshold_rel", type=float, default=1e-4)
    p.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/exp_check_intensity_grid_parity",
    )
    return p.parse_args()


def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        out[k[6:] if k.startswith("model.") else k] = v
    return out


def _load_model_from_result(
    result_entry: Dict[str, Any],
    *,
    device: str,
) -> torch.nn.Module:
    ckpt_path = Path(result_entry["checkpoint_best"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ck.get("state_dict", ck)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")
    stripped = _strip_state_dict_prefix(state_dict)

    if "encoder.embed.weight" not in stripped:
        raise KeyError(f"Missing encoder.embed.weight in checkpoint: {ckpt_path}")
    embed_w = stripped["encoder.embed.weight"]
    hidden_dim = int(embed_w.shape[0])
    enc_in = int(embed_w.shape[1])
    spatial_dim = max(1, enc_in - 1)  # expected 2 here

    model = build_model(
        config=result_entry.get("overrides", {}),
        spatial_dim=spatial_dim,
        hidden_dim=hidden_dim,
        event_cov_dim=0,
        field_cov_dim=0,
        preset=result_entry["preset"],
        n_marks=0,
    )
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch\n"
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''}\n"
            f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )
    model.to(device)
    model.eval()
    return model


def _generate_long_sequence(
    alpha: float,
    beta: float,
    mu: float,
    g0_cov: np.ndarray,
    g2_cov: np.ndarray,
    *,
    seed: int,
    t_end: float,
) -> Dict[str, np.ndarray]:
    gen = STHPDataset(
        s_mu=np.array([0.0, 0.0], dtype=np.float64),
        g0_cov=np.asarray(g0_cov, dtype=np.float64),
        g2_cov=np.asarray(g2_cov, dtype=np.float64),
        alpha=float(alpha),
        beta=float(beta),
        mu=float(mu),
        seed=int(seed),
        covariate_fn=lambda t, s: np.array([0.0], dtype=np.float32),
    )
    np.random.seed(int(seed))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gen.generate(t_start=0.0, t_end=float(t_end), verbose=False)
    return {
        "times": np.asarray(gen.his_t, dtype=np.float32),
        "locations": np.asarray(gen.his_s, dtype=np.float32).reshape(-1, 2),
    }


def _split_long_sequence(
    seq: Dict[str, np.ndarray],
    *,
    n_windows: int,
    window_T: float,
    reset_time_to_window: bool = True,
) -> List[Dict[str, np.ndarray]]:
    times = np.asarray(seq["times"], dtype=np.float64)
    locs = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, 2)
    out: List[Dict[str, np.ndarray]] = []
    for i in range(int(n_windows)):
        t0 = i * float(window_T)
        t1 = (i + 1) * float(window_T)
        mask = (times >= t0) & (times < t1)
        tw = times[mask]
        sw = locs[mask]
        if reset_time_to_window:
            tw = tw - t0
        out.append({"times": tw.astype(np.float32), "locations": sw.astype(np.float32)})
    return out


def _build_repro_splits(results: Dict[str, Any]) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    ds = results["dataset"]
    args = results["args"]
    long_seq = _generate_long_sequence(
        alpha=float(ds["alpha"]),
        beta=float(ds["beta"]),
        mu=float(ds["mu"]),
        g0_cov=np.asarray(ds["g0_cov"], dtype=np.float64),
        g2_cov=np.asarray(ds["g2_cov"], dtype=np.float64),
        seed=int(args["seed"]),
        t_end=float(ds["long_horizon"]),
    )
    split = ds["split"]
    n_train = int(split["train"])
    n_val = int(split["val"])
    n_test = int(split["test"])
    windows = _split_long_sequence(
        long_seq,
        n_windows=int(ds["n_windows"]),
        window_T=float(ds["window_T"]),
        reset_time_to_window=True,
    )
    train = windows[:n_train]
    val = windows[n_train : n_train + n_val]
    test = windows[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def _history_index(
    times_raw: np.ndarray,
    t_query: float,
    *,
    include_equal: bool,
    exclude_last: bool,
) -> int:
    times = times_raw[:-1] if exclude_last and len(times_raw) > 1 else times_raw
    side = "right" if include_equal else "left"
    return int(np.searchsorted(times, float(t_query), side=side) - 1)


def _build_history_context(
    model: torch.nn.Module,
    hist_times_norm: np.ndarray,
    hist_locs_norm: np.ndarray,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(hist_times_norm.shape[0])
    if n <= 0:
        raise ValueError("History length must be >= 1.")
    t = torch.tensor(hist_times_norm, dtype=torch.float32, device=device).unsqueeze(0)
    s = torch.tensor(hist_locs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    lengths = torch.tensor([n], dtype=torch.long, device=device)
    with torch.no_grad():
        events = torch.cat([t.unsqueeze(-1), s], dim=-1)
        z, _ = model.encoder(events, lengths, x_event=None)
        t_prev = t[:, -1:].contiguous()
    return z, t_prev


def _normal_pdf_cdf(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z * inv_sqrt2))
    return pdf, cdf


def _deep_temporal_hazard_from_tau_norm(
    model: torch.nn.Module,
    z: torch.Tensor,
    tau_norm: float,
) -> float:
    temporal = model.decoder.temporal
    with torch.no_grad():
        logits, mu, sigma = temporal._get_params(z, None)  # (1,K)
        log_pi = torch.log_softmax(logits, dim=-1)
        tau = max(float(tau_norm), 1e-6)
        log_tau = torch.tensor([[math.log(tau)]], dtype=torch.float32, device=z.device)
        zscore = (log_tau - mu) / sigma
        pdf_n, cdf_n = _normal_pdf_cdf(zscore)
        lognormal = pdf_n / (sigma * tau + 1e-12)
        f_tau = torch.sum(torch.exp(log_pi) * lognormal, dim=-1)  # (1,)
        Ft = torch.sum(torch.exp(log_pi) * cdf_n, dim=-1)         # (1,)
        St = (1.0 - Ft).clamp(min=1e-8)
        lam_t = f_tau / St
    return float(lam_t.item())


def _make_grid(
    seq: SequenceData,
    *,
    scales: np.ndarray,
    biases: np.ndarray,
    x_nstep: int,
    y_nstep: int,
    t_nstep: int,
    round_time: bool,
    use_unit_box_default: bool,
    xmin: Optional[float],
    xmax: Optional[float],
    ymin: Optional[float],
    ymax: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if xmin is None or xmax is None or ymin is None or ymax is None:
        if use_unit_box_default:
            x_min_norm, x_max_norm = 0.0, 1.0
            y_min_norm, y_max_norm = 0.0, 1.0
        else:
            qx = np.percentile(seq.locs_raw[:, 0], [1.0, 99.0])
            qy = np.percentile(seq.locs_raw[:, 1], [1.0, 99.0])
            x_pad = 0.05 * max(float(qx[1] - qx[0]), 1e-6)
            y_pad = 0.05 * max(float(qy[1] - qy[0]), 1e-6)
            xmin_raw = float(qx[0] - x_pad)
            xmax_raw = float(qx[1] + x_pad)
            ymin_raw = float(qy[0] - y_pad)
            ymax_raw = float(qy[1] + y_pad)
            x_min_norm = (xmin_raw - biases[0]) / scales[0]
            x_max_norm = (xmax_raw - biases[0]) / scales[0]
            y_min_norm = (ymin_raw - biases[1]) / scales[1]
            y_max_norm = (ymax_raw - biases[1]) / scales[1]
    else:
        x_min_norm = (float(xmin) - biases[0]) / scales[0]
        x_max_norm = (float(xmax) - biases[0]) / scales[0]
        y_min_norm = (float(ymin) - biases[1]) / scales[1]
        y_max_norm = (float(ymax) - biases[1]) / scales[1]

    x_norm = np.linspace(x_min_norm, x_max_norm, int(x_nstep), dtype=np.float64)
    y_norm = np.linspace(y_min_norm, y_max_norm, int(y_nstep), dtype=np.float64)
    x_raw = x_norm * scales[0] + biases[0]
    y_raw = y_norm * scales[1] + biases[1]

    t_start = float(seq.times_raw[0])
    t_end = float(seq.times_raw[-1])
    if t_nstep <= 1 or t_end <= t_start:
        t_raw = np.array([t_start], dtype=np.float64)
    else:
        t_step = (t_end - t_start) / max(int(t_nstep) - 1, 1)
        if round_time:
            t_raw = np.arange(round(t_start), round(t_end) + 1e-5, t_step, dtype=np.float64)
            if t_raw.size < 2:
                t_raw = np.linspace(t_start, t_end, int(t_nstep), dtype=np.float64)
        else:
            t_raw = np.arange(t_start, t_end + 1e-5, t_step, dtype=np.float64)
            if t_raw.size < 2:
                t_raw = np.linspace(t_start, t_end, int(t_nstep), dtype=np.float64)

    return x_norm, y_norm, x_raw, y_raw, t_raw


def _prepare_grid_points(x_norm: np.ndarray, y_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x_norm, y_norm, indexing="ij")
    s_flat = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
    return xx, yy, s_flat


def _compute_unified_grid(
    model: torch.nn.Module,
    seq: SequenceData,
    *,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    t_raw: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    device: torch.device,
    include_equal: bool = True,
) -> np.ndarray:
    xx, yy, s_flat = _prepare_grid_points(x_norm, y_norm)
    n_pts = int(s_flat.shape[0])
    s_tensor = torch.tensor(s_flat, dtype=torch.float32, device=device)
    jac = float(np.prod(scales))
    out: List[np.ndarray] = []

    for t in t_raw:
        i = _history_index(seq.times_raw, float(t), include_equal=include_equal, exclude_last=False)
        i = max(i, 0)
        hist_t = seq.times_raw[: i + 1]
        hist_s = seq.locs_raw[: i + 1]
        hist_t_norm = (hist_t - biases[2]) / scales[2]
        hist_s_norm = (hist_s - biases[:2]) / scales[:2]
        z, t_prev = _build_history_context(model, hist_t_norm, hist_s_norm, device=device)
        evaluator = IntensityEvaluator(model, z=z, t_prev=t_prev)

        t_norm = (float(t) - biases[2]) / scales[2]
        t_tensor = torch.full((n_pts, 1), float(t_norm), dtype=torch.float32, device=device)
        with torch.no_grad():
            z0, tp0 = evaluator.z, evaluator.t_prev
            evaluator.z = z.expand(n_pts, -1)
            evaluator.t_prev = t_prev.expand(n_pts, -1)
            lam_norm = evaluator.intensity(t_tensor, s_tensor, x_field=None)
            evaluator.z, evaluator.t_prev = z0, tp0
        lam_raw = lam_norm.detach().cpu().numpy().reshape(xx.shape) / max(jac, EPS)
        out.append(np.clip(lam_raw, a_min=0.0, a_max=None))
    return np.stack(out, axis=0)  # (T,X,Y)


def _compute_deep_reference_grid(
    model: torch.nn.Module,
    seq: SequenceData,
    *,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    t_raw: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    device: torch.device,
    include_equal: bool = True,
    tau_with_bias: bool = True,
    trunc_k: Optional[int] = None,
    batch_points: int = 8192,
) -> np.ndarray:
    xx, yy, s_flat = _prepare_grid_points(x_norm, y_norm)
    n_pts = int(s_flat.shape[0])
    s_tensor = torch.tensor(s_flat, dtype=torch.float32, device=device)
    jac = float(np.prod(scales))
    out: List[np.ndarray] = []

    for t in t_raw:
        i = _history_index(seq.times_raw, float(t), include_equal=include_equal, exclude_last=False)
        i = max(i, 0)
        h0 = max(0, i + 1 - int(trunc_k)) if trunc_k is not None else 0
        hist_t = seq.times_raw[h0 : i + 1]
        hist_s = seq.locs_raw[h0 : i + 1]
        hist_t_norm = (hist_t - biases[2]) / scales[2]
        hist_s_norm = (hist_s - biases[:2]) / scales[:2]
        z, t_prev = _build_history_context(model, hist_t_norm, hist_s_norm, device=device)

        tau_raw = float(t - hist_t[-1])
        tau_norm = (tau_raw - biases[2]) / scales[2] if tau_with_bias else (tau_raw / scales[2])
        lam_t_norm = _deep_temporal_hazard_from_tau_norm(model, z, tau_norm)

        t_query_norm = torch.full((n_pts, 1), float((float(t) - biases[2]) / scales[2]), dtype=torch.float32, device=device)
        t_prev_rep = t_prev.expand(n_pts, -1)
        fs_parts: List[np.ndarray] = []
        with torch.no_grad():
            for j in range(0, n_pts, int(batch_points)):
                s_j = s_tensor[j : j + int(batch_points)]
                z_j = z.expand(s_j.shape[0], -1)
                t_j = t_query_norm[j : j + int(batch_points)]
                tp_j = t_prev_rep[j : j + int(batch_points)]
                log_fs = model.decoder.spatial.log_prob(z_j, t_j, s_j, tp_j, x_field=None)
                fs_parts.append(torch.exp(log_fs).detach().cpu().numpy())
        fs_norm = np.concatenate(fs_parts, axis=0).reshape(xx.shape)
        lam_raw = (lam_t_norm * fs_norm) / max(jac, EPS)
        out.append(np.clip(lam_raw, a_min=0.0, a_max=None))

    return np.stack(out, axis=0)


def _compute_auto_reference_grid(
    model: torch.nn.Module,
    seq: SequenceData,
    *,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    t_raw: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    device: torch.device,
    include_equal: bool = True,
    trunc: bool = False,
    trunc_k: int = 20,
    bg_mode: str = "pre_jac",   # pre_jac | post_jac | none
    t_diff_mode: str = "history",  # history | last_only
    batch_points: int = 4096,
) -> np.ndarray:
    if not hasattr(model.decoder, "prodnet"):
        raise TypeError("Auto reference adapter expects AutoIntDecoder with prodnet.")

    xx, yy, s_flat = _prepare_grid_points(x_norm, y_norm)
    n_pts = int(s_flat.shape[0])
    jac = float(np.prod(scales))
    decoder = model.decoder
    mu_bg = float(decoder._mu().detach().cpu().item())
    s_grids = torch.tensor(s_flat, dtype=torch.float32, device=device)  # normalized grid
    out: List[np.ndarray] = []

    for t in t_raw:
        i = _history_index(seq.times_raw, float(t), include_equal=include_equal, exclude_last=True)
        i = max(i, 0)
        his_raw = np.column_stack([seq.locs_raw[: i + 1], seq.times_raw[: i + 1]]).astype(np.float64)
        if trunc and his_raw.shape[0] > int(trunc_k):
            his_raw = his_raw[-int(trunc_k) :]

        hist_t = his_raw[:, 2]
        hist_s = his_raw[:, :2]
        hist_t_norm = (hist_t - biases[2]) / scales[2]
        hist_s_norm = (hist_s - biases[:2]) / scales[:2]
        z, _ = _build_history_context(model, hist_t_norm, hist_s_norm, device=device)

        st_x = his_raw.copy()
        if st_x.shape[0] > 1:
            st_x[1:, 2] = np.diff(st_x[:, 2])  # inter-event conversion
        st_x_scaled = (st_x - biases.reshape(1, 3)) / scales.reshape(1, 3)

        s_hist = st_x_scaled[:, :2].astype(np.float32)  # (H,2)
        h = int(s_hist.shape[0])
        s_diff = s_grids.unsqueeze(1) - torch.tensor(s_hist, dtype=torch.float32, device=device).unsqueeze(0)  # (N,H,2)

        if t_diff_mode == "history":
            t_diff = (float(t) - hist_t) / scales[2]  # (H,)
        elif t_diff_mode == "last_only":
            t_last = float(hist_t[-1])
            t_diff = np.full((h,), (float(t) - t_last) / scales[2], dtype=np.float64)
        else:
            raise ValueError(f"Unknown t_diff_mode: {t_diff_mode}")
        t_diff_t = torch.tensor(t_diff, dtype=torch.float32, device=device).view(1, h, 1).expand(n_pts, -1, -1)

        st_diff = torch.cat([s_diff, t_diff_t], dim=-1)  # (N,H,3)
        temp = st_diff.reshape(-1, 3)  # (N*H,3)

        vals: List[np.ndarray] = []
        with torch.no_grad():
            for j in range(0, temp.shape[0], int(batch_points)):
                chunk = temp[j : j + int(batch_points)]
                h_rep = z.expand(chunk.shape[0], -1)
                f = decoder.prodnet.intensity(chunk[:, 0:1], chunk[:, 1:2], chunk[:, 2:3], h_rep)
                vals.append(torch.clamp(f, min=0.0).detach().cpu().numpy())
        f_hist = np.concatenate(vals, axis=0).reshape(n_pts, h)
        lam_norm = f_hist.sum(axis=-1).reshape(xx.shape)

        if bg_mode == "pre_jac":
            lam_raw = (lam_norm + mu_bg) / max(jac, EPS)
        elif bg_mode == "post_jac":
            lam_raw = lam_norm / max(jac, EPS) + mu_bg
        elif bg_mode == "none":
            lam_raw = lam_norm / max(jac, EPS)
        else:
            raise ValueError(f"Unknown bg_mode: {bg_mode}")

        out.append(np.clip(lam_raw, a_min=0.0, a_max=None))

    return np.stack(out, axis=0)


def _summaries(arr: np.ndarray) -> Dict[str, float]:
    flat = arr.reshape(-1)
    return {
        "min": float(np.min(flat)),
        "median": float(np.median(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
    }


def _integral_over_space(arr_txy: np.ndarray, x_raw: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
    if len(x_raw) < 2 or len(y_raw) < 2:
        return np.sum(arr_txy, axis=(1, 2))
    dx = float(np.mean(np.diff(x_raw)))
    dy = float(np.mean(np.diff(y_raw)))
    return np.sum(arr_txy, axis=(1, 2)) * dx * dy


def _compare_grids(name: str, a: np.ndarray, b: np.ndarray, x_raw: np.ndarray, y_raw: np.ndarray) -> Dict[str, Any]:
    if a.shape != b.shape:
        raise ValueError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    abs_err = np.abs(a - b)
    rel_err = abs_err / np.maximum(np.abs(b), EPS)
    int_a = _integral_over_space(a, x_raw, y_raw)
    int_b = _integral_over_space(b, x_raw, y_raw)
    int_abs = np.abs(int_a - int_b)
    int_rel = int_abs / np.maximum(np.abs(int_b), EPS)
    return {
        "shape": list(a.shape),
        "mae": float(np.mean(abs_err)),
        "max_abs": float(np.max(abs_err)),
        "mean_rel": float(np.mean(rel_err)),
        "max_rel": float(np.max(rel_err)),
        "summary_a": _summaries(a),
        "summary_b": _summaries(b),
        "integral_a": [float(v) for v in int_a.tolist()],
        "integral_b": [float(v) for v in int_b.tolist()],
        "integral_abs_err_mean": float(np.mean(int_abs)),
        "integral_abs_err_max": float(np.max(int_abs)),
        "integral_rel_err_mean": float(np.mean(int_rel)),
        "integral_rel_err_max": float(np.max(int_rel)),
    }


def _print_compare(title: str, c: Dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"shape={c['shape']}  "
        f"mae={c['mae']:.6e}  max_abs={c['max_abs']:.6e}  "
        f"mean_rel={c['mean_rel']:.6e}  max_rel={c['max_rel']:.6e}"
    )
    sa, sb = c["summary_a"], c["summary_b"]
    print(
        "lambda summary (A vs ref): "
        f"min {sa['min']:.6g}/{sb['min']:.6g}, "
        f"med {sa['median']:.6g}/{sb['median']:.6g}, "
        f"max {sa['max']:.6g}/{sb['max']:.6g}"
    )
    print(
        "space integral error: "
        f"mean_abs={c['integral_abs_err_mean']:.6e}, "
        f"max_abs={c['integral_abs_err_max']:.6e}, "
        f"mean_rel={c['integral_rel_err_mean']:.6e}, "
        f"max_rel={c['integral_rel_err_max']:.6e}"
    )


def _diagnose_deep(
    *,
    base_unified: np.ndarray,
    base_ref: np.ndarray,
    model: torch.nn.Module,
    seq: SequenceData,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    t_raw: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    base = _compare_grids("deep_base", base_unified, base_ref, x_raw, y_raw)

    alt_hist = _compute_deep_reference_grid(
        model,
        seq,
        x_norm=x_norm,
        y_norm=y_norm,
        t_raw=t_raw,
        scales=scales,
        biases=biases,
        device=device,
        include_equal=False,
        tau_with_bias=True,
    )
    c_hist = _compare_grids("deep_hist_lt", base_unified, alt_hist, x_raw, y_raw)

    alt_tau = _compute_deep_reference_grid(
        model,
        seq,
        x_norm=x_norm,
        y_norm=y_norm,
        t_raw=t_raw,
        scales=scales,
        biases=biases,
        device=device,
        include_equal=True,
        tau_with_bias=False,
    )
    c_tau = _compare_grids("deep_tau_nobias", base_unified, alt_tau, x_raw, y_raw)

    jac = float(np.prod(scales))
    c_nojac = _compare_grids("deep_no_jac", base_unified, base_ref * jac, x_raw, y_raw)
    c_double = _compare_grids("deep_double_jac", base_unified, base_ref / max(jac, EPS), x_raw, y_raw)

    checks = {
        "history_leq_vs_lt": {
            "base_mean_rel": base["mean_rel"],
            "alt_mean_rel": c_hist["mean_rel"],
            "improves": bool(c_hist["mean_rel"] < base["mean_rel"]),
        },
        "cumulative_vs_interevent_time_conversion": {
            "base_mean_rel": base["mean_rel"],
            "alt_mean_rel": c_tau["mean_rel"],
            "improves": bool(c_tau["mean_rel"] < base["mean_rel"]),
        },
        "jacobian_once_vs_zero_or_twice": {
            "base_mean_rel": base["mean_rel"],
            "alt_no_jac_mean_rel": c_nojac["mean_rel"],
            "alt_double_jac_mean_rel": c_double["mean_rel"],
            "best_alt": float(min(c_nojac["mean_rel"], c_double["mean_rel"])),
            "improves": bool(min(c_nojac["mean_rel"], c_double["mean_rel"]) < base["mean_rel"]),
        },
        "background_term_placement": {
            "note": "DeepSTPP factorized decoder has no explicit additive background term in this repo.",
        },
        "truncation": {
            "note": "No explicit truncation path in deep reference adapter.",
        },
    }

    return {"base": base, "checks": checks}


def _diagnose_auto(
    *,
    base_unified: np.ndarray,
    base_ref: np.ndarray,
    model: torch.nn.Module,
    seq: SequenceData,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    t_raw: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    device: torch.device,
    trunc: bool,
    trunc_k: int,
) -> Dict[str, Any]:
    base = _compare_grids("auto_base", base_unified, base_ref, x_raw, y_raw)

    alt_hist = _compute_auto_reference_grid(
        model,
        seq,
        x_norm=x_norm,
        y_norm=y_norm,
        t_raw=t_raw,
        scales=scales,
        biases=biases,
        device=device,
        include_equal=False,
        trunc=trunc,
        trunc_k=trunc_k,
        bg_mode="pre_jac",
        t_diff_mode="history",
    )
    c_hist = _compare_grids("auto_hist_lt", base_unified, alt_hist, x_raw, y_raw)

    alt_last = _compute_auto_reference_grid(
        model,
        seq,
        x_norm=x_norm,
        y_norm=y_norm,
        t_raw=t_raw,
        scales=scales,
        biases=biases,
        device=device,
        include_equal=True,
        trunc=trunc,
        trunc_k=trunc_k,
        bg_mode="pre_jac",
        t_diff_mode="last_only",
    )
    c_last = _compare_grids("auto_last_only_time", base_unified, alt_last, x_raw, y_raw)

    jac = float(np.prod(scales))
    c_nojac = _compare_grids("auto_no_jac", base_unified, base_ref * jac, x_raw, y_raw)
    c_double = _compare_grids("auto_double_jac", base_unified, base_ref / max(jac, EPS), x_raw, y_raw)

    alt_bg_post = _compute_auto_reference_grid(
        model,
        seq,
        x_norm=x_norm,
        y_norm=y_norm,
        t_raw=t_raw,
        scales=scales,
        biases=biases,
        device=device,
        include_equal=True,
        trunc=trunc,
        trunc_k=trunc_k,
        bg_mode="post_jac",
        t_diff_mode="history",
    )
    c_bg_post = _compare_grids("auto_bg_post_jac", base_unified, alt_bg_post, x_raw, y_raw)

    alt_bg_none = _compute_auto_reference_grid(
        model,
        seq,
        x_norm=x_norm,
        y_norm=y_norm,
        t_raw=t_raw,
        scales=scales,
        biases=biases,
        device=device,
        include_equal=True,
        trunc=trunc,
        trunc_k=trunc_k,
        bg_mode="none",
        t_diff_mode="history",
    )
    c_bg_none = _compare_grids("auto_bg_none", base_unified, alt_bg_none, x_raw, y_raw)

    alt_trunc = _compute_auto_reference_grid(
        model,
        seq,
        x_norm=x_norm,
        y_norm=y_norm,
        t_raw=t_raw,
        scales=scales,
        biases=biases,
        device=device,
        include_equal=True,
        trunc=True,
        trunc_k=int(trunc_k),
        bg_mode="pre_jac",
        t_diff_mode="history",
    )
    c_trunc = _compare_grids("auto_trunc", base_unified, alt_trunc, x_raw, y_raw)

    checks = {
        "history_leq_vs_lt": {
            "base_mean_rel": base["mean_rel"],
            "alt_mean_rel": c_hist["mean_rel"],
            "improves": bool(c_hist["mean_rel"] < base["mean_rel"]),
        },
        "cumulative_vs_interevent_time_conversion": {
            "base_mean_rel": base["mean_rel"],
            "alt_last_only_mean_rel": c_last["mean_rel"],
            "improves": bool(c_last["mean_rel"] < base["mean_rel"]),
        },
        "jacobian_once_vs_zero_or_twice": {
            "base_mean_rel": base["mean_rel"],
            "alt_no_jac_mean_rel": c_nojac["mean_rel"],
            "alt_double_jac_mean_rel": c_double["mean_rel"],
            "best_alt": float(min(c_nojac["mean_rel"], c_double["mean_rel"])),
            "improves": bool(min(c_nojac["mean_rel"], c_double["mean_rel"]) < base["mean_rel"]),
        },
        "background_term_placement": {
            "base_mean_rel": base["mean_rel"],
            "alt_post_jac_mean_rel": c_bg_post["mean_rel"],
            "alt_no_bg_mean_rel": c_bg_none["mean_rel"],
            "best_alt": float(min(c_bg_post["mean_rel"], c_bg_none["mean_rel"])),
            "improves": bool(min(c_bg_post["mean_rel"], c_bg_none["mean_rel"]) < base["mean_rel"]),
        },
        "truncation": {
            "base_mean_rel": base["mean_rel"],
            "alt_trunc_mean_rel": c_trunc["mean_rel"],
            "improves": bool(c_trunc["mean_rel"] < base["mean_rel"]),
        },
    }

    return {"base": base, "checks": checks}


def _rank_likely_causes(checks: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    ranked: List[Tuple[str, float, float]] = []
    for k, v in checks.items():
        if isinstance(v, dict) and "base_mean_rel" in v:
            base = float(v["base_mean_rel"])
            alts = [float(v[x]) for x in v.keys() if x.endswith("mean_rel") and x != "base_mean_rel"]
            if not alts:
                continue
            best = min(alts)
            ranked.append((k, base, best))
    ranked.sort(key=lambda x: (x[2], x[2] / max(x[1], EPS)))
    return ranked


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.results_json).open("r") as f:
        results = json.load(f)

    train_seqs, val_seqs, test_seqs = _build_repro_splits(results)
    split_map = {"train": train_seqs, "val": val_seqs, "test": test_seqs}
    chosen_split = split_map[args.split]
    if args.start_idx < 0 or args.start_idx >= len(chosen_split):
        raise IndexError(f"--start_idx out of range for split={args.split}: {args.start_idx}")

    dm = STPPDataModule(
        train_seqs,
        val_seqs,
        test_seqs=test_seqs,
        batch_size=int(args.batch_size),
        num_workers=0,
        normalize=True,
        seed=int(results["args"]["seed"]),
    )
    dm.setup()
    train_ds = dm._train_dataset
    scales = np.array(
        [float(train_ds.loc_std[0]), float(train_ds.loc_std[1]), float(train_ds.time_std)],
        dtype=np.float64,
    )
    biases = np.array(
        [float(train_ds.loc_mean[0]), float(train_ds.loc_mean[1]), float(train_ds.time_mean)],
        dtype=np.float64,
    )

    seq_raw = chosen_split[args.start_idx]
    seq = SequenceData(
        times_raw=np.asarray(seq_raw["times"], dtype=np.float64).reshape(-1),
        locs_raw=np.asarray(seq_raw["locations"], dtype=np.float64).reshape(-1, 2),
    )
    if seq.times_raw.shape[0] < 3:
        raise RuntimeError("Selected sequence has <3 events.")

    x_norm, y_norm, x_raw, y_raw, t_raw = _make_grid(
        seq,
        scales=scales,
        biases=biases,
        x_nstep=args.x_nstep,
        y_nstep=args.y_nstep,
        t_nstep=args.t_nstep,
        round_time=bool(args.round_time),
        use_unit_box_default=bool(args.use_unit_box_default),
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
    )

    res_by_preset = {r["preset"]: r for r in results["results"]}
    if "deep_stpp" not in res_by_preset or "auto_stpp" not in res_by_preset:
        raise KeyError("results.json must include both deep_stpp and auto_stpp entries.")
    deep_model = _load_model_from_result(res_by_preset["deep_stpp"], device=str(device))
    auto_model = _load_model_from_result(res_by_preset["auto_stpp"], device=str(device))

    print("Unified intensity pathway located at: unified_stpp/models/sampling.py::IntensityEvaluator")
    print("Inputs/semantics:")
    print("  - history represented by latent z + last event time t_prev")
    print("  - query points in normalized units")
    print("  - scale conversion to native units by dividing by prod(scales)")
    print("  - no explicit mask in evaluator (history comes from chosen prefix)")
    print(f"scales=[sx, sy, st]={scales.tolist()}  biases=[bx, by, bt]={biases.tolist()}")
    print(
        f"split={args.split}, start_idx={args.start_idx}, shuffle=False, "
        f"grid=({len(t_raw)} times, {len(x_raw)} x, {len(y_raw)} y)"
    )

    # (A) Unified grids
    deep_A = _compute_unified_grid(
        deep_model, seq, x_norm=x_norm, y_norm=y_norm, t_raw=t_raw,
        scales=scales, biases=biases, device=device, include_equal=True,
    )
    auto_A = _compute_unified_grid(
        auto_model, seq, x_norm=x_norm, y_norm=y_norm, t_raw=t_raw,
        scales=scales, biases=biases, device=device, include_equal=True,
    )

    # (B) Deep reference-style adapter
    deep_B = _compute_deep_reference_grid(
        deep_model,
        seq,
        x_norm=x_norm,
        y_norm=y_norm,
        t_raw=t_raw,
        scales=scales,
        biases=biases,
        device=device,
        include_equal=True,
        tau_with_bias=(args.deep_tau_bias_mode == "with_bias"),
    )

    # (C) Auto reference-style adapter
    auto_C = _compute_auto_reference_grid(
        auto_model,
        seq,
        x_norm=x_norm,
        y_norm=y_norm,
        t_raw=t_raw,
        scales=scales,
        biases=biases,
        device=device,
        include_equal=True,
        trunc=bool(args.auto_trunc),
        trunc_k=int(args.auto_trunc_k),
        bg_mode="pre_jac",
        t_diff_mode="history",
    )

    deep_cmp = _compare_grids("deep_A_vs_B", deep_A, deep_B, x_raw, y_raw)
    auto_cmp = _compare_grids("auto_A_vs_C", auto_A, auto_C, x_raw, y_raw)
    _print_compare("DeepSTPP: unified (A) vs reference-style (B)", deep_cmp)
    _print_compare("AutoSTPP: unified (A) vs reference-style (C)", auto_cmp)

    deep_diag = None
    auto_diag = None
    if deep_cmp["mean_rel"] > float(args.threshold_rel) or deep_cmp["max_rel"] > float(args.threshold_rel):
        deep_diag = _diagnose_deep(
            base_unified=deep_A,
            base_ref=deep_B,
            model=deep_model,
            seq=seq,
            x_norm=x_norm,
            y_norm=y_norm,
            t_raw=t_raw,
            x_raw=x_raw,
            y_raw=y_raw,
            scales=scales,
            biases=biases,
            device=device,
        )
    if auto_cmp["mean_rel"] > float(args.threshold_rel) or auto_cmp["max_rel"] > float(args.threshold_rel):
        auto_diag = _diagnose_auto(
            base_unified=auto_A,
            base_ref=auto_C,
            model=auto_model,
            seq=seq,
            x_norm=x_norm,
            y_norm=y_norm,
            t_raw=t_raw,
            x_raw=x_raw,
            y_raw=y_raw,
            scales=scales,
            biases=biases,
            device=device,
            trunc=bool(args.auto_trunc),
            trunc_k=int(args.auto_trunc_k),
        )

    if deep_diag is not None:
        print("\nDeep mismatch diagnosis (requested checks i-v):")
        for k, v in deep_diag["checks"].items():
            print(f"  - {k}: {v}")
        ranked = _rank_likely_causes(deep_diag["checks"])
        if ranked:
            print("  likely causes (best alt mean_rel):")
            for k, b, a in ranked[:5]:
                print(f"    {k}: base={b:.3e}, best_alt={a:.3e}")
    else:
        print("\nDeep mismatch diagnosis: PASS (within threshold).")

    if auto_diag is not None:
        print("\nAuto mismatch diagnosis (requested checks i-v):")
        for k, v in auto_diag["checks"].items():
            print(f"  - {k}: {v}")
        ranked = _rank_likely_causes(auto_diag["checks"])
        if ranked:
            print("  likely causes (best alt mean_rel):")
            for k, b, a in ranked[:5]:
                print(f"    {k}: base={b:.3e}, best_alt={a:.3e}")
    else:
        print("\nAuto mismatch diagnosis: PASS (within threshold).")

    payload = {
        "args": vars(args),
        "results_json": str(Path(args.results_json).resolve()),
        "unified_pathway": {
            "module": "unified_stpp.models.sampling.IntensityEvaluator",
            "inputs": {
                "history_representation": "latent z + t_prev from encoded history prefix",
                "query_units": "normalized (z-scored) internally",
                "jacobian_conversion": "divide by prod(scales) to native units",
                "mask_semantics": "history prefix built explicitly; no pad mask at query time",
                "background_handling": "decoder-specific (AutoInt includes mu inside log_prob path)",
            },
        },
        "sequence": {
            "split": args.split,
            "start_idx": int(args.start_idx),
            "n_events": int(seq.times_raw.shape[0]),
            "time_range": [float(seq.times_raw[0]), float(seq.times_raw[-1])],
        },
        "grid": {
            "shape": [int(len(t_raw)), int(len(x_raw)), int(len(y_raw))],
            "t_range": [float(x) for x in t_raw.tolist()],
            "x_range": [float(x) for x in x_raw.tolist()],
            "y_range": [float(y) for y in y_raw.tolist()],
            "scales_xyz_t": [float(x) for x in scales.tolist()],
            "biases_xyz_t": [float(x) for x in biases.tolist()],
        },
        "deep": {
            "comparison_A_vs_B": deep_cmp,
            "diagnosis": deep_diag,
        },
        "auto": {
            "comparison_A_vs_C": auto_cmp,
            "diagnosis": auto_diag,
        },
        "threshold_rel": float(args.threshold_rel),
    }

    out_json = out_dir / f"{Path(args.results_json).stem}_split{args.split}_idx{args.start_idx}_parity.json"
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved parity artifact: {out_json}")


if __name__ == "__main__":
    main()
