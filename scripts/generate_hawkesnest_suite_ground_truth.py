#!/usr/bin/env python3
"""Backfill synthetic ground-truth bundles for HawkesNest suite evals.

This script writes the per-config files expected by
`resolve_hawkesnest_campaign_eval_targets.py`:

  ground_truth/<CONFIG>_intensity_grid_r0.npz
  ground_truth/<CONFIG>_params.json

The current synthetic surface-eval path compares model intensity against a
single representative test-sequence grid per config. We follow that contract:
for each config we use the first row of `test.jsonl` as the conditioning
history, reconstruct an analytic intensity grid from the suite metadata, and
save a matching `background_grid` for decomposition metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_GRID_SPEC = {
    "x_resolution": 50,
    "y_resolution": 50,
    "t_resolution": 100,
}

SUITE3_DEFAULTS = {
    "background": {"type": "constant", "rate": 2.0},
    "kernel": {
        "type": "traveling_wave",
        "theta_wave": math.pi / 4.0,
        "sigma": 0.08,
        "temporal_scale": 1.0,
        "tau_max": 6.0,
    },
}

SUITE4_DEFAULTS = {
    "tau_max": 5.0,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite-root",
        default="data/hawkesnest_suitesv2",
        help="Root containing suite3_entanglement/, suite4_heterogeneity/, ...",
    )
    p.add_argument(
        "--suite",
        action="append",
        dest="suites",
        help="Optional suite name under --suite-root. Repeat to select multiple.",
    )
    p.add_argument(
        "--suite-path",
        action="append",
        dest="suite_paths",
        help="Optional explicit suite path. Repeat to select multiple.",
    )
    p.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Optional config filter like L0 or H2. Repeat to select multiple.",
    )
    p.add_argument("--x-resolution", type=int, default=DEFAULT_GRID_SPEC["x_resolution"])
    p.add_argument("--y-resolution", type=int, default=DEFAULT_GRID_SPEC["y_resolution"])
    p.add_argument("--t-resolution", type=int, default=DEFAULT_GRID_SPEC["t_resolution"])
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing ground_truth files.",
    )
    return p.parse_args()


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _suite_paths(args: argparse.Namespace) -> list[Path]:
    out: list[Path] = []
    if args.suite_paths:
        out.extend(Path(p).expanduser().resolve() for p in args.suite_paths)
    if args.suites:
        root = Path(args.suite_root).expanduser().resolve()
        out.extend((root / name).resolve() for name in args.suites)
    if not out:
        root = Path(args.suite_root).expanduser().resolve()
        out.extend(sorted(p.resolve() for p in root.iterdir() if p.is_dir()))
    return out


def _gaussian2d(x: np.ndarray, y: np.ndarray, cx: float, cy: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-8)
    r2 = (x - float(cx)) ** 2 + (y - float(cy)) ** 2
    return np.exp(-0.5 * r2 / (sigma * sigma)).astype(np.float64)


def _background_field(
    bg_spec: dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    bg_type = str(bg_spec.get("type", "constant"))

    if bg_type == "constant":
        rate = float(bg_spec["rate"])
        return np.full((len(ts), len(xs), len(ys)), rate, dtype=np.float32)

    if bg_type == "function":
        name = str(bg_spec.get("name", ""))
        if name == "cluster_mix":
            field = np.full((len(xs), len(ys)), float(bg_spec["a0"]), dtype=np.float64)
            sigma = float(bg_spec["sigma"])
            amp = float(bg_spec["amp"])
            for cx, cy in bg_spec["centers"]:
                field += amp * _gaussian2d(xx, yy, float(cx), float(cy), sigma)
            return np.repeat(field[None].astype(np.float32), len(ts), axis=0)

        if name == "moving_hotspots":
            start = np.asarray(bg_spec["start"], dtype=np.float64)
            velocity = np.asarray(bg_spec["v"], dtype=np.float64)
            sigma = float(bg_spec["sigma"])
            amp = float(bg_spec["amp"])
            a0 = float(bg_spec["a0"])
            frames = []
            for t_query in ts.astype(np.float64):
                center = start + velocity * float(t_query)
                frame = a0 + amp * _gaussian2d(xx, yy, center[0], center[1], sigma)
                frames.append(frame.astype(np.float32))
            return np.stack(frames, axis=0)

        if name == "gabor_travel":
            start = np.asarray(bg_spec["start"], dtype=np.float64)
            sigma = float(bg_spec["sigma"])
            amp = float(bg_spec["amp"])
            a0 = float(bg_spec["a0"])
            freq = float(bg_spec["freq"])
            freq_t = float(bg_spec["freq_t"])
            u = np.asarray([1.0, 1.0], dtype=np.float64)
            u /= np.linalg.norm(u)
            proj = (xx - start[0]) * u[0] + (yy - start[1]) * u[1]
            envelope = _gaussian2d(xx, yy, start[0], start[1], sigma)
            frames = []
            for t_query in ts.astype(np.float64):
                phase = 2.0 * math.pi * (freq * proj - freq_t * float(t_query))
                carrier = 0.5 * (1.0 + np.cos(phase))
                frame = a0 + amp * envelope * carrier
                frames.append(frame.astype(np.float32))
            return np.stack(frames, axis=0)

    raise ValueError(f"Unsupported background spec: {bg_spec}")


def _trigger_kernel_frame(
    kernel_spec: dict[str, Any],
    adj: float,
    history_times: np.ndarray,
    history_locs: np.ndarray,
    t_query: float,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    if history_times.size == 0:
        return np.zeros_like(xx, dtype=np.float64)

    dt = float(t_query) - history_times.astype(np.float64)
    active = dt >= 0.0
    tau_max = kernel_spec.get("tau_max")
    if tau_max is not None:
        active &= dt <= float(tau_max)
    if not np.any(active):
        return np.zeros_like(xx, dtype=np.float64)

    dt = dt[active]
    locs = history_locs.astype(np.float64)[active]
    kind = str(kernel_spec["type"])

    if kind == "separable":
        beta = float(kernel_spec["temporal_decay"])
        sigma = max(float(kernel_spec["spatial_sigma"]), 1e-8)
        weights = np.exp(-beta * dt).astype(np.float64)
        centers = locs
    elif kind == "traveling_wave":
        sigma = max(float(kernel_spec["sigma"]), 1e-8)
        tau = max(float(kernel_spec["temporal_scale"]), 1e-8)
        theta = float(kernel_spec["theta_wave"])
        velocity = float(kernel_spec["v"])
        direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
        weights = np.exp(-dt / tau).astype(np.float64)
        centers = locs + (velocity * dt)[:, None] * direction[None, :]
    else:
        raise ValueError(f"Unsupported kernel spec: {kernel_spec}")

    total = np.zeros_like(xx, dtype=np.float64)
    for weight, (cx, cy) in zip(weights, centers, strict=False):
        total += float(weight) * _gaussian2d(xx, yy, float(cx), float(cy), sigma)
    return float(adj) * total


def _config_spec(
    suite_name: str,
    suite_meta: dict[str, Any],
    level_meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], float]:
    if suite_name == "suite3_entanglement":
        kernel = dict(SUITE3_DEFAULTS["kernel"])
        kernel["v"] = float(level_meta["v"])
        background = dict(SUITE3_DEFAULTS["background"])
        return background, kernel, float(level_meta["adj"])

    if suite_name == "suite4_heterogeneity":
        kernel = dict(suite_meta.get("kernel") or level_meta.get("kernel") or {})
        if not kernel:
            raise ValueError("suite4_heterogeneity metadata must include kernel.")
        kernel.setdefault("tau_max", SUITE4_DEFAULTS["tau_max"])
        background = dict(level_meta["bg"])
        return background, kernel, float(level_meta["adj"])

    raise ValueError(f"Unsupported suite for GT backfill: {suite_name}")


def _domain_bounds(suite_name: str, suite_path: Path, config_id: str) -> tuple[list[float], list[float]]:
    sequence_root = suite_path / "sequences"
    source_paths = sorted(sequence_root.glob(f"{config_id}_r*.npz"))
    if source_paths:
        with np.load(source_paths[0]) as npz:
            if "domain_bounds" in npz:
                bounds = np.asarray(npz["domain_bounds"], dtype=np.float64)
                return [float(bounds[0, 0]), float(bounds[0, 1])], [float(bounds[1, 0]), float(bounds[1, 1])]
    if suite_name in {"suite3_entanglement", "suite4_heterogeneity"}:
        return [0.0, 1.0], [0.0, 1.0]
    raise ValueError(f"Could not infer domain bounds for {suite_name}/{config_id}")


def generate_suite_ground_truth(
    *,
    suite_path: Path,
    config_filter: set[str] | None = None,
    x_resolution: int = DEFAULT_GRID_SPEC["x_resolution"],
    y_resolution: int = DEFAULT_GRID_SPEC["y_resolution"],
    t_resolution: int = DEFAULT_GRID_SPEC["t_resolution"],
    overwrite: bool = False,
) -> list[Path]:
    suite_path = suite_path.expanduser().resolve()
    metadata = _load_json(suite_path / "metadata.json")
    suite_name = str(metadata["suite"])
    gt_dir = suite_path / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for level_meta in metadata.get("levels", []):
        config_id = str(level_meta["label"])
        if config_filter is not None and config_id not in config_filter:
            continue

        test_rows = _load_jsonl(suite_path / "jsonl" / config_id / "test.jsonl")
        if not test_rows:
            raise ValueError(f"No test rows found for {suite_name}/{config_id}")
        history_row = test_rows[0]

        manifest_rows = _load_jsonl(suite_path / "jsonl" / config_id / "manifest.jsonl")
        provenance = next((row for row in manifest_rows if row.get("split") == "test"), None)
        if provenance is None:
            raise ValueError(f"No test manifest row found for {suite_name}/{config_id}")

        x_range, y_range = _domain_bounds(suite_name, suite_path, config_id)
        xs = np.linspace(x_range[0], x_range[1], int(x_resolution), dtype=np.float32)
        ys = np.linspace(y_range[0], y_range[1], int(y_resolution), dtype=np.float32)

        times = np.asarray(history_row["times"], dtype=np.float64)
        locs = np.asarray(history_row["locations"], dtype=np.float64)
        ts = np.linspace(float(times[0]), float(times[-1]), int(t_resolution), dtype=np.float32)

        background_spec, kernel_spec, adj = _config_spec(suite_name, metadata, level_meta)
        background = _background_field(background_spec, xs, ys, ts).astype(np.float32)

        xx, yy = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64), indexing="ij")
        lambda_true = np.empty_like(background, dtype=np.float32)
        for ti, t_query in enumerate(ts.astype(np.float64)):
            trigger = _trigger_kernel_frame(
                kernel_spec,
                adj,
                times,
                locs,
                float(t_query),
                xx,
                yy,
            )
            lambda_true[ti] = background[ti].astype(np.float64) + trigger

        npz_path = gt_dir / f"{config_id}_intensity_grid_r0.npz"
        params_path = gt_dir / f"{config_id}_params.json"
        if (npz_path.exists() or params_path.exists()) and not overwrite:
            raise FileExistsError(
                f"Ground-truth bundle already exists for {suite_name}/{config_id}. "
                "Pass --overwrite to replace it."
            )

        np.savez_compressed(
            npz_path,
            lambda_=lambda_true.astype(np.float32),
            x_grid=xs.astype(np.float32),
            y_grid=ys.astype(np.float32),
            t_grid=ts.astype(np.float32),
        )
        params_payload = {
            "suite": suite_name,
            "config": config_id,
            "ground_truth_method": "reconstructed_from_suite_metadata",
            "reference_test_seed": provenance.get("seed"),
            "reference_source_npz": provenance.get("source_npz"),
            "reference_chunk_idx": provenance.get("chunk_idx"),
            "adjacency": adj,
            "kernel": kernel_spec,
            "background": background_spec,
            "x_range": x_range,
            "y_range": y_range,
            "t_range": [float(ts[0]), float(ts[-1])],
            "background_grid": background.tolist(),
        }
        with params_path.open("w") as f:
            json.dump(params_payload, f)

        written.extend([npz_path, params_path])
    return written


def main() -> int:
    args = _parse_args()
    config_filter = set(args.configs or []) or None
    written: list[Path] = []
    for suite_path in _suite_paths(args):
        written.extend(
            generate_suite_ground_truth(
                suite_path=suite_path,
                config_filter=config_filter,
                x_resolution=int(args.x_resolution),
                y_resolution=int(args.y_resolution),
                t_resolution=int(args.t_resolution),
                overwrite=bool(args.overwrite),
            )
        )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
