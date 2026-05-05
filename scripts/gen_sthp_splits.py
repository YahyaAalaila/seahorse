"""
Generate train/val/test JSONL splits for STHP synthetic ST-Hawkes datasets.

Three variants (from AutoSTPP paper Table 3):

  sthp0  α=0.5, β=1.0, μ=0.2  g0_cov=diag(0.2) g2_cov=diag(0.5)
  sthp1  α=0.3, β=0.2, μ=1.0  g0_cov=diag(0.4) g2_cov=diag(0.3)
  sthp2  α=0.4, β=0.2, μ=1.0  g0_cov=diag(0.25) g2_cov=diag(0.2)

sthp1/sthp2 parameters correspond to STSCPDataset (original paper) with gamma=0,
which is equivalent to STHPDataset (gamma=0 drops the self-correction term).

Output: data/sthp{N}/{train,val,test}.jsonl + data/sthp{N}/dataset_meta.json
Usage : python scripts/gen_sthp_splits.py --variant 0 [--out_dir data/sthp0] [--seed 42]
        python scripts/gen_sthp_splits.py --variant 1 --plot              # save intensity_plot.png
        python scripts/gen_sthp_splits.py --variant 2 --plot --plot_window 3  # zoom to window 3
        python scripts/gen_sthp_splits.py --variant 0 --plot3d            # animated HTML surface
        python scripts/gen_sthp_splits.py --variant 0 --plot3d --plot3d_mode surface  # 3-D mesh
"""
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from unified_stpp.data.synthetic import STHPDataset, plot_intensity_surface


# ---------------------------------------------------------------------------
# Variant parameter tables
# ---------------------------------------------------------------------------

_VARIANTS: dict[int, dict] = {
    0: dict(
        alpha=0.5,
        beta=1.0,
        mu=0.2,
        g0_cov=np.array([[0.2, 0.0], [0.0, 0.2]], dtype=np.float64),
        g2_cov=np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64),
    ),
    1: dict(
        alpha=0.3,
        beta=0.5,
        mu=0.5,
        g0_cov=np.array([[0.4, 0.0], [0.0, 0.4]], dtype=np.float64),
        g2_cov=np.array([[0.3, 0.0], [0.0, 0.3]], dtype=np.float64),
    ),
    2: dict(
        alpha=0.4,
        beta=0.2,
        mu=1.0,
        g0_cov=np.array([[0.25, 0.0], [0.0, 0.25]], dtype=np.float64),
        g2_cov=np.array([[0.2,  0.0], [0.0, 0.2 ]], dtype=np.float64),
    ),
}

# Windowing constants (match exp_repro pipeline)
T_END      = 10_000.0
WINDOW_T   = 200.0
N_WINDOWS  = int(T_END / WINDOW_T)   # 50
TRAIN_FRAC = 0.8                      # 40 train / 5 val / 5 test
VAL_FRAC   = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate(params: dict, seed: int) -> tuple[STHPDataset, dict]:
    gen = STHPDataset(
        s_mu=np.array([0.0, 0.0], dtype=np.float64),
        g0_cov=params["g0_cov"],
        g2_cov=params["g2_cov"],
        alpha=params["alpha"],
        beta=params["beta"],
        mu=params["mu"],
        seed=seed,
        covariate_fn=lambda t, s: np.array([0.0], dtype=np.float32),
    )
    np.random.seed(seed)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gen.generate(t_start=0.0, t_end=T_END, verbose=False)
    seq = {
        "times":     np.asarray(gen.his_t, dtype=np.float32),
        "locations": np.asarray(gen.his_s, dtype=np.float32).reshape(-1, 2),
    }
    return gen, seq


def _split_windows(seq: dict) -> list[dict]:
    times = np.asarray(seq["times"], dtype=np.float64)
    locs  = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, 2)
    windows = []
    for i in range(N_WINDOWS):
        t0, t1 = i * WINDOW_T, (i + 1) * WINDOW_T
        mask = (times >= t0) & (times < t1)
        windows.append({
            "times":     (times[mask] - t0).astype(np.float32).tolist(),
            "locations": locs[mask].astype(np.float32).tolist(),
        })
    return windows


def _plot_intensity(gen: STHPDataset, out_dir: Path, window_idx: int) -> None:
    """Call STHPDataset.plot_intensity() over one window and save to out_dir."""
    import matplotlib.pyplot as plt

    t_start = window_idx * WINDOW_T
    t_end   = t_start + WINDOW_T
    gen.plot_intensity(t_start=t_start, t_end=t_end)
    plt.suptitle(
        f"True intensity at s=({gen.s_mu[0]:.2f}, {gen.s_mu[1]:.2f}), "
        f"window {window_idx} (t={t_start:.0f}–{t_end:.0f})",
        y=1.01,
    )
    plt.tight_layout()
    save_path = out_dir / "intensity_plot.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  intensity plot → {save_path}")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate STHP benchmark splits (variants 0, 1, or 2)."
    )
    parser.add_argument(
        "--variant", type=int, required=True, choices=[0, 1, 2],
        help="Dataset variant: 0 = sthp0, 1 = sthp1, 2 = sthp2",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: data/sthp{variant})",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot", action="store_true",
        help="Save an intensity plot (intensity_plot.png) to out_dir after generation",
    )
    parser.add_argument(
        "--plot_window", type=int, default=0,
        help="Window index to visualise with --plot (default: 0, i.e. t=0–200)",
    )
    parser.add_argument(
        "--plot3d", action="store_true",
        help="Save an animated plotly HTML surface (intensity_surface.html) to out_dir",
    )
    parser.add_argument(
        "--plot3d_mode", default="heatmap", choices=["heatmap", "surface"],
        help="Render mode for --plot3d: 2-D heatmap (default) or 3-D mesh surface",
    )
    parser.add_argument(
        "--plot3d_window", type=int, default=4,
        help="Window index to visualise with --plot3d (default: 10, i.e. t=2000–2200)",
    )
    parser.add_argument(
        "--plot3d_n_t", type=int, default=40,
        help="Number of time slices for the animated surface (default: 20)",
    )
    parser.add_argument(
        "--plot3d_n_grid", type=int, default=50,
        help="Spatial grid resolution per axis for --plot3d (default: 50)",
    )
    args = parser.parse_args()

    params  = _VARIANTS[args.variant]
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/sthp{args.variant}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = f"sthp{args.variant}"
    print(f"Simulating {dataset_id} sequence up to T={T_END:.0f} (seed={args.seed})…")
    gen, seq = _generate(params, args.seed)
    n_events = len(seq["times"])
    print(f"  Total events: {n_events}  →  splitting into {N_WINDOWS} windows of T={WINDOW_T}")

    windows = _split_windows(seq)
    n_train = int(N_WINDOWS * TRAIN_FRAC)
    n_val   = int(N_WINDOWS * VAL_FRAC)

    splits = {
        "train": windows[:n_train],
        "val":   windows[n_train : n_train + n_val],
        "test":  windows[n_train + n_val :],
    }
    for split_name, seqs in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for s in seqs:
                f.write(json.dumps(s) + "\n")
        avg_len = np.mean([len(s["times"]) for s in seqs])
        print(f"  {split_name}: {len(seqs)} seqs  avg_len={avg_len:.1f}  → {path}")

    # Dataset provenance metadata
    meta = {
        "dataset_id":        dataset_id,
        "generator":         "STHPDataset",
        "generator_file":    "unified_stpp/data/synthetic.py",
        "params": {
            "alpha":   float(params["alpha"]),
            "beta":    float(params["beta"]),
            "mu":      float(params["mu"]),
            "g0_cov":  params["g0_cov"].tolist(),
            "g2_cov":  params["g2_cov"].tolist(),
            "s_mu":    [0.0, 0.0],
        },
        "seed":              args.seed,
        "T_end":             T_END,
        "window_T":          WINDOW_T,
        "n_windows":         N_WINDOWS,
        "n_train":           n_train,
        "n_val":             n_val,
        "n_test":            N_WINDOWS - n_train - n_val,
        "generated_at":      datetime.now().isoformat(timespec="seconds"),
        "git_sha":           _git_sha(),
        "true_intensity_fn": "STHPDataset.lamb_st",
    }
    meta_path = out_dir / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  metadata → {meta_path}")

    if args.plot:
        if args.plot_window >= N_WINDOWS:
            print(f"  WARNING: --plot_window {args.plot_window} out of range "
                  f"(0–{N_WINDOWS - 1}); using 0")
            args.plot_window = 0
        _plot_intensity(gen, out_dir, args.plot_window)

    if args.plot3d:
        win = min(args.plot3d_window, N_WINDOWS - 1)
        t0, t1 = win * WINDOW_T, (win + 1) * WINDOW_T
        print(f"  Computing intensity surface for window {win} "
              f"(t={t0:.0f}–{t1:.0f}, "
              f"n_t={args.plot3d_n_t}, n_grid={args.plot3d_n_grid})…")
        lambs, x_range, y_range, t_range = gen.get_lamb_st(
            t_start=t0, t_end=t1,
            n_x=args.plot3d_n_grid, n_y=args.plot3d_n_grid,
            n_t=args.plot3d_n_t,
        )
        save_path = str(out_dir / "intensity_surface.html")
        plot_intensity_surface(
            lambs, x_range, y_range, t_range,
            mode=args.plot3d_mode,
            title=f"λ*(t,s|H) — {dataset_id} window {win}",
            save_path=save_path,
        )
        print(f"  intensity surface → {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
