"""
Generate train/val/test JSONL splits for the sthp0 synthetic ST-Hawkes dataset.

sthp0 = AutoSTPP paper Table 3, DS1:
  alpha=0.5, beta=1.0, mu=0.2
  g0_cov = diag(0.2, 0.2)   (background spread)
  g2_cov = diag(0.5, 0.5)   (offspring spread)

Output: data/sthp0/{train,val,test}.jsonl
Usage : python scripts/gen_sthp0_splits.py [--out_dir data/sthp0] [--seed 42]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from unified_stpp.data.synthetic import STHPDataset


STHP0 = dict(
    alpha=0.5,
    beta=1.0,
    mu=0.2,
    g0_cov=np.array([[0.2, 0.0], [0.0, 0.2]], dtype=np.float64),
    g2_cov=np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64),
)

# Matches exp_repro_autostpp_synth_sthp.py: long sequence split into 200-unit windows
T_END = 10_000.0       # total simulation horizon → ~50 windows
WINDOW_T = 200.0       # each window is 200 time units (matches exp_repro)
N_WINDOWS = int(T_END / WINDOW_T)  # 50 windows total
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1   # 40 train / 5 val / 5 test


def _generate(seed: int) -> dict:
    gen = STHPDataset(
        s_mu=np.array([0.0, 0.0], dtype=np.float64),
        g0_cov=STHP0["g0_cov"],
        g2_cov=STHP0["g2_cov"],
        alpha=STHP0["alpha"],
        beta=STHP0["beta"],
        mu=STHP0["mu"],
        seed=seed,
        covariate_fn=lambda t, s: np.array([0.0], dtype=np.float32),
    )
    np.random.seed(seed)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gen.generate(t_start=0.0, t_end=T_END, verbose=False)
    return {
        "times": np.asarray(gen.his_t, dtype=np.float32),
        "locations": np.asarray(gen.his_s, dtype=np.float32).reshape(-1, 2),
    }


def _split_windows(seq: dict) -> list[dict]:
    times = np.asarray(seq["times"], dtype=np.float64)
    locs = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, 2)
    windows = []
    for i in range(N_WINDOWS):
        t0, t1 = i * WINDOW_T, (i + 1) * WINDOW_T
        mask = (times >= t0) & (times < t1)
        windows.append({
            "times": (times[mask] - t0).astype(np.float32).tolist(),
            "locations": locs[mask].astype(np.float32).tolist(),
        })
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/sthp0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Simulating sthp0 sequence up to T={T_END:.0f} (seed={args.seed})…")
    seq = _generate(args.seed)
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
    for split, seqs in splits.items():
        path = out / f"{split}.jsonl"
        with open(path, "w") as f:
            for s in seqs:
                f.write(json.dumps(s) + "\n")
        avg_len = np.mean([len(s["times"]) for s in seqs])
        print(f"  {split}: {len(seqs)} seqs  avg_len={avg_len:.1f}  → {path}")

    print("Done.")


if __name__ == "__main__":
    main()
