"""
Benchmark script: reproduce the PinwheelHawkes results from
Chen et al. (2021) "Neural Spatio-Temporal Point Processes" (ICLR 2021).

Trains multiple seeds and reports mean ± std NLL on the held-out test set,
matching the evaluation protocol from the original paper.

Usage
-----
    # Quick smoke test (1 seed, few epochs)
    python scripts/benchmark_pinwheel.py --n_epochs 5 --n_seeds 1

    # Full reproduction (matches paper: deep_stpp preset)
    python scripts/benchmark_pinwheel.py --preset deep_stpp --n_epochs 200 --n_seeds 5

    # NeuralSTPP preset (ODE model)
    python scripts/benchmark_pinwheel.py --preset neural_stpp --n_epochs 200 --n_seeds 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_stpp.data import STPPDataset, collate_fn
from unified_stpp.data.synthetic import generate_pinwheel_hawkes_stpp
from unified_stpp.registry import build_model
from unified_stpp.training import Trainer


# ── Original benchmark parameters (Chen et al. 2021) ────────────────────────
PINWHEEL_T = 30.0
PINWHEEL_NUM_ARMS = 10
PINWHEEL_MU = 0.05
PINWHEEL_ALPHA = 0.6
PINWHEEL_OMEGA = 10.0
PINWHEEL_N_TRAIN = 2000
PINWHEEL_N_VAL = 200
PINWHEEL_N_TEST = 200
PINWHEEL_SEED = 13579   # matches original ``temporary_seed(13579)``


def make_loaders(batch_size: int, seed: int):
    """Generate the PinwheelHawkes split and return DataLoaders."""
    total = PINWHEEL_N_TRAIN + PINWHEEL_N_VAL + PINWHEEL_N_TEST
    all_seqs = generate_pinwheel_hawkes_stpp(
        n_sequences=total,
        T=PINWHEEL_T,
        num_arms=PINWHEEL_NUM_ARMS,
        mu_per_arm=PINWHEEL_MU,
        alpha_offdiag=PINWHEEL_ALPHA,
        omega=PINWHEEL_OMEGA,
        seed=PINWHEEL_SEED,
    )
    train_seqs = all_seqs[:PINWHEEL_N_TRAIN]
    val_seqs   = all_seqs[PINWHEEL_N_TRAIN : PINWHEEL_N_TRAIN + PINWHEEL_N_VAL]
    test_seqs  = all_seqs[PINWHEEL_N_TRAIN + PINWHEEL_N_VAL :]

    train_ds = STPPDataset(train_seqs, normalize_time=True, normalize_space=True)
    val_ds   = STPPDataset(val_seqs,   normalize_time=True, normalize_space=True,
                           cov_mean=train_ds.cov_mean, cov_std=train_ds.cov_std)
    test_ds  = STPPDataset(test_seqs,  normalize_time=True, normalize_space=True,
                           cov_mean=train_ds.cov_mean, cov_std=train_ds.cov_std)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, train_ds


def run_seed(seed: int, args) -> dict:
    """Train one seed and return metrics dict."""
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader, train_ds = make_loaders(
        args.batch_size, seed=seed
    )

    model = build_model(
        config={},
        spatial_dim=2,
        hidden_dim=args.hidden_dim,
        preset=args.preset,
        n_marks=PINWHEEL_NUM_ARMS if args.use_marks else 0,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainer = Trainer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=device,
    )

    t0 = time.time()
    history = trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.n_epochs,
        log_every=args.log_every,
        early_stopping_patience=args.early_stopping_patience,
        restore_best=True,
    )
    elapsed = time.time() - t0

    # Evaluate on held-out test set
    test_metrics = trainer.evaluate(test_loader)
    test_nll = float(test_metrics["nll"])

    # Convert back to original (unnormalized) scale:
    # NLL_orig = NLL_norm - log(1 / (time_std * prod(loc_std)))
    #          = NLL_norm + log(time_std * prod(loc_std))
    # This matches the Jacobian correction for the change-of-variables.
    time_std = float(train_ds.time_std)
    loc_std  = train_ds.loc_std  # np.ndarray (d,)
    log_jacobian = float(np.log(time_std * np.prod(loc_std)))
    test_nll_orig = test_nll + log_jacobian

    result = {
        "seed": seed,
        "preset": args.preset,
        "hidden_dim": args.hidden_dim,
        "n_params": n_params,
        "test_nll_norm": test_nll,
        "test_nll_orig": test_nll_orig,
        "log_jacobian": log_jacobian,
        "best_val_nll": history.get("best_val_nll"),
        "best_epoch": history.get("best_epoch"),
        "train_time_sec": elapsed,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Pinwheel-Hawkes benchmark")
    parser.add_argument("--preset", default="deep_stpp",
                        choices=["deep_stpp", "neural_stpp", "dstpp"],
                        help="Model preset to benchmark")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--early_stopping_patience", type=int, default=20,
                        help="Early-stop patience (epochs). 0 disables.")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of seeds to average over")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit seed list (overrides --n_seeds)")
    parser.add_argument("--use_marks", action="store_true",
                        help="Model the arm index as a discrete mark")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    print(f"Benchmarking preset={args.preset}, hidden_dim={args.hidden_dim}")
    print(f"Dataset: PinwheelHawkes  T={PINWHEEL_T}  arms={PINWHEEL_NUM_ARMS}")
    print(f"Train/Val/Test: {PINWHEEL_N_TRAIN}/{PINWHEEL_N_VAL}/{PINWHEEL_N_TEST}")
    print(f"Seeds: {seeds}\n")

    all_results = []
    for seed in seeds:
        print(f"{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        result = run_seed(seed, args)
        all_results.append(result)
        print(
            f"  test NLL (norm)  : {result['test_nll_norm']:.4f}\n"
            f"  test NLL (orig)  : {result['test_nll_orig']:.4f}\n"
            f"  best val NLL     : {result['best_val_nll']}\n"
            f"  best epoch       : {result['best_epoch']}\n"
            f"  train time       : {result['train_time_sec']:.1f}s\n"
        )

    # ── Summary ─────────────────────────────────────────────────────────────
    norm_nlls = [r["test_nll_norm"] for r in all_results]
    orig_nlls = [r["test_nll_orig"] for r in all_results]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  preset          : {args.preset}")
    print(f"  n_seeds         : {len(seeds)}")
    print(f"  test NLL (norm) : {np.mean(norm_nlls):.4f} ± {np.std(norm_nlls):.4f}")
    print(f"  test NLL (orig) : {np.mean(orig_nlls):.4f} ± {np.std(orig_nlls):.4f}")
    print()
    print("  Individual results:")
    for r in all_results:
        print(f"    seed {r['seed']:2d}  norm={r['test_nll_norm']:.4f}  orig={r['test_nll_orig']:.4f}")

    summary = {
        "preset": args.preset,
        "hidden_dim": args.hidden_dim,
        "n_epochs": args.n_epochs,
        "seeds": seeds,
        "test_nll_norm_mean": float(np.mean(norm_nlls)),
        "test_nll_norm_std":  float(np.std(norm_nlls)),
        "test_nll_orig_mean": float(np.mean(orig_nlls)),
        "test_nll_orig_std":  float(np.std(orig_nlls)),
        "per_seed": all_results,
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.out}")

    return summary


if __name__ == "__main__":
    main()
