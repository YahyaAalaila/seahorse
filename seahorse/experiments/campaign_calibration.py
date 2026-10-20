"""``seahorse campaign calibrate``: compensator-based calibration of one campaign cell."""

from __future__ import annotations

import argparse
import json


def add_subparser(commands) -> None:
    parser = commands.add_parser(
        "calibrate",
        help="Temporal PIT/CRPS and spatial HPD-PIT of one completed cell from the NLL compensator",
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--model-dir", default=None,
                        help="default: <job>/model if present, else status.json['checkpoint']")
    parser.add_argument("--out", required=True)
    parser.add_argument("--c-hours", type=float, default=None,
                        help="CRPS threshold in hours (default: max train inter-event time rounded up to 0.1 raw unit)")
    parser.add_argument("--tau-grid-n", type=int, default=1024)
    parser.add_argument("--tau-factor", type=float, default=4.0)
    parser.add_argument("--crps-scope", choices=["auto", "all", "subset"], default="auto")
    parser.add_argument("--crps-subset-size", type=int, default=20000)
    parser.add_argument("--crps-subset-seed", type=int, default=20261020)
    parser.add_argument("--spatial-grid", type=int, default=128)
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument("--no-crps", action="store_true")
    parser.add_argument("--mask-npz", default=None,
                        help="optional unobserved-span mask (sequence_index, target_event_index, excluded)")
    parser.add_argument("--solver-factor", type=int, default=1,
                        help="neural-ODE family: tighten the temporal solver by this factor")
    parser.add_argument("--skip-eval-forward", action="store_true",
                        help="neural-ODE family: do not re-run the campaign NLL forward")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--commit-every", type=int, default=64)
    parser.add_argument("--allow-binding-mismatch", action="store_true")
    parser.add_argument("--closed-form-check", action="store_true",
                        help="statistical Poisson/Hawkes: compare CRPS_c with analytic/quadrature references")


def execute(args) -> dict:
    from seahorse.evaluation.v1.calibration import evaluate_calibration

    summary = evaluate_calibration(
        args.manifest, args.job, args.out, model_dir=args.model_dir, c_hours=args.c_hours,
        tau_grid_n=args.tau_grid_n, tau_factor=args.tau_factor, crps_scope=args.crps_scope,
        crps_subset_size=args.crps_subset_size, crps_subset_seed=args.crps_subset_seed,
        spatial_grid=args.spatial_grid, spatial=not args.no_spatial, crps=not args.no_crps,
        mask_npz=args.mask_npz, solver_factor=args.solver_factor, skip_eval_forward=args.skip_eval_forward,
        limit=args.limit, batch_size=args.batch_size, device=args.device, threads=args.threads,
        commit_every=args.commit_every, allow_binding_mismatch=args.allow_binding_mismatch,
        closed_form_check=args.closed_form_check,
    )
    brief = {
        "preset": summary["preset"], "dataset": summary["dataset"], "n": summary["n_targets_scored"],
        "c_hours": summary.get("c_hours"), "crps_scope": summary.get("crps_scope", {}).get("policy"),
        "pit_ks": summary["pit"]["ks"]["ks"], "crps_c_hours": summary.get("crps", {}).get("crps_c_mean_hours"),
        "median_abs_err_hours": summary.get("point_forecast", {}).get("median_abs_error_hours_median"),
        "hpd_pit_ks": summary.get("spatial", {}).get("hpd_pit_ks", {}).get("ks"),
        "nll_repro_max_abs": summary.get("campaign_nll_reproduction", {}).get("max_abs_diff"),
        "errors": len(summary["errors"]), "wall_s": summary["timing"]["wall_seconds_loop"],
    }
    print(json.dumps(brief, indent=1, default=str), flush=True)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    execute(parser.parse_args(argv))


if __name__ == "__main__":
    main()
