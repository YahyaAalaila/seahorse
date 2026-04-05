"""``evaluate`` subcommand — post-fit analysis on a saved run."""

from __future__ import annotations

from pathlib import Path


def add_subparser(sub) -> None:
    p = sub.add_parser(
        "evaluate",
        help="Post-fit analysis on a saved run (primary evaluation path)",
    )
    p.add_argument("--run", required=True,
                   help="Path to a saved run directory (produced by fit)")
    p.add_argument("--val", default=None,
                   help="Path to val .jsonl — required for --surface_viz or val-split upstream intensity plots")
    p.add_argument("--test", default=None,
                   help="Optional path to test .jsonl — used for notebook-faithful upstream intensity plots")
    p.add_argument("--out", default=None,
                   help="Output directory for artifacts (default: the run directory)")
    p.add_argument("--surface_viz", action="store_true",
                   help="Enable surface visualization workflow")
    p.add_argument("--surface_n_grid", type=int, default=50,
                   metavar="N", help="Grid resolution per axis (default: 50)")
    p.add_argument("--surface_n_time_steps", type=int, default=3,
                   metavar="N", help="Number of time steps to query (default: 3)")
    p.add_argument("--surface_render_mode", default="2d", choices=["2d", "3d"],
                   help="Render mode: '2d' heatmap (default) or '3d' surface")
    p.add_argument("--surface_animate", action="store_true",
                   help="Also save a GIF animation of the surface sequence")
    p.add_argument("--surface_history_length", type=int, default=10,
                   metavar="N", help="Number of history events to condition on (default: 10)")
    p.add_argument("--surface_history_split", default="val",
                   choices=["train", "val", "test"],
                   help="Which data split to draw history from (default: val)")
    p.add_argument(
        "--surface_history_mode",
        choices=["fixed", "rolling"], default=None,
        help=(
            "History strategy per animation frame. "
            "'rolling' (recommended for animation) uses all events strictly before "
            "each t_query, giving genuinely different surfaces per frame. "
            "Default when --surface_animate is set: 'rolling'. "
            "Default otherwise: 'fixed'."
        ),
    )
    p.add_argument(
        "--surface_t_query_mode",
        choices=["after_history", "uniform"], default="after_history",
        help=(
            "How to place query times across the sequence. "
            "'uniform' spans the entire sequence time range and typically gives "
            "more visibly distinct frames. "
            "'after_history' with small --surface_horizon can yield subtle changes."
        ),
    )
    p.add_argument(
        "--surface_horizon",
        type=float, default=1.0,
        help="Time horizon past the last history event for 'after_history' mode (default: 1.0).",
    )
    p.add_argument(
        "--surface_reference_mode",
        choices=["none", "empirical_kde", "sthp_gt"], default="none",
        help=(
            "Reference surface to show alongside the model. "
            "'empirical_kde': marginal spatial KDE from the sequence events (time-independent proxy). "
            "'sthp_gt': exact conditional intensity λ*(t,s|H) from an STHP model — "
            "requires --surface_sthp_meta pointing to the dataset_meta.json produced by "
            "gen_sthp_splits.py. "
            "'none' (default): no reference."
        ),
    )
    p.add_argument(
        "--surface_sthp_meta",
        default=None,
        metavar="PATH",
        help=(
            "Path to dataset_meta.json for STHP ground-truth reference. "
            "Required when --surface_reference_mode sthp_gt is set."
        ),
    )
    p.add_argument(
        "--surface_reference_first",
        action="store_true", default=False,
        help=(
            "Put the reference surface on the LEFT and the model on the RIGHT in "
            "all outputs (animation frames, multi-panel figures, individual files). "
            "Default: model left, reference right."
        ),
    )
    p.add_argument(
        "--surface_no_share_colorscale",
        action="store_true", default=False,
        help=(
            "Disable the fixed global colorscale in GIF animations. "
            "By default every frame uses the same vmin/vmax per surface type so "
            "colors are comparable across time steps. "
            "Pass this flag to let each frame auto-scale independently."
        ),
    )
    p.add_argument(
        "--upstream_intensity_viz",
        action="store_true",
        help=(
            "Generate a notebook-faithful calc_lamb -> plot_lambst_interactive HTML "
            "artifact for deep_stpp or auto_stpp."
        ),
    )
    p.add_argument(
        "--upstream_intensity_split",
        default="test",
        choices=["val", "test"],
        help="Which split to use for notebook-faithful intensity plotting (default: test)",
    )
    p.add_argument(
        "--upstream_intensity_seq_idx",
        type=int,
        default=2,
        metavar="N",
        help="Sequence index within the chosen split for notebook-faithful intensity plotting (default: 2)",
    )
    p.add_argument(
        "--upstream_intensity_x_nstep",
        type=int,
        default=101,
        metavar="N",
        help="Number of x-grid points for notebook-faithful intensity plotting (default: 101)",
    )
    p.add_argument(
        "--upstream_intensity_y_nstep",
        type=int,
        default=101,
        metavar="N",
        help="Number of y-grid points for notebook-faithful intensity plotting (default: 101)",
    )
    p.add_argument(
        "--upstream_intensity_t_nstep",
        type=int,
        default=201,
        metavar="N",
        help="Number of time frames for notebook-faithful intensity plotting (default: 201)",
    )
    p.add_argument(
        "--upstream_intensity_no_round_time",
        dest="upstream_intensity_round_time",
        action="store_false",
        help="Disable the upstream notebook-style rounded time-grid start/end behavior",
    )
    p.add_argument(
        "--upstream_intensity_trunc",
        dest="upstream_intensity_trunc",
        action="store_true",
        default=None,
        help="Override the preset and truncate history for notebook-faithful intensity plotting",
    )
    p.add_argument(
        "--upstream_intensity_no_trunc",
        dest="upstream_intensity_trunc",
        action="store_false",
        help="Override the preset and disable history truncation for notebook-faithful intensity plotting",
    )
    p.add_argument(
        "--upstream_intensity_xmin",
        type=float,
        default=None,
        help="Optional original-space xmin override for notebook-faithful intensity plotting",
    )
    p.add_argument(
        "--upstream_intensity_xmax",
        type=float,
        default=None,
        help="Optional original-space xmax override for notebook-faithful intensity plotting",
    )
    p.add_argument(
        "--upstream_intensity_ymin",
        type=float,
        default=None,
        help="Optional original-space ymin override for notebook-faithful intensity plotting",
    )
    p.add_argument(
        "--upstream_intensity_ymax",
        type=float,
        default=None,
        help="Optional original-space ymax override for notebook-faithful intensity plotting",
    )
    p.add_argument(
        "--upstream_intensity_heatmap",
        action="store_true",
        help="Render notebook-faithful intensity as a heatmap instead of a 3D surface",
    )
    p.add_argument(
        "--upstream_intensity_out_html",
        default=None,
        help="Optional explicit HTML output path for notebook-faithful intensity plotting",
    )
    p.set_defaults(upstream_intensity_round_time=True)


def execute(args) -> None:
    from unified_stpp.runner import STPPRunner
    from unified_stpp.evaluation.intensity import calc_lamb_from_runner
    from unified_stpp.evaluation.surface import SurfaceEvalSpec
    from unified_stpp.viz import plot_lambst_interactive
    from unified_stpp.viz.workflow import SurfaceVizConfig
    from unified_stpp.utils import load_jsonl

    runner = STPPRunner.load(args.run)
    val_seqs = load_jsonl(args.val) if args.val else None
    test_seqs = load_jsonl(args.test) if args.test else None

    surface_viz = None
    if args.surface_viz:
        if val_seqs is None:
            raise ValueError("--val PATH is required when --surface_viz is set.")
        # When --surface_animate is set and the user did not explicitly pass
        # --surface_history_mode, default to rolling (each frame conditions on
        # events before its query time, giving genuinely distinct surfaces).
        # For static panels the conservative 'fixed' default applies.
        history_mode = args.surface_history_mode
        if history_mode is None:
            history_mode = "rolling" if args.surface_animate else "fixed"
        eval_spec = SurfaceEvalSpec(
            split=args.surface_history_split,
            history_mode=history_mode,
            history_length=args.surface_history_length,
            t_query_mode=args.surface_t_query_mode,
            n_time_steps=args.surface_n_time_steps,
            horizon=args.surface_horizon,
            n_grid=args.surface_n_grid,
        )
        reference_provider = None
        if args.surface_reference_mode == "sthp_gt":
            if not args.surface_sthp_meta:
                raise ValueError(
                    "--surface_sthp_meta PATH is required with "
                    "--surface_reference_mode sthp_gt"
                )
            from unified_stpp.viz.reference import STHPGroundTruthProvider
            reference_provider = STHPGroundTruthProvider.from_meta_file(
                args.surface_sthp_meta
            )
        surface_viz = SurfaceVizConfig(
            eval_spec=eval_spec,
            enabled=True,
            render_mode=args.surface_render_mode,
            animate=args.surface_animate,
            save_panel=True,
            save_individual=True,
            reference_mode=args.surface_reference_mode,
            reference_provider=reference_provider,
            reference_first=args.surface_reference_first,
            animate_share_colorscale=not args.surface_no_share_colorscale,
        )

    run_dir = Path(args.out) if args.out else None
    artifacts = {}
    if surface_viz is not None:
        artifacts.update(
            runner.evaluate(val_seqs=val_seqs, surface_viz=surface_viz, run_dir=run_dir)
        )

    if args.upstream_intensity_viz:
        split_to_sequences = {"val": val_seqs, "test": test_seqs}
        chosen_split = args.upstream_intensity_split
        chosen_sequences = split_to_sequences.get(chosen_split)
        if chosen_sequences is None:
            flag = "--val" if chosen_split == "val" else "--test"
            raise ValueError(
                f"{flag} PATH is required when --upstream_intensity_split {chosen_split!r} is selected."
            )

        cube = calc_lamb_from_runner(
            runner=runner,
            sequences=chosen_sequences,
            seq_idx=args.upstream_intensity_seq_idx,
            split=chosen_split,
            x_nstep=args.upstream_intensity_x_nstep,
            y_nstep=args.upstream_intensity_y_nstep,
            t_nstep=args.upstream_intensity_t_nstep,
            round_time=args.upstream_intensity_round_time,
            xmin=args.upstream_intensity_xmin,
            xmax=args.upstream_intensity_xmax,
            ymin=args.upstream_intensity_ymin,
            ymax=args.upstream_intensity_ymax,
            trunc=args.upstream_intensity_trunc,
        )
        fig = plot_lambst_interactive(
            cube.lambs,
            cube.x_range,
            cube.y_range,
            cube.t_range,
            heatmap=args.upstream_intensity_heatmap,
            show=False,
            master_title=(
                f"{runner.config.model.preset} notebook-faithful intensity "
                f"({chosen_split} seq {args.upstream_intensity_seq_idx})"
            ),
        )

        base_dir = Path(args.out) if args.out else Path(args.run)
        if args.upstream_intensity_out_html:
            html_path = Path(args.upstream_intensity_out_html)
        else:
            html_path = (
                base_dir
                / "surfaces"
                / (
                    f"upstream_intensity_{runner.config.model.preset}_"
                    f"{chosen_split}_seq{args.upstream_intensity_seq_idx}.html"
                )
            )
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        artifacts["upstream_intensity_html"] = html_path

    if artifacts:
        for name, path in artifacts.items():
            print(f"  {name}: {path}")
    else:
        print(
            "No evaluation workflows were enabled. Pass --surface_viz and/or "
            "--upstream_intensity_viz to generate artifacts."
        )
