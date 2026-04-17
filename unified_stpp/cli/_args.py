"""Reusable argparse argument-group helpers for the unified_stpp CLI."""

from __future__ import annotations


def add_config_source_args(p) -> None:
    """Add ``--preset`` / ``--config`` as a mutually exclusive required group."""
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", help="Model preset name (e.g. auto_stpp)")
    g.add_argument("--config", help="Path to YAML config file")


def add_data_args(p, *, include_test: bool = True) -> None:
    """Add named-dataset and explicit split-file arguments.

    Validation happens in the config/data-resolution layer so the same contract
    is shared by the CLI, tests, and programmatic callers:

    - ``--dataset [--dataset-revision]`` for curated/local dataset resolution
    - ``--train --val [--test]`` for explicit JSONL file paths
    """
    p.add_argument(
        "--dataset",
        default=None,
        help=(
            "Named curated dataset, local dataset directory, or Hugging Face "
            "dataset repo/path like owner/repo[/subdir]."
        ),
    )
    p.add_argument(
        "--dataset-revision",
        default=None,
        help="Optional dataset revision when --dataset resolves through the hub.",
    )
    p.add_argument("--train", default=None, help="Path to train .jsonl")
    p.add_argument("--val",   default=None, help="Path to val .jsonl")
    if include_test:
        p.add_argument("--test", default=None, help="Path to test .jsonl (optional)")


def add_hpo_args(p, *, sentinel_defaults: bool = False) -> None:
    """Add ``--n_trials``, ``--search-alg``, ``--scheduler``.

    Parameters
    ----------
    sentinel_defaults:
        ``True`` (used by ``tune``) — all defaults are ``None`` so the command
        can distinguish "explicitly provided" from "not provided", enabling the
        YAML ``tuning:`` section to serve as the base config.
        ``False`` (used by ``bench``) — concrete defaults: 50 / "random" / "asha".
    """
    sfx = " (overrides YAML tuning.* if set)" if sentinel_defaults else ""
    p.add_argument(
        "--n_trials", type=int,
        default=None if sentinel_defaults else 50,
        help=f"Max HPO trials{sfx}",
    )
    p.add_argument(
        "--search-alg", dest="search_alg",
        default=None if sentinel_defaults else "random",
        choices=["random", "bayesian"],
        help=f"Proposal algorithm{sfx}",
    )
    p.add_argument(
        "--scheduler",
        default=None if sentinel_defaults else "asha",
        choices=["asha", "none"],
        help=f"Early-stopping policy{sfx}",
    )
