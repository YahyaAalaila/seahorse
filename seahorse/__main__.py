"""Module CLI entrypoint for ``seahorse``.

Stable v1 modes:

    python -m seahorse fit
    python -m seahorse tune
    python -m seahorse bench
    python -m seahorse evaluate

Datasets are resolved from Hugging Face dataset repositories or user-provided
local JSONL split paths/directories. Use ``--help`` on each mode for arguments.
"""

from __future__ import annotations

import argparse

from seahorse.cli import fit, tune, evaluate, bench


def main():
    parser = argparse.ArgumentParser(
        prog="python -m seahorse",
        description="Seahorse STPP CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fit.add_subparser(sub)
    tune.add_subparser(sub)
    evaluate.add_subparser(sub)
    bench.add_subparser(sub)
    args = parser.parse_args()
    {
        "fit":      fit.execute,
        "tune":     tune.execute,
        "evaluate": evaluate.execute,
        "bench":    bench.execute,
    }[args.command](args)


if __name__ == "__main__":
    main()
