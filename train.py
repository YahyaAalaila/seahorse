"""
Legacy compatibility wrapper for the archived root training script.

The maintained entrypoint is ``python -m unified_stpp`` with subcommands
``fit``, ``evaluate``, ``bench``, and ``tune``.

The pre-cleanup implementation has been preserved at ``archive/train.py``.
"""

from __future__ import annotations

import sys

from unified_stpp.__main__ import main as cli_main


_LIVE_SUBCOMMANDS = {"fit", "evaluate", "bench", "tune"}


def main() -> None:
    print(
        "train.py is a legacy wrapper. Prefer `python -m unified_stpp ...`.",
        file=sys.stderr,
    )

    if len(sys.argv) <= 1:
        raise SystemExit("Use `python -m unified_stpp fit ...`.")

    # Preserve the common old `python train.py --preset ...` workflow by
    # routing argument lists without an explicit subcommand to `fit`.
    if sys.argv[1] not in _LIVE_SUBCOMMANDS:
        sys.argv.insert(1, "fit")

    cli_main()


if __name__ == "__main__":
    main()
