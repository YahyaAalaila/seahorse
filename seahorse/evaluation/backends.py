"""Backend-neutral access to evaluation capabilities.

Statistical runners expose capabilities directly. Neural runners keep their
existing event-model contract; no statistical object needs a mock Torch module.
"""

from __future__ import annotations

from typing import Any


def is_statistical_runner(runner: Any) -> bool:
    return getattr(runner, "runner_name", None) == "statstpp"


def evaluation_capabilities(runner: Any) -> Any:
    caps = getattr(runner, "capabilities", None)
    if caps is not None:
        return caps
    return runner.model.event_model.capabilities
