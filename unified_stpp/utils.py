"""Shared utility functions for the unified_stpp package."""

from __future__ import annotations


def deep_update(base: dict, override: dict) -> None:
    """Recursively update *base* in-place with values from *override*.

    Nested dicts are merged rather than replaced; all other types are
    overwritten directly.
    """
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            deep_update(base[key], val)
        else:
            base[key] = val
