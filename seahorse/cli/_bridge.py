"""CLI-only helpers for translating argparse values into config inputs."""

from __future__ import annotations


def _set_dotted(mapping: dict, dotted_key: str, value) -> None:
    """Set ``mapping[a][b][c] = value`` for ``dotted_key="a.b.c"``."""
    parts = dotted_key.split(".")
    node = mapping
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def extract_explicit_cli_values(args, field_map: dict[str, str]) -> dict:
    """Return only explicitly provided CLI values mapped to config-style keys.

    ``field_map`` maps argparse destination names to dotted config paths.
    A value is considered explicit when it is not ``None``. This allows
    ``False`` from ``--no-*`` style flags to remain representable while keeping
    the CLI layer free of merge logic.
    """
    values: dict = {}
    for arg_name, config_path in field_map.items():
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        _set_dotted(values, config_path, value)
    return values


__all__ = ["extract_explicit_cli_values"]
