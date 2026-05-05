#!/usr/bin/env python3
"""Resolve HawkesNest campaign/preset targets for predictive KDE evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SUPPORTED_PRESETS = frozenset({"auto_stpp", "deep_stpp", "diffusion_stpp", "smash"})


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _discover_campaign_roots(inputs: list[str]) -> list[Path]:
    roots: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            continue
        manifest_path = path / "manifests" / "campaign_manifest.json"
        if manifest_path.exists():
            roots.add(path)
            continue
        if path.is_dir():
            roots.update(
                candidate.parent.parent.resolve()
                for candidate in path.rglob("manifests/campaign_manifest.json")
            )
    return sorted(roots)


def build_targets(roots: list[str]) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for campaign_root in _discover_campaign_roots(roots):
        manifest_path = campaign_root / "manifests" / "campaign_manifest.json"
        run_index_path = campaign_root / "manifests" / "run_index.jsonl"
        if not manifest_path.exists() or not run_index_path.exists():
            continue
        manifest = _load_json(manifest_path)
        run_rows = _load_jsonl(run_index_path)
        suite = str(manifest.get("suite") or campaign_root.parent.name)
        suite_path_raw = manifest.get("suite_path")
        suite_path = (
            Path(str(suite_path_raw)).expanduser().resolve()
            if suite_path_raw
            else None
        )
        if suite_path is not None and not suite_path.exists():
            suite_path = None
        discovered_presets = sorted(
            {
                str(row.get("preset"))
                for row in run_rows
                if str(row.get("preset") or "") in SUPPORTED_PRESETS
            }
        )
        for preset in discovered_presets:
            targets.append(
                {
                    "campaign_id": campaign_root.name,
                    "campaign_root": str(campaign_root),
                    "suite": suite,
                    "suite_path": "" if suite_path is None else str(suite_path),
                    "preset": preset,
                }
            )
    return sorted(
        targets,
        key=lambda row: (
            row["suite"],
            row["campaign_id"],
            row["preset"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Campaign root(s) or parent directories.")
    parser.add_argument("--format", choices=("json", "tsv", "usv"), default="tsv")
    args = parser.parse_args()

    rows = build_targets(list(args.roots))
    if args.format == "json":
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    delim = "\x1f" if args.format == "usv" else "\t"
    for row in rows:
        fields = (
            row["campaign_id"],
            row["campaign_root"],
            row["suite"],
            row["suite_path"],
            row["preset"],
        )
        sys.stdout.write(delim.join(fields) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
