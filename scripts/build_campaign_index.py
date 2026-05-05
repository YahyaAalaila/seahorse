#!/usr/bin/env python
"""Build a campaign-level index over HPO, bench, and evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    with open(path) as f:
        return json.load(f)


def _sorted_unique_paths(paths: list[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths})


def _find_hpo_manifests(campaign_root: Path) -> list[Path]:
    hpo_root = campaign_root / "hpo"
    if not hpo_root.exists():
        return []
    return _sorted_unique_paths(list(hpo_root.glob("*_hpo_manifest.json")))


def _find_bench_roots(campaign_root: Path) -> list[Path]:
    bench_root = campaign_root / "bench"
    if not bench_root.exists():
        return []
    return _sorted_unique_paths([path.parent for path in bench_root.rglob("bench_meta.json")])


def _find_evaluation_roots(campaign_root: Path) -> list[Path]:
    roots: list[Path] = []
    roots.extend(path.parent for path in campaign_root.rglob("evaluation_manifest.json"))
    for summary_path in campaign_root.rglob("summary.json"):
        parent = summary_path.parent
        if (parent / "derived_surfaces.npz").exists() or (parent / "data.npz").exists():
            roots.append(parent)
    return _sorted_unique_paths(roots)


def _classify_evaluation_root(root: Path) -> str:
    if (root / "evaluation_manifest.json").exists():
        return "metrics"
    if (root / "derived_surfaces.npz").exists():
        return "predictive_compare"
    if (root / "data.npz").exists():
        return "surface"
    return "unknown"


def _bench_entry(root: Path) -> dict[str, Any]:
    meta_path = root / "bench_meta.json"
    cell_index_path = root / "cell_index.json"
    results_path = root / "results.json"
    meta = _load_json(meta_path) if meta_path.exists() else {}
    cell_index = _load_json(cell_index_path) if cell_index_path.exists() else []
    hpo_provenance = meta.get("hpo_provenance") if isinstance(meta, dict) else None
    image_tags = sorted(
        {
            value.get("container_image_tag")
            for value in (hpo_provenance or {}).values()
            if isinstance(value, dict) and value.get("container_image_tag")
        }
    )
    return {
        "bench_root": str(root),
        "bench_id": meta.get("bench_id"),
        "git_sha": meta.get("git_sha"),
        "container_image_tags": image_tags,
        "datasets": meta.get("datasets"),
        "presets": meta.get("presets"),
        "results_path": str(results_path) if results_path.exists() else None,
        "cell_index_path": str(cell_index_path) if cell_index_path.exists() else None,
        "cell_count": len(cell_index) if isinstance(cell_index, list) else None,
        "meta": meta,
    }


def _evaluation_entry(root: Path) -> dict[str, Any]:
    eval_type = _classify_evaluation_root(root)
    summary = {}
    if eval_type == "metrics":
        summary = _load_json(root / "evaluation_manifest.json")
    elif (root / "summary.json").exists():
        summary = _load_json(root / "summary.json")
    return {
        "evaluation_root": str(root),
        "type": eval_type,
        "summary": summary,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_campaign_index(campaign_root: Path) -> dict[str, Any]:
    hpo_manifest_paths = _find_hpo_manifests(campaign_root)
    bench_roots = _find_bench_roots(campaign_root)
    evaluation_roots = _find_evaluation_roots(campaign_root)

    hpo = [
        {
            "manifest_path": str(path),
            "manifest": _load_json(path),
        }
        for path in hpo_manifest_paths
    ]
    benches = [_bench_entry(root) for root in bench_roots]
    cells: list[dict[str, Any]] = []
    for bench in benches:
        cell_index_path = bench.get("cell_index_path")
        if cell_index_path:
            for cell in _load_json(Path(cell_index_path)):
                row = dict(cell)
                row["bench_root"] = bench["bench_root"]
                row["bench_id"] = bench["bench_id"]
                cells.append(row)
    evaluations = [_evaluation_entry(root) for root in evaluation_roots]

    return {
        "campaign_root": str(campaign_root.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hpo": hpo,
        "benchmarks": benches,
        "cells": cells,
        "evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        required=True,
        help="Root directory for one experiment campaign, e.g. runs/<campaign_id>.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Defaults to <campaign-root>/campaign_index.json.",
    )
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root).expanduser().resolve()
    if not campaign_root.exists():
        raise SystemExit(f"Campaign root does not exist: {campaign_root}")

    index = build_campaign_index(campaign_root)
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out is not None
        else campaign_root / "campaign_index.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2, default=str)

    _write_csv(
        out_path.with_suffix(".cells.csv"),
        index["cells"],
        [
            "bench_id",
            "bench_root",
            "preset",
            "dataset_id",
            "seed",
            "run_dir",
            "run_result_path",
            "artifacts_path",
            "resolved_config_path",
            "checkpoint_path",
            "checkpoint_select",
            "nll_kind",
            "nll_report_space",
            "hpo_source",
            "hpo_manifest_path",
            "data_source_fingerprint",
        ],
    )
    _write_csv(
        out_path.with_suffix(".evaluations.csv"),
        [
            {
                "evaluation_root": row["evaluation_root"],
                "type": row["type"],
            }
            for row in index["evaluations"]
        ],
        ["evaluation_root", "type"],
    )


if __name__ == "__main__":
    main()
