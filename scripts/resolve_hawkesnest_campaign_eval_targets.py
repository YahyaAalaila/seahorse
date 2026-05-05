#!/usr/bin/env python
"""Resolve per-run evaluation targets from one synthetic campaign directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PREDICTIVE_PRESETS = frozenset(
    {
        "auto_stpp",
        "deep_stpp",
        "nsmpp",
        "rmtpp_gmm",
        "thp_gmm",
        "smash",
        "diffusion_stpp",
        "njsde",
        "neural_jumpcnf",
        "neural_attncnf",
        "poisson_gmm",
        "hawkes_gmm",
        "selfcorrecting_gmm",
        "poisson_cnf",
        "hawkes_cnf",
        "selfcorrecting_cnf",
        "poisson_tvcnf",
        "hawkes_tvcnf",
        "selfcorrecting_tvcnf",
    }
)

SURFACE_HISTORY_PRESETS = frozenset({"auto_stpp", "deep_stpp"})
SURFACE_FUTURE_EXACT_PRESETS = frozenset(
    {"njsde", "neural_jumpcnf", "neural_attncnf"}
)

GROUP_PRESETS = {
    "predictive": PREDICTIVE_PRESETS,
    "surface_history": SURFACE_HISTORY_PRESETS,
    "surface_future_exact": SURFACE_FUTURE_EXACT_PRESETS,
    "all": None,
}


def _load_json(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _level_index(config_id: str) -> int | None:
    digits = []
    for ch in reversed(config_id):
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return None
    return int("".join(reversed(digits)))


def build_targets(
    *,
    campaign_root: Path,
    split: str,
    group: str,
    presets: set[str] | None = None,
) -> list[dict[str, Any]]:
    campaign_root = campaign_root.expanduser().resolve()
    manifest_path = campaign_root / "manifests" / "campaign_manifest.json"
    run_index_path = campaign_root / "manifests" / "run_index.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"campaign_manifest.json not found under {campaign_root}")
    if not run_index_path.exists():
        raise FileNotFoundError(f"run_index.jsonl not found under {campaign_root}")

    manifest = _load_json(manifest_path)
    rows = _load_jsonl(run_index_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object.")
    if not isinstance(rows, list):
        raise ValueError(f"{run_index_path} must contain JSON-lines objects.")

    allowed = GROUP_PRESETS[group]
    if presets is not None:
        allowed = frozenset(presets) if allowed is None else frozenset(allowed) & frozenset(presets)

    out: list[dict[str, Any]] = []
    seen_run_dirs: set[Path] = set()
    campaign_id = campaign_root.name
    suite = str(manifest.get("suite") or campaign_root.parent.name)
    manifest_suite_path = manifest.get("suite_path")
    suite_path = (
        Path(str(manifest_suite_path)).expanduser().resolve()
        if manifest_suite_path
        else None
    )
    if suite_path is not None and not suite_path.exists():
        suite_path = None

    for row in rows:
        if not isinstance(row, dict):
            continue
        preset = str(row.get("preset") or "")
        if not preset:
            continue
        if allowed is not None and preset not in allowed:
            continue

        run_dir = Path(str(row.get("run_dir") or "")).expanduser().resolve()
        if not run_dir:
            continue
        if run_dir in seen_run_dirs:
            continue
        seen_run_dirs.add(run_dir)

        run_result_path = Path(str(row.get("run_result_path") or (run_dir / "run_result.json"))).expanduser().resolve()
        if not run_result_path.exists():
            continue
        run_result = _load_json(run_result_path)

        config_id = str(row.get("config_id") or run_result.get("dataset_id") or "")
        test_path = Path(str(row.get("test_path") or "")).expanduser().resolve()
        train_path_raw = row.get("train_path")
        train_path = None if not train_path_raw else Path(str(train_path_raw)).expanduser().resolve()
        if not test_path.exists():
            continue
        if train_path is not None and not train_path.exists():
            train_path = None

        row_suite_path = suite_path
        if row_suite_path is None and test_path.exists():
            candidate = test_path.parent.parent
            if candidate.exists():
                row_suite_path = candidate
        gt_intensity_path = None
        gt_params_path = None
        if row_suite_path is not None:
            gt_dir = row_suite_path / "ground_truth"
            intensity_candidate = gt_dir / f"{config_id}_intensity_grid_r0.npz"
            params_candidate = gt_dir / f"{config_id}_params.json"
            if intensity_candidate.exists():
                gt_intensity_path = intensity_candidate.resolve()
            if params_candidate.exists():
                gt_params_path = params_candidate.resolve()

        seed = int(row.get("seed") if row.get("seed") is not None else run_result.get("seed", 0))
        out.append(
            {
                "campaign_id": campaign_id,
                "campaign_root": str(campaign_root),
                "suite": suite,
                "suite_path": None if row_suite_path is None else str(row_suite_path),
                "config_id": config_id,
                "level_index": _level_index(config_id),
                "preset": preset,
                "seed": seed,
                "run_dir": str(run_dir),
                "run_result_path": str(run_result_path),
                "train_path": None if train_path is None else str(train_path),
                "test_path": str(test_path),
                "ground_truth_intensity_path": (
                    None if gt_intensity_path is None else str(gt_intensity_path)
                ),
                "ground_truth_params_path": (
                    None if gt_params_path is None else str(gt_params_path)
                ),
            }
        )
    return sorted(
        out,
        key=lambda row: (
            str(row["suite"]),
            10**9 if row["level_index"] is None else int(row["level_index"]),
            str(row["config_id"]),
            str(row["preset"]),
            int(row["seed"]),
            str(row["run_dir"]),
        ),
    )


def _emit_delimited(rows: list[dict[str, Any]], delimiter: str) -> None:
    for row in rows:
        fields = (
            row["campaign_id"],
            row["suite"],
            row["suite_path"] or "",
            row["config_id"],
            "" if row["level_index"] is None else str(int(row["level_index"])),
            row["preset"],
            str(int(row["seed"])),
            row["run_dir"],
            row["test_path"],
            row["train_path"] or "",
            row["ground_truth_intensity_path"] or "",
            row["ground_truth_params_path"] or "",
        )
        print(delimiter.join(fields))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, help="Synthetic campaign root.")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Evaluation split label. Currently informational; test_path remains the run-index test split.",
    )
    parser.add_argument(
        "--group",
        default="predictive",
        choices=tuple(GROUP_PRESETS),
        help="Preset capability group to resolve.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        help="Optional explicit preset filter. Repeat to keep multiple presets.",
    )
    parser.add_argument(
        "--format",
        default="tsv",
        choices=("tsv", "json", "usv"),
        help="Output format.",
    )
    args = parser.parse_args()

    targets = build_targets(
        campaign_root=Path(args.campaign_root),
        split=str(args.split),
        group=str(args.group),
        presets=set(args.preset or []) or None,
    )
    if args.format == "json":
        json.dump(targets, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.format == "usv":
        _emit_delimited(targets, "\x1f")
    else:
        _emit_delimited(targets, "\t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
