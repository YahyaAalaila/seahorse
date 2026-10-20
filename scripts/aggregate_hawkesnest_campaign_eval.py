#!/usr/bin/env python
"""Aggregate synthetic campaign evaluation artifacts into tables and plots."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _discover_campaign_roots(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if (root / "manifests" / "campaign_manifest.json").exists():
        return [root.resolve()]
    return sorted(
        {
            path.parent.parent.resolve()
            for path in root.rglob("manifests/campaign_manifest.json")
        }
    )


def _level_index(config_id: str) -> int | None:
    match = re.search(r"(\d+)$", str(config_id))
    if match is None:
        return None
    return int(match.group(1))


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._") or "item"


PRESET_LABELS: dict[str, str] = {
    "auto_stpp": "AutoSTPP",
    "deep_stpp": "DeepSTPP",
    "diffusion_stpp": "DSTPP",
    "smash": "SMASH",
    "nsmpp": "NSMPP",
    "neural_cond_gmm": "NJSDE",
    "neural_attncnf": "Attn-CNF",
    "neural_jumpcnf": "Jump-CNF",
    "rmtpp_gmm": "RMTPP",
    "thp_gmm": "THP",
    "poisson_gmm": "Poisson+GMM",
    "poisson_cnf": "Poisson+CNF",
    "poisson_tvcnf": "Poisson+TV-CNF",
    "hawkes_gmm": "Hawkes+GMM",
    "hawkes_cnf": "Hawkes+CNF",
    "hawkes_tvcnf": "Hawkes+TV-CNF",
    "selfcorrecting_gmm": "SelfCorr+GMM",
    "selfcorrecting_cnf": "SelfCorr+CNF",
    "selfcorrecting_tvcnf": "SelfCorr+TV-CNF",
}


def _preset_label(preset: str) -> str:
    return PRESET_LABELS.get(preset, preset)


def _suite_level_descriptions(suite_path: Path | None) -> tuple[dict[str, int], dict[str, str]]:
    if suite_path is None:
        return {}, {}
    metadata_path = suite_path / "metadata.json"
    if not metadata_path.exists():
        return {}, {}
    payload = _load_json(metadata_path)
    if not isinstance(payload, dict):
        return {}, {}
    level_order: dict[str, int] = {}
    descriptions: dict[str, str] = {}
    for idx, row in enumerate(payload.get("levels", [])):
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        if label is None:
            continue
        level_order[str(label)] = idx
        description = row.get("description")
        if description is not None:
            descriptions[str(label)] = str(description)
    return level_order, descriptions


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _metric_profile_from_manifest(manifest: dict[str, Any], eval_dir: Path) -> str:
    metrics = manifest.get("metrics")
    if isinstance(metrics, dict):
        profile = metrics.get("profile")
        if profile:
            return str(profile)
    name = eval_dir.name
    if "_" in name:
        return name.split("_", 1)[0]
    return name


def _resolve_transferred_run_dir(campaign_root: Path, raw_run_dir: Any) -> Path:
    run_dir = Path(str(raw_run_dir or "")).expanduser()
    if run_dir.exists():
        return run_dir.resolve()

    parts = run_dir.parts
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] != campaign_root.name:
            continue
        candidate = campaign_root.joinpath(*parts[idx + 1 :])
        if candidate.exists():
            return candidate.resolve()
    return run_dir.resolve()


def _collect_campaign_rows(
    campaign_root: Path,
    *,
    requested_profiles: set[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = _load_json(campaign_root / "manifests" / "campaign_manifest.json")
    suite = str(manifest.get("suite") or campaign_root.parent.name)
    suite_path_raw = manifest.get("suite_path")
    suite_path = (
        Path(str(suite_path_raw)).expanduser().resolve()
        if suite_path_raw
        else None
    )
    if suite_path is not None and not suite_path.exists():
        suite_path = None
    level_order, level_descriptions = _suite_level_descriptions(suite_path)

    metric_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    run_count = 0

    run_index_path = campaign_root / "manifests" / "run_index.jsonl"
    if not run_index_path.exists():
        return [], [], {
            "campaign_id": campaign_root.name,
            "campaign_root": str(campaign_root),
            "suite": suite,
            "suite_path": None if suite_path is None else str(suite_path),
            "run_count": 0,
            "metric_row_count": 0,
            "surface_row_count": 0,
            "reason": "missing_run_index",
        }

    with open(run_index_path) as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                continue
            run_dir = _resolve_transferred_run_dir(campaign_root, row.get("run_dir"))
            if not run_dir.exists():
                continue
            run_count += 1
            config_id = str(row.get("config_id") or "")
            preset = str(row.get("preset") or "")
            seed = int(row.get("seed", 0))
            level_idx = level_order.get(config_id)
            if level_idx is None:
                level_idx = _level_index(config_id)
            config_description = level_descriptions.get(config_id)

            for metrics_path in sorted((run_dir / "evaluate" / "metrics").glob("*/metrics.json")):
                eval_dir = metrics_path.parent
                eval_manifest_path = eval_dir / "evaluation_manifest.json"
                if not eval_manifest_path.exists():
                    continue
                eval_manifest = _load_json(eval_manifest_path)
                if not isinstance(eval_manifest, dict):
                    continue
                profile = _metric_profile_from_manifest(eval_manifest, eval_dir)
                if requested_profiles is not None and profile not in requested_profiles:
                    continue
                metrics = _load_json(metrics_path)
                if not isinstance(metrics, dict):
                    continue
                for metric_name, payload in sorted(metrics.items()):
                    if not isinstance(payload, dict):
                        continue
                    value = _to_float(payload.get("value"))
                    row_out = {
                        "campaign_id": campaign_root.name,
                        "campaign_root": str(campaign_root),
                        "suite": suite,
                        "suite_path": None if suite_path is None else str(suite_path),
                        "config_id": config_id,
                        "config_description": config_description,
                        "level_index": level_idx,
                        "preset": preset,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "profile": profile,
                        "eval_dir": str(eval_dir),
                        "metric": metric_name,
                        "value": value,
                        "available": bool(payload.get("available", False)),
                        "method": payload.get("method"),
                        "reason": payload.get("reason"),
                    }
                    metric_rows.append(row_out)

            for artifacts_path in sorted((run_dir / "evaluate" / "surface").glob("*/artifacts.json")):
                summary_path = artifacts_path.parent / "summary.json"
                if not summary_path.exists():
                    continue
                summary = _load_json(summary_path)
                if not isinstance(summary, dict):
                    continue
                surface_rows.append(
                    {
                        "campaign_id": campaign_root.name,
                        "campaign_root": str(campaign_root),
                        "suite": suite,
                        "config_id": config_id,
                        "config_description": config_description,
                        "level_index": level_idx,
                        "preset": preset,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "surface_profile": str(summary.get("profile") or artifacts_path.parent.name),
                        "surface_dir": str(artifacts_path.parent),
                        "artifacts_json": str(artifacts_path),
                        "summary_json": str(summary_path),
                        "interactive_html": (
                            str((artifacts_path.parent / "intensity_interactive.html").resolve())
                            if (artifacts_path.parent / "intensity_interactive.html").exists()
                            else (
                                str((artifacts_path.parent / f"{summary.get('primary_value_name', 'surface')}_interactive.html").resolve())
                                if (artifacts_path.parent / f"{summary.get('primary_value_name', 'surface')}_interactive.html").exists()
                                else None
                            )
                        ),
                    }
                )

    campaign_summary = {
        "campaign_id": campaign_root.name,
        "campaign_root": str(campaign_root),
        "suite": suite,
        "suite_path": None if suite_path is None else str(suite_path),
        "run_count": run_count,
        "metric_row_count": len(metric_rows),
        "surface_row_count": len(surface_rows),
    }
    return metric_rows, surface_rows, campaign_summary


def _summarize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    extra: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not row.get("available"):
            continue
        value = _to_float(row.get("value"))
        if value is None:
            continue
        key = (
            row["suite"],
            row["config_id"],
            row["level_index"],
            row["preset"],
            row["profile"],
            row["metric"],
        )
        grouped[key].append(value)
        extra[key] = {
            "suite": row["suite"],
            "config_id": row["config_id"],
            "config_description": row.get("config_description"),
            "level_index": row["level_index"],
            "preset": row["preset"],
            "profile": row["profile"],
            "metric": row["metric"],
        }

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = np.asarray(grouped[key], dtype=np.float64)
        meta = dict(extra[key])
        meta.update(
            {
                "n_values": int(values.shape[0]),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
        summary_rows.append(meta)
    return summary_rows


def _plot_metric_curves(summary_rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    by_suite_profile_metric: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_suite_profile_metric[(row["suite"], row["profile"], row["metric"])].append(row)

    for (suite, profile, metric), rows in sorted(by_suite_profile_metric.items()):
        rows = sorted(
            rows,
            key=lambda row: (
                10**9 if row["level_index"] is None else int(row["level_index"]),
                str(row["config_id"]),
                str(row["preset"]),
            ),
        )
        x_labels = []
        x_lookup: dict[str, int] = {}
        for row in rows:
            key = str(row["config_id"])
            if key not in x_lookup:
                x_lookup[key] = len(x_labels)
                x_labels.append(key)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        by_preset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_preset[str(row["preset"])].append(row)

        for preset, preset_rows in sorted(by_preset.items()):
            preset_rows = sorted(
                preset_rows,
                key=lambda row: (
                    10**9 if row["level_index"] is None else int(row["level_index"]),
                    str(row["config_id"]),
                ),
            )
            xs = np.asarray([x_lookup[str(row["config_id"])] for row in preset_rows], dtype=np.float64)
            ys = np.asarray([float(row["mean"]) for row in preset_rows], dtype=np.float64)
            stds = np.asarray([float(row["std"]) for row in preset_rows], dtype=np.float64)
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=_preset_label(preset))
            if np.any(stds > 0.0):
                ax.fill_between(xs, ys - stds, ys + stds, alpha=0.18)

        ax.set_title(f"{suite} | {profile} | {metric}")
        ax.set_xlabel("config")
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(x_labels)), x_labels)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncols=2 if len(by_preset) > 6 else 1)
        fig.tight_layout()

        plot_path = out_dir / "plots" / _slugify(suite) / _slugify(profile) / f"{_slugify(metric)}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        plots.append(
            {
                "suite": suite,
                "profile": profile,
                "metric": metric,
                "plot_path": str(plot_path),
            }
        )
    return plots


def _write_report_html(
    out_dir: Path,
    *,
    campaign_summaries: list[dict[str, Any]],
    plots: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
) -> None:
    suites = sorted({row["suite"] for row in plots})
    plots_by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plots:
        plots_by_suite[row["suite"]].append(row)

    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>HawkesNest Eval Aggregate</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} img{max-width:100%;border:1px solid #ddd;} table{border-collapse:collapse;} th,td{border:1px solid #ddd;padding:6px 8px;} h2,h3{margin-top:28px;}</style>",
        "</head><body>",
        "<h1>HawkesNest Campaign Evaluation</h1>",
        "<h2>Campaigns</h2>",
        "<table><thead><tr><th>campaign</th><th>suite</th><th>runs</th><th>metric rows</th><th>surface rows</th></tr></thead><tbody>",
    ]
    for row in campaign_summaries:
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['campaign_id']))}</td>"
            f"<td>{html.escape(str(row['suite']))}</td>"
            f"<td>{int(row['run_count'])}</td>"
            f"<td>{int(row['metric_row_count'])}</td>"
            f"<td>{int(row['surface_row_count'])}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")

    for suite in suites:
        lines.append(f"<h2>{html.escape(suite)}</h2>")
        for row in sorted(plots_by_suite[suite], key=lambda item: (item["profile"], item["metric"])):
            rel = Path(row["plot_path"]).resolve().relative_to(out_dir.resolve())
            lines.append(f"<h3>{html.escape(row['profile'])} | {html.escape(row['metric'])}</h3>")
            lines.append(f"<img src=\"{html.escape(str(rel))}\" alt=\"{html.escape(row['metric'])}\">")

    if surface_rows:
        lines.append("<h2>Surface Artifacts</h2>")
        lines.append("<table><thead><tr><th>suite</th><th>config</th><th>preset</th><th>seed</th><th>profile</th><th>dir</th></tr></thead><tbody>")
        for row in sorted(surface_rows, key=lambda item: (item["suite"], item["config_id"], item["preset"], int(item["seed"]))):
            lines.append(
                "<tr>"
                f"<td>{html.escape(str(row['suite']))}</td>"
                f"<td>{html.escape(str(row['config_id']))}</td>"
                f"<td>{html.escape(str(row['preset']))}</td>"
                f"<td>{int(row['seed'])}</td>"
                f"<td>{html.escape(str(row['surface_profile']))}</td>"
                f"<td>{html.escape(str(row['surface_dir']))}</td>"
                "</tr>"
            )
        lines.append("</tbody></table>")

    lines.append("</body></html>")
    (out_dir / "report.html").write_text("\n".join(lines), encoding="utf-8")


def build_aggregate(
    *,
    roots: list[Path],
    out_dir: Path,
    profiles: set[str] | None,
) -> dict[str, Any]:
    campaign_roots: list[Path] = []
    for root in roots:
        campaign_roots.extend(_discover_campaign_roots(root))
    campaign_roots = sorted(set(path.resolve() for path in campaign_roots))

    metric_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    campaign_summaries: list[dict[str, Any]] = []

    for campaign_root in campaign_roots:
        rows, surface, campaign_summary = _collect_campaign_rows(
            campaign_root,
            requested_profiles=profiles,
        )
        metric_rows.extend(rows)
        surface_rows.extend(surface)
        campaign_summaries.append(campaign_summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _summarize_metric_rows(metric_rows)
    plots = _plot_metric_curves(summary_rows, out_dir)

    _write_csv(
        out_dir / "table_metrics_long.csv",
        metric_rows,
        [
            "campaign_id",
            "campaign_root",
            "suite",
            "suite_path",
            "config_id",
            "config_description",
            "level_index",
            "preset",
            "seed",
            "run_dir",
            "profile",
            "eval_dir",
            "metric",
            "value",
            "available",
            "method",
            "reason",
        ],
    )
    _write_csv(
        out_dir / "table_metrics_summary.csv",
        summary_rows,
        [
            "suite",
            "config_id",
            "config_description",
            "level_index",
            "preset",
            "profile",
            "metric",
            "n_values",
            "mean",
            "std",
            "min",
            "max",
        ],
    )
    _write_csv(
        out_dir / "table_surface_artifacts.csv",
        surface_rows,
        [
            "campaign_id",
            "campaign_root",
            "suite",
            "config_id",
            "config_description",
            "level_index",
            "preset",
            "seed",
            "run_dir",
            "surface_profile",
            "surface_dir",
            "artifacts_json",
            "summary_json",
            "interactive_html",
        ],
    )
    _write_csv(
        out_dir / "table_plots.csv",
        plots,
        ["suite", "profile", "metric", "plot_path"],
    )
    _write_report_html(
        out_dir,
        campaign_summaries=campaign_summaries,
        plots=plots,
        surface_rows=surface_rows,
    )

    manifest = {
        "roots": [str(root.resolve()) for root in roots],
        "campaign_count": len(campaign_roots),
        "metric_row_count": len(metric_rows),
        "summary_row_count": len(summary_rows),
        "surface_row_count": len(surface_rows),
        "plot_count": len(plots),
        "campaigns": campaign_summaries,
        "profiles": None if profiles is None else sorted(profiles),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Campaign root or parent directory to scan. Repeatable.",
    )
    parser.add_argument(
        "--out",
        default="runs/hawkesnest_campaign_eval_analysis",
        help="Output directory for aggregate tables and plots.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=None,
        help="Optional metric profile filter (for example: predictive, surface). Repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = (
        [Path(root).expanduser().resolve() for root in args.root]
        if args.root
        else [Path("runs/hawkesnest_campaigns").resolve()]
    )
    build_aggregate(
        roots=roots,
        out_dir=Path(args.out).expanduser().resolve(),
        profiles=None if not args.profile else set(args.profile),
    )


if __name__ == "__main__":
    main()
