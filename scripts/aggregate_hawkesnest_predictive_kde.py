#!/usr/bin/env python3
"""Aggregate HawkesNest predictive-KDE outputs into suite-level tables and plots."""

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


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _level_index(config_id: str) -> int | None:
    match = re.search(r"(\d+)$", str(config_id))
    if match is None:
        return None
    return int(match.group(1))


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._") or "item"


def _discover_campaign_roots(inputs: list[Path]) -> list[Path]:
    roots: set[Path] = set()
    for root in inputs:
        if not root.exists():
            continue
        manifest_path = root / "manifests" / "campaign_manifest.json"
        if manifest_path.exists():
            roots.add(root.resolve())
            continue
        if root.is_dir():
            roots.update(
                manifest.parent.parent.resolve()
                for manifest in root.rglob("manifests/campaign_manifest.json")
            )
    return sorted(roots)


def _suite_metadata(suite_path: Path | None) -> tuple[dict[str, int], dict[str, str]]:
    if suite_path is None:
        return {}, {}
    metadata_path = suite_path / "metadata.json"
    if not metadata_path.exists():
        return {}, {}
    payload = _load_json(metadata_path)
    if not isinstance(payload, dict):
        return {}, {}
    order: dict[str, int] = {}
    descriptions: dict[str, str] = {}
    for idx, row in enumerate(payload.get("levels", [])):
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        if label is None:
            continue
        order[str(label)] = idx
        if row.get("description") is not None:
            descriptions[str(label)] = str(row["description"])
    return order, descriptions


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _collect_campaign_rows(campaign_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = _load_json(campaign_root / "manifests" / "campaign_manifest.json")
    suite = str(manifest.get("suite") or campaign_root.parent.name)
    suite_path_raw = manifest.get("suite_path")
    suite_path = Path(str(suite_path_raw)).expanduser().resolve() if suite_path_raw else None
    if suite_path is not None and not suite_path.exists():
        suite_path = None
    level_order, level_descriptions = _suite_metadata(suite_path)

    metric_rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    preset_eval_roots = sorted((campaign_root / "evaluate" / "predictive_kde").glob("*"))
    for preset_eval_root in preset_eval_roots:
        if not preset_eval_root.is_dir():
            continue
        metrics_by_run = preset_eval_root / "metrics_by_run.csv"
        if not metrics_by_run.exists():
            continue
        for row in _read_csv_rows(metrics_by_run):
            dataset_id = str(row.get("dataset_id") or "")
            level_idx = level_order.get(dataset_id)
            if level_idx is None:
                level_idx = _level_index(dataset_id)
            out_row = {
                "campaign_id": campaign_root.name,
                "campaign_root": str(campaign_root),
                "suite": suite,
                "suite_path": None if suite_path is None else str(suite_path),
                "config_id": dataset_id,
                "config_description": level_descriptions.get(dataset_id),
                "level_index": level_idx,
                "family": row.get("family"),
                "preset": row.get("preset"),
                "seed": int(row["seed"]),
                "run_dir": row.get("run_dir"),
                "bundle_dir": row.get("bundle_dir"),
                "render_dir": row.get("render_dir") or None,
                "panel_frame0_2d": row.get("panel_frame0_2d") or None,
                "panel_frame0_3d": row.get("panel_frame0_3d") or None,
                "split": row.get("split"),
                "seq_idx": int(row["seq_idx"]),
                "start_event_idx": int(row["start_event_idx"]),
                "n_frames": int(row["n_frames"]),
                "n_rollouts": int(row["n_rollouts"]),
                "grid_size": int(row["grid_size"]),
                "horizon": _to_float(row.get("horizon")),
                "step_size": _to_float(row.get("step_size")),
                "test_nll": _to_float(row.get("test_nll")),
                "rmse_mean": _to_float(row.get("rmse_mean")),
                "mae_mean": _to_float(row.get("mae_mean")),
                "count_mae_mean": _to_float(row.get("count_mae_mean")),
                "count_bias_mean": _to_float(row.get("count_bias_mean")),
                "eval_elapsed_sec": _to_float(row.get("eval_elapsed_sec")),
            }
            metric_rows.append(out_row)
            if out_row["render_dir"] or out_row["panel_frame0_2d"] or out_row["panel_frame0_3d"]:
                render_rows.append(
                    {
                        "campaign_id": campaign_root.name,
                        "campaign_root": str(campaign_root),
                        "suite": suite,
                        "config_id": dataset_id,
                        "level_index": level_idx,
                        "preset": row.get("preset"),
                        "seed": int(row["seed"]),
                        "render_dir": out_row["render_dir"],
                        "panel_frame0_2d": out_row["panel_frame0_2d"],
                        "panel_frame0_3d": out_row["panel_frame0_3d"],
                        "bundle_dir": out_row["bundle_dir"],
                    }
                )

    return (
        metric_rows,
        render_rows,
        {
            "campaign_id": campaign_root.name,
            "campaign_root": str(campaign_root),
            "suite": suite,
            "suite_path": None if suite_path is None else str(suite_path),
            "metric_row_count": len(metric_rows),
            "render_row_count": len(render_rows),
        },
    )


def _summarize_metric_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        key = (
            row["suite"],
            row["config_id"],
            row["level_index"],
            row["preset"],
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0], 10**9 if item[2] is None else item[2], item[1], item[3])):
        rows = grouped[key]
        sample = rows[0]

        def summarize(name: str) -> tuple[float | None, float | None]:
            values = np.asarray(
                [row[name] for row in rows if row.get(name) is not None],
                dtype=np.float64,
            )
            if values.size == 0:
                return None, None
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            return mean, std

        rmse_mean, rmse_std = summarize("rmse_mean")
        mae_mean, mae_std = summarize("mae_mean")
        count_mae_mean, count_mae_std = summarize("count_mae_mean")
        count_bias_mean, count_bias_std = summarize("count_bias_mean")
        test_nll_mean, test_nll_std = summarize("test_nll")
        elapsed_mean, elapsed_std = summarize("eval_elapsed_sec")
        out.append(
            {
                "suite": sample["suite"],
                "config_id": sample["config_id"],
                "config_description": sample.get("config_description"),
                "level_index": sample["level_index"],
                "preset": sample["preset"],
                "n_runs": len(rows),
                "seeds": json.dumps(sorted(int(row["seed"]) for row in rows)),
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "count_mae_mean": count_mae_mean,
                "count_mae_std": count_mae_std,
                "count_bias_mean": count_bias_mean,
                "count_bias_std": count_bias_std,
                "test_nll_mean": test_nll_mean,
                "test_nll_std": test_nll_std,
                "eval_elapsed_sec_mean": elapsed_mean,
                "eval_elapsed_sec_std": elapsed_std,
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_suite_plots(summary_rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, str]]:
    metric_names = [
        ("rmse_mean", "Predictive KDE Surface RMSE", "RMSE"),
        ("mae_mean", "Predictive KDE Surface MAE", "MAE"),
        ("count_mae_mean", "Predictive KDE Count MAE", "count MAE"),
        ("count_bias_mean", "Predictive KDE Count Bias", "count bias"),
    ]
    plot_rows: list[dict[str, str]] = []
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_suite[str(row["suite"])].append(row)

    for suite, suite_rows in sorted(by_suite.items()):
        level_map = {
            row["config_id"]: (10**9 if row["level_index"] is None else int(row["level_index"]))
            for row in suite_rows
        }
        ordered_config_ids = [item[0] for item in sorted(level_map.items(), key=lambda item: (item[1], item[0]))]
        x = np.arange(len(ordered_config_ids), dtype=np.float64)
        xtick_lookup = {config_id: idx for idx, config_id in enumerate(ordered_config_ids)}

        for metric_name, title, ylabel in metric_names:
            if not any(row.get(metric_name) is not None for row in suite_rows):
                continue
            fig, ax = plt.subplots(figsize=(8.0, 4.5))
            by_preset: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in suite_rows:
                if row.get(metric_name) is None:
                    continue
                by_preset[str(row["preset"])].append(row)
            for preset, rows in sorted(by_preset.items()):
                rows = sorted(rows, key=lambda row: xtick_lookup[str(row["config_id"])])
                xs = [xtick_lookup[str(row["config_id"])] for row in rows]
                ys = [float(row[metric_name]) for row in rows]
                ax.plot(xs, ys, marker="o", label=preset)
            ax.set_title(f"{suite}: {title}")
            ax.set_xlabel("configuration")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(ordered_config_ids)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()

            suite_dir = out_dir / "plots" / _slugify(suite)
            suite_dir.mkdir(parents=True, exist_ok=True)
            plot_path = suite_dir / f"{metric_name}.png"
            fig.savefig(plot_path, dpi=160)
            plt.close(fig)
            plot_rows.append(
                {
                    "suite": suite,
                    "metric": metric_name,
                    "path": str(plot_path),
                }
            )
    return plot_rows


def _write_report(
    *,
    out_dir: Path,
    campaign_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    plot_rows: list[dict[str, str]],
) -> None:
    plots_by_suite: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in plot_rows:
        plots_by_suite[row["suite"]].append(row)

    summary_preview = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['suite']))}</td>"
        f"<td>{html.escape(str(row['config_id']))}</td>"
        f"<td>{html.escape(str(row['preset']))}</td>"
        f"<td>{'' if row.get('rmse_mean') is None else f'{float(row['rmse_mean']):.4f}'}</td>"
        f"<td>{'' if row.get('mae_mean') is None else f'{float(row['mae_mean']):.4f}'}</td>"
        f"<td>{'' if row.get('count_mae_mean') is None else f'{float(row['count_mae_mean']):.4f}'}</td>"
        "</tr>"
        for row in sorted(
            summary_rows,
            key=lambda row: (
                str(row["suite"]),
                10**9 if row["level_index"] is None else int(row["level_index"]),
                str(row["config_id"]),
                str(row["preset"]),
            ),
        )[:80]
    )
    suite_sections = []
    for suite, rows in sorted(plots_by_suite.items()):
        images = "".join(
            f"<div><h3>{html.escape(plot['metric'])}</h3><img src='{Path(plot['path']).relative_to(out_dir)}' style='max-width:100%; border:1px solid #ccc;'></div>"
            for plot in sorted(rows, key=lambda item: item["metric"])
        )
        suite_sections.append(f"<section><h2>{html.escape(suite)}</h2>{images}</section>")

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HawkesNest Predictive KDE Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    img {{ display: block; margin: 8px 0 24px; }}
  </style>
</head>
<body>
  <h1>HawkesNest Predictive KDE Analysis</h1>
  <p>Campaigns processed: {len(campaign_rows)}. Summary rows: {len(summary_rows)}.</p>
  <p>This report covers the predictive-KDE synthetic comparison path currently supported by the rollout stack: auto_stpp, deep_stpp, smash, diffusion_stpp.</p>
  <h2>Summary Preview</h2>
  <table>
    <thead>
      <tr>
        <th>suite</th>
        <th>config</th>
        <th>preset</th>
        <th>rmse</th>
        <th>mae</th>
        <th>count_mae</th>
      </tr>
    </thead>
    <tbody>{summary_preview}</tbody>
  </table>
  {''.join(suite_sections)}
</body>
</html>
"""
    (out_dir / "report.html").write_text(body)


def build_aggregate(*, roots: list[Path], out_dir: Path) -> dict[str, Any]:
    campaign_roots = _discover_campaign_roots(roots)
    metric_rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    campaign_rows: list[dict[str, Any]] = []
    for campaign_root in campaign_roots:
        rows, renders, campaign = _collect_campaign_rows(campaign_root)
        metric_rows.extend(rows)
        render_rows.extend(renders)
        campaign_rows.append(campaign)

    summary_rows = _summarize_metric_rows(metric_rows)
    plot_rows = _write_suite_plots(summary_rows, out_dir)

    metric_fieldnames = [
        "campaign_id",
        "campaign_root",
        "suite",
        "suite_path",
        "config_id",
        "config_description",
        "level_index",
        "family",
        "preset",
        "seed",
        "run_dir",
        "bundle_dir",
        "render_dir",
        "panel_frame0_2d",
        "panel_frame0_3d",
        "split",
        "seq_idx",
        "start_event_idx",
        "n_frames",
        "n_rollouts",
        "grid_size",
        "horizon",
        "step_size",
        "test_nll",
        "rmse_mean",
        "mae_mean",
        "count_mae_mean",
        "count_bias_mean",
        "eval_elapsed_sec",
    ]
    summary_fieldnames = [
        "suite",
        "config_id",
        "config_description",
        "level_index",
        "preset",
        "n_runs",
        "seeds",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "count_mae_mean",
        "count_mae_std",
        "count_bias_mean",
        "count_bias_std",
        "test_nll_mean",
        "test_nll_std",
        "eval_elapsed_sec_mean",
        "eval_elapsed_sec_std",
    ]
    render_fieldnames = [
        "campaign_id",
        "campaign_root",
        "suite",
        "config_id",
        "level_index",
        "preset",
        "seed",
        "render_dir",
        "panel_frame0_2d",
        "panel_frame0_3d",
        "bundle_dir",
    ]
    plot_fieldnames = ["suite", "metric", "path"]

    _write_csv(out_dir / "table_metrics_by_run.csv", metric_rows, metric_fieldnames)
    _write_csv(out_dir / "table_metrics_by_level.csv", summary_rows, summary_fieldnames)
    _write_csv(out_dir / "table_render_artifacts.csv", render_rows, render_fieldnames)
    _write_csv(out_dir / "table_plots.csv", plot_rows, plot_fieldnames)
    _write_report(out_dir=out_dir, campaign_rows=campaign_rows, summary_rows=summary_rows, plot_rows=plot_rows)

    manifest = {
        "campaign_count": len(campaign_rows),
        "metric_row_count": len(metric_rows),
        "summary_row_count": len(summary_rows),
        "render_row_count": len(render_rows),
        "plot_count": len(plot_rows),
        "roots": [str(root) for root in campaign_roots],
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Campaign root or parent directory. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for merged tables and plots.",
    )
    args = parser.parse_args()

    roots = [Path(root).expanduser().resolve() for root in (args.root or ["runs/hawkesnest_campaigns"])]
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_aggregate(roots=roots, out_dir=out_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
