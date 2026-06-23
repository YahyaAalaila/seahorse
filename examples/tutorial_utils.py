"""Shared helpers for Seahorse release tutorials.

These helpers intentionally live under ``examples/`` so they are easy to read
alongside the tutorials and do not become part of the stable package API.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def _configure_plot_cache(root: Path) -> None:
    cache_root = Path(root) / ".plot_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_root / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")


def write_jsonl(path: Path, records: Iterable[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            clean = {
                "times": np.asarray(record["times"], dtype=float).round(6).tolist(),
                "locations": np.asarray(record["locations"], dtype=float).round(6).tolist(),
            }
            f.write(json.dumps(clean) + "\n")
    return path


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_event_sequences(
    *,
    n_sequences: int,
    seed: int,
    min_events: int = 14,
    max_events: int = 24,
) -> list[dict]:
    """Generate a small but structured ST event dataset.

    The process has three moving hotspots and a daily-like temporal rhythm. It
    is synthetic by design: stable enough for tutorials, nontrivial enough for
    EDA and model comparison.
    """
    rng = np.random.default_rng(seed)
    sequences: list[dict] = []
    base_centers = np.asarray(
        [[0.25, 0.28], [0.68, 0.35], [0.52, 0.76]],
        dtype=np.float32,
    )
    for seq_idx in range(n_sequences):
        n_events = int(rng.integers(min_events, max_events + 1))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        regime = int(rng.integers(0, 3))
        times = []
        locs = []
        t = float(rng.uniform(0.0, 0.15))
        for event_idx in range(n_events):
            rhythm = 1.0 + 0.45 * np.sin(phase + event_idx / max(n_events - 1, 1) * np.pi)
            dt = rng.gamma(shape=1.8, scale=0.095 / max(rhythm, 0.25))
            t += float(dt)
            mixture_logits = np.asarray([0.15, 0.15, 0.15], dtype=np.float64)
            mixture_logits[regime] += 0.65
            mixture_logits[(regime + (event_idx // 6)) % 3] += 0.35
            weights = mixture_logits / mixture_logits.sum()
            center_idx = int(rng.choice(3, p=weights))
            drift = np.asarray(
                [
                    0.055 * np.sin(phase + 0.55 * t + center_idx),
                    0.045 * np.cos(phase * 0.5 + 0.40 * t + center_idx),
                ]
            )
            loc = base_centers[center_idx] + drift + rng.normal(0.0, 0.045, size=2)
            locs.append(np.clip(loc, 0.02, 0.98))
            times.append(t)
        sequences.append(
            {
                "times": np.asarray(times, dtype=np.float32),
                "locations": np.asarray(locs, dtype=np.float32),
            }
        )
    return sequences


def write_tutorial_dataset(
    root: Path,
    *,
    seed: int = 42,
    n_train: int = 24,
    n_val: int = 8,
    n_test: int = 8,
) -> dict[str, Path]:
    root = Path(root)
    splits = {
        "train": generate_event_sequences(n_sequences=n_train, seed=seed),
        "val": generate_event_sequences(n_sequences=n_val, seed=seed + 1),
        "test": generate_event_sequences(n_sequences=n_test, seed=seed + 2),
    }
    return {name: write_jsonl(root / f"{name}.jsonl", seqs) for name, seqs in splits.items()}


def load_tutorial_splits(dataset_dir: Path) -> dict[str, list[dict]]:
    dataset_dir = Path(dataset_dir)
    return {
        "train": load_jsonl(dataset_dir / "train.jsonl"),
        "val": load_jsonl(dataset_dir / "val.jsonl"),
        "test": load_jsonl(dataset_dir / "test.jsonl"),
    }


def _all_lengths(splits: dict[str, list[dict]]) -> np.ndarray:
    return np.asarray([len(seq["times"]) for seqs in splits.values() for seq in seqs])


def _all_deltas(splits: dict[str, list[dict]]) -> np.ndarray:
    chunks = []
    for seqs in splits.values():
        for seq in seqs:
            times = np.asarray(seq["times"], dtype=np.float32)
            if times.size > 1:
                chunks.append(np.diff(times))
    return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)


def _all_locations(splits: dict[str, list[dict]]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(seq["locations"], dtype=np.float32) for seqs in splits.values() for seq in seqs],
        axis=0,
    )


def _svg_text(x: float, y: float, value: str, *, size: int = 18, weight: int = 500, fill: str = "#0f172a") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(value)}</text>'
    )


def _panel(x: float, y: float, w: float, h: float, title: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="#ffffff" '
        f'stroke="#dbe4ef" stroke-width="1.4"/>'
        + _svg_text(x + 24, y + 42, title, size=22, weight=800)
    )


def _histogram_svg(data: np.ndarray, *, x: float, y: float, w: float, h: float, color: str, bins: int = 16) -> str:
    if data.size == 0:
        return ""
    counts, edges = np.histogram(data, bins=bins)
    max_count = max(int(counts.max()), 1)
    pad_l, pad_b, pad_t = 44, 44, 62
    plot_x, plot_y = x + pad_l, y + pad_t
    plot_w, plot_h = w - pad_l - 24, h - pad_t - pad_b
    parts = [
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#94a3b8"/>',
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#94a3b8"/>',
    ]
    bar_w = plot_w / max(len(counts), 1)
    for i, count in enumerate(counts):
        bh = plot_h * float(count) / max_count
        bx = plot_x + i * bar_w + 2
        by = plot_y + plot_h - bh
        parts.append(
            f'<rect x="{bx:.2f}" y="{by:.2f}" width="{max(bar_w - 4, 1):.2f}" height="{bh:.2f}" '
            f'rx="4" fill="{color}" opacity="0.86"/>'
        )
    parts.append(_svg_text(plot_x, y + h - 14, f"{edges[0]:.2f} → {edges[-1]:.2f}", size=13, fill="#64748b"))
    return "\n".join(parts)


def _spatial_density_svg(locs: np.ndarray, *, x: float, y: float, w: float, h: float) -> str:
    pad_l, pad_b, pad_t = 44, 40, 62
    plot_x, plot_y = x + pad_l, y + pad_t
    plot_w, plot_h = w - pad_l - 28, h - pad_t - pad_b
    counts, xedges, yedges = np.histogram2d(locs[:, 0], locs[:, 1], bins=18, range=[[0, 1], [0, 1]])
    max_count = max(float(counts.max()), 1.0)
    parts = [
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" rx="12" fill="#0f172a"/>'
    ]
    for ix in range(counts.shape[0]):
        for iy in range(counts.shape[1]):
            count = counts[ix, iy]
            if count <= 0:
                continue
            cx = plot_x + (ix / counts.shape[0]) * plot_w
            cy = plot_y + plot_h - ((iy + 1) / counts.shape[1]) * plot_h
            cw = plot_w / counts.shape[0] + 0.5
            ch = plot_h / counts.shape[1] + 0.5
            opacity = 0.16 + 0.78 * float(count) / max_count
            parts.append(
                f'<rect x="{cx:.2f}" y="{cy:.2f}" width="{cw:.2f}" height="{ch:.2f}" '
                f'fill="#38bdf8" opacity="{opacity:.3f}"/>'
            )
    parts.append(_svg_text(plot_x, y + h - 14, "x/y event density", size=13, fill="#64748b"))
    return "\n".join(parts)


def _trajectory_svg(sequence: dict, *, x: float, y: float, w: float, h: float) -> str:
    loc = np.asarray(sequence["locations"], dtype=np.float32)
    times = np.asarray(sequence["times"], dtype=np.float32)
    pad_l, pad_b, pad_t = 44, 40, 62
    plot_x, plot_y = x + pad_l, y + pad_t
    plot_w, plot_h = w - pad_l - 28, h - pad_t - pad_b

    def px(v):
        return plot_x + float(v) * plot_w

    def py(v):
        return plot_y + plot_h - float(v) * plot_h

    points = " ".join(f"{px(a):.2f},{py(b):.2f}" for a, b in loc)
    parts = [
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" rx="12" fill="#f8fafc" stroke="#cbd5e1"/>',
        f'<polyline points="{points}" fill="none" stroke="#64748b" stroke-width="2.3" opacity="0.85"/>',
    ]
    denom = max(float(times.max() - times.min()), 1e-8)
    for i, ((lx, ly), t) in enumerate(zip(loc, times, strict=True)):
        frac = float((t - times.min()) / denom)
        fill = "#0ea5e9" if i < len(times) - 1 else "#f97316"
        radius = 5.0 + 4.0 * frac
        parts.append(
            f'<circle cx="{px(lx):.2f}" cy="{py(ly):.2f}" r="{radius:.2f}" fill="{fill}" '
            f'fill-opacity="{0.45 + 0.5 * frac:.3f}" stroke="#0f172a" stroke-width="1"/>'
        )
    parts.append(_svg_text(plot_x, y + h - 14, "held-out history, colored by time", size=13, fill="#64748b"))
    return "\n".join(parts)


def plot_eda_panel(splits: dict[str, list[dict]], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lengths = _all_lengths(splits)
    deltas = _all_deltas(splits)
    locs = _all_locations(splits)
    panels = [
        (55, 100, 520, 330, "Sequence lengths"),
        (625, 100, 520, 330, "Temporal rhythm"),
        (55, 490, 520, 330, "Spatial event density"),
        (625, 490, 520, 330, "One held-out event history"),
    ]
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 880" role="img">',
        '<rect width="1200" height="880" fill="#f8fafc"/>',
        _svg_text(55, 58, "Modeling Events in Space and Time", size=34, weight=900),
        _svg_text(55, 84, "A compact fingerprint of the tutorial ST event dataset", size=16, fill="#64748b"),
    ]
    for p in panels:
        body.append(_panel(*p))
    body.append(_histogram_svg(lengths, x=55, y=100, w=520, h=330, color="#0ea5e9", bins=10))
    body.append(_histogram_svg(deltas, x=625, y=100, w=520, h=330, color="#7c3aed", bins=18))
    body.append(_spatial_density_svg(locs, x=55, y=490, w=520, h=330))
    body.append(_trajectory_svg(splits["test"][0], x=625, y=490, w=520, h=330))
    body.append("</svg>")
    output_path.write_text("\n".join(body))
    return output_path


def write_event_movie_html(sequence: dict, output_path: Path) -> Path:
    """Write a lightweight browser movie for one spatiotemporal event sequence."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(sequence["times"], dtype=np.float32)
    locs = np.asarray(sequence["locations"], dtype=np.float32)
    frame_ids = np.unique(np.linspace(1, len(times), num=min(len(times), 36), dtype=int))
    frames = []
    for n in frame_ids:
        circles = []
        for i, (x, y) in enumerate(locs[:n]):
            px = 40 + float(x) * 520
            py = 560 - float(y) * 520
            opacity = 0.30 + 0.70 * (i + 1) / n
            radius = 5.0 + 3.5 * (i + 1 == n)
            circles.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" '
                f'fill="#0ea5e9" fill-opacity="{opacity:.3f}" stroke="#0f172a" stroke-width="1.2" />'
            )
        frames.append(
            {
                "time": float(times[n - 1]),
                "events": int(n),
                "svg": "\n".join(circles),
            }
        )

    payload = json.dumps(frames)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Spatiotemporal event movie</title>
  <style>
    body {{ margin: 0; background: #020617; color: #e2e8f0; font-family: Inter, system-ui, sans-serif; }}
    .wrap {{ max-width: 860px; margin: 28px auto; padding: 24px; background: #0f172a; border-radius: 24px; box-shadow: 0 24px 80px rgba(2, 6, 23, 0.45); }}
    h1 {{ margin: 0 0 4px; font-size: 24px; }}
    p {{ margin: 0 0 18px; color: #94a3b8; }}
    svg {{ width: 100%; height: auto; background: radial-gradient(circle at 30% 25%, #1e3a8a 0, #0f172a 34%, #020617 100%); border-radius: 18px; }}
    .toolbar {{ display: flex; gap: 12px; align-items: center; margin-top: 16px; }}
    button {{ background: #0ea5e9; color: white; border: 0; padding: 10px 18px; border-radius: 999px; font-weight: 700; cursor: pointer; }}
    input[type=range] {{ flex: 1; accent-color: #0ea5e9; }}
    .badge {{ color: #bae6fd; font-variant-numeric: tabular-nums; min-width: 170px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Events unfold across space and time</h1>
    <p>Each frame reveals the observed history available to an STPP model.</p>
    <svg viewBox="0 0 600 600" role="img" aria-label="Animated spatiotemporal event history">
      <defs>
        <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
          <path d="M 60 0 L 0 0 0 60" fill="none" stroke="#334155" stroke-width="1" opacity="0.5"/>
        </pattern>
      </defs>
      <rect x="25" y="25" width="550" height="550" rx="22" fill="url(#grid)" stroke="#64748b" />
      <g id="events"></g>
      <text x="40" y="55" fill="#e0f2fe" font-size="18" font-weight="700" id="label"></text>
    </svg>
    <div class="toolbar">
      <button id="play">Pause</button>
      <input id="slider" type="range" min="0" max="{len(frames) - 1}" value="0" />
      <span class="badge" id="badge"></span>
    </div>
  </div>
  <script>
    const frames = {payload};
    const events = document.getElementById("events");
    const label = document.getElementById("label");
    const badge = document.getElementById("badge");
    const slider = document.getElementById("slider");
    const play = document.getElementById("play");
    let i = 0;
    let running = true;
    function render(idx) {{
      const f = frames[idx];
      events.innerHTML = f.svg;
      label.textContent = `t = ${{f.time.toFixed(2)}}`;
      badge.textContent = `${{f.events}} events observed`;
      slider.value = idx;
    }}
    slider.addEventListener("input", () => {{ i = Number(slider.value); render(i); }});
    play.addEventListener("click", () => {{ running = !running; play.textContent = running ? "Pause" : "Play"; }});
    render(0);
    setInterval(() => {{ if (running) {{ i = (i + 1) % frames.length; render(i); }} }}, 420);
  </script>
</body>
</html>
"""
    output_path.write_text(html_doc)
    return output_path


def write_results_table_html(rows: list[dict], output_path: Path, *, title: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cells = []
    for row in rows:
        cells.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{float(row['test_nll']):.4f}</td>"
            f"<td>{float(row.get('mean_seq_nll', row['test_nll'])):.4f}</td>"
            f"<td>{html.escape(str(row.get('note', '')))}</td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <style>
    body {{ font-family: Inter, system-ui, sans-serif; background: #f8fafc; color: #0f172a; }}
    .card {{ max-width: 780px; margin: 24px auto; background: white; border-radius: 20px; box-shadow: 0 18px 45px rgba(15, 23, 42, .12); overflow: hidden; }}
    h1 {{ margin: 0; padding: 22px 26px; background: linear-gradient(135deg, #0f172a, #0369a1); color: white; font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 14px 18px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
    th {{ color: #475569; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    td:nth-child(2), td:nth-child(3) {{ font-variant-numeric: tabular-nums; font-weight: 700; color: #0369a1; }}
  </style>
</head>
<body><div class="card"><h1>{html.escape(title)}</h1><table>
<thead><tr><th>Model</th><th>Test NLL</th><th>Mean sequence NLL</th><th>Note</th></tr></thead>
<tbody>{''.join(cells)}</tbody>
</table></div></body></html>
"""
    output_path.write_text(doc)
    return output_path


def plot_model_comparison(rows: list[dict], output_path: Path, *, title: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(r["model"]) for r in rows]
    values = [float(r["test_nll"]) for r in rows]
    finite = [v for v in values if np.isfinite(v)]
    vmin = min([0.0] + finite) if finite else 0.0
    vmax = max([1.0] + finite) if finite else 1.0
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1.0
    plot_x, plot_y, plot_w, plot_h = 90, 110, 720, 290

    def y_for(v: float) -> float:
        return plot_y + plot_h - (float(v) - vmin) / (vmax - vmin) * plot_h

    zero_y = y_for(0.0)
    bar_w = plot_w / max(len(values), 1) * 0.55
    colors = ["#0ea5e9", "#7c3aed", "#14b8a6", "#f97316"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 500" role="img">',
        '<rect width="900" height="500" fill="#f8fafc"/>',
        '<rect x="40" y="40" width="820" height="420" rx="26" fill="#ffffff" stroke="#dbe4ef"/>',
        _svg_text(72, 86, title, size=24, weight=900),
        _svg_text(72, 422, "test NLL (lower is better)", size=14, fill="#64748b"),
        f'<line x1="{plot_x}" y1="{zero_y:.2f}" x2="{plot_x + plot_w}" y2="{zero_y:.2f}" stroke="#94a3b8"/>',
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#94a3b8"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        cx = plot_x + (i + 0.5) * plot_w / max(len(values), 1)
        yv = y_for(value)
        by = min(yv, zero_y)
        bh = abs(zero_y - yv)
        parts.append(
            f'<rect x="{cx - bar_w / 2:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" '
            f'rx="10" fill="{colors[i % len(colors)]}" stroke="#0f172a"/>'
        )
        parts.append(_svg_text(cx - 38, by - 10, f"{value:.3f}", size=14, weight=800))
        parts.append(_svg_text(cx - 58, 438, label, size=14, weight=700))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts))
    return output_path
