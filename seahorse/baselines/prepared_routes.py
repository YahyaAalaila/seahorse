"""Convert frozen prepared JSONL windows for external fixed-horizon baselines.

The converter is deliberately lossless about source-window identity and
observation bounds.  It makes no jitter, event filtering, or train/test
statistics: it only maps each source window's local time coordinate to
``(0, 1)`` and writes the corresponding time-Jacobian metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _convert(row: dict[str, Any], *, source: Path) -> dict[str, Any]:
    times = [float(x) for x in row["times"]]
    locations = row["locations"]
    if len(times) != len(locations):
        raise ValueError(f"{source}: paired times/locations length mismatch for {row.get('sequence_id')}")
    t0, t1 = float(row["t0"]), float(row["t1"])
    width = t1 - t0
    if not math.isfinite(width) or width <= 0:
        raise ValueError(f"{source}: nonpositive finite window width for {row.get('sequence_id')}")
    normalized = [(t - t0) / width for t in times]
    if any(not math.isfinite(t) or t <= 0.0 or t >= 1.0 for t in normalized):
        raise ValueError(f"{source}: external fixed-horizon contract needs 0 < u < 1; use a compatible jittered view")
    if any(b <= a for a, b in zip(normalized, normalized[1:])):
        raise ValueError(f"{source}: external fixed-horizon contract needs strictly increasing times")
    return {
        "sequence_id": str(row["sequence_id"]),
        "times": normalized,
        "locations": locations,
        "t0": 0.0,
        "t1": 1.0,
        "source_t0": t0,
        "source_t1": t1,
        "source_window_width": width,
        "time_log_jacobian": math.log(width),
        "source_window_start": row.get("window_start"),
        "source_window_end": row.get("window_end"),
        "source_event_ids": row.get("event_ids", row.get("event_id")),
        "source_route": str(source),
    }


def materialize_fixed_window_routes(spec: dict[str, Any], *, output: str | Path | None = None) -> dict[str, Any]:
    """Materialize train/val/test route JSONL from a frozen external spec."""
    converter = spec.get("fixed_window_converter", {})
    out = Path(output or converter["output_directory"])
    source_splits = spec["source_prepared_route"]["splits"]
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"kind": spec.get("kind"), "horizon": 1.0, "splits": {}}
    for split in ("train", "val", "test"):
        source = Path(source_splits[split])
        destination = out / f"{split}.jsonl"
        digest = hashlib.sha256()
        n_windows = n_events = n_empty = 0
        with destination.open("w") as handle:
            for row in _rows(source):
                converted = _convert(row, source=source)
                encoded = json.dumps(converted, separators=(",", ":"), allow_nan=False) + "\n"
                handle.write(encoded); digest.update(encoded.encode())
                n_windows += 1; n_events += len(converted["times"]); n_empty += not bool(converted["times"])
        manifest["splits"][split] = {"source": str(source), "output": str(destination),
                                     "windows": n_windows, "events": n_events, "empty_windows": n_empty,
                                     "sha256": digest.hexdigest()}
    (out / "converter_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize external fixed-window JSONL routes from a frozen spec.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", help="override fixed_window_converter.output_directory")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    print(json.dumps(materialize_fixed_window_routes(spec, output=args.out), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
