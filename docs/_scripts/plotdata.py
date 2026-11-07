"""Compact plot data for one dataset: split statistics, a location subsample, daily counts, gaps.

Standalone (stdlib + numpy) so it can run next to data that cannot be copied, e.g. on the
cluster that holds the campaign's prepared copies of the script-only datasets:

    python plotdata.py --out citibike_nyc.npz \
        train=/path/train.jsonl val=/path/val.jsonl test=/path/test.jsonl

The .npz holds no raw records: at most MAX_POINTS (x, y) locations without times, per-day event
counts, a subsample of inter-event times and the split statistics (JSON).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

SPLITS = ("train", "val", "test")
MAX_POINTS = 30_000
MAX_GAPS = 200_000


def _events(rec: dict):
    if "events" in rec:
        ev = rec["events"]
        return [e["t"] for e in ev], [(e["x"], e["y"]) for e in ev]
    return rec.get("times", []), [tuple(loc[:2]) for loc in rec.get("locations", [])]


def _start(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=None)
    except ValueError:
        return None


def plotdata_from_files(paths: dict, seed: int = 20261107) -> dict:
    """paths: {split: jsonl path}. Returns arrays and statistics (see module docstring)."""
    rng = np.random.default_rng(seed)
    xs, ys, codes, gaps_all, stats = [], [], [], {}, {}
    day_counts: dict[str, dict] = {}
    calendar = True
    for code, split in enumerate(SPLITS):
        n_seq, counts, gaps, per_day = 0, [], [], {}
        with open(paths[split]) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                times, locs = _events(rec)
                n_seq += 1
                counts.append(len(times))
                start = _start(rec.get("window_start"))
                if start is None:
                    calendar = False
                else:
                    key = start.date().isoformat()
                    per_day[key] = per_day.get(key, 0) + len(times)
                if len(times) > 1:
                    gaps.append(np.diff(np.asarray(times, dtype=float)))
                if locs:
                    arr = np.asarray(locs, dtype=float)
                    xs.append(arr[:, 0])
                    ys.append(arr[:, 1])
                    codes.append(np.full(len(arr), code, dtype=np.int8))
        c = np.asarray(counts)
        stats[split] = {
            "sequences": int(n_seq),
            "events": int(c.sum()),
            "events_per_seq": [int(c.min()), float(np.median(c)), int(c.max())] if n_seq else None,
        }
        g = np.concatenate(gaps) if gaps else np.zeros(0)
        g = g[g > 0]
        gaps_all[split] = rng.choice(g, size=min(len(g), MAX_GAPS), replace=False) if len(g) else g
        day_counts[split] = per_day
        stats[split]["_seq_counts"] = counts
    x, y, s = np.concatenate(xs), np.concatenate(ys), np.concatenate(codes)
    idx = rng.choice(len(x), size=min(len(x), MAX_POINTS), replace=False)
    out = {"x": x[idx], "y": y[idx], "split": s[idx], "calendar": calendar}
    for split in SPLITS:
        out[f"gaps_{split}"] = gaps_all[split]
        out[f"seqcounts_{split}"] = np.asarray(stats[split].pop("_seq_counts"), dtype=np.int64)
        if calendar:
            days = sorted(day_counts[split])
            out[f"days_{split}"] = np.asarray(days, dtype="datetime64[D]")
            out[f"daycounts_{split}"] = np.asarray([day_counts[split][d] for d in days], dtype=np.int64)
    out["stats"] = stats
    return out


def save(pd: dict, path) -> None:
    arrays = {k: v for k, v in pd.items() if k not in ("stats", "calendar")}
    np.savez_compressed(path, stats=json.dumps(pd["stats"]), calendar=pd["calendar"], **arrays)


def load(path) -> dict:
    z = np.load(path, allow_pickle=False)
    pd = {k: z[k] for k in z.files if k not in ("stats", "calendar")}
    pd["stats"] = json.loads(str(z["stats"]))
    pd["calendar"] = bool(z["calendar"])
    return pd


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("splits", nargs=3, help="train=PATH val=PATH test=PATH")
    a = ap.parse_args()
    save(plotdata_from_files(dict(s.split("=", 1) for s in a.splits)), a.out)
    print("wrote", a.out)
