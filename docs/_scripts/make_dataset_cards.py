"""Generate the dataset pages of the documentation from the released Hub repositories.

For every dataset in ``seahorse.data.hf.RELEASED_DATASETS`` the script downloads the dataset
card and (for hosted datasets) the default-config splits at the pinned revision, computes split
statistics, draws one PNG (event locations + events per window), and writes

    docs/datasets/index.md              index with one tile per dataset and a summary table
    docs/datasets/cards/<name>.md       one page per dataset
    docs/assets/datasets/<name>.png     one figure per hosted dataset
    docs/_scripts/dataset_stats.json    the computed statistics (re-used with --offline)

Paper facts (period and split points, Table 2 best presets, HawkesNest Table 3) are copied from
the NeurIPS 2026 camera-ready and kept in the PAPER dict below; everything else comes from the Hub.

Usage (repository root):

    python docs/_scripts/make_dataset_cards.py              # download, compute, plot, write
    python docs/_scripts/make_dataset_cards.py --offline    # re-write pages from dataset_stats.json
    python docs/_scripts/make_dataset_cards.py --only austin311 covid

Requires huggingface_hub, numpy, matplotlib. The download is about 1.5 GB (cached by huggingface_hub).
Per-dataset plot data (location subsample, daily counts, gaps; no raw records) is cached in
~/.cache/seahorse-docs/plotdata/<name>.npz and re-used by --offline; it is never committed.

Script-only datasets (citibike_nyc, citibike_legacy, seattle_fire911) are not on the Hub. Their
plot data was computed with docs/_scripts/plotdata.py next to the campaign's prepared copies
(read-only) and copied to the cache above; every train split's SHA-256 equals the dataset card's
expected rebuild hash:
  citibike_nyc, seattle_fire911: Pegasus, /netscratch/aalaila/seahorse_iclr2027_20260916/data/
      seahorse_data_v2/processed/<corpus>/jittered_seed20260910/{train,val,test}.jsonl
  citibike_legacy (days): Unicorn, /scratch/aalaila/seahorse_campaigns/iclr2027/
      nnstat_legacy3_v1_inputs/data/citibike_stpp/{train,val,test}.jsonl
Alternatively run plotdata.py on a local rebuild. Without a
cached .npz a script-only card gets no plot.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from seahorse.data.hf import RELEASED_DATASETS, ReleasedDataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotdata as PD  # noqa: E402

PLOTDATA_CACHE = Path.home() / ".cache" / "seahorse-docs" / "plotdata"
THUMB_INK = "#5b8ac2"  # mid-tone blue: >= 3:1 against both the light (#fff) and dark (#141b26) card surface

DOCS = ROOT / "docs"
CARDS_DIR = DOCS / "datasets" / "cards"
IMG_DIR = DOCS / "assets" / "datasets"
THUMB_DIR = IMG_DIR / "thumbs"
STATS_PATH = DOCS / "_scripts" / "dataset_stats.json"
SPLITS = ("train", "val", "test")
MAX_POINTS = 30_000  # plotted locations per dataset (subsampled)

# Split colours (categorical slots 1-3 of the docs palette, on the white figure surface).
COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
LABELS = {"train": "train", "val": "validation", "test": "test"}

# --------------------------------------------------------------------------------------------
# Facts copied from the NeurIPS 2026 camera-ready (Table 2, Table 3, appendix Table "Real datasets").
# best / second: best and second-best exact- or numerical-likelihood preset by mean test NLL
# (nats/event, raw units) over three seeds; generative presets are not ranked.
# --------------------------------------------------------------------------------------------
GROUPS = {
    "chrono": "Real data, chronological splits",
    "nstpp": "Real data, Neural STPP benchmark splits",
    "synthetic": "Synthetic stress test",
    "released": "Released, not benchmarked in the paper",
}

PAPER: dict[str, dict] = {
    "comcat_california": dict(
        title="ComCat · California", group="chrono", column="ComCat",
        what="Earthquakes in California (USGS ANSS ComCat, as packaged by EarthquakeNPP)",
        period="1981–2020; validation from 1998, test from 2007", unit="one day",
        best=("AutoSTPP", "-0.469"), second=("DeepSTPP", "-0.004")),
    "geonet_nz": dict(
        title="GeoNet · New Zealand", group="chrono", column="GeoNet",
        what="Earthquakes in New Zealand (GeoNet catalogue)",
        period="2018–2020; validation from 2019-01, test from 2019-07", unit="one day",
        best=("AutoSTPP", "0.315"), second=("Jump-CNF", "0.340")),
    "citibike_nyc": dict(
        title="Citi Bike · hourly", group="chrono", column="Citi Bike (h)",
        what="Bike-share trip starts in New York City, hourly sequences",
        period="2019-04 – 2019-08; validation from 08-01, test from 08-16", unit="one hour",
        best=("AutoSTPP", "-15.717"), second=("Attn-CNF", "-15.467")),
    "austin311": dict(
        title="Austin 311", group="chrono", column="Austin",
        what="311 service requests in Austin, Texas",
        period="2023; validation from 09-14, test from 10-20", unit="one day",
        best=("DeepSTPP", "-8.941"), second=("AutoSTPP", "-8.873")),
    "seattle_fire911": dict(
        title="Seattle fire 911", group="chrono", column="Seattle",
        what="Fire-department 911 dispatches in Seattle",
        period="2023; validation from 09-14, test from 10-20", unit="one day",
        best=("Attn-CNF", "-9.241"), second=("Jump-CNF", "-9.208")),
    "nyc311": dict(
        title="NYC 311", group="chrono", column="NYC 311",
        what="311 service requests in New York City",
        period="2023; validation from 09-14, test from 10-20", unit="one day",
        best=("AutoSTPP", "-12.203"), second=("DeepSTPP", "-11.573")),
    "nyc_collisions": dict(
        title="NYC collisions", group="chrono", column="NYC coll.",
        what="Police-reported motor-vehicle collisions in New York City",
        period="2023; validation from 09-14, test from 10-20", unit="one day",
        best=("Attn-CNF", "-7.349"), second=("Jump-CNF", "-7.226")),
    "chicago_crime": dict(
        title="Chicago crime", group="chrono", column="Chicago",
        what="Crime reports in Chicago",
        period="2023; validation from 09-14, test from 10-20", unit="one day",
        best=("Attn-CNF", "-9.264"), second=("Jump-CNF", "-9.178")),
    "la_crime": dict(
        title="LA crime", group="chrono", column="LA",
        what="Crime reports in Los Angeles",
        period="2023; validation from 09-14, test from 10-20", unit="one day",
        best=("Attn-CNF", "-9.163"), second=("Jump-CNF", "-8.976")),
    "uber_nyc": dict(
        title="Uber pickups · NYC", group="chrono", column="Uber",
        what="Ride-hailing pickups in New York City",
        period="2014-04 – 2014-09; validation from 08-01, test from 09-01", unit="one hour",
        best=("Jump-CNF", "-14.114"), second=("Attn-CNF", "-13.906")),
    "covid": dict(
        title="COVID-19 · New Jersey", group="nstpp", column="COVID",
        what="COVID-19 case reports in New Jersey counties (The New York Times)",
        period="2020-03 – 2020-07; original Neural STPP split", unit="see card",
        best=("Attn-CNF", "-2.513"), second=("Jump-CNF", "-2.496")),
    "earthquakes": dict(
        title="Earthquakes · Japan", group="nstpp", column="Earthquakes",
        what="Earthquakes of magnitude ≥ 2.5 around Japan (USGS ComCat)",
        period="1990–2019; original Neural STPP split", unit="see card",
        best=("DeepSTPP", "3.411"), second=("AutoSTPP", "3.655")),
    "citibike_legacy": dict(
        title="Citi Bike · daily", group="nstpp", column="Citi Bike (d)",
        what="Bike-share trip starts in New York City, daily sequences (thinned, Neural STPP construction)",
        period="2019-04 – 2019-08; original Neural STPP split", unit="one day",
        best=("Attn-CNF", "-9.335"), second=("Jump-CNF", "-9.326")),
    "hawkesnest_entanglement": dict(
        title="HawkesNest entanglement suite", group="synthetic", column=None,
        what="Spatiotemporal Hawkes process on the unit square with tunable space–time coupling, levels L0–L3",
        period="ten realizations of 8,000 events per level, cut into 20 windows each", unit="one window"),
    "synthetic": dict(
        title="Synthetic controls", group="released", column=None,
        what="Homogeneous Poisson, moving seasonal Poisson and gamma renewal processes on the unit square",
        period="independent windows on [0, 1)", unit="one window"),
    "bold5000": dict(
        title="BOLD5000", group="released", column=None,
        what="Eventized responses derived from BOLD5000 fMRI recordings (space is not geographic)",
        period="see card", unit="see card"),
    "us_wildfires": dict(
        title="US wildfires", group="released", column=None,
        what="Wildfire ignitions in the United States (USFS RDS-2013-0009.4)",
        period="see card", unit="see card"),
    "us_accidents": dict(
        title="US traffic accidents", group="released", column=None,
        what="Road traffic accident reports in the United States (Moosavi et al.)",
        period="see card", unit="see card"),
}

# HawkesNest entanglement suite: paper Table 3 (excess test NLL over the true process, nats/event,
# mean over three seeds) and the generator table of the appendix.
HAWKESNEST_LEVELS = [("L0", "0", "0"), ("L1", "0.1", "1"), ("L2", "0.2", "2"), ("L3", "0.3", "3")]
HAWKESNEST_TABLE3 = [
    ("Separable Hawkes fit (reference)", "0.00", "0.08", "0.17", "0.25"),
    ("AutoSTPP", "0.04", "0.06", "0.07", "0.10"),
    ("DeepSTPP", "0.21", "0.35", "0.50", "0.63"),
    ("NSMPP", "1.45", "1.46", "1.50", "1.56"),
    ("NJSDE", "0.96", "0.97", "0.93", "0.95"),
    ("RMTPP", "2.25", "2.29", "2.32", "2.40"),
    ("THP", "2.14", "2.18", "2.21", "2.29"),
    ("DSTPP (variational bound)", "3.29", "3.32", "3.33", "3.35"),
]

NOTICES = {
    "covid": "Event data © The New York Times, for non-commercial use only, with credit to "
             "“The New York Times”. The governing terms are the NYT licence in the repository.",
    "us_accidents": "CC BY-NC-SA 4.0: non-commercial, share-alike. Cite both source papers listed on the card.",
}


# --------------------------------------------------------------------------------------------
# Hub access and statistics
# --------------------------------------------------------------------------------------------
def fetch(spec: ReleasedDataset, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(spec.repo_id, filename, repo_type="dataset", revision=spec.revision))


def card_text(spec: ReleasedDataset) -> str:
    return fetch(spec, "README.md").read_text()


def card_section(text: str, heading: str) -> str | None:
    m = re.search(rf"^## {re.escape(heading)}[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else None


def card_source(text: str) -> tuple[str | None, str | None]:
    prov = re.search(r"Publisher / provider:\s*(.+)", text)
    url = re.search(r"^- Source:\s*(\S+)", text, re.M)
    return (prov.group(1).strip() if prov else None, url.group(1).strip() if url else None)


def iter_events(rec: dict):
    """(times, locations) of one record, for both JSONL layouts on the Hub."""
    if "events" in rec:
        ev = rec["events"]
        return [e["t"] for e in ev], [(e["x"], e["y"]) for e in ev]
    return rec.get("times", []), [tuple(loc[:2]) for loc in rec.get("locations", [])]


def parse_start(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def hub_plotdata(spec: ReleasedDataset, config: str) -> dict:
    return PD.plotdata_from_files({sp: fetch(spec, spec.split_path(config, sp)) for sp in SPLITS})


def stats_of(pd: dict, config: str) -> dict:
    return {"config": config, "calendar": pd["calendar"], "splits": pd["stats"]}


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#52514e",
        "text.color": "#0b0b0b", "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": 8.5,
    })
    return plt


def _limits(x, y):
    import numpy as np

    lo_x, hi_x = np.nanpercentile(x, [0.5, 99.5])
    lo_y, hi_y = np.nanpercentile(y, [0.5, 99.5])
    pad_x, pad_y = 0.04 * (hi_x - lo_x or 1), 0.04 * (hi_y - lo_y or 1)
    return (lo_x - pad_x, hi_x + pad_x), (lo_y - pad_y, hi_y + pad_y)


def _geo_aspect(y0, y1):
    import numpy as np

    return 1.0 / max(np.cos(np.deg2rad((y0 + y1) / 2)), 0.2)


def _daily(pd: dict, split: str):
    return pd[f"days_{split}"].astype(object), pd[f"daycounts_{split}"]


def plot_dataset(name: str, pd: dict, geographic: bool = True) -> Path:
    """Card-page figure: locations coloured by split, and daily counts (or gaps) by split."""
    import numpy as np

    plt = _style()
    fig, (ax_map, ax_time) = plt.subplots(1, 2, figsize=(9.0, 3.3), gridspec_kw={"width_ratios": [1, 1.55]})
    x, y, code = pd["x"], pd["y"], pd["split"]
    for i, split in enumerate(SPLITS):  # test drawn last, on top
        m = code == i
        ax_map.scatter(x[m], y[m], s=1.2, lw=0, alpha=0.45, color=COLORS[split], rasterized=True)
    (x0, x1), (y0, y1) = _limits(x, y)
    ax_map.set_xlim(x0, x1)
    ax_map.set_ylim(y0, y1)
    if geographic and -90 <= y0 and y1 <= 90:
        ax_map.set_aspect(_geo_aspect(y0, y1))
        ax_map.set_xlabel("longitude")
        ax_map.set_ylabel("latitude")
    else:
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("x")
        ax_map.set_ylabel("y")
    ax_map.set_title("Event locations (subsample)", loc="left")
    ax_map.tick_params(labelsize=7.5)

    if pd["calendar"]:
        import matplotlib.dates as mdates

        for split in SPLITS:
            days, counts = _daily(pd, split)
            ax_time.plot(days, counts, lw=1.0, color=COLORS[split], label=LABELS[split])
        ax_time.set_ylabel("events per day")
        ax_time.set_title("Events per day, coloured by split", loc="left")
        loc = mdates.AutoDateLocator(maxticks=7)
        ax_time.xaxis.set_major_locator(loc)
        ax_time.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax_time.set_ylim(bottom=0)
    else:
        gaps = [pd[f"gaps_{s}"] for s in SPLITS]
        allg = np.concatenate([g for g in gaps if len(g)])
        lo, hi = np.percentile(np.log10(allg), [0.5, 99.9])
        bins = np.linspace(lo, hi, 41)
        for split, g in zip(SPLITS, gaps):
            if len(g):
                ax_time.hist(np.log10(g), bins=bins, histtype="step", lw=1.3, color=COLORS[split],
                             label=LABELS[split], density=True)
        ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
        ax_time.set_xticks(ticks)
        ax_time.set_xticklabels([f"$10^{{{int(t)}}}$" for t in ticks])
        ax_time.set_xlabel("time between consecutive events (dataset time unit, log scale)")
        ax_time.set_ylabel("density")
        ax_time.set_title("Inter-event times, by split", loc="left")
    ax_time.grid(axis="y", color="#e4e3df", lw=0.6)
    ax_time.legend(loc="upper left", ncol=3, handlelength=1.4)
    ax_time.tick_params(labelsize=7.5)

    fig.tight_layout(w_pad=2.0)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / f"{name}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_thumb(name: str, pd: dict, geographic: bool = True) -> Path:
    """Index thumbnail: location silhouette in one ink (4:3 box) over a sparkline of activity.

    Transparent background, no axes or text. The sparkline is events per day for calendar
    datasets, otherwise events per sequence in file order (train, val, test).
    """
    import numpy as np

    plt = _style()
    fig = plt.figure(figsize=(4.0, 3.4), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.03, 0.15, 0.94, 0.83])   # about 4:3 in pixels
    sp = fig.add_axes([0.03, 0.01, 0.94, 0.11])
    for a in (ax, sp):
        a.set_axis_off()
        a.patch.set_alpha(0)
    x, y = pd["x"], pd["y"]
    (x0, x1), (y0, y1) = _limits(x, y)
    keep = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    ax.scatter(x[keep], y[keep], s=1.0, lw=0, alpha=0.55, color=THUMB_INK, rasterized=True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(_geo_aspect(y0, y1) if geographic and -90 <= y0 and y1 <= 90 else 1.0, adjustable="datalim")
    if pd["calendar"]:
        series = np.concatenate([_daily(pd, s)[1] for s in SPLITS])
    else:
        series = np.concatenate([pd[f"seqcounts_{s}"] for s in SPLITS])
    series = series.astype(float)
    if len(series) > 400:  # smooth very long series into 400 bins
        series = np.array([b.mean() for b in np.array_split(series, 400)])
    sp.fill_between(np.arange(len(series)), series, color=THUMB_INK, alpha=0.35, lw=0)
    sp.plot(np.arange(len(series)), series, color=THUMB_INK, lw=0.8)
    sp.set_xlim(0, max(len(series) - 1, 1))
    sp.set_ylim(0, series.max() * 1.05 if series.max() > 0 else 1)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    path = THUMB_DIR / f"{name}.png"
    fig.savefig(path, dpi=100, transparent=True)
    plt.close(fig)
    return path


def plot_hawkesnest(spec: ReleasedDataset) -> Path:
    """One training window per level: locations coloured by time within the window."""
    import numpy as np

    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues_trunc", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.6))
    for ax, (cfg, (lvl, v, rho)) in zip(axes, zip(spec.configs, HAWKESNEST_LEVELS)):
        with open(fetch(spec, spec.split_path(cfg, "train"))) as fh:
            rec = json.loads(fh.readline())
        t, locs = iter_events(rec)
        xy = np.asarray(locs)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=np.asarray(t), s=3, lw=0, cmap=cmap)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_title(f"{lvl}: v = {v}, ρ = {rho}", loc="left")
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("time in window")
    cb.outline.set_visible(False)
    path = IMG_DIR / "hawkesnest_entanglement.png"
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------------
def fmt_int(n) -> str:
    return f"{n:,}" if isinstance(n, int) else "–"


def chips(name: str, spec: ReleasedDataset) -> str:
    """At most one status mark: script-only > synthetic > paper > not in paper."""
    meta = PAPER[name]
    if spec.form == "script-only":
        return '<span class="sh-ds-chip sh-ds-chip--muted">script-only</span>'
    if name in ("hawkesnest_entanglement", "synthetic"):
        return '<span class="sh-ds-chip sh-ds-chip--synth">synthetic</span>'
    if meta["group"] in ("chrono", "nstpp"):
        return '<span class="sh-ds-chip">paper</span>'
    return '<span class="sh-ds-chip sh-ds-chip--muted">not in paper</span>'


def load_snippet(name: str, spec: ReleasedDataset) -> str:
    if spec.form == "script-only":
        return (
            "```python\nfrom seahorse.data import load_released_dataset\n\n"
            f"# after rebuilding into data/{name} (see below); the splits are verified against the card\n"
            f'splits = load_released_dataset("{name}", local_root="data/{name}")\n```\n'
        )
    lines = ["```python", "from seahorse.data import load_released_dataset", ""]
    lines.append(f'splits = load_released_dataset("{name}")  # config "{spec.default_config}"')
    for cfg in spec.configs:
        if cfg != spec.default_config:
            lines.append(f'other = load_released_dataset("{name}", config="{cfg}")')
            break
    lines += ["train, val, test = splits[\"train\"], splits[\"val\"], splits[\"test\"]", "```", "",
              "```bash", f"python -m seahorse fit --preset poisson_gmm --dataset {name} --out runs/{name}_poisson",
              "```"]
    return "\n".join(lines) + "\n"


def rebuild_block(text: str, spec: ReleasedDataset) -> str:
    sec = card_section(text, "How to rebuild")
    code = None
    if sec:
        m = re.search(r"```bash\n(.*?)```", sec, re.S)
        code = m.group(1).strip() if m else None
    if code is None:
        code = ("pip install -r rebuild/requirements.txt\n"
                "python rebuild/rebuild.py --out ./rebuilt\n"
                "python verify.py --out ./rebuilt")
    why = ("The records contain address-level emergency call types, so the data are not redistributed."
           if spec.name == "seattle_fire911" else
           "The publisher's licence does not allow hosting the data as a stand-alone dataset.")
    return (
        f"{why} The repository ships a rebuild script, a verifier and the expected split hashes:\n\n"
        "```bash\n"
        f"huggingface-cli download {spec.repo_id} --repo-type dataset --revision {spec.revision} --local-dir {spec.repo}\n"
        f"cd {spec.repo}\n{code}\n```\n\n"
        "Please do not re-host the rebuilt files. The exact arguments and the expected outputs are on the "
        f"[dataset card](https://huggingface.co/datasets/{spec.repo_id}).\n"
    )


def stats_table(stats: dict | None, unit: str, source: str = "the Hub files") -> str:
    if not stats:
        return ""
    rows = ["| Split | Sequences | Events | Events per sequence (min / median / max) |",
            "| --- | ---: | ---: | --- |"]
    for split in SPLITS:
        sp = stats["splits"][split]
        e = sp["events_per_seq"]
        eps = f"{e[0]:,} / {e[1]:,.0f} / {e[2]:,}" if e else "–"
        rows.append(f"| {LABELS[split]} | {fmt_int(sp['sequences'])} | {fmt_int(sp['events'])} | {eps} |")
    note = f"\nOne sequence = {unit}. Counts of the `{stats['config']}` configuration, computed from {source}.\n"
    return "\n".join(rows) + "\n" + note


def write_card(name: str, spec: ReleasedDataset, stats: dict | None, text: str, extra_stats: dict | None) -> None:
    meta = PAPER[name]
    provider, src_url = card_source(text)
    hub = f"https://huggingface.co/datasets/{spec.repo_id}"
    lines = [f"# {meta['title']}", "", f'<p class="sh-ds-eyebrow">{GROUPS[meta["group"]]}</p>', "",
             f'<div class="sh-card-chips" markdown="0">{chips(name, spec)}<code class="sh-ds-id">{name}</code></div>',
             "", meta["what"] + ".", ""]
    img = IMG_DIR / f"{name}.png"
    if img.exists():
        if name == "hawkesnest_entanglement":
            alt = "Four unit-square scatter plots, one training window per entanglement level, coloured by time"
            cap = ("One training window per level; colour: time within the window. With growing v the offspring "
                   "of an event drift along the diagonal as they age.")
        else:
            alt = f"{meta['title']}: event locations and the temporal profile of the three splits"
            cap = ("Left: event locations, a random subsample of at most 30,000 events, coloured by split. Right: "
                   + ("events per calendar day." if (stats or {}).get("calendar")
                      else "distribution of the time between consecutive events within a sequence.")
                   + f" Configuration <code>{spec.default_config}</code>."
                   + (" Computed from the campaign's prepared copy, whose split hashes equal the card's expected"
                      " rebuild hashes; the data are not hosted." if spec.form == "script-only" else ""))
        lines += [f'<figure class="sh-figure" markdown="0"><img src="../../../assets/datasets/{name}.png" '
                  f'alt="{alt}" loading="lazy"><figcaption>{cap}</figcaption></figure>', ""]

    facts = [("Period and splits", meta["period"])]
    if provider:
        facts.append(("Source", provider + (f" ([link]({src_url}))" if src_url else "")))
    facts.append(("Licence", spec.license))
    facts.append(("Hub repository", f"[`{spec.repo_id}`]({hub}) at revision `{spec.revision[:7]}`"))
    facts.append(("Configurations", ", ".join(f"`{c}`" + (" (default)" if c == spec.default_config else "")
                                               for c in spec.configs)))
    facts.append(("Availability", "rebuild script only (data not hosted)" if spec.form == "script-only"
                  else "hosted; splits hash-verified on load"))
    if meta["column"]:
        facts.append(("In the paper", f"Table 2 column “{meta['column']}”; all 19 presets × 3 seeds"))
    elif name == "hawkesnest_entanglement":
        facts.append(("In the paper", "Table 3 and Figures 5–6; eight presets × 3 seeds per level"))
    else:
        facts.append(("In the paper", "not benchmarked"))
    lines += ["## Facts", "", "| Field | Value |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in facts] + [""]
    if name in NOTICES:
        lines += ['!!! warning "Licence terms"', f"    {NOTICES[name]}", ""]

    if stats:
        lines += ["## Size", "", stats_table(stats, meta["unit"], "the campaign's prepared copy (hashes equal the card's)"
                                             if spec.form == "script-only" else "the Hub files")]
    elif spec.form == "script-only":
        sec = card_section(text, "Statistics (of the rebuilt corpus)") or card_section(text, "Splits")
        table = None
        if sec:
            m = re.search(r"((?:^\|.*\|\n?)+)", sec, re.M)
            table = m.group(1).strip() if m else None
        if table:
            lines += ["## Size", "", "From the dataset card (statistics of the rebuilt corpus):", "", table, ""]
    if extra_stats:
        lines += ["Other configurations:", "", "| Configuration | train / val / test sequences | train / val / test events |",
                  "| --- | --- | --- |"]
        for cfg, st in extra_stats.items():
            seqs = " / ".join(fmt_int(st["splits"][s]["sequences"]) for s in SPLITS)
            evs = " / ".join(fmt_int(st["splits"][s]["events"]) for s in SPLITS)
            lines.append(f"| `{cfg}` | {seqs} | {evs} |")
        lines.append("")

    lines += ["## Load it", "", load_snippet(name, spec)]
    if spec.form == "script-only":
        lines += ["## Rebuild it", "", rebuild_block(text, spec)]

    if meta.get("best"):
        b, s = meta["best"], meta["second"]
        lines += ["## In the paper", "",
                  "All 19 benchmarked presets were trained on this dataset with three seeds under the shared budget "
                  "(three tuning trials of 5,000 steps, refit of at most 20,000 steps). Best and second-best "
                  "exact- or numerical-likelihood presets by mean test NLL (nats per event, raw units; comparable "
                  "only within this dataset):", "",
                  "| Rank | Preset | Test NLL |", "| --- | --- | ---: |",
                  f"| 1 | {b[0]} | {b[1].replace('-', '−')} |", f"| 2 | {s[0]} | {s[1].replace('-', '−')} |", "",
                  "Full column, standard deviations and per-cell caveats: Table 2 and Table 10 of the paper; "
                  "see [Paper results](../../results.md). To place your model here, see "
                  "[Add your model and benchmark it](../../benchmark/your-model.md).", ""]
    if name == "hawkesnest_entanglement":
        lines += ["## Levels", "",
                  "Triggering kernel: exponential in time (β = 0.3), Gaussian in space (σ = 0.03) with a centre "
                  "that drifts at speed v; entanglement ρ = vβ/σ. Background and branching ratio (realized ≈ 0.70) "
                  "are held fixed across levels.", "",
                  "| Level | v | ρ | Config |", "| --- | --- | --- | --- |"]
        lines += [f"| {l} | {v} | {r} | `hawkesnest_entanglement_{l}` |" for l, v, r in HAWKESNEST_LEVELS]
        lines += ["", "Excess test NLL over the true process (nats/event, mean over three seeds; paper Table 3):", "",
                  "| Model | L0 | L1 | L2 | L3 |", "| --- | ---: | ---: | ---: | ---: |"]
        lines += [f"| {r[0]} | {' | '.join(r[1:])} |" for r in HAWKESNEST_TABLE3]
        lines += ["", "AutoSTPP is the only model below the separable reference at L1–L3. SMASH has no likelihood. "
                  "The generator code, the true-process parameters and SHA-256 checksums are in the folder "
                  f"`hawkesnest_entanglement/` of the [repository]({hub}).", ""]

    lines += ["---", "", f"Generated by `docs/_scripts/make_dataset_cards.py` from the [dataset card]({hub}) "
              "and the Hub files. Provenance, cleaning counts and known issues are on the card.", ""]
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    (CARDS_DIR / f"{name}.md").write_text("\n".join(lines))


def write_index(all_stats: dict) -> None:
    lines = ["# Dataset cards", "",
             '<p class="sh-ds-eyebrow">13 real datasets · HawkesNest entanglement suite · released extras</p>', "",
             "The NeurIPS 2026 paper benchmarks 19 presets on **13 real datasets**: ten with chronological "
             "train/validation/test splits and the three datasets of the Neural STPP benchmark (COVID, Earthquakes, "
             "Citi Bike daily) under their original splits. The **HawkesNest entanglement suite** is the controlled "
             "synthetic stress test. All are on the Hugging Face Hub under "
             "[`seahorse-stpp`](https://huggingface.co/seahorse-stpp), pinned to a release revision and verified "
             "against the card hashes on load. Three repositories are script-only: they rebuild the data from the "
             "source instead of hosting it.", "",
             "Each card shows where and when the events fall, the split sizes, the licence, how to load the data and "
             "the best presets of the paper.", ""]
    for group, title in GROUPS.items():
        names = [n for n in PAPER if PAPER[n]["group"] == group and n in RELEASED_DATASETS]
        lines += [f"## {title}", "", '<div class="sh-ds-grid sh-ds-grid--tiles" markdown="0">']
        for n in names:
            spec, meta = RELEASED_DATASETS[n], PAPER[n]
            st = (all_stats.get(n) or {}).get("default")
            thumb = (f'<img class="sh-ds-thumb" src="../assets/datasets/thumbs/{n}.png" alt="" loading="lazy">'
                     if (THUMB_DIR / f"{n}.png").exists()
                     else '<span class="sh-ds-thumb sh-ds-thumb--empty">rebuild script</span>')
            size = "&nbsp;"
            if st:
                ev = sum(st["splits"][s]["events"] for s in SPLITS)
                sq = sum(st["splits"][s]["sequences"] for s in SPLITS)
                size = f"{ev:,} events · {sq:,} sequences"
            lines.append(
                f'<a class="sh-ds-tile" href="cards/{n}/">'
                f"{thumb}"
                f'<span class="sh-ds-tile-head"><span class="sh-ds-name">{meta["title"]}</span>{chips(n, spec)}</span>'
                f'<span class="sh-ds-tile-id">{n}</span>'
                f'<span class="sh-ds-tile-desc">{meta["what"]}</span>'
                f'<span class="sh-ds-size">{size}</span>'
                "</a>")
        lines += ["</div>", ""]

    lines += ["## Summary table", "",
              "| Dataset | Short name | Paper column | Period and splits | Sequences (train / val / test) | Events | Licence | Hosted |",
              "| --- | --- | --- | --- | --- | ---: | --- | --- |"]
    for n, meta in PAPER.items():
        if n not in RELEASED_DATASETS:
            continue
        spec, st = RELEASED_DATASETS[n], (all_stats.get(n) or {}).get("default")
        seqs = " / ".join(fmt_int(st["splits"][s]["sequences"]) for s in SPLITS) if st else "see card"
        ev = fmt_int(sum(st["splits"][s]["events"] for s in SPLITS)) if st else "see card"
        hosted = "script" if spec.form == "script-only" else "yes"
        lines.append(f"| [{meta['title']}](cards/{n}.md) | `{n}` | {meta['column'] or '–'} | {meta['period']} | "
                     f"{seqs} | {ev} | {spec.license} | {hosted} |")
    lines += ["", "Counts are for the default configuration; the HawkesNest row counts level L0. Regenerate this page "
              "and the cards with `python docs/_scripts/make_dataset_cards.py`.", ""]
    (DOCS / "datasets" / "index.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------------------------
def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="re-write pages from dataset_stats.json (cards still read)")
    ap.add_argument("--only", nargs="*", help="restrict to these short names")
    args = ap.parse_args()

    cached = json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else {}
    names = [n for n in PAPER if n in RELEASED_DATASETS and (not args.only or n in args.only)]
    for name in names:
        spec = RELEASED_DATASETS[name]
        text = card_text(spec)
        stats, extra = cached.get(name, {}).get("default"), cached.get(name, {}).get("extra")
        cache = PLOTDATA_CACHE / f"{name}.npz"
        pd = None
        if cache.exists() and (args.offline or spec.form == "script-only"):
            pd = PD.load(cache)
        elif spec.form != "script-only" and not args.offline:
            print(f"[{name}] downloading and counting ...", flush=True)
            pd = hub_plotdata(spec, spec.default_config)
            PLOTDATA_CACHE.mkdir(parents=True, exist_ok=True)
            PD.save(pd, cache)
        if pd is not None:
            geographic = name not in ("synthetic", "bold5000", "hawkesnest_entanglement")
            if name == "hawkesnest_entanglement":
                plot_hawkesnest(spec)
            else:
                plot_dataset(name, pd, geographic=geographic)
            plot_thumb(name, pd, geographic=geographic)
            stats = stats_of(pd, spec.default_config)
            if not args.offline and spec.form != "script-only" and name in ("hawkesnest_entanglement", "synthetic"):
                extra = {c: stats_of(hub_plotdata(spec, c), c) for c in spec.configs if c != spec.default_config}
            cached[name] = {"default": stats, "extra": extra}
        write_card(name, spec, stats, text, extra)
        print(f"[{name}] card written", flush=True)
    STATS_PATH.write_text(json.dumps(cached, indent=1, default=str))
    write_index(cached)


if __name__ == "__main__":
    main()
