# Dataset Overview

Seahorse reads event sequence data from **JSONL split files** — one JSON object per line, one line per sequence. The same format works for single-model runs, benchmark campaigns, and Hugging Face sources.

## What a Dataset Needs

A complete dataset for a Seahorse run consists of three split files:

| File | Required for | Notes |
| --- | --- | --- |
| `train.jsonl` | `fit`, `tune`, `bench` | Training sequences |
| `val.jsonl` | `fit`, `tune`, `bench` | Validation sequences — drives early stopping |
| `test.jsonl` | evaluation | Held-out test sequences |

Each file is one JSON object per line. Each object is one event sequence.

## Minimal Record

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

`times[i]` and `locations[i]` describe the same event. `locations` must be a list of 2D coordinate pairs.

## Data Sources

| Source | How to use |
| --- | --- |
| Released `seahorse-stpp` datasets | `load_dataset("covid")` or `--dataset covid` (short name, pinned revision) |
| Any Hugging Face repo | Pass `--dataset owner/repo[/subdir]` — Seahorse downloads and caches |
| Local JSONL files | Pass `--train`, `--val`, `--test` flags or a directory with `--dataset` |
| Local split collection | Point `--splits_dir` at a root with one subdirectory per dataset |

See [Data Format](../data-format.md) for the full contract and command support matrix.

## Ready-to-use Datasets

The datasets of the Seahorse benchmark are released in the
[`seahorse-stpp`](https://huggingface.co/seahorse-stpp) Hugging Face
organization: the 13 real datasets of the paper (urban service and incident
records, bike-share and ride-hailing demand, earthquake catalogs, public health),
the HawkesNest entanglement suite, and a few released datasets outside the paper,
all in the same JSONL split format.
Load any of them by its short name; every repository is pinned to its release
revision and each split is verified against the hashes on the dataset card:

```python
from seahorse.data import load_dataset, load_released_dataset

splits = load_dataset("covid")                   # downloads + caches
splits = load_released_dataset("austin311")      # same, with hash verification
```

Three datasets (`citibike_nyc`, `citibike_legacy`, `seattle_fire911`) are not
hosted, because of the source licence or, for Seattle, the content; their
repositories ship a rebuild script and the expected hashes instead.

Every dataset has a **[card](index.md)** with a plot of where and when its events
fall, split sizes, licence, load snippet and the best presets of the paper.

Need controlled ground truth? The paper's synthetic stress test is the
**HawkesNest entanglement suite**, generated with
[HawkesNest](https://github.com/YahyaAalaila/HawkesNest): a spatiotemporal Hawkes
process whose space–time coupling grows over four levels L0–L3, with the true
intensity stored for recovery diagnostics. It loads with
`load_released_dataset("hawkesnest_entanglement", config="hawkesnest_entanglement_L0")`;
see its [card](cards/hawkesnest_entanglement.md).

## Next Steps

- [Dataset Cards](index.md) — one page per released dataset, with plots and statistics.
- [Dataset Catalog](catalog.md) — the 17 released repositories, grouped by domain.
- [Data Format](../data-format.md) — detailed format specification and command matrix.
- [Ready-to-use HF Datasets](hf-datasets.md) — load by repo id, or host your own.
- [Add Your Dataset](add-dataset.md) — how to prepare your own data.
- [Conversion Standard](conversion.md) — how to convert from common formats.
