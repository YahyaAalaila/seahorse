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
| Curated `seahorse-stpp` datasets | `load_dataset("citibike")` or `--dataset citibike` (short name) |
| Any Hugging Face repo | Pass `--dataset owner/repo[/subdir]` — Seahorse downloads and caches |
| Local JSONL files | Pass `--train`, `--val`, `--test` flags or a directory with `--dataset` |
| Local split collection | Point `--splits_dir` at a root with one subdirectory per dataset |

See [Data Format](../data-format.md) for the full contract and command support matrix.

## Ready-to-use Datasets

Seahorse curates **13 real-world STPP datasets** in the
[`seahorse-stpp`](https://huggingface.co/seahorse-stpp) Hugging Face
organization — spanning urban mobility, crime, natural hazards, public health,
social check-ins, and even neuroimaging — all in the same JSONL split format.
Load any of them by its short name:

```python
from seahorse.data import load_dataset

splits = load_dataset("citibike")  # downloads + caches
```

Browse the full collection, with load snippets and each dataset's space/time
axes, in the **[Dataset Catalog](catalog.md)**.

Need controlled ground truth? Seahorse's synthetic benchmark sequences are
generated with [HawkesNest](https://github.com/YahyaAalaila/HawkesNest) — its
**entanglement suite** produces spatio-temporal data with tunable space–time
coupling for stress-testing models. See
[Synthetic benchmark suites](catalog.md#synthetic-benchmark-suites).

## Next Steps

- [Dataset Catalog](catalog.md) — browse the 13 ready-to-use datasets.
- [Data Format](../data-format.md) — detailed format specification and command matrix.
- [Ready-to-use HF Datasets](hf-datasets.md) — load by repo id, or host your own.
- [Add Your Dataset](add-dataset.md) — how to prepare your own data.
- [Conversion Standard](conversion.md) — how to convert from common formats.
