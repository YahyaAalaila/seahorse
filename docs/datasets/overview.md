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
| Local JSONL files | Pass `--train`, `--val`, `--test` flags or a directory with `--dataset` |
| Local split collection | Point `--splits_dir` at a root with one subdirectory per dataset |
| Hugging Face Hub | Pass `--dataset owner/repo[/subdir]` — Seahorse downloads and caches |

See [Data Format](../data-format.md) for the full contract and command support matrix.

## Ready-to-use Datasets

Seahorse is designed to work with Hugging Face-hosted STPP datasets that follow the JSONL split convention. Built-in dataset aliases resolve to the project-controlled `seahorse-stpp` namespace. See [Ready-to-use HF Datasets](hf-datasets.md) for sources.

## Next Steps

- [Data Format](../data-format.md) — detailed format specification and command matrix.
- [Ready-to-use HF Datasets](hf-datasets.md) — pre-formatted datasets you can use immediately.
- [Add Your Dataset](add-dataset.md) — how to prepare your own data.
- [Conversion Standard](conversion.md) — how to convert from common formats.
