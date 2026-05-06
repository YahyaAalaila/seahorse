# Data Format

The same JSONL split format is the public data contract for the CLI and the
Python-first API.

## Minimal JSONL Record

Each line in a split file is one event sequence. The minimal public record has
matching `times` and `locations` arrays:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

`times[i]` and `locations[i]` describe the same event. Optional per-event arrays
such as `marks`, `event_covariates`, and `field_covariates` must have the same
length as `times`.

## Single-Dataset Layout

Use this layout for `fit`, `tune`, or one-dataset `bench` runs:

```text
dataset_root/
  train.jsonl
  val.jsonl
  test.jsonl
```

`train.jsonl` and `val.jsonl` are required. `test.jsonl` is optional for data
resolution, but most evaluation workflows need a test or held-out split.

## Benchmark Layout

Use a split collection when benchmarking multiple datasets:

```text
splits_root/
  dataset_a/
    train.jsonl
    val.jsonl
    test.jsonl
  dataset_b/
    train.jsonl
    val.jsonl
    test.jsonl
```

Run all datasets found under the root:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits_root \
  --seeds 1 2 3 \
  --out runs/bench
```

Restrict a split collection to selected dataset names:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits_root \
  --datasets dataset_a dataset_b \
  --seeds 1 \
  --out runs/bench_subset
```

## Data Sources By Command

`fit` accepts either:

```bash
python -m unified_stpp fit --preset poisson_gmm --dataset owner/repo[/subdir]
```

or:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl
```

`tune` accepts the same data forms, without `--test`.

`bench` accepts either `--dataset` or `--splits_dir`. It does not accept
explicit `--train`, `--val`, or `--test` paths.

`evaluate metrics` reads evaluation data from `--data`.
