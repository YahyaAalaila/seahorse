# Evaluate And Visualize

Use the CLI after `fit` or `bench` has produced saved run directories. These
commands create benchmark-aligned metrics and visualization artifacts.

## Placeholder Notebook

[Colab badge placeholder]

A Colab notebook can be linked here when a maintained notebook exists.

## Core Metrics

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/evaluate/core_test
```

Use `python -m unified_stpp evaluate metrics --help` to see the profiles and
metric controls supported by the installed version.

## Predictive Metrics

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile predictive \
  --k-pred 64 \
  --out runs/evaluate/predictive_test
```

Predictive profiles create sampling artifacts. Use `--max-seqs` or
`--seq-shard` for large datasets.

## Predictive Comparison

```bash
python -m unified_stpp evaluate predictive-compare \
  --run path/to/run_a \
  --run path/to/run_b \
  --label model_a \
  --label model_b \
  --history data/my_dataset/test.jsonl \
  --split test \
  --horizon 1.0 \
  --out runs/evaluate/predictive_compare
```

## Surface Diagnostics

```bash
python -m unified_stpp evaluate surface \
  --run path/to/run_dir \
  --history data/my_dataset/test.jsonl \
  --split test \
  --seq-idx 0 \
  --profile history_frame \
  --out runs/evaluate/surface
```

The `surface` command supports profiles such as `history_frame` and
`future_exact`; use `python -m unified_stpp evaluate surface --help` for exact
options in your environment.
