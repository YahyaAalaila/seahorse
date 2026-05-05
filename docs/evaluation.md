# Evaluation

Use `evaluate` after `fit` or `bench` has produced a saved run directory. This
is the current supported path for benchmark-aligned metrics and diagnostic
artifacts.

The planned Python API will eventually expose a normal programmatic evaluation
workflow for one fitted model. That wrapper is not stable on this branch yet.

## Metric Reports

Run the core metric profile on a saved run:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data path/to/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/evaluate/core_test
```

Use `--metric-profile` when you want a predefined profile. The installed CLI
prints the current profile names:

```bash
python -m unified_stpp evaluate metrics --help
```

Use `--metric` when you want explicit metric names:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data path/to/test.jsonl \
  --split test \
  --metric nll \
  --out runs/evaluate/nll_test
```

Useful controls:

- `--max-seqs`: cap the number of sequences.
- `--max-events`: cap events per sequence.
- `--k-pred`: next-event sample count.
- `--k-gen`: full-rollout sample count.
- `--n-context-events`: observed prefix length for rollout metrics.
- `--device`: `auto`, `cpu`, `cuda`, `cuda:0`, `mps`, or another supported device string.
- `--artifact-dir`: root directory for persisted evaluation artifacts.

## Sharded Metric Evaluation

Evaluate sequence ranges with `--seq-shard`:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data path/to/test.jsonl \
  --split test \
  --metric-profile predictive \
  --seq-shard 0:50 \
  --out runs/evaluate/shard_0
```

Merge predictive sample artifacts after running shards:

```bash
python -m unified_stpp evaluate merge-artifacts \
  --artifact-dir runs/evaluate/shard_0/artifacts \
  --artifact-dir runs/evaluate/shard_1/artifacts \
  --out runs/evaluate/merged_artifacts
```

Repeat `--artifact-dir` in shard order.

## Predictive Comparison

Compare sampled future windows across one or more saved runs:

```bash
python -m unified_stpp evaluate predictive-compare \
  --run path/to/run_a \
  --run path/to/run_b \
  --label model_a \
  --label model_b \
  --history path/to/test.jsonl \
  --split test \
  --horizon 1.0 \
  --out runs/evaluate/predictive_compare
```

`predictive-compare` is a qualitative diagnostic workflow. Use
`evaluate metrics --metric-profile predictive` for benchmark-aligned predictive
metric artifacts.

## Surface Diagnostics

Render a surface diagnostic for one saved run:

```bash
python -m unified_stpp evaluate surface \
  --run path/to/run_dir \
  --history path/to/test.jsonl \
  --split test \
  --seq-idx 0 \
  --profile history_frame \
  --out runs/evaluate/surface
```

The `surface` command supports `history_frame` and `future_exact` profiles. Run
`python -m unified_stpp evaluate surface --help` for the exact options in your
installed version.
