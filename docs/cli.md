# CLI Reference

The CLI/config interface is the stable path for reproducible runs, HPO,
benchmark campaigns, and paper-style artifacts.

Use the Python API when you want a normal programmatic workflow for one model.
See [Python API](python-api.md).

The public CLI is the module entrypoint:

```bash
python -m unified_stpp --help
```

It exposes:

```text
fit
tune
bench
evaluate
```

Use `--help` on each command for the exact arguments supported by the installed
version.

## fit

Train one model from a registered preset or YAML config:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl \
  --out runs/fit \
  --override training.n_epochs=10 training.batch_size=64
```

Use a YAML config instead of a preset:

```bash
python -m unified_stpp fit \
  --config path/to/config.yaml \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl
```

Key options:

- `--preset` or `--config`: required config source.
- `--dataset`: named, local, or Hugging Face dataset source.
- `--dataset-revision`: optional Hugging Face revision.
- `--train`, `--val`, `--test`: explicit local JSONL paths.
- `--out`: output directory for logs and run artifacts.
- `--save`: directory for saving the runner.
- `--override`: dotted config overrides such as `training.lr=1e-4`.

## tune

Run HPO and write a best-config YAML:

```bash
python -m unified_stpp tune \
  --preset poisson_gmm \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --n_trials 20 \
  --search-alg random \
  --scheduler asha \
  --out runs/hpo/poisson_gmm_best.yaml
```

`tune` requires HPO dependencies:

```bash
python -m pip install -e ".[hpo]"
```

Key options:

- `--preset` or `--config`: required config source.
- `--dataset` or `--train --val`: data source.
- `--n_trials`: maximum HPO trials.
- `--search-alg`: `random` or `bayesian`.
- `--scheduler`: `asha` or `none`.
- `--seed`: HPO seed.
- `--max-concurrent-trials`: concurrency cap.
- `--out`: best-config YAML path.

## bench

Run a benchmark grid:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits_root \
  --seeds 1 2 3 \
  --out runs/bench \
  --n_workers 1
```

Benchmark one dataset directory or Hugging Face dataset source:

```bash
python -m unified_stpp bench \
  --preset poisson_gmm \
  --dataset owner/repo[/subdir] \
  --seeds 1 \
  --out runs/bench_one
```

Key options:

- `--preset` or `--presets`: required model preset selection.
- `--dataset` or `--splits_dir`: required benchmark data source.
- `--datasets`: filter dataset names from a split collection.
- `--seeds`: one or more seeds.
- `--out`: benchmark output directory.
- `--n_workers`: worker count.
- `--tune`: run HPO before evaluation.
- `--tune-dataset`: dataset used for benchmark HPO.
- `--hpo_configs_dir`: directory containing `{preset}_best.yaml` files.
- `--normalize` or `--no-normalize`: benchmark normalization policy.

## evaluate

`evaluate` is for post-fit analysis on saved runs:

```bash
python -m unified_stpp evaluate --help
```

Supported modes:

- `metrics`: metric reports with artifact-backed profiles.
- `predictive-compare`: qualitative future-window predictive comparisons.
- `surface`: single-run exact or factorized surface diagnostics.
- `merge-artifacts`: merge predictive sample artifacts from sharded runs.

Metric evaluation:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data path/to/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/evaluate/core_test
```

The available metric profiles are reported by:

```bash
python -m unified_stpp evaluate metrics --help
```
