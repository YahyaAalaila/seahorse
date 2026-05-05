# Getting Started

## Install

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install development tooling when you want to run tests and lint checks:

```bash
python -m pip install -e ".[dev]"
```

Install HPO support when you use `tune` or benchmark HPO:

```bash
python -m pip install -e ".[hpo]"
```

## Verify The CLI

```bash
python -m unified_stpp --help
python -m unified_stpp fit --help
python -m unified_stpp tune --help
python -m unified_stpp bench --help
python -m unified_stpp evaluate --help
```

The top-level CLI exposes four modes: `fit`, `tune`, `bench`, and `evaluate`.

## Train A Tiny Local Run

The repository includes tiny JSONL splits for command-shape checks:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train examples/tiny_jsonl/train.jsonl \
  --val examples/tiny_jsonl/val.jsonl \
  --test examples/tiny_jsonl/test.jsonl \
  --out runs/quickstart \
  --override training.n_epochs=1 training.batch_size=2 data.num_workers=0
```

`fit` writes a timestamped run directory under:

```text
runs/quickstart/fit/poisson_gmm/<run_id>/
```

Use that run directory for post-fit evaluation:

```bash
python -m unified_stpp evaluate metrics \
  --run runs/quickstart/fit/poisson_gmm/<run_id> \
  --data examples/tiny_jsonl/test.jsonl \
  --split test \
  --metric-profile core
```

## Use A Hugging Face Dataset

Pass a dataset repository, optionally with a subdirectory:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --dataset owner/repo[/subdir] \
  --dataset-revision main \
  --out runs/hf_fit
```

The resolved dataset path must contain `train.jsonl` and `val.jsonl`. `test.jsonl`
is used when available for `fit`.

## Use Explicit Local Paths

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl \
  --out runs/local_fit
```

Use explicit paths when the data is private, generated locally, or not hosted on
Hugging Face.
