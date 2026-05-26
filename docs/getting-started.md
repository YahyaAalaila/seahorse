# Getting Started

This page gets you from a fresh checkout to the two main Seahorse workflows:
one-model Python experiments and reproducible CLI runs.

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

## Prepare Data

For local runs, create JSONL split files:

```text
data/my_dataset/
  train.jsonl
  val.jsonl
  test.jsonl
```

Each line is one sequence with `times` and `locations`:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

See [Data Format](data-format.md) for local and Hugging Face data sources.

## Run One Model With Python

Use the Python API when you want one model in a script or notebook:

```python
from unified_stpp import AutoSTPP, load_jsonl

train = load_jsonl("data/my_dataset/train.jsonl")
val = load_jsonl("data/my_dataset/val.jsonl")
test = load_jsonl("data/my_dataset/test.jsonl")

model = AutoSTPP(device="cpu", seed=42)
model.fit(train, val, test, epochs=10, batch_size=64)
scores = model.evaluate(test)
samples = model.predict_next(test, n_samples=32)
```

The Python `evaluate()` method currently reports the implemented likelihood
metrics exposed by the estimator. The predictive method is `predict_next()`.
There is no generic `predict()` method in this API.

Continue with [Python API](python-api.md) or
[Train One Model](examples/train-one-model.md).

## Use The CLI For Reproducible Runs

Use the CLI when you need saved run artifacts, HPO, benchmark campaigns, or
paper-style reproducibility.

Verify the installed commands:

```bash
python -m unified_stpp --help
python -m unified_stpp fit --help
python -m unified_stpp tune --help
python -m unified_stpp bench --help
python -m unified_stpp evaluate --help
```

The top-level CLI exposes four modes: `fit`, `tune`, `bench`, and `evaluate`.

## Train A Local CLI Run

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train data/my_dataset/train.jsonl \
  --val data/my_dataset/val.jsonl \
  --test data/my_dataset/test.jsonl \
  --out runs/quickstart \
  --override training.n_epochs=1 training.batch_size=2 data.num_workers=0
```

`fit` writes a timestamped run directory under:

```text
runs/quickstart/fit/poisson_gmm/<run_id>/
```

Evaluate that saved run:

```bash
python -m unified_stpp evaluate metrics \
  --run runs/quickstart/fit/poisson_gmm/<run_id> \
  --data data/my_dataset/test.jsonl \
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

The resolved dataset path must contain `train.jsonl` and `val.jsonl`.
`test.jsonl` is used when available for `fit` and is normally required for
post-fit evaluation.
