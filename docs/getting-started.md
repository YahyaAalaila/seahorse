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

## Run One Model With The Python API

Use the Python API when you want to train and evaluate one model
programmatically.

```python
from unified_stpp import AutoSTPP, PoissonGMM

model = AutoSTPP(device="cpu")
baseline = PoissonGMM()
```

Load JSONL splits and fit the model:

```python
from unified_stpp import load_jsonl

train = load_jsonl("path/to/train.jsonl")
val = load_jsonl("path/to/val.jsonl")
test = load_jsonl("path/to/test.jsonl")

model.fit(train, val, test, epochs=10, batch_size=64)
scores = model.evaluate(test)
samples = model.predict_next(test, n_samples=32)
```

`evaluate()` currently reports implemented likelihood metrics. The implemented
predictive method is `predict_next()`; there is no generic `predict()` method in
this API.

See [Python API](python-api.md) for method details.

## Use The CLI For Reproducible Runs

Use the CLI when you want reproducible runs, HPO, benchmark campaigns, and
paper-style artifacts.

Verify the CLI:

```bash
python -m unified_stpp --help
python -m unified_stpp fit --help
python -m unified_stpp tune --help
python -m unified_stpp bench --help
python -m unified_stpp evaluate --help
```

The top-level CLI exposes four modes: `fit`, `tune`, `bench`, and `evaluate`.

## CLI: Train A Local Run

Start with explicit JSONL split files:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl \
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
  --data path/to/test.jsonl \
  --split test \
  --metric-profile core
```

## CLI: Use A Hugging Face Dataset

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

## CLI: Use Explicit Local Paths

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
