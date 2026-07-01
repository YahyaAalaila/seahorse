# Getting Started

Two main workflows: one-model Python experiments and reproducible CLI runs.

## Install

=== "PyPI"

    ```bash
    python -m venv .venv && source .venv/bin/activate
    python -m pip install seahorse-stpp
    ```

=== "With HPO (Ray Tune)"

    ```bash
    python -m pip install "seahorse-stpp[hpo]"
    ```

=== "From source (development)"

    ```bash
    git clone https://github.com/YahyaAalaila/seahorse.git
    cd seahorse
    python -m pip install -e ".[dev]"
    ```

## Prepare Data

Each JSONL file has one sequence per line:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

Splits go in one directory:

```text
data/my_dataset/  train.jsonl  val.jsonl  test.jsonl
```

See [Data Format](data-format.md) for full details and Hugging Face sources.

## Run One Model With Python

```python
from seahorse import AutoSTPP, load_jsonl

train = load_jsonl("data/my_dataset/train.jsonl")
val   = load_jsonl("data/my_dataset/val.jsonl")
test  = load_jsonl("data/my_dataset/test.jsonl")

model = AutoSTPP(device="cpu", seed=42)
model.fit(train, val, test, epochs=10, batch_size=64)
scores  = model.evaluate(test)           # likelihood metrics
samples = model.predict_next(test, n_samples=32)
```

!!! note
    `evaluate()` returns likelihood metrics. Use `predict_next()` for predictive
    samples. There is no generic `predict()` method.

Continue with [Python API](python-api.md) or [Train One Model](examples/train-one-model.md).

## Use The CLI For Reproducible Runs

Use the CLI when you need saved run artifacts, HPO, benchmark campaigns, or
paper-style reproducibility.

Verify the installed commands:

```bash
python -m seahorse --help
python -m seahorse fit --help
python -m seahorse tune --help
python -m seahorse bench --help
python -m seahorse evaluate --help
```

The top-level CLI exposes four modes: `fit`, `tune`, `bench`, and `evaluate`.

## Train A Local CLI Run

```bash
python -m seahorse fit \
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
python -m seahorse evaluate metrics \
  --run runs/quickstart/fit/poisson_gmm/<run_id> \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core
```

## Use A Hugging Face Dataset

Pass a dataset repository, optionally with a subdirectory:

```bash
python -m seahorse fit \
  --preset poisson_gmm \
  --dataset owner/repo[/subdir] \
  --dataset-revision main \
  --out runs/hf_fit
```

The resolved dataset path must contain `train.jsonl` and `val.jsonl`.
`test.jsonl` is used when available for `fit` and is normally required for
post-fit evaluation.
