# Seahorse / unified-stpp

Seahorse, packaged as `unified-stpp` and imported as `unified_stpp`, is a
research framework for spatio-temporal point process models.

Use the Python API when you want to train and evaluate one model
programmatically. Use the CLI when you want reproducible runs, HPO, benchmark
campaigns, and paper-style artifacts.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Python Quickstart

```python
from unified_stpp import AutoSTPP, PoissonGMM, load_jsonl

train = load_jsonl("path/to/train.jsonl")
val = load_jsonl("path/to/val.jsonl")
test = load_jsonl("path/to/test.jsonl")

model = AutoSTPP(device="cpu")
baseline = PoissonGMM()

model.fit(train, val, test, epochs=10, batch_size=64)
scores = model.evaluate(test)
samples = model.predict_next(test, n_samples=32)
```

`evaluate()` currently reports implemented likelihood metrics such as
`test_nll` and `mean_seq_nll`. `predict_next()` samples held-out next-event
contexts when the fitted model supports the required sampling path.

## CLI Quickstart

The CLI/config interface is the reproducibility and benchmarking path:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl \
  --out runs/quickstart

python -m unified_stpp evaluate metrics \
  --run runs/quickstart/fit/poisson_gmm/<run_id> \
  --data path/to/test.jsonl \
  --split test \
  --metric-profile core
```

Benchmark campaigns use:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm njsde auto_stpp \
  --splits_dir splits_root \
  --seeds 1 2 3 \
  --out runs/bench
```

## Data

The canonical local layout is:

```text
dataset_root/
  train.jsonl
  val.jsonl
  test.jsonl
```

Each JSONL line is one event sequence:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

## Documentation

Start at [docs/index.md](docs/index.md). The docs site builds with:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs build --strict
```

## Citation

Citation details will be added before publication.

## License

Seahorse is distributed under the MIT License. See [LICENSE](LICENSE).
