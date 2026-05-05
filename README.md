# Seahorse / unified-stpp

Seahorse, packaged as `unified-stpp` and imported as `unified_stpp`, is a
research framework for spatio-temporal point process models.

Seahorse is intended to support two user-facing interfaces:

- Use the Python API when you want to train and evaluate one model
  programmatically.
- Use the CLI when you want reproducible runs, HPO, benchmark campaigns, and
  paper-style artifacts.

The Python-first wrapper API is under active integration and is not exposed as a
stable public API on this branch yet. The stable interface available today is
the module CLI:

```bash
python -m unified_stpp fit --help
python -m unified_stpp tune --help
python -m unified_stpp bench --help
python -m unified_stpp evaluate --help
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For development checks:

```bash
python -m pip install -e ".[dev]"
```

For HPO workflows that use Ray Tune:

```bash
python -m pip install -e ".[hpo]"
```

For the documentation site:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs build --strict
```

## Python API

The planned normal-user workflow is:

```text
load data -> instantiate a model -> fit -> predict/evaluate
```

This branch does not yet contain the lightweight public wrapper that provides
that workflow, so the documentation does not publish stable Python API calls
yet. See [docs/python-api.md](docs/python-api.md) for the current API status.

## CLI Quickstart

Train one model on local JSONL splits:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl \
  --out runs/quickstart \
  --override training.n_epochs=1 training.batch_size=2 data.num_workers=0
```

Run a benchmark grid over presets, datasets, and seeds:

```bash
python -m unified_stpp bench \
  --preset poisson_gmm \
  --dataset path/to/dataset_root \
  --seeds 1 \
  --out runs/quickstart_bench \
  --n_workers 1 \
  --override training.n_epochs=1 training.batch_size=2 data.num_workers=0
```

Evaluate a saved run:

```bash
python -m unified_stpp evaluate metrics \
  --run runs/quickstart/fit/poisson_gmm/<run_id> \
  --data path/to/test.jsonl \
  --split test \
  --metric-profile core
```

Replace `<run_id>` with the timestamped run directory created by `fit`.

## Data

Seahorse resolves data from either Hugging Face dataset repositories or explicit
local JSONL paths. Use `--dataset owner/repo[/subdir]` for Hugging Face-backed
data, or pass `--train`, `--val`, and optionally `--test` for local files.

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

The documentation source starts at [docs/index.md](docs/index.md). This
repository includes `mkdocs.yml` for a dark MkDocs Material site.

## Citation

Citation details will be added before publication.

## License

Seahorse is distributed under the MIT License. See [LICENSE](LICENSE).
