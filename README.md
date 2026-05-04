# Seahorse / unified-stpp

Seahorse (`unified-stpp`, imported as `unified_stpp`) is a research framework
for training, tuning, benchmarking, and evaluating spatio-temporal point process
models through one package interface.

The public v1 surface is the Python package and module CLI:

```bash
python -m unified_stpp fit
python -m unified_stpp tune
python -m unified_stpp bench
python -m unified_stpp evaluate
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Development checks:

```bash
python -m pip install -e ".[dev]"
```

All optional extras, including HPO support:

```bash
python -m pip install -e ".[all]"
```

`pyproject.toml` is the dependency source of truth.

## CLI Modes

- `fit`: train one preset or YAML config on local JSONL splits or a Hugging Face dataset.
- `tune`: run HPO and write a best-config YAML.
- `bench`: run one or more presets across datasets and seeds.
- `evaluate`: compute post-fit metrics, predictive comparisons, surfaces, or merge evaluation artifacts.

Use `--help` for the exact arguments:

```bash
python -m unified_stpp fit --help
python -m unified_stpp tune --help
python -m unified_stpp bench --help
python -m unified_stpp evaluate --help
```

## Data Policy

Benchmark datasets are resolved from Hugging Face dataset repositories. Users
with custom or private data should pass their own local JSONL split paths or a
local split directory. The repository does not bundle public datasets.

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

Real paper datasets are hosted on Hugging Face. Processed HawkesNest suite 3
and suite 4 synthetic datasets will also be uploaded to Hugging Face. Synthetic
generation will be documented separately.
