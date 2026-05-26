# Seahorse

Seahorse, packaged as `unified-stpp`, is a framework for spatio-temporal point
process experiments. The docs are organized around the task you are trying to
complete.

## What Do You Want To Do?

### Run One Model

Use the Python API when you want a script or notebook that trains one model,
evaluates it, and samples next events.

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

Start with [Train One Model](examples/train-one-model.md) or the
[Python API](python-api.md).

### Benchmark Many Models

Use the CLI when you need reproducible grids across presets, datasets, and
seeds.

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm neural_attncnf \
  --splits_dir splits \
  --seeds 1 2 3 \
  --out runs/bench
```

Start with [Benchmark Models](examples/benchmark-models.md) or
[Benchmark Campaigns](benchmarks.md).

### Add A Model

Use the registry/config path when a model should become a preset that works
with `fit`, `bench`, and `evaluate`.

```python
from unified_stpp import STPPEstimator

model = STPPEstimator("my_preset", device="cpu")
```

Start with [Adding A Model](adding-a-model.md).

### Reproduce Paper Results

Use pinned datasets, explicit presets/configs, fixed seeds, and CLI evaluation
profiles. The reproduction page is the public checklist for paper-grade runs.

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --dataset owner/repo[/subdir] \
  --dataset-revision <revision> \
  --seeds 1 2 3 \
  --out runs/paper_bench
```

Start with [Paper Reproduction](paper-reproduction.md).

## Core Concepts

- Data is JSONL split files with one event sequence per line.
- The Python API is the shortest path for one-model experiments.
- The CLI is the supported path for reproducible runs, HPO, benchmarks,
  artifact-backed metrics, visual diagnostics, and paper reproduction.
- Model developers integrate through registries and preset configs so new
  models can use the same CLI and benchmark machinery.

## Local Documentation

Serve these docs locally from the repository root:

```bash
python -m mkdocs serve
```

If port `8000` is busy, choose another port:

```bash
python -m mkdocs serve -a 127.0.0.1:8002
```
