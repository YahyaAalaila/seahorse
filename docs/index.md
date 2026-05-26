# Seahorse

Seahorse, packaged as `unified-stpp`, is a framework for spatio-temporal point
process experiments. The docs are organized around the task you are trying to
complete.

## What Do You Want To Do?

<div class="grid cards" markdown>

- **Run One Model**

    Use the Python API for one-model scripts and notebooks.

    [Start Here](examples/train-one-model.md)

- **Benchmark Many Models**

    Use the CLI for reproducible preset, dataset, and seed grids.

    [Run A Benchmark](examples/run-a-small-benchmark.md)

- **Add A Model**

    Register a preset so it works with `fit`, `bench`, and `evaluate`.

    [Developer Guide](adding-a-model.md)

- **Reproduce Paper Results**

    Use pinned data, explicit presets, fixed seeds, and saved artifacts.

    [Reproduction Guide](paper-reproduction.md)

</div>

## Quick Taste

```python
from unified_stpp import AutoSTPP, load_jsonl

train = load_jsonl("data/my_dataset/train.jsonl")
val = load_jsonl("data/my_dataset/val.jsonl")
test = load_jsonl("data/my_dataset/test.jsonl")

model = AutoSTPP(device="cpu", seed=42)
model.fit(train, val, test, epochs=1, batch_size=64)
print(model.evaluate(test))
```

For the full walkthrough, open the [tutorial notebooks](examples/colabs.md).

## Core Concepts

- Data is JSONL split files with one event sequence per line.
- The Python API is the shortest path for one-model experiments.
- The CLI is the supported path for reproducible runs, HPO, benchmarks,
  artifact-backed metrics, visual diagnostics, and paper reproduction.
- Model developers integrate through registries and preset configs so new
  models can use the same CLI and benchmark machinery.

## Serve The Docs

Serve these docs locally from the repository root:

```bash
python -m mkdocs serve
```

If port `8000` is busy, choose another port:

```bash
python -m mkdocs serve -a 127.0.0.1:8002
```
