# End-to-End Case Study

This case study shows the complete Seahorse workflow on a small JSONL dataset:
inspect the data contract, fit a model, run a benchmark, and inspect saved
artifacts. It is the recommended path for new users who want to understand how
the repository works before moving to larger datasets.

## 1. Start From The Data Contract

Seahorse expects three JSONL split files:

```text
data/my_dataset/
  train.jsonl
  val.jsonl
  test.jsonl
```

Each line is one event sequence:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

The Colab tutorials generate this layout automatically, so a new user can run
the workflow without downloading data.

## 2. Fit One Model

Use the Python API for a focused single-model experiment:

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

Open the executable walkthrough:
[Run One Model With The Python API](colabs.md).

## 3. Run A Benchmark Campaign

Use the CLI when comparing presets or preserving benchmark artifacts:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm auto_stpp deep_stpp \
  --dataset data/my_dataset \
  --seeds 1 \
  --out runs/examples/small_benchmark \
  --n_workers 1
```

The benchmark path records the preset, dataset source, seed, overrides, and run
directory for each benchmark cell. Open the executable walkthrough:
[Benchmark Models With The CLI](colabs.md).

## 4. Inspect The Artifacts

A benchmark directory contains campaign-level tables and per-run directories:

```text
runs/examples/small_benchmark/
  bench_meta.json
  cell_index.json
  results.json
  table_test_nll_all.csv
```

Use `cell_index.json` to find a saved run and evaluate additional metric
profiles:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/examples/small_benchmark/evaluate_core
```

## 5. Scale The Same Pattern

After the case study runs, scale in three directions:

- replace the demo data with your own JSONL splits;
- add presets and seeds to the benchmark campaign;
- pin dataset revisions and commit hashes for paper-grade reproducibility.

The same data contract, CLI commands, and artifact layout apply to larger
experiments.
