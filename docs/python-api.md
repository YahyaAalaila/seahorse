# Python API

## Status

The Python-first wrapper API is under active integration. It is not exposed as a
stable public API in this branch yet.

The intended normal-user workflow is:

```text
load data -> instantiate a model -> fit -> predict/evaluate
```

No stable class name, import path, or method contract for that wrapper is
documented here because it does not exist in the current branch.

## What To Use Today

Use the CLI for supported training, evaluation, HPO, and benchmark workflows:

```bash
python -m unified_stpp fit --help
python -m unified_stpp evaluate --help
```

For one-model experiments, start with:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl \
  --out runs/quickstart
```

Then evaluate the saved run:

```bash
python -m unified_stpp evaluate metrics \
  --run runs/quickstart/fit/poisson_gmm/<run_id> \
  --data path/to/test.jsonl \
  --split test \
  --metric-profile core
```

## Current Lower-Level Exports

The package currently exports lower-level building blocks such as `STPPConfig`,
`STPPRunner`, `RunResult`, `Benchmark`, and `BenchmarkTable`. These are useful
for internal orchestration and advanced experiments, but they are not the
planned sklearn-style normal-user wrapper.

The public Python API page will be updated once the lightweight wrapper is
integrated on this branch.
