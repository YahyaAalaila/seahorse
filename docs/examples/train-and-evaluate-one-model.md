# Train And Evaluate One Model

For the complete executable walkthrough, use
<a href="https://colab.research.google.com/github/YahyaAalaila/uni-stpp/blob/release/v1-integration/docs/notebooks/01_run_one_model_python_api.ipynb">01 Run One Model With The Python API</a>
in Google Colab.

## Goal

Train one model with the Python API, evaluate it on held-out JSONL data, and
sample possible next events. This is the normal-user path for scripts and
notebooks. Benchmarking workflows use the CLI.

## Required JSONL Files

```text
data/my_dataset/
  train.jsonl
  val.jsonl
  test.jsonl
```

Each line is one sequence:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

## Minimal Python API Flow

```python
from unified_stpp import AutoSTPP, PoissonGMM, load_jsonl

train = load_jsonl("data/my_dataset/train.jsonl")
val = load_jsonl("data/my_dataset/val.jsonl")
test = load_jsonl("data/my_dataset/test.jsonl")

model = AutoSTPP(device="cpu", seed=42)
baseline = PoissonGMM(device="cpu", seed=42)

model.fit(train, val, test, epochs=10, batch_size=64, dataset_id="my_dataset")
baseline.fit(train, val, test, epochs=5, batch_size=64, dataset_id="my_dataset")

print(model.evaluate(test))
print(baseline.evaluate(test))

samples = model.predict_next(test, n_samples=32)
print(samples["next_times"].shape)
print(samples["next_locations"].shape)
```

## Optional Visualization

For a lightweight sanity check, plot arrays returned by `predict_next`. The
notebook shows a simple sampled-location scatter plot. For benchmark-aligned
visual artifacts, use the CLI evaluation workflow.

## Expected Outputs

- score dictionaries from `evaluate(test)`.
- predictive arrays from `predict_next(test, n_samples=32)`.
- optional plots built from the sampled arrays.

## Common Errors

- `ValueError: val_seqs is required`: pass an explicit validation split.
- `FileNotFoundError`: check the paths passed to `load_jsonl`.
- `RuntimeError: Model is not fitted`: call `fit()` before evaluation or sampling.
- `NotImplementedError` from `predict_next`: choose a model with sampling support.
