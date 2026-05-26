# Train And Evaluate One Model

## Goal

This example trains one Seahorse model with the Python API, evaluates it on a
held-out test split, and samples possible next events. It is intended for normal
local scripts and notebooks.

Benchmarking workflows use the CLI; this page stays with the Python API.

## Required JSONL Files

Prepare three JSONL split files:

```text
data/my_dataset/
  train.jsonl
  val.jsonl
  test.jsonl
```

Each line is one sequence. A minimal line looks like:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

`times` and `locations` must have the same length. Validation data is required
by `fit()`.

## Load Data

```python
from unified_stpp import load_jsonl

train = load_jsonl("data/my_dataset/train.jsonl")
val = load_jsonl("data/my_dataset/val.jsonl")
test = load_jsonl("data/my_dataset/test.jsonl")

print(f"train sequences: {len(train)}")
print(f"val sequences: {len(val)}")
print(f"test sequences: {len(test)}")
print(test[0].keys())
```

## Instantiate Model

Start with `AutoSTPP` for the main model and `PoissonGMM` as a simple baseline.

```python
from unified_stpp import AutoSTPP, PoissonGMM

model = AutoSTPP(device="cpu", seed=42)
baseline = PoissonGMM(device="cpu", seed=42)
```

Use `device="auto"` when you want Seahorse to choose the available accelerator.
Use `device="cpu"` for the most portable first run.

## Fit

```python
model.fit(
    train,
    val,
    test,
    epochs=10,
    batch_size=64,
    dataset_id="my_dataset",
)

baseline.fit(
    train,
    val,
    test,
    epochs=5,
    batch_size=64,
    dataset_id="my_dataset",
)
```

For a smoke test, reduce `epochs` and `batch_size`.

## Evaluate

```python
model_scores = model.evaluate(test)
baseline_scores = baseline.evaluate(test)

print("AutoSTPP:", model_scores)
print("PoissonGMM:", baseline_scores)
```

The Python API currently reports likelihood-oriented metrics such as
`test_nll` and `mean_seq_nll`. Use the CLI evaluation workflow when you need
benchmark metric profiles or artifact-backed reports.

## Predict/Sample Next Events

```python
samples = model.predict_next(test, n_samples=32)

print(samples.keys())
print(samples["next_times"].shape)
print(samples["next_locations"].shape)
print(samples["sampling_backend"])
```

The returned arrays are organized by held-out next-event contexts. That means
the first dimension is not necessarily the same as `len(test)`.

Useful fields include:

- `next_times`
- `next_locations`
- `true_next_times`
- `true_next_locations`
- `sequence_index`
- `target_event_index`
- `history_length`
- `sampling_succeeded`
- `sampling_backend`

## Optional Visualization

`plot_kde_surface()` renders a simple HTML summary of sampled next-event times
and locations. It requires `plotly`.

```python
kde = model.plot_kde_surface(
    test[0],
    n_samples=128,
    output_path="runs/examples/auto_stpp_kde",
)

print(kde["html"])
print(kde["sampling_backend"])
```

For intensity surfaces, `plot_intensity()` is also available on fitted
estimators, but support depends on the model family and surface profile.

## Expected Outputs

After running the example, you should have:

- printed split sizes from `load_jsonl`.
- two fitted estimator objects: `model` and `baseline`.
- score dictionaries from `evaluate(test)`.
- a predictive sample dictionary from `predict_next(test, n_samples=32)`.
- optionally, an HTML file such as `runs/examples/auto_stpp_kde/kde_surface.html`.

Example score dictionaries look like:

```python
{"test_nll": 1.23, "mean_seq_nll": 12.34}
```

Exact values depend on the dataset, seed, device, and training settings.

## Common Errors

`ValueError: val_seqs is required`

: Pass an explicit validation split to `fit(train, val, test, ...)`.

`FileNotFoundError`

: Check the paths passed to `load_jsonl`. The example assumes files under
  `data/my_dataset/`.

`NotImplementedError` from `predict_next`

: The fitted model does not support the required sampling path. Try `AutoSTPP`
  or `PoissonGMM`, or use a model listed with sampling support in the model
  capability matrix.

`RuntimeError: Model is not fitted`

: Call `fit()` before `evaluate()`, `predict_next()`, or visualization helpers.

`RuntimeError: plot_kde_surface requires plotly`

: Install `plotly` or use the arrays returned by `predict_next()` directly.

`ValueError: No held-out next-event context was available for plotting`

: Use a test sequence with enough events to form a history and a next-event
  target.
