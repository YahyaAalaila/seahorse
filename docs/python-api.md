# Python API

The Python-first API is for running one model programmatically. It wraps the
same presets and runner used by the CLI while presenting model classes with a
small sklearn-style surface.

Use the CLI for benchmark campaign orchestration, paper reproduction, and
artifact-backed metric profiles.

## Imports

```python
from seahorse import AutoSTPP, DeepSTPP, PoissonGMM, STPPEstimator, load_dataset, load_jsonl
```

Use concrete classes when they exist:

```python
model = AutoSTPP(device="cpu")
baseline = PoissonGMM(device="cpu")
```

Use `STPPEstimator` when you want to choose by model name or registered preset:

```python
model = STPPEstimator("AutoSTPP", device="cpu")
same_model = STPPEstimator("auto_stpp", device="cpu")
```

## Supported Model Aliases

The Python API exposes friendly classes for registered presets, including:

- `PoissonGMM`, `PoissonCNF`, `PoissonTVCNF`
- `HawkesGMM`, `HawkesCNF`, `HawkesTVCNF`
- `SelfCorrectingGMM`, `SelfCorrectingCNF`, `SelfCorrectingTVCNF`
- `RMTPPGMM`, `THPGMM`
- `NeuralSTPP`, `NeuralJumpSC`, `NeuralAttnSC`
- `NeuralJumpCNF`, `NeuralAttnCNF`
- `DeepSTPP`, `AutoSTPP`, `SMASH`, `DiffusionSTPP`, `NSMPP`

The canonical NJSDE preset is available through `STPPEstimator("njsde")`.

Programmatic discovery is available:

```python
from seahorse import list_available_models

print(list_available_models())
```

## Data

Load a Seahorse-ready Hugging Face dataset with `load_dataset`:

```python
splits = load_dataset("yahya021/citibike-stpp")
train = splits["train"]
val = splits["val"]
test = splits["test"]
```

Load your own canonical JSONL split files with `load_jsonl`:

```python
train = load_jsonl("data/my_dataset/train.jsonl")
val = load_jsonl("data/my_dataset/val.jsonl")
test = load_jsonl("data/my_dataset/test.jsonl")
```

Each split is a list of sequence dictionaries with `times` and `locations`.
See [Data Format](data-format.md) for the full data contract.

## Fit

`fit` trains from in-memory train, validation, and optional test sequences. A
validation split is required.

```python
model = AutoSTPP(device="cpu", seed=42)
model.fit(
    train,
    val,
    test,
    epochs=10,
    lr=1e-3,
    batch_size=64,
    dataset_id="my_dataset",
)
```

`fit` returns the estimator itself. The fitted runner is available as
`model.runner`, and the underlying model is available as `model.model`.

## Evaluate

`evaluate` currently supports implemented likelihood metrics:

```python
scores = model.evaluate(test)
```

The default `core` profile returns:

- `test_nll`
- `mean_seq_nll`

You can request supported metrics explicitly:

```python
scores = model.evaluate(test, metrics=["test_nll"])
```

Unsupported estimator metrics raise `NotImplementedError`. Use
`python -m seahorse evaluate metrics ...` for the full artifact-backed
evaluation profiles.

## Predict Next Events

The implemented predictive method is `predict_next`, not `predict`:

```python
samples = model.predict_next(test, n_samples=32)
```

The returned dictionary includes arrays such as:

- `next_times`
- `next_locations`
- `true_next_times`
- `true_next_locations`
- `sequence_index`
- `target_event_index`
- `sampling_succeeded`
- `sampling_backend`

`predict_next` raises `NotImplementedError` when the fitted model does not
support the required native or exact-intensity sampling path.

## Tune

The Python API exposes a thin HPO wrapper:

```python
best_config = model.tune(train, val, n_trials=10, max_epochs=20)
```

This uses the existing Ray Tune path. Install HPO dependencies before using it:

```bash
python -m pip install -e ".[hpo]"
```

## Save And Load

Save a fitted estimator through the underlying runner:

```python
save_dir = model.save("runs/api/auto_stpp")
```

Load through the base estimator or a matching concrete class:

```python
loaded = AutoSTPP.load(save_dir)
```

## Visualization Helpers

Fitted estimators expose plotting helpers:

```python
surface = model.plot_intensity(test[0], output_path="runs/plots/intensity")
kde = model.plot_kde_surface(test[0], n_samples=128, output_path="runs/plots/kde")
```

`plot_intensity` requires a fitted or loaded runner with a run directory.
`plot_kde_surface` requires `plotly`.

For benchmark-aligned visual artifacts, use the CLI workflows in
[Evaluation And Visualization](evaluation.md).
