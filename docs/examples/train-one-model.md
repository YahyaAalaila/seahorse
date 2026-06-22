# Train One Model

Use the Python API when you want to train, evaluate, and sample from one model
inside a script or notebook.

## Notebook

Use <a href="https://colab.research.google.com/github/YahyaAalaila/seahorse/blob/main/docs/notebooks/01_run_one_model_python_api.ipynb">01 Run One Model With The Python API</a>
for an executable walkthrough in Google Colab.

## Python API Example

```python
from seahorse import AutoSTPP, load_jsonl

train = load_jsonl("data/my_dataset/train.jsonl")
val   = load_jsonl("data/my_dataset/val.jsonl")
test  = load_jsonl("data/my_dataset/test.jsonl")

model = AutoSTPP(device="cpu", seed=42)
model.fit(train, val, test, epochs=10, batch_size=64, dataset_id="my_dataset")

scores  = model.evaluate(test)
samples = model.predict_next(test, n_samples=32)

print(scores)
print(samples["next_times"].shape)
```

`fit()` requires a validation split. `evaluate()` supports the Python-facing
likelihood metrics `test_nll` and `mean_seq_nll`. For benchmark metric profiles
and artifact-backed reports, use the CLI.

??? example "Show Python example — try a baseline"
    ```python
    from seahorse import PoissonGMM

    baseline = PoissonGMM(device="cpu", seed=42)
    baseline.fit(train, val, test, epochs=5, batch_size=64)
    print(baseline.evaluate(test))
    ```

??? example "Show Python example — save and reload"
    ```python
    save_dir = model.save("runs/api/auto_stpp")
    loaded   = AutoSTPP.load(save_dir)
    print(loaded.evaluate(test))
    ```

Use [Python API](../python-api.md) for the full Python-facing surface.
