# FAQ / Common Errors

## Installation and Setup

**`No module named seahorse`**
: Install Seahorse — `pip install seahorse-stpp` (or `pip install -e .` from a source checkout).

**`No module named ray` or HPO errors**
: Install the HPO extras: `pip install "seahorse-stpp[hpo]"`

**`No module named plotly`**
: `plot_kde_surface` requires Plotly: `python -m pip install plotly`

---

## Data Problems

**`Missing train.jsonl or val.jsonl`**
: Check the directory you passed to `--dataset` or `--splits_dir`. Both files are required for fitting.

**`FileNotFoundError` for `test.jsonl`**
: Ensure `test.jsonl` is in the dataset directory. It is required for post-fit evaluation with `--split test`.

**`Unknown split source` or path errors in `bench`**
: The `bench` command does not accept `--train`, `--val`, `--test`. Use `--dataset` for a single dataset directory or `--splits_dir` for a collection.

**`JSONDecodeError` when loading JSONL**
: Each line must be a standalone JSON object, not part of a JSON array. Check for trailing commas or wrapped arrays.

---

## Fitting

**`Unknown model preset: my_preset`**
: The preset is not registered. Check spelling with `list_available_models()`. If you wrote a custom preset, make sure the config module is imported in `seahorse/models/configs/__init__.py`.

**Out of memory during fit**
: Reduce `--override training.batch_size=16` (or smaller). Use `--n_workers 1` and fewer concurrent runs. For `auto_stpp` and `deep_stpp`, also reduce hidden dim with `--override model.build_overrides.hidden_dim=64`.

**`RuntimeError: autograd` or `enable_grad` errors**
: Some models (AutoSTPP, neural ODE families) use `torch.autograd.grad` internally. Make sure the runner is using `inference_mode=False`. This is set automatically in `STPPRunner` but can be disrupted by manual Trainer construction.

**Validation NLL does not decrease**
: Try a smaller learning rate (`--override training.lr=1e-4`) or more epochs. For AutoSTPP, confirm the bounding box was computed from training data — the `PresetDescriptor` handles this automatically.

---

## Evaluation

**`Path ... is not a saved run directory`**
: Pass a per-model run directory, not the top-level benchmark directory. Look up the correct path in `cell_index.json`.

**Metric `available: false` in `metrics.json`**
: The model does not support the requested metric. Check the [Model Capability Matrix](../model-capability-matrix.md). This is not a failure — it is intentional.

**`Requested metrics require unplanned heavy artifacts`**
: You requested a metric that needs sampling or grid work but used `--metric-profile core`. Re-run with `--metric-profile predictive`, `generative`, or `surface`.

**`predictive-compare requires --horizon`**
: Pass a positive duration such as `--horizon 1.0`.

**NLL values differ between Python API and CLI**
: The Python API `evaluate()` uses in-memory state from the current `fit()` call. The CLI `evaluate metrics` loads from the saved checkpoint. Minor differences can arise from batch ordering. Large differences indicate a save/load or normalization mismatch.

---

## Benchmarking

**HPO dependency errors with `--tune`**
: Remove `--tune` unless you intend to run HPO, or install HPO extras: `pip install "seahorse-stpp[hpo]"`

**Benchmark cells have inconsistent NLL**
: Check that all presets were run under the same `--normalize` / `--no-normalize` setting. The benchmark contract enforces this when using `bench` directly, but manual `fit` runs do not automatically apply the contract.

**`Unknown search-alg: bayesian`**
: Make sure Ray Tune and its optional search backends are installed. Use `--search-alg random` as a fallback.

---

## Results and Comparison

**NLL values look very different from a published paper**
: Check normalization. Published results may use z-scored time and space (normalized NLL) or raw-coordinate NLL. See [Evaluation Semantics](evaluation-semantics.md) for the conversion formula.

**SMASH or Diffusion STPP NLL looks unexpectedly low**
: These families optimize surrogate objectives (score-matching, ELBO), not exact log-likelihood. Their `test_nll` values are not directly comparable to exact-NLL families. See [Model Capability Matrix](../model-capability-matrix.md).

**`predict_next` raises `NotImplementedError`**
: The fitted model does not support next-event sampling. Check the Sampling column in the [Model Capability Matrix](../model-capability-matrix.md).
