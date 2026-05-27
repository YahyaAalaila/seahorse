# Execution Contract

The Seahorse execution contract defines what the framework guarantees before any model-specific code runs. It is the mechanism that makes benchmark results trustworthy across model families.

## What the Contract Enforces

For every `bench` run, Seahorse forces:

1. **Same dataset** — all presets receive identical train/val/test splits.
2. **Same normalization** — `--normalize` / `--no-normalize` is applied uniformly; presets cannot override it internally.
3. **Same metric definition** — per-event NLL is computed identically for all models via the shared training harness.
4. **Same config resolution order** — YAML defaults → preset defaults → `--override`; no preset can inject values after the override step.

This runs in `Benchmark._apply_data_contract()`, called in both `tune_all()` and `run()`. No preset receives the evaluation data before this step.

## Shared Layer Components

| Component | Module | What it owns |
| --- | --- | --- |
| Dataset loading | `training/data_module.py` | Load JSONL; apply `protocol` and `normalize` settings |
| Benchmark policy | `benchmark/benchmark.py` | Force `protocol="unified"` and single `normalize` across all presets |
| Config resolution | `config/schema.py` | Merge YAML → preset → override into validated `STPPConfig` |
| Training harness | `training/lightning_module.py` | Per-event NLL aggregation, checkpointing, early stopping |

## Family-Owned Layer

The family-owned layer runs **after** the contract is applied. It is the only place where model-specific behavior is permitted:

- `PresetDescriptor.data_init_overrides(dm)` — compute training-data-dependent quantities (bounding box, float64 ODE fallback). This is called after the data module is initialized but before `build_model()`.
- `ModelFamilyConfig.from_dict()` and `build_model()` — construct the model from the fully resolved config.

Critically, `data_init_overrides` has access to **training partition statistics only** — never validation or test data.

## NLL Definition

`val_nll` and `test_nll` in `RunResult` are **per-event NLL in the training coordinate space**:

```
NLL = -1/N Σ log p(t, s | history)
```

where N is the total number of events in the split.

This is computed identically for all presets. Values are comparable across presets only when:
- All presets were run under the same normalization setting.
- All presets compute exact (not approximate) log-likelihood.

To convert to original-coordinate NLL when normalization was applied:

```
NLL_original = NLL_normalised − log(time_std × loc_std_x × loc_std_y)
```

`norm_stats` in `run_result.json` carries the normalization parameters needed for this conversion.

## Why This Matters

Without an explicit contract, each paper reproduces its own metric using its own data split and normalization. The numbers cannot be compared without manually re-running everything under a common setup. The execution contract eliminates this problem at the framework level rather than relying on convention.

See [Architecture](../architecture.md) for the full model layer documentation.
