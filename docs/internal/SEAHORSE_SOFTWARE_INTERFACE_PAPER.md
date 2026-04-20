# 5  Software Interface

Seahorse is designed for two interaction modes: a **command-line interface
(CLI)** for experiment execution and benchmark sweeps, and a **Python API**
for programmatic use in notebooks and research scripts. Both modes share the
same configuration layer: experiments are fully specified by YAML files that
declare the model preset, training hyperparameters, and data protocol, and
all fields can be selectively overridden at invocation time without modifying
the base config.

## 5.1  YAML Configuration and CLI

A minimal experiment configuration identifies a preset and the fields the
user wishes to override. The remaining fields inherit from preset defaults,
which are themselves registered as YAML files in the package:

```yaml
# my_experiment.yaml
model:
  preset: auto_stpp
  hidden_dim: 256

training:
  lr: 5.0e-3
  n_epochs: 100
  batch_size: 64

data:
  protocol: raw
  normalize: true
```

Experiments are launched through a single CLI entry point with four
subcommands that correspond directly to the benchmark execution modes
described in Section 4.4:

```bash
# Train a single preset, writing a RunResult artifact
python -m unified_stpp fit  --config my_experiment.yaml \
                             --train data/train.jsonl --val data/val.jsonl

# Hyperparameter search (benchmark contract frozen)
python -m unified_stpp tune --preset auto_stpp \
                             --dataset hawkesnest/topology_T5 --n_trials 50

# Multi-preset, multi-seed benchmark sweep
python -m unified_stpp bench --presets auto_stpp deep_stpp smash \
                              --splits_dir data/ --seeds 42 43 44

# Post-fit evaluation on a stored artifact
python -m unified_stpp evaluate --run runs/auto_stpp_20240415/
```

The `--preset` flag resolves a named preset with its registered defaults;
`--config` accepts a full or partial YAML that overrides any subset of
fields. These two modes compose: `--preset P --config overrides.yaml` merges
the override file on top of the preset defaults, so users can maintain
experiment-specific overrides as small diffs rather than full config copies.

## 5.2  Python API

For users who prefer programmatic control, `STPPRunner` exposes the same
fit–evaluate–save–load cycle as a Python object:

```python
from unified_stpp import STPPRunner

runner  = STPPRunner.from_preset("auto_stpp", hidden_dim=256, lr=5e-3)
result  = runner.fit(train_seqs, val_seqs, test_seqs)
runner.save("runs/my_run/")

# Load and query a stored run
runner2 = STPPRunner.load("runs/my_run/")
grid    = runner2.intensity_grid(t_query, s_query)
```

`RunResult` is a plain dataclass and can be inspected, serialized, or passed
directly to `BenchmarkTable` for aggregation. The `norm_stats` attribute
carries the coordinate normalization statistics fitted on the training
partition, enabling downstream denormalization without re-loading the
training data.

## 5.3  Extensibility

Adding a new STPP family to Seahorse requires five concrete artifacts:

1. **A config dataclass** (`models/configs/my_family.py`) decorated with
   `@ConfigRegistry.register("preset_name")`, declaring `_STATE_MODEL` and
   `_EVENT_MODEL` string keys, and implementing `from_dict()`,
   `_state_kwargs()`, and `_event_kwargs()`. `build_model()` is inherited
   from `BaseModelConfig` and assembles `UnifiedSTPP` for free.
   Optionally override `data_init_overrides()` for data-dependent init
   (e.g., spatial support bounds computed from training statistics).

2. **A state model** (`models/state_models/my_family_state.py`) decorated
   with `@register_state("my_family")` and implementing one abstract method,
   `encode_history()`, which packages observed history into a `StateContext`.
   All other `StateModel` methods have no-op or passthrough defaults.

3. **An event model** (`models/event_models/my_family_event.py`) decorated
   with `@register_event("my_family")` and implementing one abstract method,
   `training_loss()`. For exact-NLL families, `eval_nll()` delegates to
   `training_loss()` for free. Capability flags (`nll_kind`, `has_intensity`,
   `has_density`, etc.) are declared as a plain `EventCapabilities` dataclass
   and drive all downstream evaluation dispatch automatically.

4. **Three import lines** — one each in `models/configs/__init__.py`,
   `models/state_models/__init__.py`, and `models/event_models/__init__.py`.

5. **A YAML defaults file** (`configs/my_preset.yaml`) declaring the preset
   name and hyperparameter defaults.

Once registered, the new preset requires no further changes to the framework:
`Benchmark._apply_data_contract()` enforces the raw-first policy
automatically, `bench` includes it in sweeps, `tune` searches it under the
frozen benchmark contract, and `evaluate` dispatches capability-aware metrics
based on the declared `EventCapabilities`. This extension path is exercised
in the current codebase by the `nsmpp` family (direct joint-intensity
DeepBasis model), which adds no framework-level code beyond the five artifacts
above.

The goal of Seahorse is to provide a controlled, extensible framework for
benchmarking heterogeneous STPP families under a shared, reproducible
contract. For researchers, integrating a new model family requires three
concrete artifacts: a `ModelFamilyConfig` dataclass decorated with
`@ConfigRegistry.register`, which implements `from_dict()` to parse the
YAML config dict and `_state_kwargs()` / `_event_kwargs()` to forward
parameters to the state and event model constructors; an optional
`data_init_overrides()` classmethod for families that require
data-dependent initialization, such as spatial bounding boxes or coordinate
statistics; and a YAML defaults file declaring the preset name and
hyperparameter defaults. The base `build_model()` implementation is inherited
for free, and the benchmark contract — raw-first policy, fixed splits,
normalized reporting space — requires no changes from the researcher. A new
family is automatically covered by `bench`, `tune`, and `evaluate` upon
registration; there is no framework-level bookkeeping. For practitioners, the
configuration-driven interface and pre-implemented presets allow STPP models
to be applied to new datasets with minimal code, while the semantically
annotated run artifacts make results interpretable and auditable. Full API
reference, dataset format specifications, and worked extension examples are
available in the package documentation.

---

<!-- DRAFT NOTE — internal, do not include in submission -->

**Extensibility audit (2026-04-16):**

**What is true today:**
- `BaseModelConfig.build_model()` is genuinely free (`base.py:297–309`).
- `EventModel.eval_nll()` delegates to `training_loss()` for free for
  exact-NLL families (`abstractions.py:372–405`).
- `StateModel` has one abstract method (`encode_history`); everything else
  is opt-in with defaults.
- `EventCapabilities` is a plain frozen dataclass — declare it, evaluation
  dispatch happens automatically.
- The benchmark contract (`_apply_data_contract()`) requires zero changes
  from a new family author.
- The `nsmpp` family is a live proof of this path: 3 Python files, 1 YAML.

**One incomplete integration found (pre-submission fix required):**
`unified_stpp/models/configs/__init__.py` does NOT import
`NSMPPDeepBasisConfig`, meaning `@ConfigRegistry.register("nsmpp")` never
fires at import time — the preset is silently invisible to the CLI and
benchmark. The state and event `__init__.py` files DO import their
counterparts. Fix: add `from .nsmpp_deepbasis import NSMPPDeepBasisConfig`
to `models/configs/__init__.py` and `__all__`. ~5 minutes of work.

**Verdict for the paper:**
No design work needed. The extensibility claim is structurally sound and the
implementation is complete modulo the one missing import. Fix that, and the
section above is fully backed by the current codebase. The five-artifact
breakdown IS the argument — no documentation or tutorial needed to make this
claim in the paper, though an appendix code listing would strengthen it.

<!-- END DRAFT NOTE -->

