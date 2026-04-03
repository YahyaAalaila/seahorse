# unified_stpp

A research framework for neural spatio-temporal point processes (STPPs).
Multiple model families — exact-likelihood, score-based, and diffusion-based — share
a common training pipeline, artifact layout, and CLI.

---

## Framework Design

Every STPP model is decomposed into two framework-facing pieces.

**`StateModel`** encodes the observed event history into a `StateContext`:

```python
ctx = state_model.encode_history(
    times=times,          # (B, N)
    locations=locations,  # (B, N, d)
    lengths=lengths,      # (B,)
)
```

**`EventModel`** consumes that state to compute the training objective:

```python
out = event_model.training_loss(
    times=times, locations=locations, lengths=lengths,
    state=ctx,
)  # must return {"loss": Tensor}; may also include "nll", sub-term matrices, etc.
```

**`UnifiedSTPP`** is the thin wrapper that routes a forward pass through both:

```python
out = model(times=times, locations=locations, lengths=lengths)
# {"nll": Tensor, ...}
```

The framework standardizes this outer interface. Internal components — sequence encoders,
Hawkes kernels, normalizing flows, diffusion networks — are unconstrained.

## What the Framework Standardizes

| Concern | How it is standardized |
|---------|------------------------|
| Training entry point | `python -m unified_stpp fit --preset <name> ...` |
| Optimizer, scheduler, gradient clipping | `TrainingConfig` → Lightning `Trainer` |
| Validation metric and early stopping | Lightning `EarlyStopping` on `val/nll` |
| Artifact layout | `artifacts/<preset>_<run-id>/` — same structure for every model |
| State wrapper contract | `StateModel.encode_history(...) → StateContext` |
| Event wrapper contract | `EventModel.training_loss(state, ...) → {"loss": Tensor}` |

A Hawkes-process temporal model and a self-attentive CNF spatial model train through
the same loop because they satisfy the same outer contracts.

---

## Available Presets

| Preset | Family | Spatial model | Objective |
|--------|--------|---------------|-----------|
| `poisson_gmm` | Factorized — Homogeneous Poisson | Gaussian mixture | Exact NLL |
| `hawkes_gmm` | Factorized — Hawkes | Gaussian mixture | Exact NLL |
| `selfcorrecting_gmm` | Factorized — Self-correcting | Gaussian mixture | Exact NLL |
| `poisson_cnf` | Factorized — Homogeneous Poisson | CNF, time-invariant | Exact NLL |
| `hawkes_cnf` | Factorized — Hawkes | CNF, time-invariant | Exact NLL |
| `selfcorrecting_cnf` | Factorized — Self-correcting | CNF, time-invariant | Exact NLL |
| `poisson_tvcnf` | Factorized — Homogeneous Poisson | CNF, time-varying | Exact NLL |
| `hawkes_tvcnf` | Factorized — Hawkes | CNF, time-varying | Exact NLL |
| `selfcorrecting_tvcnf` | Factorized — Self-correcting | CNF, time-varying | Exact NLL |
| `deep_stpp` | DeepSTPP | VAE latent + Gaussian | ELBO |
| `auto_stpp` | AutoSTPP | AutoInt integrator | Exact NLL |
| `neural_stpp_attn_sc` | Neural STPP | Self-attentive CNF | Exact NLL |
| `neural_stpp_jump_sc` | Neural STPP | Jump CNF | Exact NLL |
| `smash` | SMASH | Score-based | Score matching |
| `diffusion_stpp` | Diffusion STPP | Diffusion decoder | Diffusion ELBO |

Factorized models treat the temporal and spatial marginals as independent.
All other families learn joint spatio-temporal structure.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional: install pre-commit hooks for linting and formatting:

```bash
pre-commit install
```

---

## Data Format

Input files are newline-delimited JSON (JSONL), one sequence per line:

```json
{"times": [0.1, 0.4, 1.2, 2.7], "locations": [[0.3, 0.7], [0.1, 0.5], [0.8, 0.2], [0.4, 0.9]]}
{"times": [0.3, 1.1], "locations": [[0.6, 0.3], [0.2, 0.8]]}
```

- `times`: list of floats (absolute or relative; variable sequence length is supported)
- `locations`: list of `[x, y]` coordinate pairs (or higher-dimensional)
- `marks` (optional): list of integer class labels

Normalization (zero-mean, unit-variance per axis) is computed on the training split
and applied consistently to val and test. No pre-processing is required.

---

## Quickstart

```bash
python -m unified_stpp fit \
    --preset hawkes_cnf \
    --train  data/train.jsonl \
    --val    data/val.jsonl
```

Add `--test data/test.jsonl` to evaluate on a held-out set.
Training logs per-epoch validation NLL and prints the final result on exit.

### Artifacts

Every run writes to `artifacts/<preset>_<run-id>/`:

```
artifacts/hawkes_cnf_<id>/
├── config.yaml          # full resolved config (preset defaults + any overrides)
├── run_result.json      # val_nll, test_nll, n_params, norm_stats, runtime, ...
├── metrics.csv          # per-epoch train/val metrics (Lightning CSV logger)
└── checkpoints/
    ├── best.ckpt        # checkpoint at best validation NLL
    └── last.ckpt        # checkpoint after the final epoch
```

`run_result.json` includes `norm_stats` (training-split time and location statistics),
so results can be de-normalized without re-running.

### Reloading a run

```python
from unified_stpp import STPPRunner

runner = STPPRunner.load("artifacts/hawkes_cnf_<id>/")
```

`load` looks for `checkpoints/best.ckpt` first, then falls back to `model.ckpt`.

---

## Configuration

**Preset only** — no YAML needed; runs with preset defaults:

```bash
python -m unified_stpp fit --preset hawkes_cnf --train data/train.jsonl --val data/val.jsonl
```

**YAML config** — override specific fields; everything else inherits preset defaults:

```yaml
# my_run.yaml
model:
  preset: hawkes_cnf
  hidden_dim: 256

training:
  n_epochs: 100
  lr: 5.0e-4
  batch_size: 32
  patience: 20
```

```bash
python -m unified_stpp fit --config my_run.yaml --train data/train.jsonl --val data/val.jsonl
```

See [`unified_stpp/config/schema.py`](unified_stpp/config/schema.py) for the full schema
(`DataConfig`, `ModelConfig`, `TrainingConfig`, `LoggingConfig`).

---

## Repository Layout

```
unified_stpp/
├── models/
│   ├── abstractions.py          # StateModel, EventModel, StateContext (ABCs)
│   ├── unified_model.py         # UnifiedSTPP — wires state + event models
│   ├── configs/                 # BaseModelConfig, ConfigRegistry, per-family configs
│   ├── state_models/            # StateModel implementations
│   ├── event_models/            # EventModel implementations
│   ├── temporal_models/         # Internal: parametric temporal processes
│   ├── spatial_models/          # Internal: spatial density / flow models
│   └── ...                      # Other internal components
├── training/
│   ├── lightning_module.py      # STPPLightningModule
│   └── data_module.py           # STPPDataModule (loads, normalizes, batches)
├── config/
│   └── schema.py                # Pydantic STPPConfig
├── runner/
│   └── runner.py                # STPPRunner — fit / save / load
├── data/
│   └── dataset.py               # STPPDataset (normalization, padding, collation)
└── __main__.py                  # CLI entry point
```

---

## How to Add a New Model Family

Five steps to integrate a new STPP family into the framework.

**1. Implement internal components** (optional)

Add temporal models, spatial density models, encoders, or decoders anywhere under
`models/temporal_models/`, `models/spatial_models/`, or a new subdirectory. No
constraints — only the outer `StateModel` and `EventModel` wrappers need to satisfy
the framework interface.

**2. Implement a `StateModel` subclass**

```python
# models/state_models/my_state.py
from unified_stpp.models.abstractions import StateModel, StateContext

class MyStateModel(StateModel):
    def encode_history(self, *, times, locations, lengths, **kwargs) -> StateContext:
        # build state from observed history
        return StateContext(payload={"h": h})
```

**3. Implement an `EventModel` subclass**

```python
# models/event_models/my_event.py
from unified_stpp.models.abstractions import EventModel

class MyEventModel(EventModel):
    def training_loss(self, *, times, locations, lengths, state, **kwargs):
        h = state.payload["h"]
        # compute loss
        return {"loss": loss, "nll": nll}
```

**4. Write a config class and register it**

```python
# models/configs/my_model.py
import dataclasses
from .base import BaseModelConfig, ConfigRegistry

@ConfigRegistry.register("my_preset")
@dataclasses.dataclass
class MyModelConfig(BaseModelConfig):
    my_param: float = 1.0

    @classmethod
    def from_dict(cls, d, *, hidden_dim=128, spatial_dim=2, **kwargs):
        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            my_param=d.get("my_param", 1.0),
        )

    def build_model(self):
        from unified_stpp.models.unified_model import UnifiedSTPP
        return UnifiedSTPP(
            state_model=MyStateModel(...),
            event_model=MyEventModel(...),
            hidden_dim=self.hidden_dim,
        )
```

**5. Import in `models/configs/__init__.py`**

```python
from .my_model import MyModelConfig
```

Done. `python -m unified_stpp fit --preset my_preset ...` now trains the new model
through the same pipeline as every other preset.

---

Benchmark comparison (`bench`) and hyperparameter search (`tune`) are available as
separate CLI subcommands; native sampling and intensity-evaluation metrics are planned.
