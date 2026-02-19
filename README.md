# Unified Neural Spatiotemporal Point Process Framework

A modular, research-oriented framework that unifies neural STPP methods under a common
$\mathcal{M} = (\mathcal{E}, \mathcal{D}, \mathcal{U}, \mathcal{G})$ architecture with
systematic covariate augmentation.

## Dev Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

Run a tiny end-to-end training smoke:

```bash
bash scripts/tiny_train.sh
```

Or directly:

```bash
python train.py --preset deep_stpp --data hawkes --n_train 8 --n_val 2 --n_epochs 1 --batch_size 4 --no_save_metrics
```

## Framework Overview

Every neural STPP is decomposed into four components:

| Component | Role | Interface |
|-----------|------|-----------|
| **Encoder** $\mathcal{E}$ | History → latent state | `(events, lengths, x_event) → (z, states)` |
| **Dynamics** $\mathcal{D}$ | Inter-event state evolution | `(z_n, Δt, x_field) → z(t)` |
| **Updater** $\mathcal{U}$ | State update at events | `(z⁻, t, s, x_event, x_field) → z⁺` |
| **Decoder** $\mathcal{G}$ | State → intensity/density | `(z, t, s, x_field) → log f*(t,s)` |

Covariates inject at four points:
- **(I) Encoder**: event-level covariates enrich history representation
- **(II) Dynamics**: field covariates modulate state evolution (non-trivial only for ODE dynamics)
- **(III) Updater**: covariates at event location inform state transitions
- **(IV) Decoder**: field covariates directly modulate intensity/density

## Implemented Methods

| Method | Encoder | Dynamics | Updater | Decoder |
|--------|---------|----------|---------|---------|
| **NeuralSTPP** | GRU | Neural ODE | GRU Jump | Cumulative hazard + CNF |
| **DeepSTPP** | Attention | Identity | Attention | Log-normal mixture + Gaussian mixture |
| **DSTPP** | Transformer | Identity | Attention | Score-based diffusion (joint) |

## Project Structure

```
unified_stpp/
├── train.py                          # Entry point
├── configs/
│   ├── neural_stpp.yaml
│   ├── deep_stpp.yaml
│   ├── dstpp.yaml
│   └── deep_stpp_covariates.yaml     # Covariate-augmented example
├── unified_stpp/
│   ├── registry.py                   # Model factory + presets
│   ├── models/
│   │   ├── base.py                   # Abstract base classes (Encoder, Dynamics, Updater, Decoder)
│   │   ├── unified_model.py          # UnifiedSTPP: composes E, D, U, G
│   │   ├── encoders/
│   │   │   ├── gru.py                # GRU (NeuralSTPP)
│   │   │   ├── attention.py          # Self-attention (DeepSTPP)
│   │   │   └── transformer.py        # Transformer + continuous time encoding (DSTPP)
│   │   ├── dynamics/
│   │   │   ├── identity.py           # z(t) = z_n (most methods)
│   │   │   └── neural_ode.py         # dz/dt = f(z,t,X) (NeuralSTPP)
│   │   ├── updaters/
│   │   │   ├── gru_jump.py           # GRU cell update (NeuralSTPP)
│   │   │   └── attention_update.py   # Cross-attention update (DeepSTPP, DSTPP)
│   │   ├── decoders/
│   │   │   ├── factorized.py         # f*(t)·f*(s|t) wrapper
│   │   │   ├── temporal.py           # Cumulative hazard, log-normal mixture
│   │   │   ├── spatial.py            # CNF, Gaussian mixture
│   │   │   └── diffusion.py          # Score-based joint decoder (DSTPP)
│   │   └── covariates/
│   │       └── __init__.py           # LiftingMap, FieldCovariateEncoder
│   ├── data/
│   │   ├── dataset.py                # STPPDataset + collation
│   │   └── synthetic.py              # Hawkes + inhomogeneous Poisson generators
│   └── training/
│       └── trainer.py                # Training loop
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Train DeepSTPP on synthetic Hawkes data
python train.py --preset deep_stpp --n_epochs 50

# Train NeuralSTPP (requires torchdiffeq)
python train.py --preset neural_stpp --n_epochs 50

# Train DSTPP (diffusion decoder)
python train.py --preset dstpp --n_epochs 50

# Use a YAML config
python train.py --config configs/deep_stpp.yaml

# Covariate-augmented model on inhomogeneous data
python train.py --preset deep_stpp --field_cov_dim 1 --data inhomogeneous

# Moving hotspot benchmark (with explicit switch at t1)
python train.py --config unified_stpp/configs/moving_hotspot_with_covariates.yaml
python train.py --config unified_stpp/configs/moving_hotspot_without_covariates.yaml

# Regime-gated Hawkes benchmark (with/without covariates)
python train.py --config unified_stpp/configs/regime_gated_hawkes_with_covariates.yaml
python train.py --config unified_stpp/configs/regime_gated_hawkes_without_covariates.yaml
```

## Mix-and-Match

The modular design allows combining components across methods:

```python
from unified_stpp.registry import build_model

# DeepSTPP encoder + NeuralSTPP decoder (CNF spatial)
model = build_model(
    config={
        "encoder": {"type": "attention", "num_heads": 4, "num_layers": 2},
        "dynamics": {"type": "identity"},
        "updater": {"type": "attention", "num_heads": 4},
        "decoder": {
            "type": "factorized",
            "temporal": {"type": "cumulative_hazard"},
            "spatial": {"type": "cnf"},
        },
    },
    spatial_dim=2,
    hidden_dim=64,
)
```

**Compatibility note**: Not all combinations are valid. See Remark 2 in the paper.
The main constraint is dimensional: all components must agree on `hidden_dim`.

## Adding Covariates

```python
# With field covariates (e.g., temperature field)
model = build_model(
    config={},
    preset="deep_stpp",
    spatial_dim=2,
    hidden_dim=64,
    field_cov_dim=3,  # 3-dimensional covariate field
)

# Covariates automatically injected at points (I), (III), (IV).
# Point (II) is vacuous for identity dynamics.
```

## Sampling from Any Model

All decoders implement `sample()`, enabling autoregressive event generation from any model configuration:

```python
# After training:
sampled_times, sampled_locs, mask = model.sample(
    history_times, history_locations, history_lengths,
    n_samples=50, t_max=20.0,
)
```

Sampling strategies per decoder:

| Decoder | Temporal sampling | Spatial sampling |
|---------|------------------|-----------------|
| **Log-normal mixture** (DeepSTPP) | Ancestral: sample component k ~ π, then τ ~ LogNormal(μ_k, σ_k²) | Ancestral: sample component, then s ~ N(μ_k, σ_k² I) |
| **Cumulative hazard + CNF** (NeuralSTPP) | Inverse CDF via bisection: find τ s.t. Λ*(τ) = -log(U) | CNF forward pass: z₀ ~ N(0,I), solve ODE τ: 0→1 |
| **Diffusion** (DSTPP) | Joint annealed Langevin dynamics | (joint with temporal) |

For intensity-based decoders or validation, use the **thinning sampler**:

```python
from unified_stpp.models.sampling import thinning_sample, IntensityEvaluator

# Wrap model for intensity queries
evaluator = IntensityEvaluator(model, z=z_state, t_prev=t_last)

# Thinning-based sampling (works with any model)
times, locs, counts = thinning_sample(
    intensity_fn=evaluator.intensity,
    t_start=t_last,
    t_max=20.0,
    spatial_bounds=(s_min, s_max),
)

# Intensity heatmap for visualization
grid_x, grid_y, lam_grid = evaluator.intensity_grid(
    t=5.0, s_min=s_min, s_max=s_max, n_grid=50,
)
```

## Adding a New Component

1. Subclass the appropriate base in `models/base.py`
2. Implement the required interface
3. Register in `registry.py`

Example: adding a new encoder:

```python
# In models/encoders/my_encoder.py
from ..base import Encoder

class MyEncoder(Encoder):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__(input_dim, hidden_dim)
        # ... your architecture ...
    
    def forward(self, events, lengths, x_event=None):
        # ... your forward pass ...
        return z_final, all_states

# In registry.py, add:
ENCODER_REGISTRY["my_encoder"] = MyEncoder
```

## Citation

If you use this framework, please cite the methodology paper (in preparation).
