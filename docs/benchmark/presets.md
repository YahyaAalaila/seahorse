# Presets and Configs

A **preset** is a named model configuration that works immediately with `fit`, `bench`, `evaluate`, and the Python API. Seahorse ships presets for all registered model families.

## Available Presets

| Family | CLI preset | Python class | NLL type |
| --- | --- | --- | --- |
| Poisson + GMM | `poisson_gmm` | `PoissonGMM` | Exact (factorized) |
| Hawkes + GMM | `hawkes_gmm` | `HawkesGMM` | Exact (factorized) |
| Self-correcting + GMM | `selfcorrecting_gmm` | `SelfCorrectingGMM` | Exact (factorized) |
| Poisson + CNF | `poisson_cnf` | `PoissonCNF` | Exact (factorized) |
| Hawkes + CNF | `hawkes_cnf` | `HawkesCNF` | Exact (factorized) |
| Self-correcting + CNF | `selfcorrecting_cnf` | `SelfCorrectingCNF` | Exact (factorized) |
| Poisson + TVCNF | `poisson_tvcnf` | `PoissonTVCNF` | Exact (factorized) |
| Hawkes + TVCNF | `hawkes_tvcnf` | `HawkesTVCNF` | Exact (factorized) |
| Self-correcting + TVCNF | `selfcorrecting_tvcnf` | `SelfCorrectingTVCNF` | Exact (factorized) |
| RMTPP + GMM | `rmtpp_gmm` | `RMTPPGMM` | Exact (factorized) |
| THP + GMM | `thp_gmm` | `THPGMM` | Exact (factorized) |
| AutoSTPP | `auto_stpp` | `AutoSTPP` | Exact |
| DeepSTPP | `deep_stpp` | `DeepSTPP` | Exact |
| NSMPP DeepBasis | `nsmpp` | `NSMPP` | Exact |
| NJSDE | `njsde` | `NJSDE` | Exact |
| Neural JumpCNF | `neural_jumpcnf` | `NeuralJumpCNF` | Exact |
| Neural AttnCNF | `neural_attncnf` | `NeuralAttnCNF` | Exact |
| SMASH | `smash` | `SMASH` | Approximate |
| Diffusion STPP | `diffusion_stpp` | `DiffusionSTPP` | Approx (ELBO) |

Discover all registered presets at runtime:

```python
from seahorse import list_available_models
print(list_available_models())
```

## Config Sources

Every run needs a config source. In order of precedence when combined:

1. **Preset** (`--preset <name>`): loads defaults from the bundled YAML under `seahorse/configs/<preset>.yaml`.
2. **Config file** (`--config path/to/config.yaml`): loads a complete or partial YAML config.
3. **Overrides** (`--override key=value`): dotted-path overrides applied last.

## Override Keys

Overrides use dotted paths into `STPPConfig`. Common groups:

```bash
# Training
--override training.n_epochs=20 training.lr=5e-4 training.batch_size=32

# Data
--override data.normalize=false data.num_workers=2

# Model-specific build params
--override model.build_overrides.hidden_dim=128
```

Run `python -m seahorse fit --help` for the full config schema.

## Using a Preset

=== "Python API"

    ```python
    from seahorse import AutoSTPP

    model = AutoSTPP(device="cpu", seed=42)
    model.fit(train, val, test, epochs=10, batch_size=64)
    ```

=== "CLI fit"

    ```bash
    python -m seahorse fit \
      --preset auto_stpp \
      --train data/my_dataset/train.jsonl \
      --val data/my_dataset/val.jsonl \
      --test data/my_dataset/test.jsonl \
      --out runs/fit
    ```

=== "CLI bench"

    ```bash
    python -m seahorse bench \
      --presets poisson_gmm hawkes_gmm auto_stpp \
      --splits_dir splits \
      --seeds 1 2 3 \
      --out runs/bench
    ```

## Saving and Loading a Config

After fitting, the resolved config is written to `{run_dir}/config.yaml`. Load it for a reproducible re-run:

```bash
python -m seahorse fit \
  --config runs/fit/auto_stpp/<run_id>/config.yaml \
  --train data/my_dataset/train.jsonl \
  --val data/my_dataset/val.jsonl \
  --out runs/rerun
```

See [Add a Model](../adding-a-model.md) to register your own preset.
