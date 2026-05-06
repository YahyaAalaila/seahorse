# Adding A Model

Seahorse models are connected through registries and preset configs. Follow the
nearest existing model family first; it is the best source of truth for required
interfaces and artifact behavior.

New model families should continue to integrate with the registry/config path
because the CLI uses it for reproducibility. The Python-first wrapper sits on
top of the same model and data contracts.

Use the Python API for quick local testing of an already registered model. Use
the config/registry path when the model should enter benchmark comparisons.

```python
from unified_stpp import STPPEstimator

model = STPPEstimator("my_preset", device="cpu")
```

## Extension Checklist

1. Add or reuse model components under `unified_stpp/models/`.
2. Register component classes with the model registries when they are selected
   by key.
3. Add a construction config under `unified_stpp/models/configs/`.
4. Register the preset with `ConfigRegistry`.
5. Import the config module from `unified_stpp/models/configs/__init__.py`.
6. Add a bundled YAML config under `unified_stpp/configs/` when the preset needs
   public defaults.
7. Add focused tests for config loading, model construction, and a tiny fit path.

## Component Registries

Component registry decorators live in `unified_stpp/models/model_registry.py`:

```python
from unified_stpp.models.model_registry import register_event, register_spatial, register_state
```

Use the registry only for components that need keyed construction. Many preset
configs reuse existing registered components.

## Preset Config Registry

Preset configs live under `unified_stpp/models/configs/` and register with
`ConfigRegistry`:

```python
from unified_stpp.models.configs.base import BaseModelConfig, ConfigRegistry


@ConfigRegistry.register("my_preset")
class MyPresetConfig(BaseModelConfig):
    ...
```

A preset config owns construction-time parameters and builds a `UnifiedSTPP`
model. Existing configs show the supported patterns:

- `factorized.py` for compact temporal plus spatial families.
- `auto_stpp.py` and `deep_stpp.py` for paper-style model families.
- `neural_stpp.py` for neural exact-family presets.
- `nsmpp_deepbasis.py` for the public `nsmpp` preset.

## Bundled YAML

If users should run the preset directly, add:

```text
unified_stpp/configs/my_preset.yaml
```

The CLI can then load it with:

```bash
python -m unified_stpp fit \
  --preset my_preset \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl \
  --override training.n_epochs=1 training.batch_size=2 data.num_workers=0
```

Use `--config path/to/config.yaml` when the model is experimental and should not
be exposed as a bundled preset yet.

## Tests To Add

Keep the first tests narrow:

- The preset name is registered and resolves through `ConfigRegistry`.
- `STPPConfig.from_preset("my_preset")` or `STPPConfig.from_yaml(...)` loads.
- The model builds for a tiny config.
- A one-epoch fit on a small local JSONL split completes or fails with a clear,
  intentional unsupported-capability error.

Do not add a preset to benchmark examples until the fit, save/load, and
evaluation path you document has been exercised.
