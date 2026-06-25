# Register a Preset

A **preset** is a named entry in the config registry — and that registry is the
quiet trick that makes Seahorse composable. You register your model **once**, and
every entry point resolves it by name through the same lookup. No import-path
changes, no per-command wiring.

<div class="sh-hub" markdown="0">
  <div class="sh-hub-center">
    <span class="sh-hub-k">register once</span>
    <code class="sh-hub-token">@ConfigRegistry.register("my_preset")</code>
  </div>
  <span class="sh-hub-fan" aria-hidden="true"></span>
  <div class="sh-hub-spokes">
    <span class="sh-hub-spoke">fit</span>
    <span class="sh-hub-spoke">bench</span>
    <span class="sh-hub-spoke">evaluate</span>
    <span class="sh-hub-spoke">predict_next</span>
    <span class="sh-hub-spoke">CLI&nbsp;·&nbsp;--preset my_preset</span>
    <span class="sh-hub-spoke">STPPEstimator("my_preset")</span>
  </div>
  <span class="sh-hub-cap">One name, reachable everywhere. The registry is the single source every entry point shares — so one decorator is the only wiring you write.</span>
</div>

## Step 1: Create a ModelFamilyConfig

```python
from dataclasses import dataclass, field
from seahorse.models.configs.base import BaseModelConfig, ConfigRegistry
from seahorse.models.unified_model import UnifiedSTPP

@ConfigRegistry.register("my_preset")
@dataclass
class MyPresetConfig(BaseModelConfig):
    hidden_dim: int = 64
    n_layers: int = 2

    @classmethod
    def from_dict(cls, d: dict, *, hidden_dim: int, **kwargs) -> "MyPresetConfig":
        return cls(
            hidden_dim=hidden_dim,
            n_layers=d.get("n_layers", 2),
        )

    def build_model(self) -> UnifiedSTPP:
        from .my_family import MyStateModel, MyEventModel
        state = MyStateModel(self.hidden_dim, self.n_layers)
        event = MyEventModel(self.hidden_dim)
        return UnifiedSTPP(state, event, hidden_dim=self.hidden_dim)
```

## Step 2: Import the Config Module

Add your config module to `seahorse/models/configs/__init__.py` so it is registered on import:

```python
# seahorse/models/configs/__init__.py
from . import my_preset_config  # noqa: F401 — triggers @ConfigRegistry.register
```

## Step 3: Add a Bundled YAML (Optional)

For a preset users should run directly, add defaults at:

```text
seahorse/configs/my_preset.yaml
```

Example minimal YAML:

```yaml
model:
  preset: my_preset
  hidden_dim: 64
  n_layers: 2
training:
  n_epochs: 100
  lr: 5.0e-4
  batch_size: 64
```

## Step 4: Verify Registration

```python
from seahorse import STPPEstimator, list_available_models

print("my_preset" in list_available_models())  # True

model = STPPEstimator("my_preset", device="cpu")
```

## Step 5: CLI Smoke Test

```bash
python -m seahorse fit \
  --preset my_preset \
  --train data/my_dataset/train.jsonl \
  --val data/my_dataset/val.jsonl \
  --test data/my_dataset/test.jsonl \
  --out runs/smoke \
  --override training.n_epochs=1 training.batch_size=4 data.num_workers=0
```

## Optional: PresetDescriptor

If your preset needs training-data-dependent initialization (bounding box, coordinate statistics, device fallback), implement a `PresetDescriptor`:

```python
from seahorse.presets.base import PresetDescriptor

class MyDescriptor(PresetDescriptor):
    def data_init_overrides(self, dm) -> dict:
        # dm is the fitted STPPDataModule; access dm._train_dataset
        bbox = compute_bbox(dm._train_dataset)
        return {"bbox": bbox}
```

The runner calls `descriptor.data_init_overrides(dm)` before `build_model()` and merges the result into `build_overrides`.

See [existing descriptors](https://github.com/YahyaAalaila/seahorse/tree/main/seahorse/presets) for reference implementations.
