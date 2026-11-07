# Add Your Model and Benchmark It

<p class="sh-ds-eyebrow">register a preset · run it on the released datasets · compare on the same footing</p>

The paper's tables are a starting point, not a leaderboard to close. If you have an STPP model,
whether new or from a paper we have not wrapped, Seahorse lets you train it on the same 13 real
datasets and the HawkesNest suite and score it the way the 19 benchmarked presets were scored. We
welcome pull requests that add presets, and issues that report results.

## 1. Register a preset

Implement the encoder / evolution / decoder interface (or wrap an existing PyTorch module) and
register a config under a name:

```python
from dataclasses import dataclass
from seahorse.models.configs.base import BaseModelConfig, ConfigRegistry
from seahorse.models.unified_model import UnifiedSTPP

@ConfigRegistry.register("my_preset")
@dataclass
class MyPresetConfig(BaseModelConfig):
    hidden_dim: int = 64

    def build_model(self) -> UnifiedSTPP:
        from .my_family import MyStateModel, MyEventModel
        return UnifiedSTPP(MyStateModel(self.hidden_dim), MyEventModel(self.hidden_dim),
                           hidden_dim=self.hidden_dim)
```

Declare what your model can do (exact, quadrature or approximate NLL; sampling) so the evaluation
layer reports it under the right label. Details: [Add a Model](../adding-a-model.md),
[Wrap an Existing Model](../extend/wrap.md), [Register a Preset](../extend/preset.md),
[Declare Capabilities](../extend/capabilities.md), [Testing Checklist](../extend/testing.md).

## 2. Run it against the baselines

The released datasets load by short name at their pinned revisions, so one command trains your
preset next to the paper's strongest models on a paper dataset:

```bash
python -m seahorse bench \
  --presets my_preset auto_stpp deep_stpp static_hawkes poisson_gmm \
  --dataset nyc_collisions \
  --seeds 1 2 3 \
  --out runs/my_preset_nyc_collisions
```

Repeat for the other datasets (short names on the [dataset cards](../datasets/index.md)); the
script-only datasets (`citibike_nyc`, `seattle_fire911`, `citibike_legacy`) must be rebuilt first.
Every preset in the run sees the same splits and is scored by the same harness, so the resulting
table (`table_test_nll_all.csv`, `report.html`) is a like-for-like comparison.

## 3. Get a number comparable with the paper

The paper's NLLs come from the frozen campaign protocol, which fixes more than `bench` does by
default:

| Fixed by the paper protocol | Value |
| --- | --- |
| Targets | every model scores the same target events, each conditioned on its 20 preceding events |
| Units | raw coordinates: time in days, space in degrees (log-Jacobian correction for models trained on transformed data) |
| Budget | 3 tuning trials of 5,000 steps (learning rate × 1, 0.3, 3), refit of at most 20,000 steps with early stopping |
| Seeds | three |

To put your model in that table, add it to the `models` list of a campaign specification in
`experiments/campaign_2026/` (one entry per preset, like the existing `auto_stpp` entry, with your
`preset` name and its `nll_path`) and run the campaign:

```bash
python -m seahorse campaign prepare --spec my_spec.json --out campaigns/my_preset
python -m seahorse campaign verify  --manifest campaigns/my_preset/manifest.json
python -m seahorse campaign run     --manifest campaigns/my_preset/manifest.json --job <job_id>
python -m seahorse campaign report  --manifest campaigns/my_preset/manifest.json --out campaigns/my_preset/report
```

Your numbers then sit next to the published values of Table 2 (the best and second-best presets of
each dataset are listed on its card). The full workflow is in
[Reproduce Paper Results](../paper-reproduction.md).

!!! note "Approximate likelihoods are labelled, not ranked"
    If your model only provides samples or a surrogate objective, report it like SMASH and DSTPP:
    its value is labelled approximate, kept out of the ranking, and the sample-based next-event
    and calibration metrics carry the comparison ([Evaluation Profiles](../evaluation.md)).

## 4. Share it

Open a pull request with the preset, its YAML defaults and a smoke test, or an issue with your
campaign report, at [github.com/YahyaAalaila/seahorse](https://github.com/YahyaAalaila/seahorse).
