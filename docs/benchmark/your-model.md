# Add Your Model and Benchmark It

<p class="sh-ds-eyebrow">register a preset · run it on the released datasets · compare on the same footing</p>

The paper's tables are a starting point, not a leaderboard to close. If you have an STPP model,
whether new or from a paper we have not wrapped, Seahorse lets you train it on the same 13 real
datasets and the HawkesNest suite and score it the way the 19 benchmarked presets were scored. We
welcome pull requests that add presets, and issues that report results.

## 1. Register a preset

Implement the encoder / evolution / decoder interface (or wrap an existing PyTorch module) and
register a config under a name, in a module of your own (`my_preset.py`):

```python
from dataclasses import dataclass
from seahorse.models.configs.base import BaseModelConfig, ConfigRegistry
from seahorse.models.unified_model import UnifiedSTPP

@ConfigRegistry.register("my_preset")
@dataclass
class MyPresetConfig(BaseModelConfig):
    hidden_dim: int = 64

    def build_model(self) -> UnifiedSTPP:
        from my_family import MyStateModel, MyEventModel
        return UnifiedSTPP(MyStateModel(self.hidden_dim), MyEventModel(self.hidden_dim),
                           hidden_dim=self.hidden_dim)
```

Declare what your model can do (exact, quadrature or approximate NLL; sampling) so the evaluation
layer reports it under the right label. Details: [Add a Model](../adding-a-model.md),
[Wrap an Existing Model](../extend/wrap.md), [Register a Preset](../extend/preset.md),
[Declare Capabilities](../extend/capabilities.md), [Testing Checklist](../extend/testing.md).

??? example "Try the route first with a stand-in preset (runs on CPU in seconds)"
    A preset that reuses a built-in family, here a Hawkes temporal process with a conditional
    Gaussian mixture over space, is enough to exercise every step below:

    ```python
    # my_preset.py
    import dataclasses
    from typing import ClassVar

    from seahorse.models.configs.base import ConfigRegistry
    from seahorse.models.configs.factorized import FactorizedConfig

    @ConfigRegistry.register("my_preset")
    @dataclasses.dataclass
    class MyPresetConfig(FactorizedConfig):
        _TEMPORAL_TYPE: ClassVar[str] = "hawkes"
    ```

## 2. Make the CLI see it

`python -m seahorse` only knows the presets that ship with the package; a preset registered in
your own module raises `Unknown preset 'my_preset'`. Either add the module to
`seahorse/models/configs/` and import it in `seahorse/models/configs/__init__.py` (the route for a
pull request, see [Register a Preset](../extend/preset.md)), or start the CLI from a two-line
launcher that imports your module first:

```python
# seahorse_cli.py
import my_preset  # noqa: F401  registers "my_preset"
from seahorse.__main__ import main

main()
```

`python seahorse_cli.py <mode> ...` then accepts every argument of `python -m seahorse <mode> ...`.

## 3. Run it against the baselines

The released datasets load by short name at their pinned revisions, so one command trains your
preset next to built-in baselines on a paper dataset:

```bash
python seahorse_cli.py bench \
  --presets my_preset auto_stpp deep_stpp static_hawkes poisson_gmm \
  --dataset nyc_collisions \
  --seeds 1 2 3 \
  --n_workers 1 \
  --out runs/my_preset_nyc_collisions
```

Keep `--n_workers 1` with the launcher: parallel workers start fresh processes that do not import
your module. For a quick smoke test, append `--override training.n_epochs=1 data.num_workers=0`
and use a small dataset such as `geonet_nz`. Repeat for the other datasets (short names on the
[dataset cards](../datasets/index.md)); the script-only datasets (`citibike_nyc`,
`seattle_fire911`, `citibike_legacy`) must be rebuilt first. Every preset in the run sees the same
splits and is scored by the same harness, so the resulting table (`table_test_nll_all.csv`,
`report.html`) is a like-for-like comparison within the run. A single fit works the same way:
`python seahorse_cli.py fit --preset my_preset --dataset geonet_nz --out runs/fit_my_preset`.

## 4. Get a number comparable with the paper

The paper's NLLs come from the frozen campaign protocol, which fixes more than `bench` does by
default:

| Fixed by the paper protocol | Value |
| --- | --- |
| Targets | every model scores the same target events, each conditioned on its 20 preceding events |
| Units | raw coordinates: time in days, space in degrees (log-Jacobian correction for models trained on transformed data) |
| Budget | 3 tuning trials of 5,000 steps (learning rate × 1, 0.3, 3), refit of at most 20,000 steps with early stopping |
| Seeds | three |

A campaign reads its datasets from a **prepared-data root** (`SEAHORSE_PREPARED_DATA_ROOT`), laid
out as `<root>/<corpus>/dataset_meta.json` plus `<root>/<corpus>/<view>/{train,val,test}.jsonl`,
and needs a train-only coordinate artifact. Both can be assembled from the released repositories,
which carry `metadata/dataset_meta.json` and `metadata/coordinate_transform.json`:

```python
# build_prepared_root.py  —  usage: python build_prepared_root.py <root> geonet_nz [austin311 ...]
import json, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from seahorse.data.hf import RELEASED_DATASETS

root, names = Path(sys.argv[1]), sys.argv[2:]
routes = {}
for name in names:
    spec = RELEASED_DATASETS[name]
    snap = Path(snapshot_download(
        spec.repo_id, repo_type="dataset", revision=spec.revision,
        allow_patterns=["metadata/dataset_meta.json", "metadata/coordinate_transform.json",
                        *[f"{c}/*.jsonl" for c in spec.configs]]))
    meta = json.loads((snap / "metadata/dataset_meta.json").read_text())
    corpus = root / meta["dataset_name"]
    corpus.mkdir(parents=True, exist_ok=True)
    shutil.copy(snap / "metadata/dataset_meta.json", corpus / "dataset_meta.json")
    for c in spec.configs:
        (corpus / c).mkdir(exist_ok=True)
        for s in ("train", "val", "test"):
            dst = corpus / c / f"{s}.jsonl"
            if not dst.exists():
                dst.symlink_to((snap / c / f"{s}.jsonl").resolve())
    ct = json.loads((snap / "metadata/coordinate_transform.json").read_text())
    routes[ct["route"]] = ct["entry"]
(root / "COORDINATE_TRANSFORMS.json").write_text(json.dumps({"routes": routes}, indent=1))
```

This works for the hosted datasets; the script-only ones must be rebuilt into the same layout.
Then copy `experiments/campaign_2026/spec_nnstat_v2.json` to `my_spec.json` and edit it:

- `datasets`: keep the entries you want (e.g. `geonet_nz`);
- `models`: add an entry for your preset, copied from a built-in one with the same NLL type, e.g.
  `{"id": "my_preset", "preset": "my_preset", "label": "My preset", "stage": "core", "track": "conditional_next_event", "nll_path": "closed_form", "runner": "nnstpp"}`;
- `coordinate_artifact`: the path of `COORDINATE_TRANSFORMS.json` written above;
- remove `data_audit` and `synthetic_manifest`, which point to author-side files that are not
  released.

```bash
export SEAHORSE_PREPARED_DATA_ROOT=$PWD/prepared
python build_prepared_root.py prepared geonet_nz
python seahorse_cli.py campaign prepare   --spec my_spec.json --out campaigns/my_preset
python seahorse_cli.py campaign verify    --manifest campaigns/my_preset/manifest.json
python seahorse_cli.py campaign preflight --manifest campaigns/my_preset/manifest.json --models my_preset poisson_gmm
python seahorse_cli.py campaign run       --manifest campaigns/my_preset/manifest.json --job geonet_nz__my_preset__hpo0
python seahorse_cli.py campaign report    --manifest campaigns/my_preset/manifest.json --out campaigns/my_preset/report
```

`prepare` verifies the split hashes against the dataset manifests and writes one job per tuning
trial, refit and seed (job ids are listed in `manifest.json`); it records `pip freeze`, so the
environment needs `pip` (a `uv` virtual environment may not have it). `campaign run` executes the
neural runner on a CUDA GPU in the environment recorded by `preflight`; it refuses CPU-only
evidence. Your numbers then sit next to the published values of Table 2 (the best and second-best
presets of each dataset are listed on its card). The full workflow is in
[Reproduce Paper Results](../paper-reproduction.md).

!!! note "Approximate likelihoods are labelled, not ranked"
    If your model only provides samples or a surrogate objective, report it like SMASH and DSTPP:
    its value is labelled approximate, kept out of the ranking, and the sample-based next-event
    and calibration metrics carry the comparison ([Evaluation Profiles](../evaluation.md)).

## 5. Share it

Open a pull request with the preset, its YAML defaults and a smoke test, or an issue with your
campaign report, at [github.com/YahyaAalaila/seahorse](https://github.com/YahyaAalaila/seahorse).
