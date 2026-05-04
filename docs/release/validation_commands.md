# v1 Validation Commands

These commands are intended for a release candidate. Run them from the repository root. Use a fresh shell and a clean virtual environment for final validation.

## Clean Editable Install

```bash
python3 -m venv /tmp/uni-stpp-v1-venv
source /tmp/uni-stpp-v1-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
```

Development extras:

```bash
python -m pip install -e ".[dev]"
python -m pip check
```

All optional extras, including HPO:

```bash
python -m pip install -e ".[all]"
python -m pip check
```

## Import Smoke Test

```bash
python - <<'PY'
import unified_stpp
from unified_stpp import STPPConfig, STPPRunner, Benchmark
from unified_stpp.models.configs import ConfigRegistry
from unified_stpp.data import load_dataset

print("package", unified_stpp.__file__)
print("version", getattr(unified_stpp, "__version__", None))
print("public", sorted(unified_stpp.__all__))
print("presets", ConfigRegistry.canonical_preset_names())
PY
```

## CLI Smoke Test

```bash
python -m unified_stpp --help
python -m unified_stpp fit --help
python -m unified_stpp tune --help
python -m unified_stpp bench --help
python -m unified_stpp evaluate --help
python -m unified_stpp evaluate metrics --help
python -m unified_stpp evaluate predictive-compare --help
python -m unified_stpp evaluate surface --help
```

## Tiny Dataset And Loader Test

```bash
SMOKE_ROOT=/tmp/uni-stpp-v1-smoke-data
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
python - <<'PY'
import json
from pathlib import Path

root = Path("/tmp/uni-stpp-v1-smoke-data")
records = [
    {"times": [0.1, 0.4, 0.9], "locations": [[0.1, 0.2], [0.2, 0.3], [0.4, 0.5]]},
    {"times": [0.2, 0.6, 1.0], "locations": [[0.2, 0.1], [0.3, 0.4], [0.6, 0.7]]},
]
for split in ("train", "val", "test"):
    with (root / f"{split}.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
PY
python - <<'PY'
from unified_stpp.data import load_dataset
splits = load_dataset("/tmp/uni-stpp-v1-smoke-data")
assert sorted(splits) == ["test", "train", "val"]
assert len(splits["train"]) == 2
print("tiny dataset ok")
PY
```

## Hugging Face Dataset Access Check

Replace the placeholders with the final public dataset IDs and revisions before tagging.

```bash
python - <<'PY'
from unified_stpp.data import download_dataset, load_dataset

checks = [
    ("<real-dataset-hf-repo-or-path>", "<revision-or-tag>"),
    ("<suite3-hf-repo-or-path>", "<revision-or-tag>"),
    ("<suite4-hf-repo-or-path>", "<revision-or-tag>"),
]

for dataset, revision in checks:
    root = download_dataset(dataset, revision=revision)
    splits = load_dataset(root)
    assert {"train", "val", "test"}.issubset(splits), dataset
    print(dataset, {k: len(v) for k, v in splits.items()})
PY
```

## Minimal Training Run

Use a low-cost exact preset first.

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train /tmp/uni-stpp-v1-smoke-data/train.jsonl \
  --val /tmp/uni-stpp-v1-smoke-data/val.jsonl \
  --test /tmp/uni-stpp-v1-smoke-data/test.jsonl \
  --out /tmp/uni-stpp-v1-smoke-run \
  --override \
    training.n_epochs=1 \
    training.batch_size=2 \
    training.device=cpu \
    data.batch_size=2 \
    data.num_workers=0 \
    data.normalize=false
```

## Minimal Evaluation Run

Replace `<RUN_DIR>` with the fit run directory printed or written by the previous command.

```bash
python -m unified_stpp evaluate metrics \
  --run <RUN_DIR> \
  --data /tmp/uni-stpp-v1-smoke-data/test.jsonl \
  --split test \
  --metric-profile core \
  --out /tmp/uni-stpp-v1-smoke-eval
```

## Focused Pytest Suite

```bash
pytest tests/test_smoke.py
pytest tests/test_registry_compat.py tests/test_config_resolution.py
pytest tests/test_data_resolution.py tests/test_data_hub.py tests/test_hf_dataset_smoke.py
pytest tests/test_evaluate_cli.py tests/test_eval_artifacts.py tests/test_metric_profiles.py
pytest tests/test_benchmark_config.py
```

Run full tests before the final tag:

```bash
pytest
```

## Documentation Build

No MkDocs/Sphinx configuration was found. For v1, validate Markdown links and command snippets manually or add a docs build tool before tagging.

Suggested static pass:

```bash
rg -n "python -m unified_stpp (fit|tune|bench|evaluate)" README.md docs
rg -n "Hugging Face|hf|--dataset|dataset-revision|suite 3|suite 4|HawkesNest" README.md docs
rg -n "TODO|FIXME|TBD" README.md docs
```

## Large File Check

```bash
git ls-files | xargs -n 200 du -h 2>/dev/null | sort -hr | sed -n '1,100p'
git ls-files data | xargs -n 200 du -ch 2>/dev/null | tail -n 1
git status --ignored --short | sed -n '1,220p'
```

## Secrets Check

```bash
rg -n "HF_TOKEN|WANDB_API_KEY|api[_-]?key|apikey|access[_-]?token|password|secret|BEGIN RSA|PRIVATE KEY" \
  -g '!runs/**' -g '!runs2/**' -g '!runs3/**' -g '!archive/**' -g '!*.ipynb' .
```

## Hardcoded Local Path Check

```bash
rg -n "/Users/aalaila|/home/aalaila|192\\.168|PEGASUS|Pegasus|pegasus" \
  -g '!runs/**' -g '!runs2/**' -g '!runs3/**' -g '!archive/**' -g '!*.ipynb' .
```

## Benchmark Figure/Table Reproducibility Check

```bash
rg -n "make_.*figure|paper_readiness|table_|report.html|benchmark" README.md docs scripts Makefile
python scripts/paper_readiness_report.py --help
python scripts/make_post_nll_hawkesnest_figures.py --help
python -m py_compile scripts/make_suite3_paper_artifacts.py scripts/make_training_time_diagnostics.py scripts/make_post_nll_hawkesnest_main_v2.py
```

For each public figure/table, the release candidate should document:

- input data or run directory
- script path
- command
- expected output files
- expected runtime class
- whether the command is CPU, GPU, or cluster-only
