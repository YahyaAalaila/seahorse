# Pegasus Campaign Workflow

This is the execution-facing launch guide for the first real benchmark campaign.
It complements [EXPERIMENT_READINESS.md](EXPERIMENT_READINESS.md) and
[BENCHMARK.md](BENCHMARK.md) with the cluster-side workflow that should be
frozen before large runs begin.

## Scope

First-wave production presets:

- `auto_stpp`
- `deep_stpp`
- `smash`
- `diffusion_stpp`
- factorized baselines (`poisson_*`, `hawkes_*`, `selfcorrecting_*`)

Keep provisional neural and NSMPP work in a later wave unless you explicitly
decide to absorb their extra validation cost.

## Final Flow

1. `freeze`
   - commit a clean tree
   - push it
   - clone once on Pegasus at that exact SHA
   - build or select one fixed container image for the campaign
2. `tune`
   - only for presets with real packaged search spaces
   - write one flat `runs/<campaign_id>/hpo/` directory of `*_best.yaml`
3. `bench`
   - main fit/report launcher
   - run with frozen configs via `--hpo_configs_dir`
   - always pass `--no-normalize` for the frozen raw-first contract
4. `evaluate`
   - `metrics` for systematic heavy metrics and artifact-backed post-fit metrics
   - `predictive-compare` for shortlisted sample-based comparisons
   - `surface` for selected exact/factorized diagnostics only
5. `combine`
   - build a campaign index that joins HPO, bench, and evaluation outputs

Do not use `bench --tune` for the production wave. Tune first, freeze configs,
then bench separately.

## Grouping Policy

Use execution-class buckets, then split into small launch units inside each
bucket.

- `factorized_cpu`
  - factorized baselines
  - optional sub-groups: `factorized_gmm`, `factorized_cnf`, `factorized_tvcnf`
- `auto_deep_gpu`
  - `auto_stpp`, `deep_stpp`
- `approx_gpu`
  - `smash`, `diffusion_stpp`
- `provisional_neural_gpu`
  - later wave only

The recommended launch unit is one preset or one small homogeneous pair, not
one giant mixed bench invocation.

## Device Policy

- GPU bench/training:
  - `auto_stpp`
  - `deep_stpp`
  - `smash`
  - `diffusion_stpp`
- CPU bench/training:
  - factorized baselines
  - `nsmpp` unless you explicitly redesign its resource policy
- CPU evaluation:
  - exact-family `evaluate metrics`
  - exact/factorized `surface`
- GPU evaluation:
  - native-sampler `smash` / `diffusion_stpp` metrics
  - `predictive-compare` for approximate families

## Environment Strategy

Pegasus is container-first via Enroot/Pyxis. The production pattern is:

1. build one fixed `.sqsh` image outside the scheduled jobs
2. store it in persistent storage
3. mount:
   - immutable `REPO_ROOT`
   - persistent `RUN_ROOT`
   - persistent `DATA_ROOT`
4. submit jobs against that same image for the full campaign

Do not `pip install`, create a venv, or `git pull` inside each job.

The runtime dependency set should come from the repo package metadata plus the
few optional pieces you actually use in the campaign, not from ad hoc install
lines inside sbatch scripts.

## Required Saved Outputs

- Tune:
  - `*_best.yaml`
  - `*.data_manifest.json`
  - `*.trials.json`
  - `*.trials.csv`
  - `*.hpo_manifest.json`
- Bench:
  - `bench_meta.json`
  - `data_manifest.json`
  - `cell_index.json`
  - `results.json`
  - `report.html`
  - per-run `config.yaml`, `resolved_config.yaml`, `run_result.json`, `artifacts.json`
- Evaluate metrics:
  - `metrics.json`
  - `evaluation_manifest.json`
  - `artifacts.json`
- Predictive compare:
  - `summary.json`
  - `artifacts.json`
  - `derived_surfaces.npz`
  - per-frame sample payloads
- Surface:
  - `summary.json`
  - `data.npz`
  - `artifacts.json`
- Combine:
  - campaign index with bench roots, HPO manifests, evaluation roots, git SHA,
    image tag, and timestamps

## Job Templates

Use the templates in [scripts/pegasus_campaign_env.sh](/Users/aalaila/Downloads/unified_stpp/scripts/pegasus_campaign_env.sh),
[scripts/pegasus_tune_preset.sbatch](/Users/aalaila/Downloads/unified_stpp/scripts/pegasus_tune_preset.sbatch),
[scripts/pegasus_bench_group.sbatch](/Users/aalaila/Downloads/unified_stpp/scripts/pegasus_bench_group.sbatch),
[scripts/pegasus_eval_metrics.sbatch](/Users/aalaila/Downloads/unified_stpp/scripts/pegasus_eval_metrics.sbatch),
[scripts/pegasus_eval_predictive_compare.sbatch](/Users/aalaila/Downloads/unified_stpp/scripts/pegasus_eval_predictive_compare.sbatch),
and [scripts/pegasus_eval_surface.sbatch](/Users/aalaila/Downloads/unified_stpp/scripts/pegasus_eval_surface.sbatch).

All of them assume:

- the repo clone already exists
- the environment or container is already ready
- outputs live under `RUN_ROOT/<campaign_id>/...`
- jobs skip when the expected terminal artifacts already exist

## Campaign Index

After bench and evaluation finish, build a joinable index:

```bash
python scripts/build_campaign_index.py \
  --campaign-root runs/<campaign_id> \
  --out runs/<campaign_id>/campaign_index.json
```

This produces a campaign-level JSON index and companion CSVs for bench cells and
evaluation roots.

## Acceptance Gate Before Real Runs

- targeted pytest subset passes
- one smoke bench per execution bucket
- one exact metrics compute and load-only cycle
- one GPU native evaluation smoke
- one end-to-end dry run with the final container and final mount paths
