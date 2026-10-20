# Reproducible Study Checklist

This page defines the command and artifact checklist for reproducible Seahorse
studies. Use it for paper experiments, team benchmarks, or public case
studies.

Use CLI workflows for reproduction. They record the model preset or config,
dataset source, seeds, overrides, and output artifacts more explicitly than a
notebook-only workflow.

## Reproduction Inputs

A reproducible run should record:

- The Seahorse git commit or release.
- The exact model presets or YAML configs.
- The dataset source, including Hugging Face revision or local file manifest.
- The benchmark seeds.
- Any `--override` values.
- The evaluation metric profile and sampling controls.
- Hardware-relevant settings such as device, worker count, and HPO concurrency.

## Benchmark Template

Use a fixed split collection or a pinned Hugging Face dataset revision:

```bash
python -m seahorse bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits \
  --seeds 1 2 3 \
  --out runs/paper_bench \
  --n_workers 1
```

For Hugging Face-backed data:

```bash
python -m seahorse bench \
  --presets poisson_gmm hawkes_gmm \
  --dataset owner/repo[/subdir] \
  --dataset-revision <revision> \
  --seeds 1 2 3 \
  --out runs/paper_bench
```

## Evaluation Template

Run post-fit metrics on each saved run:

```bash
python -m seahorse evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/paper_eval/core_test
```

For predictive benchmark artifacts:

```bash
python -m seahorse evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile predictive \
  --out runs/paper_eval/predictive_test
```

For qualitative visual diagnostics:

```bash
python -m seahorse evaluate predictive-compare \
  --run path/to/run_a \
  --run path/to/run_b \
  --label model_a \
  --label model_b \
  --history data/my_dataset/test.jsonl \
  --split test \
  --horizon 1.0 \
  --out runs/paper_eval/predictive_compare
```

## Artifact Checklist

Keep the following with a paper reproduction bundle:

- `bench_meta.json`
- `cell_index.json`
- `results.json`
- benchmark tables such as `table_test_nll_all.csv`
- per-run `run_result.json` files
- metric output directories
- predictive, generative, or surface artifacts used in tables and figures
- exact command lines and git commit

## Paper Campaigns

The tables of the NeurIPS 2026 paper come from frozen campaigns, not from ad-hoc
`bench` runs. The campaign specifications are in `experiments/campaign_2026/`
(`spec_v1.json` for the neural roster, `spec_nnstat_v2.json` and
`spec_nnstat_pegasus_v1.json` for the neural + statistical rosters as run on the
worker clusters, `etas_spec_v1.json` for the author-ETAS adapter). A campaign is
compiled into an immutable manifest, verified, executed cell by cell, calibrated
and reported:

```bash
python -m seahorse campaign prepare   --spec experiments/campaign_2026/spec_v1.json --out campaigns/v1
python -m seahorse campaign verify    --manifest campaigns/v1/manifest.json
python -m seahorse campaign run       --manifest campaigns/v1/manifest.json --job <job_id>
python -m seahorse campaign calibrate --manifest campaigns/v1/manifest.json --job <job_id> --out campaigns/v1/calib/<job_id>
python -m seahorse campaign report    --manifest campaigns/v1/manifest.json --out campaigns/v1/report
```

Prepared-data and campaign roots are read from `SEAHORSE_*` environment variables.
The datasets load by short name at their pinned revisions
([Ready-to-use HF Datasets](datasets/hf-datasets.md)); the script-only datasets
(`citibike_nyc`, `seattle_fire911`, `citibike_legacy`) must be rebuilt locally first.
The exact-evaluation layer behind the reported NLL and calibration metrics is
documented in [Exact Evaluation Metrics](reference/exact-evaluation.md).

## Public Tutorials

For a runnable end-to-end walkthrough, use the Colab notebooks linked from
[Tutorial Notebooks](examples/colabs.md). Notebooks 01 and 02 generate demo JSONL
data and exercise the public API or CLI on CPU; the case study loads the released
NYC collisions dataset from the Hub.

## Changes from 0.1.0 (camera-ready)

Version 1.0.0 corrects the likelihood reporting of 0.1.0 (see the [changelog](changelog.md)):
NSMPP's next-event NLL omitted the compensator term, THP-GMM and RMTPP-GMM reported only the spatial
term, RMTPP's float32 compensator underflowed, and AutoSTPP/DeepSTPP averaged in zeros for targets
their window cannot score. Numbers produced with 0.1.0 for these presets are superseded; the
camera-ready tables were regenerated from run artefacts with `seahorse.evaluation.v1`, which also
checks that the eventwise and prefix-difference scoring routes agree for every exact preset
(`tests/test_nll_reporting_consistency.py`). Rollout coherence is a distance (lower is better), and
intensity recovery is evaluated on all test sequences.
