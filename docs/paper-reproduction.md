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

## Public Tutorials

For a runnable end-to-end walkthrough, use the Colab notebooks linked from
[Tutorial Notebooks](examples/colabs.md). They generate demo JSONL data, execute
the public API or CLI, and inspect the resulting artifacts on CPU.
