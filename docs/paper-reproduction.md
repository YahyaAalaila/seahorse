# Paper Reproduction

This page is the public place for paper reproduction commands once the final
artifact bundle, dataset revisions, and citation details are available.

## Reproduction Inputs

A reproducible run should record:

- The Seahorse git commit or release.
- The exact model presets or YAML configs.
- The dataset source, including Hugging Face revision or local file manifest.
- The benchmark seeds.
- Any `--override` values.
- The evaluation metric profile and sampling controls.

## Benchmark Template

Use a fixed split collection or a pinned Hugging Face dataset revision:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits_root \
  --seeds 1 2 3 \
  --out runs/paper_bench \
  --n_workers 1
```

For Hugging Face-backed data:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --dataset owner/repo[/subdir] \
  --dataset-revision <revision> \
  --seeds 1 2 3 \
  --out runs/paper_bench
```

## Evaluation Template

Run post-fit metrics on each saved run:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data path/to/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/paper_eval/core_test
```

For predictive benchmark artifacts:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data path/to/test.jsonl \
  --split test \
  --metric-profile predictive \
  --out runs/paper_eval/predictive_test
```

## Citation

Citation details will be added before publication.
