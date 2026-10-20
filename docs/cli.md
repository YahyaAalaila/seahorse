# CLI Workflows

The CLI/config interface is the stable path for reproducible runs, HPO,
benchmark campaigns, artifact-backed evaluation, visualization workflows, and
paper-style outputs.

Use the Python API when you want a normal programmatic workflow for one model.
See [Python API](python-api.md).

The public CLI is the module entrypoint:

```bash
python -m seahorse --help
```

It exposes five modes: `fit`, `tune`, `bench`, `evaluate`, `campaign`. Use `--help`
on each for exact arguments supported by the installed version.

## fit: Train One Reproducible Run

```bash
python -m seahorse fit \
  --preset poisson_gmm \
  --train data/my_dataset/train.jsonl \
  --val data/my_dataset/val.jsonl \
  --test data/my_dataset/test.jsonl \
  --out runs/fit \
  --override training.n_epochs=10 training.batch_size=64
```

??? example "Show CLI command — use a YAML config instead of a preset"
    ```bash
    python -m seahorse fit \
      --config path/to/config.yaml \
      --train data/my_dataset/train.jsonl \
      --val data/my_dataset/val.jsonl \
      --test data/my_dataset/test.jsonl
    ```

Key options:

- `--preset` or `--config`: required config source.
- `--dataset`: named, local, or Hugging Face dataset source.
- `--dataset-revision`: optional Hugging Face revision.
- `--train`, `--val`, `--test`: explicit local JSONL paths.
- `--out`: output directory for logs and run artifacts.
- `--save`: directory for saving the runner.
- `--override`: dotted config overrides such as `training.lr=1e-4`.

## tune: Search Hyperparameters

Requires HPO dependencies: `pip install "seahorse-stpp[hpo]"`

??? example "Show CLI command"
    ```bash
    python -m seahorse tune \
      --preset poisson_gmm \
      --train data/my_dataset/train.jsonl \
      --val data/my_dataset/val.jsonl \
      --n_trials 20 \
      --search-alg random \
      --scheduler asha \
      --out runs/hpo/poisson_gmm_best.yaml
    ```

Key options:

- `--preset` or `--config`: required config source.
- `--dataset` or `--train --val`: data source.
- `--n_trials`: maximum HPO trials.
- `--search-alg`: `random` or `bayesian`.
- `--scheduler`: `asha` or `none`.
- `--seed`: HPO seed.
- `--max-concurrent-trials`: concurrency cap.
- `--out`: best-config YAML path.

## bench: Run Benchmark Campaigns

```bash
python -m seahorse bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits \
  --seeds 1 2 3 \
  --out runs/bench \
  --n_workers 1
```

??? example "Show CLI command — single dataset or HuggingFace source"
    ```bash
    python -m seahorse bench \
      --preset poisson_gmm \
      --dataset owner/repo[/subdir] \
      --dataset-revision main \
      --seeds 1 \
      --out runs/bench_one
    ```

Key options:

- `--preset` or `--presets`: required model preset selection.
- `--dataset` or `--splits_dir`: required benchmark data source.
- `--datasets`: filter dataset names from a split collection.
- `--dataset-revision`: optional Hugging Face revision.
- `--seeds`: one or more seeds.
- `--out`: benchmark output directory.
- `--n_workers`: worker count.
- `--tune`: run HPO before evaluation.
- `--tune-dataset`: dataset used for benchmark HPO.
- `--hpo_configs_dir`: directory containing `{preset}_best.yaml` files.
- `--normalize` or `--no-normalize`: benchmark normalization policy.

## evaluate: Metrics And Visualizations

`evaluate` is for post-fit analysis on saved runs:

```bash
python -m seahorse evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/evaluate/core_test
```

Supported modes: `metrics`, `predictive-compare`, `surface`, `merge-artifacts`.

Continue with [Evaluation And Visualization](evaluation.md).

## campaign: Frozen Multi-Seed Campaigns

`campaign` is the path behind the paper's tables: a JSON spec is compiled into an
immutable manifest, verified, executed cell by cell, and reported.

```bash
python -m seahorse campaign prepare --spec experiments/campaign_2026/spec_v1.json --out campaigns/v1
python -m seahorse campaign verify  --manifest campaigns/v1/manifest.json
python -m seahorse campaign run     --manifest campaigns/v1/manifest.json --job <job_id>
python -m seahorse campaign report  --manifest campaigns/v1/manifest.json --out campaigns/v1/report
```

Subcommands: `prepare`, `verify`, `preflight`, `run`, `evaluate`, `calibrate`
(temporal PIT/CRPS and spatial HPD-PIT from the NLL compensator), `report`, `queue`,
`rehearse`, and `slurm` (dispatch and reconcile cluster tasks). Prepared-data and
campaign roots come from `SEAHORSE_*` environment variables. The specs used for the
paper are in `experiments/campaign_2026/`; see
[Reproduce Paper Results](paper-reproduction.md).
