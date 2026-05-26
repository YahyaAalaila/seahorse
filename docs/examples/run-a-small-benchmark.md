# Run A Small Benchmark

For the complete executable walkthrough, use
<a href="https://colab.research.google.com/github/YahyaAalaila/uni-stpp/blob/release/v1-integration/docs/notebooks/02_benchmark_models_cli.ipynb">02 Benchmark Models With The CLI</a>
in Google Colab.

## Goal

Run a small benchmark across several presets on one dataset using the CLI/config
path. This records presets, data source, seed, overrides, and benchmark artifacts.

## When To Use This Instead Of The Python API

Use this workflow when you want to compare several presets, keep reproducible
run directories, or scale later to more datasets, seeds, or HPO. Use the Python
API when you only want one model in a script or notebook.

## Input Data Layout

```text
data/my_dataset/
  train.jsonl
  val.jsonl
  test.jsonl
```

The benchmark CLI receives the dataset directory with `--dataset`; it does not
take separate `--train`, `--val`, or `--test` flags.

## Run The Benchmark

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm auto_stpp deep_stpp \
  --dataset data/my_dataset \
  --seeds 1 \
  --out runs/examples/small_benchmark \
  --n_workers 1
```

??? example "Show CLI command — quick smoke test with training overrides"
    ```bash
    python -m unified_stpp bench \
      --presets poisson_gmm hawkes_gmm auto_stpp deep_stpp \
      --dataset data/my_dataset \
      --seeds 1 \
      --out runs/examples/small_benchmark \
      --n_workers 1 \
      --override training.n_epochs=1 training.batch_size=2 data.num_workers=0
    ```

    The notebook uses additional tiny-model overrides so `auto_stpp` and
    `deep_stpp` finish quickly on CPU.

## Inspect Output Directory

Common campaign artifacts under `runs/examples/small_benchmark/`:

??? example "Show expected output"
    ```text
    runs/examples/small_benchmark/
      bench_meta.json
      cell_index.json
      results.json
      report.html
      table_test_nll_all.csv
    ```

    Use `cell_index.json` to map benchmark cells to saved run directories.

## Evaluate Metrics

??? example "Show CLI command"
    ```bash
    python -m unified_stpp evaluate metrics \
      --run path/to/run_dir \
      --data data/my_dataset/test.jsonl \
      --split test \
      --metric-profile core \
      --out runs/examples/small_benchmark/evaluate_core
    ```

## Common Errors

- Missing `train.jsonl` or `val.jsonl`: check the `--dataset` directory.
- `Unknown model preset`: check preset spelling.
- Out of memory: start with one seed, `--n_workers 1`, smaller batch size, or fewer presets.
- HPO dependency errors: remove `--tune` unless you intend to run HPO.

## Scaling Up

Add more presets to `--presets`, add more seeds with `--seeds 1 2 3`, or use
`--splits_dir` for multi-dataset benchmark collections. Use HPO only when you
have selected a tuning dataset.
