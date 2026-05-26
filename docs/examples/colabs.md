# Tutorial Notebooks

These notebooks are executable tutorials checked into the repository. The
primary links open the notebooks in Google Colab.

## Available Notebooks

| Notebook | What it covers | Runtime notes |
| --- | --- | --- |
| <a href="https://colab.research.google.com/github/YahyaAalaila/uni-stpp/blob/release/v1-integration/docs/notebooks/01_run_one_model_python_api.ipynb">01 Run One Model With The Python API</a> | Generate tiny JSONL data, fit `AutoSTPP` and `PoissonGMM`, evaluate, call `predict_next`, and plot sampled next locations. | CPU-only, no Hugging Face dependency. |
| <a href="https://colab.research.google.com/github/YahyaAalaila/uni-stpp/blob/release/v1-integration/docs/notebooks/02_benchmark_models_cli.ipynb">02 Benchmark Models With The CLI</a> | Generate tiny JSONL data, run `python -m unified_stpp bench` with `poisson_gmm`, `hawkes_gmm`, `auto_stpp`, and `deep_stpp`, then inspect benchmark tables. | CPU-only, uses one seed and one epoch. |

## Running Locally

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install notebook
jupyter notebook docs/notebooks/
```

The notebooks create their own tiny data under `runs/tutorials/`.

## Validation Status

The notebooks are validated locally with `jupyter nbconvert --execute` before
being linked from this documentation.
