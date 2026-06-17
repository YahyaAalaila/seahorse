# Tutorial Notebooks

These notebooks are executable tutorials checked into the repository. The
primary links open the notebooks in Google Colab. Each notebook clones the
public repository automatically when opened directly in Colab, installs the
package in editable mode, and uses a small CPU-friendly demo JSONL dataset generated
inside the notebook.

## Available Notebooks

| Notebook | What it covers | Runtime notes |
| --- | --- | --- |
| <a href="https://colab.research.google.com/github/YahyaAalaila/STPPGC/blob/main/docs/notebooks/01_run_one_model_python_api.ipynb">01 Run One Model With The Python API</a> | Generate demo JSONL data, fit `AutoSTPP` and `PoissonGMM`, evaluate, call `predict_next`, and plot sampled next locations. | CPU-only, no Hugging Face dependency. |
| <a href="https://colab.research.google.com/github/YahyaAalaila/STPPGC/blob/main/docs/notebooks/02_benchmark_models_cli.ipynb">02 Benchmark Models With The CLI</a> | Generate demo JSONL data, run `python -m unified_stpp bench` with `poisson_gmm`, `hawkes_gmm`, `auto_stpp`, and `deep_stpp`, then inspect benchmark tables. | CPU-only, uses one seed and one epoch. |

## Running Locally

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install notebook
jupyter notebook docs/notebooks/
```

The notebooks create their own demo data under `runs/tutorials/`.

## Execution Notes

The notebooks are designed for a fresh Colab runtime and do not require a GPU or
external datasets. They also run locally from the repository root.
