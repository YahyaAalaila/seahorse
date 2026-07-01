# Tutorial Notebooks

These notebooks are executable tutorials checked into the repository. The primary
links open them in Google Colab, where each notebook installs `seahorse-stpp` from
PyPI. Notebooks 01 and 02 generate a small demo dataset inside the notebook; the
case study loads a real dataset from the Hub.

## Available Notebooks

| Notebook | What it covers | Runtime notes |
| --- | --- | --- |
| <a href="https://colab.research.google.com/github/YahyaAalaila/seahorse/blob/main/docs/notebooks/01_run_one_model_python_api.ipynb">01 Run One Model With The Python API</a> | Generate demo JSONL data, fit `AutoSTPP` and `PoissonGMM`, evaluate, call `predict_next`, and plot sampled next locations. | CPU-only, no Hugging Face dependency. |
| <a href="https://colab.research.google.com/github/YahyaAalaila/seahorse/blob/main/docs/notebooks/02_benchmark_models_cli.ipynb">02 Benchmark Models With The CLI</a> | Generate demo JSONL data, run `python -m seahorse bench` with `poisson_gmm`, `hawkes_gmm`, `auto_stpp`, and `deep_stpp`, then inspect benchmark tables. | CPU-only, uses one seed and one epoch. |
| <a href="https://colab.research.google.com/github/YahyaAalaila/seahorse/blob/main/docs/notebooks/03_case_study_citibike.ipynb">03 Case Study: NYC Citibike</a> | Load the real **Citibike** dataset from the Hub, explore it on a map, fit `PoissonGMM` and `DeepSTPP`, compare them, and visualize predictions. | CPU-only; downloads Citibike from Hugging Face. |

## Running Locally

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install seahorse-stpp notebook
jupyter notebook docs/notebooks/
```

Notebooks 01 and 02 create their own demo data under `runs/tutorials/`; the case
study downloads Citibike from the Hub.

## Execution Notes

The notebooks are designed for a fresh Colab runtime and do not require a GPU.
Notebooks 01 and 02 need no external data; the case study downloads the Citibike
dataset from the Hub. They also run locally.
