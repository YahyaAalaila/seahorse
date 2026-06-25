# Tutorial Notebooks

These notebooks are executable tutorials checked into the repository. The
primary links open the notebooks in Google Colab. The Python API notebook
installs Seahorse as a package from the public repository and uses the
CPU-friendly `yahya021/citibike-stpp` dataset shipped on Hugging Face.

## Available Notebooks

| Notebook | What it covers | Runtime notes |
| --- | --- | --- |
| <a href="https://colab.research.google.com/github/YahyaAalaila/seahorse/blob/main/docs/notebooks/01_run_one_model_python_api.ipynb">01 Run One Model With The Python API</a> | Install Seahorse, load `yahya021/citibike-stpp`, fit `PoissonGMM` and `DeepSTPP`, evaluate, call `predict_next`, and plot results. | CPU-only, downloads the Citibike JSONL splits from Hugging Face. |
| <a href="https://colab.research.google.com/github/YahyaAalaila/seahorse/blob/main/docs/notebooks/02_benchmark_models_cli.ipynb">02 Benchmark Models With The CLI</a> | Generate demo JSONL data, run `python -m seahorse bench` with `poisson_gmm`, `hawkes_gmm`, `auto_stpp`, and `deep_stpp`, then inspect benchmark tables. | CPU-only, uses one seed and one epoch. |

## Running Locally

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install notebook
jupyter notebook docs/notebooks/
```

The Python API notebook reads the Hugging Face Citibike dataset and caps the
number of sequences for a fast Colab run. The CLI benchmark notebook creates
its own demo data under `runs/tutorials/`.

## Execution Notes

The notebooks are designed for a fresh Colab runtime and do not require a GPU.
The Python API case study downloads a small public Hugging Face dataset. They
also run locally from the repository root.
