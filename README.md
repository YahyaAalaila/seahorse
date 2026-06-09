<div align="center">

<img src="docs/assets/logofv.png" alt="Seahorse logo" width="120"/>

# Seahorse — `unified-stpp`

**A modular, research-grade framework for end-to-end development, training, and evaluation of**  
**Spatio-Temporal Point-Process (STPP) models.**  
Seahorse couples declarative YAML configuration with PyTorch Lightning execution,  
Ray Tune hyper-parameter optimisation, and version-controlled logging to deliver  
rapid prototyping and rigorous, reproducible benchmarking on streaming event data.

[![PyPI](https://img.shields.io/pypi/v/unified-stpp?label=pypi&color=blue)](https://pypi.org/project/unified-stpp/)
[![Last Commit](https://img.shields.io/github/last-commit/YahyaAalaila/uni-stpp)](https://github.com/YahyaAalaila/uni-stpp/commits)
[![Branch](https://img.shields.io/badge/branch-latest-brightgreen)](https://github.com/YahyaAalaila/uni-stpp)
[![Issues](https://img.shields.io/github/issues/YahyaAalaila/uni-stpp)](https://github.com/YahyaAalaila/uni-stpp/issues)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/lightning-2.2%2B-792ee5)](https://lightning.ai/)
[![Ray Tune](https://img.shields.io/badge/ray__tune-2.9%2B-028CF0)](https://docs.ray.io/en/latest/tune/)

</div>

---

| [News](#news) | [Features](#features) | [Model List](#model-list) | [Datasets](#datasets) | [Quick Start](#quick-start) | [CLI](#cli) | [License](#license) | [Acknowledgment](#acknowledgment) |

---

## News <sup>[[Back to Top](#seahorse--unified-stpp)]</sup>

- ![NEW](https://img.shields.io/badge/-NEW-red) Documentation site is live — install the docs extras and run `mkdocs serve`.
- ![NEW](https://img.shields.io/badge/-NEW-red) Google Colab notebooks available: [01 — Run One Model](https://colab.research.google.com/github/YahyaAalaila/uni-stpp/blob/release/v1-integration/docs/notebooks/01_run_one_model_python_api.ipynb) · [02 — Benchmark via CLI](https://colab.research.google.com/github/YahyaAalaila/uni-stpp/blob/release/v1-integration/docs/notebooks/02_benchmark_models_cli.ipynb)
- ![NEW](https://img.shields.io/badge/-NEW-red) Unified benchmark data-contract enforced across all presets — comparable NLL out of the box.

---

## Features <sup>[[Back to Top](#seahorse--unified-stpp)]</sup>

| | |
|---|---|
| **Unified Python API** | Train, evaluate, and sample any model through one consistent interface (`STPPRunner`). |
| **YAML-driven config** | Every hyperparameter is declarative; experiments are fully reproducible. |
| **Plug-and-play presets** | Switch models with `--preset auto_stpp` — no code changes required. |
| **Ray Tune HPO** | YAML search-space files feed directly into distributed hyperparameter sweeps. |
| **Benchmark campaigns** | Multi-preset × multi-dataset × multi-seed runs with a single CLI command. |
| **Data contract** | `Benchmark` enforces identical train/val/test splits across all presets so NLL scores are directly comparable. |
| **HuggingFace datasets** | Stream or cache any JSONL dataset directly from the Hub with `--dataset owner/repo`. |

---

## Model List <sup>[[Back to Top](#seahorse--unified-stpp)]</sup>

Our package includes the following state-of-the-art STPP models:

| No | Venue | Preset | Paper | Implementation |
|:--:|-------|--------|-------|:--------------:|
| 1 | NeurIPS'23 | `auto_stpp` | [Automatic Integration for Spatiotemporal Neural Point Processes](https://arxiv.org/abs/2310.01179) | PyTorch |
| 2 | L4DC'22 | `deep_stpp` | [Deep Spatiotemporal Point Process](https://proceedings.mlr.press/v168/lin22a.html) | PyTorch |
| 3 | ICLR'21 | `neural_jumpcnf` / `neural_attncnf` | [Neural Spatio-Temporal Point Processes](https://openreview.net/forum?id=XQQA6-So14) | PyTorch |
| 4 | NeurIPS'19 | `njsde` | [Neural Jump Stochastic Differential Equations](https://arxiv.org/abs/1905.10403) | PyTorch |
| 5 | ACM KDD'23 | `diffusion_stpp` | [Spatio-temporal Diffusion Point Processes](https://dl.acm.org/doi/10.1145/3580305.3599511) | PyTorch |
| 6 | ICLR'22 | `nsmpp` | [Neural Spectral Marked Point Processes](https://openreview.net/forum?id=0rcbOaoBXbg) | PyTorch |
| 7 | Arxiv | `smash` | [Embedding Event History to Vector](https://arxiv.org/abs/2310.19324) | PyTorch |
| 8 | ICML'20 | `thp_gmm` | [Transformer Hawkes Process](https://arxiv.org/abs/2002.09291) | PyTorch |
| 9 | KDD'16 | `rmtpp_gmm` | [Recurrent Marked Temporal Point Processes](https://dl.acm.org/doi/10.1145/2939672.2939875) | PyTorch |

**Parametric baselines** (fast, exact likelihood):  
`poisson_gmm` · `hawkes_gmm` · `selfcorrecting_gmm` · `poisson_cnf` · `hawkes_cnf` · `selfcorrecting_cnf` · `poisson_tvcnf` · `hawkes_tvcnf` · `selfcorrecting_tvcnf`

---

## Datasets <sup>[[Back to Top](#seahorse--unified-stpp)]</sup>

Seahorse reads any collection of JSONL event sequences. The canonical split layout is:

```text
dataset_root/
  train.jsonl
  val.jsonl
  test.jsonl
```

Each line is one sequence:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

Datasets from the original NeuralSTPP paper are directly supported:

- **Pinwheel** — Synthetic multimodal non-Gaussian process. 10 clusters in a pinwheel structure; events propagate clock-wise via a multivariate Hawkes mechanism. Tests the ability to capture drastic history-driven spatial shifts.
- **Earthquake** — Real-world seismic event catalog ([U.S. Geological Survey, 2020](https://earthquake.usgs.gov/)).
- **COVID-19** — Geo-located case reports ([New York Times, 2020](https://github.com/nytimes/covid-19-data)).
- **Citibike** — NYC bike-share ride starts; useful for dense urban mobility modelling.

Datasets can also be streamed from [HuggingFace Hub](https://huggingface.co/datasets) via `--dataset owner/repo`.

---

## Quick Start <sup>[[Back to Top](#seahorse--unified-stpp)]</sup>

**Install**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Python API**

```python
from unified_stpp import AutoSTPP, PoissonGMM, load_jsonl

train = load_jsonl("dataset_root/train.jsonl")
val   = load_jsonl("dataset_root/val.jsonl")
test  = load_jsonl("dataset_root/test.jsonl")

model    = AutoSTPP(device="cpu")
baseline = PoissonGMM()

model.fit(train, val, test, epochs=50, batch_size=64)
scores  = model.evaluate(test)          # {"test_nll": ..., "mean_seq_nll": ...}
samples = model.predict_next(test, n_samples=32)
```

**STPPRunner (lower-level)**

```python
from unified_stpp import STPPRunner

runner = STPPRunner.from_preset("auto_stpp")
result = runner.fit(train, val, test)   # returns RunResult
runner.save("/tmp/my_run/")

runner2 = STPPRunner.load("/tmp/my_run/")
grid    = runner2.intensity_grid(test[0])
```

---

## CLI <sup>[[Back to Top](#seahorse--unified-stpp)]</sup>

**Fit one model**

```bash
python -m unified_stpp fit \
  --preset auto_stpp \
  --train dataset_root/train.jsonl \
  --val   dataset_root/val.jsonl \
  --test  dataset_root/test.jsonl \
  --out   runs/quickstart
```

**Benchmark campaign** (multi-preset × multi-seed)

```bash
python -m unified_stpp bench \
  --presets auto_stpp deep_stpp njsde poisson_gmm \
  --splits_dir splits/ \
  --seeds 1 2 3 \
  --out runs/bench \
  --n_workers 4
```

**HPO sweep**

```bash
python -m unified_stpp tune \
  --preset auto_stpp \
  --search_space configs/hpo/auto_stpp_search.yaml \
  --train dataset_root/train.jsonl \
  --val   dataset_root/val.jsonl \
  --n_trials 30
```

---

## Documentation <sup>[[Back to Top](#seahorse--unified-stpp)]</sup>

```bash
pip install -e ".[docs]"
mkdocs serve          # live-reload at http://127.0.0.1:8000
mkdocs build --strict # static site → site/
```

The docs cover the [Python API](docs/python-api.md), [CLI reference](docs/cli.md), [data format](docs/data-format.md), [benchmark campaigns](docs/benchmarks.md), and [how to add a new model](docs/extend/preset.md).

---

## Citation <sup>[[Back to Top](#seahorse--unified-stpp)]</sup>

Citation details will be added before publication.

---

## License <sup>[[Back to Top](#seahorse--unified-stpp)]</sup>

Seahorse is distributed under the **MIT License**. See [LICENSE](LICENSE).

---

## Acknowledgment <sup>[[Back to Top](#seahorse--unified-stpp)]</sup>

Seahorse builds on the original implementations of the paper families it wraps. We thank the authors of AutoSTPP, DeepSTPP, NeuralSTPP, NJSDE, DiffusionSTPP, NSMPP, SMASH, THP, and RMTPP for releasing their code.
