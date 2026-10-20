<div align="center">

<img src="docs/assets/logofv.png" alt="Seahorse logo" width="200"/>

**A modular, research-grade framework for end-to-end development, training, and evaluation of**
**Spatio-Temporal Point-Process (STPP) models.**
Seahorse couples declarative YAML configuration with PyTorch Lightning execution,
Ray Tune hyper-parameter optimisation, and version-controlled logging to deliver
rapid prototyping and rigorous, reproducible benchmarking on streaming event data.

Documentation: https://yahyaaalaila.github.io/seahorse/

[![Last Commit](https://img.shields.io/github/last-commit/YahyaAalaila/seahorse)](https://github.com/YahyaAalaila/seahorse/commits)
[![Branch](https://img.shields.io/badge/branch-latest-brightgreen)](https://github.com/YahyaAalaila/seahorse)
[![Issues](https://img.shields.io/github/issues/YahyaAalaila/seahorse)](https://github.com/YahyaAalaila/seahorse/issues)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/seahorse-stpp.svg)](https://pypi.org/project/seahorse-stpp/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21078077.svg)](https://doi.org/10.5281/zenodo.21078077)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.8%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/lightning-2.3%2B-792ee5)](https://lightning.ai/)
[![Ray Tune](https://img.shields.io/badge/ray__tune-2.43%2B-028CF0)](https://docs.ray.io/en/latest/tune/)

</div>

---

| [News](#news) | [What you can build with Seahorse STPP](#what-you-can-build-with-seahorse-stpp) | [Features](#features) | [Quick Start](#quick-start) | [Supported models](#supported-models) | [Datasets](#datasets) | [CLI](#cli) | [Citation](#citation) | [License](#license) | [Acknowledgment](#acknowledgment) |

---

## 🗞️ News [[Back&nbsp;to&nbsp;Top](#top)]

- ![NEW](https://img.shields.io/badge/NEW-red?style=flat)&nbsp; The **Seahorse** paper is out on [arXiv](https://arxiv.org/abs/2607.01022) — and `v0.1.0` is on PyPI: `pip install seahorse-stpp`.
- ![NEW](https://img.shields.io/badge/NEW-red?style=flat)&nbsp; **HawkesNest**, our synthetic STPP stress-test suite, is released: [code](https://github.com/YahyaAalaila/HawkesNest) · [paper](https://arxiv.org/abs/2606.16863).
- ![NEW](https://img.shields.io/badge/NEW-red?style=flat)&nbsp; [24-05-2025] Presentation at [Machine Learning &amp; Global Health Network (MLGH)](https://mlgh.net/), London, UK.
- ![NEW](https://img.shields.io/badge/NEW-red?style=flat)&nbsp; [01-04-2025] Our [knowledgebase website](https://events2025.github.io/) is finally up.

<details>
<summary>More news</summary>

- [13-02-2025] Our review paper about [Neural Spatiotemporal Point Processes: Trends and Challenges](https://arxiv.org/abs/2502.09341) is up on [arxiv](https://arxiv.org/abs/2502.09341).

</details>

---

## 💡 What you can build with Seahorse STPP [[Back&nbsp;to&nbsp;Top](#top)]

Seahorse provides a flexible framework to model complex event dynamics. You can build applications for:

- **Epidemiology:** Modeling the spread of infectious diseases over time and regions.
- **Seismology:** Forecasting earthquake occurrences and aftershock propagation.
- **Urban Mobility:** Analyzing and predicting bike-sharing demand and traffic flows.
- **Social Networks:** Tracking information diffusion and viral content over time.
- **Finance:** Modeling high-frequency trading events and market shocks.

---

## ✨ Features [[Back&nbsp;to&nbsp;Top](#top)]

- **Unified Python API**: Train, evaluate, and sample any model through one consistent interface (`STPPRunner`).
- **YAML-driven config**: Every hyperparameter is declarative; experiments are fully reproducible.
- **Plug-and-play presets**: Switch models with `--preset auto_stpp` — no code changes required.
- **Ray Tune HPO**: YAML search-space files feed directly into distributed hyperparameter sweeps.
- **Benchmark campaigns**: Multi-preset × multi-dataset × multi-seed runs with a single CLI command.
- **Data contract**: `Benchmark` enforces identical train/val/test splits across all presets so NLL scores are directly comparable.
- **HuggingFace datasets**: Download and cache any JSONL dataset from the Hub with `--dataset owner/repo`; the released benchmark datasets load by short name with pinned revisions and hash verification (`seahorse.data.load_released_dataset`).
- **Exact evaluation and calibration**: compensator-based next-event NLL, time-rescaling PIT, CRPS, HPD-PIT and energy scores (`seahorse.evaluation.v1`), plus a cross-route NLL consistency test for every exact preset.
- **Frozen campaigns**: `seahorse campaign` compiles, verifies, runs and reports immutable multi-seed campaigns (Slurm dispatch included).

---

## 🚀 Quick Start [[Back&nbsp;to&nbsp;Top](#top)]

For more detailed guides, check out our documentation:

- [Getting Started](https://yahyaaalaila.github.io/seahorse/getting-started/)
- [Why Seahorse?](https://yahyaaalaila.github.io/seahorse/why-seahorse/)
- [End-to-End Case Study](https://yahyaaalaila.github.io/seahorse/examples/case-study/)
- [Predicting with Models](https://yahyaaalaila.github.io/seahorse/run-models/predict/)

**Install** (from PyPI):

```bash
pip install seahorse-stpp
```

Optional extras: `pip install "seahorse-stpp[hpo]"` for Ray Tune HPO.

<details>
<summary>Development install from source</summary>

*macOS / Linux:*

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

*Windows:*

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

</details>

**Python API**

```python
from seahorse import AutoSTPP, PoissonGMM, load_jsonl

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
from seahorse import STPPRunner

runner = STPPRunner.from_preset("auto_stpp")
result = runner.fit(train, val, test)   # returns RunResult
runner.save("/tmp/my_run/")

runner2 = STPPRunner.load("/tmp/my_run/")   # reloads config, normalisation and weights
# post-fit metrics: python -m seahorse evaluate metrics --run /tmp/my_run --test dataset_root/test.jsonl
```

---

## 🤖 Supported models [[Back&nbsp;to&nbsp;Top](#top)]

Our package includes the following state-of-the-art STPP models:

| No | Venue      | Preset                                  | Paper                                                                                            | Implementation |
| :-: | ---------- | --------------------------------------- | ------------------------------------------------------------------------------------------------ | :------------: |
| 1 | NeurIPS'23 | `auto_stpp`                           | [Automatic Integration for Spatiotemporal Neural Point Processes](https://arxiv.org/abs/2310.01179) |    PyTorch    |
| 2 | L4DC'22    | `deep_stpp`                           | [Deep Spatiotemporal Point Process](https://proceedings.mlr.press/v168/lin22a.html)                 |    PyTorch    |
| 3 | ICLR'21    | `neural_jumpcnf` / `neural_attncnf` | [Neural Spatio-Temporal Point Processes](https://openreview.net/forum?id=XQQA6-So14)                |    PyTorch    |
| 4 | ICLR'21    | `njsde` (alias `neural_stpp_gmm`)     | Neural Spatio-Temporal Point Processes, jump-ODE state model with the conditional-GMM spatial decoder (the preset id is historical; this is not Jia & Benson's Neural Jump SDE) |    PyTorch    |
| 5 | ACM KDD'23 | `diffusion_stpp`                      | [Spatio-temporal Diffusion Point Processes](https://dl.acm.org/doi/10.1145/3580305.3599511)         |    PyTorch    |
| 6 | ICLR'22    | `nsmpp`                               | [Neural Spectral Marked Point Processes](https://openreview.net/forum?id=0rcbOaoBXbg)               |    PyTorch    |
| 7 | Arxiv      | `smash`                               | [Embedding Event History to Vector](https://arxiv.org/abs/2310.19324)                               |    PyTorch    |
| 8 | ICML'20    | `thp_gmm`                             | [Transformer Hawkes Process](https://arxiv.org/abs/2002.09291)                                      |    PyTorch    |
| 9 | KDD'16     | `rmtpp_gmm`                           | [Recurrent Marked Temporal Point Processes](https://dl.acm.org/doi/10.1145/2939672.2939875)         |    PyTorch    |

**Parametric baselines** (fast, exact likelihood):
`poisson_gmm` · `hawkes_gmm` · `selfcorrecting_gmm` · `poisson_cnf` · `hawkes_cnf` · `selfcorrecting_cnf` · `poisson_tvcnf` · `hawkes_tvcnf` · `selfcorrecting_tvcnf` · `static_hawkes` · `gaussian_poisson`

**Statistical presets** (NumPy/SciPy, CPU, no gradient training):
`homogeneous_poisson` · `spatial_poisson` · `seasonal_poisson` · `hawkes_exp_gaussian` · `hawkes_omori_pareto`

**Available, not benchmarked in the paper:** `dnsk` (DNSK+Barrier, Dong et al. ICLR'23) · `glide` (GLIDE, diffusion objective, no point-process NLL).

NSMPP and THP-GMM evaluate their density directly with a numerical-quadrature compensator; DSTPP's benchmark NLL is a sample-KDE estimate. See the [capability matrix](https://yahyaaalaila.github.io/seahorse/model-capability-matrix/).

---

## 📊 Datasets [[Back&nbsp;to&nbsp;Top](#top)]

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

The paper's datasets are released under the [`seahorse-stpp`](https://huggingface.co/seahorse-stpp) Hugging Face organisation and load by short name with pinned revisions and card-hash verification:

```python
from seahorse.data import load_released_dataset
splits = load_released_dataset("austin311")      # {"train": [...], "val": [...], "test": [...], "__provenance__": [...]}
```

- **Real-data benchmark** — `comcat_california`, `geonet_nz`, `austin311`, `nyc311`, `nyc_collisions`, `chicago_crime`, `la_crime`, `uber_nyc`, plus the script-only `seattle_fire911` and `citibike_nyc` (rebuilt locally from the repository scripts, then verified).
- **Synthetic controls** — `synthetic` (homogeneous Poisson, moving seasonal Poisson, gamma renewal).
- **Legacy Neural-STPP splits** — `earthquakes` ([USGS](https://earthquake.usgs.gov/)), `covid` ([New York Times](https://github.com/nytimes/covid-19-data)), `citibike_legacy` (script-only).

Any other JSONL dataset on the [HuggingFace Hub](https://huggingface.co/datasets) loads via `--dataset owner/repo`.

For a complete catalog of available datasets, please visit the [Datasets Documentation](https://yahyaaalaila.github.io/seahorse/datasets/catalog/).

---

## 💻 CLI [[Back&nbsp;to&nbsp;Top](#top)]

**Fit one model**

```bash
python -m seahorse fit \
  --preset auto_stpp \
  --train dataset_root/train.jsonl \
  --val   dataset_root/val.jsonl \
  --test  dataset_root/test.jsonl \
  --out   runs/quickstart
```

**Benchmark campaign** (multi-preset × multi-seed)

```bash
python -m seahorse bench \
  --presets auto_stpp deep_stpp njsde poisson_gmm \
  --splits_dir splits/ \
  --seeds 1 2 3 \
  --out runs/bench \
  --n_workers 4
```

**HPO sweep**

```bash
python -m seahorse tune \
  --config configs/auto_stpp_hpo.yaml \
  --train dataset_root/train.jsonl \
  --val   dataset_root/val.jsonl \
  --out   runs/hpo/best.yaml
```

The search space lives in the YAML's `tuning:` block (`{min: .., max: ..}` ranges and choice lists).

---

## 📝 Citation [[Back&nbsp;to&nbsp;Top](#top)]

If Seahorse supports your work, please cite the paper:

```bibtex
@misc{aalaila2026seahorse,
  title         = {Seahorse: A Unified Benchmarking Framework for Spatiotemporal Event Modeling},
  author        = {Aalaila, Yahya and Gro{\ss}mann, Gerrit and Vollmer, Sebastian},
  year          = {2026},
  eprint        = {2607.01022},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

To cite the software release specifically, use the Zenodo DOI:

```bibtex
@software{seahorse2026,
  title   = {Seahorse: Unified Benchmarking for Spatio-Temporal Point Processes},
  author  = {Aalaila, Yahya and Gro{\ss}mann, Gerrit and Vollmer, Sebastian},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.21078077},
  url     = {https://github.com/YahyaAalaila/seahorse},
  license = {Apache-2.0}
}
```

GitHub's **Cite this repository** button (from [`CITATION.cff`](CITATION.cff)) offers these entries in APA and BibTeX.

---

## ⚖️ License [[Back&nbsp;to&nbsp;Top](#top)]

Seahorse is distributed under the **Apache License 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

**Licensing.** Apache-2.0, except the CNF spatial decoder (`seahorse/models/spatial_models/independent_cnf.py`), which is adapted from [neural_stpp](https://github.com/facebookresearch/neural_stpp) and remains under CC BY-NC 4.0 (non-commercial). The vendored Hawkes source under `seahorse/models/statistical/_hawkes_source/` is MIT.

---

## 🙏 Acknowledgment [[Back&nbsp;to&nbsp;Top](#top)]

**Funding.** The authors received support from the Bundesministerium für Bildung und Forschung (BMBF) under Grant No. 01W23005 for the project EVENTFUL.

**Contributors.** Beyond the authors, we thank Ismail Drief and Raphael Sonabend-Friend for their contributions to the codebase and its early design.

**Upstream implementations.** Seahorse builds on the original implementations of the paper families it wraps. We thank the authors of AutoSTPP, DeepSTPP, Neural STPP, DiffusionSTPP, NSMPP, SMASH, THP, and RMTPP for releasing their code.
