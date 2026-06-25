# End-to-End Case Study

Real dataset, real numbers, one click. This is the fastest way to watch Seahorse
do something real: it installs from PyPI, loads the **Citibike** spatio-temporal
dataset, trains a baseline and a neural model, scores both under one metric, and
plots the predictions — all on CPU, in a few minutes.

<a class="sh-colab-cta" href="https://colab.research.google.com/github/YahyaAalaila/seahorse/blob/main/docs/notebooks/01_run_one_model_python_api.ipynb" target="_blank" rel="noopener">
  <span class="sh-colab-badge" aria-hidden="true">▶</span>
  <span class="sh-colab-text"><strong>Open the case study in Colab</strong><span>Citibike · CPU-only · no local setup</span></span>
  <span class="sh-colab-go" aria-hidden="true">↗</span>
</a>

## What the notebook walks through

<ol class="sh-ext-steps" markdown="0">
  <li class="sh-ext-step">
    <span class="sh-ext-num">1</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Install and load real data</span>
      <p><code>pip install seahorse-stpp</code>, then <code>load_dataset("seahorse-stpp/citibike-stpp")</code> pulls the Citibike splits straight from the Hugging Face hub — no manual download.</p>
    </div>
  </li>
  <li class="sh-ext-step">
    <span class="sh-ext-num">2</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Train a baseline and a neural model</span>
      <p>Fit <code>PoissonGMM</code> as a fast parametric baseline and <code>DeepSTPP</code> as the neural model — the same <code>fit()</code> call for both.</p>
    </div>
  </li>
  <li class="sh-ext-step">
    <span class="sh-ext-num">3</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Evaluate head-to-head</span>
      <p>Score both on held-out test sequences and compare <code>test_nll</code> under one shared metric definition — comparable by construction.</p>
    </div>
  </li>
  <li class="sh-ext-step">
    <span class="sh-ext-num">4</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Visualize the predictions</span>
      <p>Sample next events with <code>predict_next</code> and plot the sampled locations against the true next event.</p>
    </div>
  </li>
</ol>

## Why this dataset

- **Real, not synthetic** — Citibike is the lightest real public dataset in the [catalog](../datasets/catalog.md), so it runs on a free CPU runtime.
- **Comparable by construction** — the baseline and the neural model share the same splits, normalization, and metric, so the numbers actually mean something.
- **One click** — the notebook installs the published package; nothing to clone or configure.

## Then go deeper

- [Dataset Catalog](../datasets/catalog.md) — swap Citibike for any of the 13 datasets.
- [Run a Benchmark](run-a-small-benchmark.md) — compare many presets and seeds.
- [Python API](../python-api.md) — the full programmatic surface behind the notebook.
