# End-to-End Case Study

A short end-to-end example on real data. The notebook installs Seahorse from PyPI,
loads the **NYC motor-vehicle collisions** dataset, explores it on a map, trains a
baseline and a neural model, scores both under one metric, and visualizes the
predictions — on CPU, in a few minutes.

<a class="sh-colab-cta" href="https://colab.research.google.com/github/YahyaAalaila/seahorse/blob/main/docs/notebooks/03_case_study_nyc_collisions.ipynb" target="_blank" rel="noopener">
  <span class="sh-colab-badge" aria-hidden="true">▶</span>
  <span class="sh-colab-text"><strong>Open the case study in Colab</strong><span>NYC collisions · CPU-only · no local setup</span></span>
  <span class="sh-colab-go" aria-hidden="true">↗</span>
</a>

## What the notebook walks through

<ol class="sh-ext-steps" markdown="0">
  <li class="sh-ext-step">
    <span class="sh-ext-num">1</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Install and load real data</span>
      <p><code>pip install seahorse-stpp</code>, then <code>load_dataset("nyc_collisions")</code> pulls the splits straight from the Hugging Face hub at their pinned release revision — no manual download.</p>
    </div>
  </li>
  <li class="sh-ext-step">
    <span class="sh-ext-num">2</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Explore it like a practitioner</span>
      <p>Inspect events per sequence, the daily rhythm of collisions, and a map of collision density across New York City — the spatial pattern a model has to capture.</p>
    </div>
  </li>
  <li class="sh-ext-step">
    <span class="sh-ext-num">3</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Train a baseline and a neural model</span>
      <p>Fit <code>PoissonGMM</code> as a fast parametric baseline and <code>DeepSTPP</code> as the neural model — the same <code>fit()</code> call for both.</p>
    </div>
  </li>
  <li class="sh-ext-step">
    <span class="sh-ext-num">4</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Evaluate head-to-head</span>
      <p>Score both on held-out test sequences and compare <code>test_nll</code> under one shared metric definition — comparable by construction.</p>
    </div>
  </li>
  <li class="sh-ext-step">
    <span class="sh-ext-num">5</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Visualize on the map</span>
      <p>Sample next events with <code>predict_next</code> and plot the model's predicted collision locations against the true next event, over the NYC map.</p>
    </div>
  </li>
</ol>

## Why this dataset

- **Real data** — NYC collisions is one of the lightest real datasets in the [catalog](../datasets/catalog.md) (under 10 MB), so it runs on a free CPU runtime.
- **Comparable by construction** — the baseline and the neural model share the same splits, normalization, and metric definition.
- **No local setup** — the notebook installs the published package; nothing to clone or configure.

## Then go deeper

- [Dataset Catalog](../datasets/catalog.md) — swap NYC collisions for any released dataset.
- [Run a Benchmark](run-a-small-benchmark.md) — compare many presets and seeds.
- [Python API](../python-api.md) — the full programmatic surface behind the notebook.
