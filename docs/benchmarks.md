# Benchmark

<p class="sh-ds-eyebrow">common datasets · shared splits · one evaluation protocol</p>

Once a model is registered, it can be evaluated against every baseline on shared
datasets under a common protocol. The headline comparison uses per-event negative
log-likelihood (NLL), and the predictive and distributional metric profiles round
out the picture, which matters for generative models that do not admit an exact
likelihood. The command below runs your preset alongside several baselines and
writes a comparable table.

<div class="sh-bench-board" markdown="0">
  <div class="sh-bench-matrix">
    <div class="sh-bench-row sh-bench-row--head">
      <span class="sh-bench-model sh-bench-model--head">preset</span>
      <span class="sh-bench-col">COVID</span>
      <span class="sh-bench-col">Earthquakes</span>
      <span class="sh-bench-col">Citibike</span>
    </div>
    <div class="sh-bench-row sh-bench-row--you">
      <span class="sh-bench-model"><span class="sh-bench-newchip">your model</span>my_preset</span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
    </div>
    <div class="sh-bench-row sh-bench-row--exact">
      <span class="sh-bench-model">auto_stpp</span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
    </div>
    <div class="sh-bench-row sh-bench-row--exact">
      <span class="sh-bench-model">deep_stpp</span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
    </div>
    <div class="sh-bench-row sh-bench-row--exact">
      <span class="sh-bench-model">poisson_gmm</span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
    </div>
    <div class="sh-bench-row sh-bench-row--approx">
      <span class="sh-bench-model">smash</span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
    </div>
    <div class="sh-bench-row sh-bench-row--approx">
      <span class="sh-bench-model">diffusion_stpp</span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
      <span class="sh-bench-cell"><span class="sh-bench-dot"></span></span>
    </div>
  </div>
  <div class="sh-bench-legend">
    <span><span class="sh-bench-rail sh-bench-rail--exact"></span>exact NLL · directly comparable</span>
    <span><span class="sh-bench-rail sh-bench-rail--bound"></span>approximate NLL · a bound (ELBO, score-matching)</span>
    <span><span class="sh-bench-keydot"></span>your model</span>
  </div>
</div>

<p class="sh-bench-cap">Each cell is one <code>(preset × dataset × seed)</code> run, trained and evaluated under the same protocol. Results are written to <code>report.html</code> and the NLL tables.</p>

```bash
python -m seahorse bench \
  --presets my_preset auto_stpp deep_stpp poisson_gmm hawkes_gmm \
  --splits_dir splits \
  --seeds 1 2 3 \
  --out runs/bench
```

Replace `my_preset` with your registered name, and point `--splits_dir` at the
core benchmark datasets (COVID, Earthquakes, Citibike). For a first run, the
[small benchmark walkthrough](examples/run-a-small-benchmark.md) is a good place
to start.

## The route

<p class="sh-ds-eyebrow">from a single run to reported results</p>

<div class="sh-ext-paths" markdown="0">
  <a class="sh-ext-path" href="../examples/run-a-small-benchmark/">
    <span class="sh-ext-path-when">first run</span>
    <span class="sh-ext-path-title">Run a small benchmark</span>
    <span class="sh-ext-path-desc">A few presets on one dataset and seed: the walkthrough and the files it produces.</span>
    <span class="sh-ext-path-go" aria-hidden="true">→</span>
  </a>
  <a class="sh-ext-path" href="../benchmark/compare/">
    <span class="sh-ext-path-when">after a campaign</span>
    <span class="sh-ext-path-title">Compare results</span>
    <span class="sh-ext-path-desc">Read the NLL tables and line up models on individual sequences, within a comparable tier.</span>
    <span class="sh-ext-path-go" aria-hidden="true">→</span>
  </a>
  <a class="sh-ext-path" href="../benchmark/tune/">
    <span class="sh-ext-path-when">selecting hyperparameters</span>
    <span class="sh-ext-path-title">Tune hyperparameters</span>
    <span class="sh-ext-path-desc">Ray Tune HPO inside a campaign, with an explicit tuning dataset held out from the comparison.</span>
    <span class="sh-ext-path-go" aria-hidden="true">→</span>
  </a>
  <a class="sh-ext-path" href="../benchmark/artifacts/">
    <span class="sh-ext-path-when">reporting results</span>
    <span class="sh-ext-path-title">Artifacts &amp; reproducibility</span>
    <span class="sh-ext-path-desc">Every run directory, config, and table on disk, so a result reproduces months later.</span>
    <span class="sh-ext-path-go" aria-hidden="true">→</span>
  </a>
</div>

## Comparability

<p class="sh-ds-eyebrow">what every preset shares before its own code runs</p>

Every `bench` run applies a shared contract: all presets receive the same train,
validation, and test splits, the same normalization policy, and a per-event NLL
computed by the same harness. A difference in the table therefore reflects the
models rather than the experimental setup. The
[execution contract](learn/execution-contract.md) records exactly what is held fixed.

Models differ in how they express likelihood. Likelihood-based models report an
exact NLL; generative families such as score-matching and diffusion models report
an approximation, typically a bound on the log-likelihood (for example an ELBO).
Both are informative once the distinction is kept in view, and they remain
comparable when read in that light. Beyond NLL, the predictive and distributional
[metric profiles](evaluation.md) give a fuller account of model quality, which is
especially useful for models whose likelihood is only approximate.

<div class="sh-tier" markdown="0">
  <div class="sh-tier-row sh-tier-row--exact">
    <div class="sh-tier-label"><span class="sh-tier-name">Exact</span><span class="sh-tier-sub">directly comparable</span></div>
    <div class="sh-tier-chips">
      <span class="sh-fam">auto_stpp</span><span class="sh-fam">deep_stpp</span><span class="sh-fam">nsmpp</span><span class="sh-fam">njsde</span><span class="sh-fam">neural_*</span><span class="sh-fam">poisson_*</span><span class="sh-fam">hawkes_*</span><span class="sh-fam">rmtpp_gmm</span><span class="sh-fam">thp_gmm</span>
    </div>
  </div>
  <div class="sh-tier-row sh-tier-row--approx">
    <div class="sh-tier-label"><span class="sh-tier-name">Approximate</span><span class="sh-tier-sub">a bound on the likelihood</span></div>
    <div class="sh-tier-chips">
      <span class="sh-fam">smash <em>· score-matching</em></span><span class="sh-fam">diffusion_stpp <em>· ELBO</em></span>
    </div>
  </div>
</div>

How a model declares its NLL type is covered in
[capabilities](extend/capabilities.md).

## What a campaign leaves behind

Every run can be reproduced from disk. Alongside each fit's config, metrics, and
checkpoint, a campaign writes:

| File | What it's for |
| --- | --- |
| `report.html` | Self-contained comparison report, viewable in a browser |
| `table_test_nll_all.csv` | Test NLL for every cell |
| `table_test_nll_exact.csv` | The exact-NLL tier only |
| `cell_index.json` | Maps each cell to its saved run directory |

Full layout and per-run files: [Artifacts and run directories](benchmark/artifacts.md).

---

If you arrived from [Add your model](extend/overview.md), your model now appears in
the same tables as the established baselines. From here, read the
[comparison](benchmark/compare.md) or
[reproduce the published results](paper-reproduction.md).
