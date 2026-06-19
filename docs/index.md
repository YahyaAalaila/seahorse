---
hide:
  - navigation
  - toc
---
<div id="hero-container">
  <img src="assets/bg.svg" class="parallax-bg" alt="Background pattern" />
  <img src="assets/fg.png" class="parallax-fg" alt="Foreground elements" />

<div class="hero-content">
    <h1 class="hero-title">STPP Easier than before</h1>
    <p class="hero-description">Discover the ultimate framework for modeling Spatial-Temporal Point Processes. Seahorse STPP simplifies complex data workflows, enabling you to build, train, and deploy models effortlessly.</p>
    <div class="hero-buttons">
      <a href="overview/" class="md-button md-button--primary">Getting Started</a>
      <a href="architecture/" class="md-button hero-btn-secondary">Learn More</a>
    </div>
  </div>
</div>

<div class="features-section" markdown="1">
<div class="features-section-inner" markdown="1">

<h2 class="features-title">Everything you would expect</h2>

<div class="features-grid" markdown="1">

<div class="feature-item" markdown="1">
<div class="feature-icon" markdown="1">

:fontawesome-brands-python:

</div>
<div class="feature-content" markdown="1">

<h3>Unified Python API</h3>

Train, evaluate, and sample any model through one consistent interface (STPPRunner).

</div>
</div>

<div class="feature-item" markdown="1">
<div class="feature-icon" markdown="1">

:fontawesome-solid-file-code:

</div>
<div class="feature-content" markdown="1">

<h3>YAML-driven config</h3>

Every hyperparameter is declarative; experiments are fully reproducible.

</div>
</div>

<div class="feature-item" markdown="1">
<div class="feature-icon" markdown="1">

:fontawesome-solid-plug:

</div>
<div class="feature-content" markdown="1">

<h3>Plug-and-play presets</h3>

Switch models with <code>--preset auto_stpp</code> — no code changes required.

</div>
</div>

<div class="feature-item" markdown="1">
<div class="feature-icon" markdown="1">

:fontawesome-solid-network-wired:

</div>
<div class="feature-content" markdown="1">

<h3>Ray Tune HPO</h3>

YAML search-space files feed directly into distributed hyperparameter sweeps.

</div>
</div>

<div class="feature-item" markdown="1">
<div class="feature-icon" markdown="1">

:fontawesome-solid-chart-line:

</div>
<div class="feature-content" markdown="1">

<h3>Benchmark campaigns</h3>

Multi-preset × multi-dataset × multi-seed runs with a single CLI command.

</div>
</div>

<div class="feature-item" markdown="1">
<div class="feature-icon" markdown="1">

:fontawesome-solid-handshake:

</div>
<div class="feature-content" markdown="1">

<h3>Data contract</h3>

Benchmark enforces identical train/val/test splits across all presets so NLL scores are directly comparable.

</div>
</div>

<div class="feature-item" markdown="1">
<div class="feature-icon" markdown="1">

:fontawesome-solid-database:

</div>
<div class="feature-content" markdown="1">

<h3>HuggingFace datasets</h3>

Stream or cache any JSONL dataset directly from the Hub with <code>--dataset owner/repo</code>.

</div>
</div>

</div>
</div>
</div>

<div class="models-section" markdown="1">
<div class="models-section-inner" markdown="1">

<h2 class="models-title">Supported Models</h2>
<p class="models-subtitle">Our package includes the following state-of-the-art STPP models.</p>

<div class="slider-wrapper" markdown="1">
<button class="slider-btn slider-btn-left" aria-label="Previous" onclick="document.getElementById('modelsSlider').scrollBy({left: -340, behavior: 'smooth'})">
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M15.41 16.59 10.83 12l4.58-4.59L14 6l-6 6 6 6 1.41-1.41z"/></svg>
</button>

<div class="models-slider" id="modelsSlider" markdown="1">

<div class="model-card" markdown="1">
<div class="card-icon" markdown="1">:fontawesome-solid-microchip:</div>
<h3 class="model-name">Automatic Integration for Spatiotemporal Neural Point Processes</h3>
<div class="model-footer" markdown="1">
<a href="https://arxiv.org/abs/2310.01179" target="_blank" class="model-link">Read More &rarr;</a>
</div>
</div>

<div class="model-card" markdown="1">
<div class="card-icon" markdown="1">:fontawesome-solid-network-wired:</div>
<h3 class="model-name">Deep Spatiotemporal Point Process</h3>
<div class="model-footer" markdown="1">
<a href="https://proceedings.mlr.press/v168/lin22a.html" target="_blank" class="model-link">Read More &rarr;</a>
</div>
</div>

<div class="model-card" markdown="1">
<div class="card-icon" markdown="1">:fontawesome-solid-code-branch:</div>
<h3 class="model-name">Neural Spatio-Temporal Point Processes</h3>
<div class="model-footer" markdown="1">
<a href="https://openreview.net/forum?id=XQQA6-So14" target="_blank" class="model-link">Read More &rarr;</a>
</div>
</div>

<div class="model-card" markdown="1">
<div class="card-icon" markdown="1">:fontawesome-solid-wave-square:</div>
<h3 class="model-name">Neural Jump Stochastic Differential Equations</h3>
<div class="model-footer" markdown="1">
<a href="https://arxiv.org/abs/1905.10403" target="_blank" class="model-link">Read More &rarr;</a>
</div>
</div>

<div class="model-card" markdown="1">
<div class="card-icon" markdown="1">:fontawesome-solid-chart-area:</div>
<h3 class="model-name">Spatio-temporal Diffusion Point Processes</h3>
<div class="model-footer" markdown="1">
<a href="https://dl.acm.org/doi/10.1145/3580305.3599511" target="_blank" class="model-link">Read More &rarr;</a>
</div>
</div>

<div class="model-card" markdown="1">
<div class="card-icon" markdown="1">:fontawesome-solid-project-diagram:</div>
<h3 class="model-name">Neural Spectral Marked Point Processes</h3>
<div class="model-footer" markdown="1">
<a href="https://openreview.net/forum?id=0rcbOaoBXbg" target="_blank" class="model-link">Read More &rarr;</a>
</div>
</div>

<div class="model-card" markdown="1">
<div class="card-icon" markdown="1">:fontawesome-solid-history:</div>
<h3 class="model-name">Embedding Event History to Vector</h3>
<div class="model-footer" markdown="1">
<a href="https://arxiv.org/abs/2310.19324" target="_blank" class="model-link">Read More &rarr;</a>
</div>
</div>

<div class="model-card" markdown="1">
<div class="card-icon" markdown="1">:fontawesome-solid-robot:</div>
<h3 class="model-name">Transformer Hawkes Process</h3>
<div class="model-footer" markdown="1">
<a href="https://arxiv.org/abs/2002.09291" target="_blank" class="model-link">Read More &rarr;</a>
</div>
</div>

<div class="model-card" markdown="1">
<div class="card-icon" markdown="1">:fontawesome-solid-layer-group:</div>
<h3 class="model-name">Recurrent Marked Temporal Point Processes</h3>
<div class="model-footer" markdown="1">
<a href="https://dl.acm.org/doi/10.1145/2939672.2939875" target="_blank" class="model-link">Read More &rarr;</a>
</div>
</div>

</div>

<button class="slider-btn slider-btn-right" aria-label="Next" onclick="document.getElementById('modelsSlider').scrollBy({left: 340, behavior: 'smooth'})">
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M8.59 16.59 13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z"/></svg>
</button>
</div>

<div class="parametric-baselines" markdown="1">

<h3>Parametric baselines (fast, exact likelihood)</h3>

<code>poisson_gmm</code> &middot; <code>hawkes_gmm</code> &middot; <code>selfcorrecting_gmm</code> &middot; <code>poisson_cnf</code> &middot; <code>hawkes_cnf</code> &middot; <code>selfcorrecting_cnf</code> &middot; <code>poisson_tvcnf</code> &middot; <code>hawkes_tvcnf</code> &middot; <code>selfcorrecting_tvcnf</code>

</div>

</div>
</div>

<div class="contribute-section" markdown="1">
<div class="contribute-inner" markdown="1">

<div class="contribute-left" markdown="1">
<h2 class="contribute-title">Become a Contributor</h2>
<p class="contribute-text">Seahorse thrives on community contributions. Whether you're adding a new state-of-the-art model, optimizing core operations, or improving documentation, we welcome your expertise to help make this project stronger. Join us in advancing Spatiotemporal Point Processes!</p>
<a href="extend/overview/" class="md-button md-button--primary contribute-btn">Extend Seahorse</a>
</div>

<div class="contribute-right" markdown="1">
<h3 class="contribute-subtitle">Let's Keep in Touch</h3>
<p class="contribute-text-small">Connect with <strong>Us</strong> to discuss research, collaboration, or STPPs.</p>
<div class="social-links" markdown="1">
<a href="https://github.com/YahyaAalaila" target="_blank" class="social-link" title="GitHub">
  <span class="social-icon" markdown="span">:fontawesome-brands-github:</span> GitHub
</a>
<a href="https://www.linkedin.com/in/yahya-aalaila-6578b41a6/" target="_blank" class="social-link" title="LinkedIn">
  <span class="social-icon" markdown="span">:fontawesome-brands-linkedin:</span> LinkedIn
</a>
<a href="https://www.researchgate.net/profile/Yahya_Aalaila" target="_blank" class="social-link" title="ResearchGate">
  <span class="social-icon" markdown="span">:fontawesome-brands-researchgate:</span> ResearchGate
</a>
</div>
</div>

</div>
</div>
