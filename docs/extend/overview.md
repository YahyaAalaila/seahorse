# Bring your model to Seahorse

Every model in Seahorse has the **same shape**. Express your freshly built STPP
model that way once, and it plugs into the entire harness — training, multi-seed
benchmarks, every metric profile, and sampling — with no other wiring.

<div class="sh-ext-contract" markdown="0">
  <span class="sh-ext-wrap">UnifiedSTPP</span>
  <div class="sh-ext-boxes">
    <div class="sh-ext-mod sh-ext-mod--state">
      <span class="sh-ext-mod-name">StateModel</span>
      <span class="sh-ext-mod-flow">event history → hidden state</span>
      <span class="sh-ext-mod-api"><code>encode</code><code>evolve</code></span>
    </div>
    <span class="sh-ext-join" aria-hidden="true">→</span>
    <div class="sh-ext-mod sh-ext-mod--event">
      <span class="sh-ext-mod-name">EventModel</span>
      <span class="sh-ext-mod-flow">hidden state → log-probability</span>
      <span class="sh-ext-mod-api"><code>log_prob</code><code>sample_next</code><code>intensity_grid</code></span>
    </div>
  </div>
  <span class="sh-ext-down" aria-hidden="true">▼</span>
  <div class="sh-ext-out">
    <span class="sh-ext-out-k">you get, for free</span>
    <span class="sh-ext-pill">fit</span>
    <span class="sh-ext-pill">bench · multi-seed</span>
    <span class="sh-ext-pill">evaluate · all profiles</span>
    <span class="sh-ext-pill">predict_next</span>
    <span class="sh-ext-pill">CLI + Python API</span>
  </div>
</div>

Your model is comparable to every baseline by construction: the same data
contract, the same normalization, the same metric definitions apply the moment
it is registered.

## Four steps from `nn.Module` to benchmarked

<ol class="sh-ext-steps" markdown="0">
  <li class="sh-ext-step">
    <span class="sh-ext-num">1</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Wrap your model as two parts</span>
      <p>Split it into a <strong>StateModel</strong> (encodes history into a hidden state) and an <strong>EventModel</strong> (turns that state into a log-likelihood). Both are plain <code>nn.Module</code>s.</p>
      <code class="sh-ext-code">class MyEventModel(nn.Module): def log_prob(self, times, locs, state, mask): ...</code>
      <a class="sh-ext-more" href="../wrap/">Wrap an existing model →</a>
    </div>
  </li>
  <li class="sh-ext-step">
    <span class="sh-ext-num">2</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Wire them in a config</span>
      <p>A <strong>ModelFamilyConfig</strong> owns the build-time parameters and returns a wired <code>UnifiedSTPP</code> from <code>build_model()</code>.</p>
      <code class="sh-ext-code">def build_model(self): return UnifiedSTPP(state, event, hidden_dim=self.hidden_dim)</code>
      <a class="sh-ext-more" href="../preset/">Register a preset →</a>
    </div>
  </li>
  <li class="sh-ext-step">
    <span class="sh-ext-num">3</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Register a preset name</span>
      <p>One decorator makes your model resolve through <code>fit</code>, <code>bench</code>, <code>evaluate</code>, and the Python API — no import-path changes anywhere else.</p>
      <code class="sh-ext-code">@ConfigRegistry.register("my_preset")</code>
      <a class="sh-ext-more" href="../../adding-a-model/">Full checklist →</a>
    </div>
  </li>
  <li class="sh-ext-step">
    <span class="sh-ext-num">4</span>
    <div class="sh-ext-step-body">
      <span class="sh-ext-step-title">Declare what it can do</span>
      <p>The methods you implement decide which metrics run. An unimplemented capability is reported as a clean skip — never a silent wrong number.</p>
      <code class="sh-ext-code">def sample_next(self, state, t_last, n_samples=1): ...  # unlocks predictive metrics</code>
      <a class="sh-ext-more" href="../capabilities/">Declare capabilities →</a>
    </div>
  </li>
</ol>

!!! tip "Then test and ship"
    Run a tiny one-epoch fit, an `evaluate`, and a save/load round-trip before
    adding the preset to benchmark examples. The
    [Testing Checklist](testing.md) is the short list to clear.

## Pick your path

<div class="sh-ext-paths" markdown="0">
  <a class="sh-ext-path" href="../wrap/">
    <span class="sh-ext-path-when">I already have a PyTorch model</span>
    <span class="sh-ext-path-title">Wrap an existing model</span>
    <span class="sh-ext-path-desc">The StateModel / EventModel adapter pattern, with minimal working stubs.</span>
    <span class="sh-ext-path-go" aria-hidden="true">→</span>
  </a>
  <a class="sh-ext-path" href="../../adding-a-model/">
    <span class="sh-ext-path-when">I'm building a new family from scratch</span>
    <span class="sh-ext-path-title">Add a model</span>
    <span class="sh-ext-path-desc">The end-to-end checklist, from components to a bundled preset.</span>
    <span class="sh-ext-path-go" aria-hidden="true">→</span>
  </a>
  <a class="sh-ext-path" href="../preset/">
    <span class="sh-ext-path-when">My components are ready to expose</span>
    <span class="sh-ext-path-title">Register a preset</span>
    <span class="sh-ext-path-desc">Config registry, YAML defaults, and CLI + Python API wiring.</span>
    <span class="sh-ext-path-go" aria-hidden="true">→</span>
  </a>
  <a class="sh-ext-path" href="../capabilities/">
    <span class="sh-ext-path-when">I'm deciding which metrics apply</span>
    <span class="sh-ext-path-title">Declare capabilities</span>
    <span class="sh-ext-path-desc">Map implemented methods to evaluation profiles and exact-vs-approximate NLL.</span>
    <span class="sh-ext-path-go" aria-hidden="true">→</span>
  </a>
</div>

## See also

- [Architecture](../architecture.md) — the full model-layer documentation.
- [Model Capability Matrix](../model-capability-matrix.md) — what each shipped preset declares.
