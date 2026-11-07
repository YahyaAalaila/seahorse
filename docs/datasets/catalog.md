# Dataset Catalog

The datasets of the Seahorse benchmark are hosted on the
[`seahorse-stpp`](https://huggingface.co/seahorse-stpp) Hugging Face organization
and load by a short name. They share one trait — events observed in **space and
time** — but the meaning of "space" stretches from a city block to the human
cortex. The catalog is ordered along that arc.

Every dataset loads the same way, by the short name shown on each card. The
repositories are pinned to their release revision and every split is verified
against the hashes on the dataset card:

```python
from seahorse.data import load_dataset, load_released_dataset

splits = load_dataset("covid")                       # {"train": [...], "val": [...], "test": [...]}
splits = load_released_dataset("austin311")          # same, with hash verification and provenance
native = load_released_dataset("austin311", config="native")
```

…or from the command line with `--dataset covid`. The full `seahorse-stpp/<id>`
reference still works everywhere a short name does. The paper benchmarks **13 real
datasets**: cards marked **paper** are the ten with chronological splits, and cards
marked **Neural STPP split** are the three datasets of the Neural STPP benchmark
(COVID, Earthquakes, Citi Bike daily), run under their original splits. The
HawkesNest entanglement suite is the paper's synthetic stress test; **not
benchmarked** marks released datasets outside the paper. Three repositories are
**script-only**: the source licence (or, for Seattle, the content) does not permit
hosting the data, so they ship a rebuild script and the expected hashes instead.
Each dataset also has a [card](index.md) with plots and split statistics.

## Moving through the city

<p class="sh-ds-eyebrow">space · docks and pickup points</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--bench" href="../cards/citibike_nyc/">
  <span class="sh-ds-head"><span class="sh-ds-name">NYC bike share · 2019 (hourly)</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">script-only</span></span>
  <code class="sh-ds-id">citibike_nyc</code>
  <span class="sh-ds-desc">Bike-share ride starts across New York City in 2019; rebuilt locally from the operator's trip files.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>dock coordinates</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>ride start</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="../cards/uber_nyc/">
  <span class="sh-ds-head"><span class="sh-ds-name">Uber pickups · NYC</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">uber_nyc</code>
  <span class="sh-ds-desc">Ride-hailing pickup requests across New York City (2014).</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>pickup location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>request time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="../cards/citibike_legacy/">
  <span class="sh-ds-head"><span class="sh-ds-name">NYC bike share · Neural STPP split</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">Neural STPP split</span><span class="sh-ds-chip sh-ds-chip--muted">script-only</span></span>
  <code class="sh-ds-id">citibike_legacy</code>
  <span class="sh-ds-desc">The Neural-STPP bike-share benchmark (times in days), reconstructed by a rebuild script.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>dock coordinates</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>ride start</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="../cards/us_accidents/">
  <span class="sh-ds-head"><span class="sh-ds-name">US traffic accidents</span><span class="sh-ds-chip sh-ds-chip--muted">not benchmarked</span></span>
  <code class="sh-ds-id">us_accidents</code>
  <span class="sh-ds-desc">Road traffic accident reports across the United States.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>accident location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>report time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

## Incidents &amp; city services

<p class="sh-ds-eyebrow">space · where something was reported</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--bench" href="../cards/chicago_crime/">
  <span class="sh-ds-head"><span class="sh-ds-name">Chicago crime</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">chicago_crime</code>
  <span class="sh-ds-desc">Reported crime incidents across the city of Chicago.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>block location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>incident time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="../cards/la_crime/">
  <span class="sh-ds-head"><span class="sh-ds-name">LA crime</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">la_crime</code>
  <span class="sh-ds-desc">Reported crime incidents across Los Angeles.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>report location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>incident time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="../cards/nyc_collisions/">
  <span class="sh-ds-head"><span class="sh-ds-name">NYC motor-vehicle collisions</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">nyc_collisions</code>
  <span class="sh-ds-desc">Police-reported motor-vehicle collisions across New York City.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>crash location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>crash time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="../cards/seattle_fire911/">
  <span class="sh-ds-head"><span class="sh-ds-name">Seattle fire 911</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">script-only</span></span>
  <code class="sh-ds-id">seattle_fire911</code>
  <span class="sh-ds-desc">Fire-department 911 dispatches in Seattle; rebuilt locally from the city's open-data export.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>dispatch address</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>dispatch time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="../cards/austin311/">
  <span class="sh-ds-head"><span class="sh-ds-name">Austin 311</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">austin311</code>
  <span class="sh-ds-desc">Non-emergency city service requests in Austin, Texas.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>request location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>request time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="../cards/nyc311/">
  <span class="sh-ds-head"><span class="sh-ds-name">NYC 311</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">nyc311</code>
  <span class="sh-ds-desc">Non-emergency service requests across New York City.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>request location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>request time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

## Earth &amp; environment

<p class="sh-ds-eyebrow">space · latitude and longitude on the globe</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--bench" href="../cards/comcat_california/">
  <span class="sh-ds-head"><span class="sh-ds-name">ComCat · California</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">comcat_california</code>
  <span class="sh-ds-desc">USGS ANSS ComCat earthquake catalog for California, native time stamps.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>epicenter</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>origin time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="../cards/geonet_nz/">
  <span class="sh-ds-head"><span class="sh-ds-name">GeoNet · New Zealand</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">geonet_nz</code>
  <span class="sh-ds-desc">GeoNet earthquake catalog for New Zealand.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>epicenter</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>origin time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="../cards/earthquakes/">
  <span class="sh-ds-head"><span class="sh-ds-name">Earthquakes · Neural STPP split</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">Neural STPP split</span></span>
  <code class="sh-ds-id">earthquakes</code>
  <span class="sh-ds-desc">The Neural-STPP earthquake benchmark split (USGS); some test windows overlap training windows in time.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>epicenter</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>origin time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="../cards/us_wildfires/">
  <span class="sh-ds-head"><span class="sh-ds-name">US wildfires</span><span class="sh-ds-chip sh-ds-chip--muted">not benchmarked</span></span>
  <code class="sh-ds-id">us_wildfires</code>
  <span class="sh-ds-desc">Wildfire ignition records across the United States.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>ignition location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>discovery date</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

## Population health

<p class="sh-ds-eyebrow">space · case geography</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card" href="../cards/covid/">
  <span class="sh-ds-head"><span class="sh-ds-name">COVID-19 · Neural STPP split</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">Neural STPP split</span></span>
  <code class="sh-ds-id">covid</code>
  <span class="sh-ds-desc">Reported COVID-19 cases over space and time (The New York Times); the Neural-STPP split, whose validation and test windows lie inside the training period.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>case location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>report date</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

## Beyond the map

<p class="sh-ds-eyebrow">space · not a map at all</p>

<div class="sh-ds-feature" markdown="0">
<a class="sh-ds-card sh-ds-card--abstract" href="../cards/bold5000/">
  <span class="sh-ds-head"><span class="sh-ds-name">BOLD5000</span><span class="sh-ds-chip sh-ds-chip--muted">non-2D</span><span class="sh-ds-chip sh-ds-chip--muted">not benchmarked</span></span>
  <code class="sh-ds-id">bold5000</code>
  <span class="sh-ds-desc">Eventized neural responses derived from BOLD5000 fMRI recordings — included to show STPP space need not be geographic.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>the brain</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>scan time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<div class="sh-ds-aside">
  <span class="sh-ds-aside-title">Supported — but not in the benchmark</span>
  <p>BOLD5000, US wildfires, US traffic accidents and the synthetic controls are released and load like every other dataset. The paper's comparison covers the 13 real datasets above (ten with chronological splits, three under the Neural STPP splits) and the HawkesNest entanglement suite, under one evaluation protocol.</p>
</div>
</div>

## Synthetic data

<p class="sh-ds-eyebrow">space · whatever you configure it to be</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--synth sh-ds-card--bench" href="../cards/synthetic/">
  <span class="sh-ds-head"><span class="sh-ds-name">Seahorse synthetic controls</span><span class="sh-ds-chip sh-ds-chip--synth">synthetic</span><span class="sh-ds-chip sh-ds-chip--muted">not benchmarked</span></span>
  <code class="sh-ds-id">synthetic</code>
  <span class="sh-ds-desc">Three generators with analytic ground truth — <code>homogeneous_poisson</code>, <code>moving_seasonal_poisson</code>, <code>gamma_renewal</code> — selected with <code>config=</code>.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>unit square</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>known process</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

When you need known ground truth, to isolate one factor that real data
confounds, the paper uses the entanglement suite generated with
[HawkesNest](https://github.com/YahyaAalaila/HawkesNest).

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--synth sh-ds-card--wide sh-ds-card--bench" href="../cards/hawkesnest_entanglement/">
  <span class="sh-ds-head"><span class="sh-ds-name">HawkesNest · entanglement suite</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--synth">synthetic</span></span>
  <code class="sh-ds-id">hawkesnest_entanglement</code>
  <span class="sh-ds-desc">A spatiotemporal Hawkes process on the unit square whose triggering kernel drifts in space with the age of the parent event (σ = 0.03, β = 0.3, drift v = 0, 0.1, 0.2, 0.3 for levels L0–L3), with fixed background and branching ratio and the true intensity stored, so a model's behaviour can be probed against a coupling that real data cannot isolate.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>unit square</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>known process</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

---

Want to add your own? See [Add Your Dataset](add-dataset.md) for the preparation
checklist and [Conversion Standard](conversion.md) for the JSONL format.
