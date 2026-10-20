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
reference still works everywhere a short name does. Cards marked **paper** are the
thirteen datasets of the Seahorse benchmark; **legacy** marks the three
Neural-STPP benchmark splits kept for comparison with earlier work. Two
repositories are **script-only**: their source licences do not permit hosting the
data, so they ship a rebuild script and the expected hashes instead.

## Moving through the city

<p class="sh-ds-eyebrow">space · docks and pickup points</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/nyc_bikeshare_2019_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">NYC bike share · 2019</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">script-only</span></span>
  <code class="sh-ds-id">citibike_nyc</code>
  <span class="sh-ds-desc">Bike-share ride starts across New York City in 2019; rebuilt locally from the operator's trip files.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>dock coordinates</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>ride start</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/uber_pickups_nyc_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Uber pickups · NYC</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">uber_nyc</code>
  <span class="sh-ds-desc">Ride-hailing pickup requests across New York City (2014).</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>pickup location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>request time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/nyc_bikeshare_legacy_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">NYC bike share · legacy split</span><span class="sh-ds-chip sh-ds-chip--muted">legacy</span><span class="sh-ds-chip sh-ds-chip--muted">script-only</span></span>
  <code class="sh-ds-id">citibike_legacy</code>
  <span class="sh-ds-desc">The Neural-STPP bike-share benchmark (times in days), reconstructed by a rebuild script.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>dock coordinates</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>ride start</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/us_accidents_stpp" target="_blank" rel="noopener">
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
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/chicago_crime_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Chicago crime</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">chicago_crime</code>
  <span class="sh-ds-desc">Reported crime incidents across the city of Chicago.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>block location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>incident time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/la_crime_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">LA crime</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">la_crime</code>
  <span class="sh-ds-desc">Reported crime incidents across Los Angeles.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>report location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>incident time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/nyc_collisions_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">NYC motor-vehicle collisions</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">nyc_collisions</code>
  <span class="sh-ds-desc">Police-reported motor-vehicle collisions across New York City.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>crash location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>crash time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/seattle_fire911_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Seattle fire 911</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">script-only</span></span>
  <code class="sh-ds-id">seattle_fire911</code>
  <span class="sh-ds-desc">Fire-department 911 dispatches in Seattle; rebuilt locally from the city's open-data export.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>dispatch address</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>dispatch time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/austin_311_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Austin 311</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">austin311</code>
  <span class="sh-ds-desc">Non-emergency city service requests in Austin, Texas.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>request location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>request time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/nyc311_stpp" target="_blank" rel="noopener">
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
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/comcat_california_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">ComCat · California</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">comcat_california</code>
  <span class="sh-ds-desc">USGS ANSS ComCat earthquake catalog for California, native time stamps.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>epicenter</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>origin time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/geonet_nz_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">GeoNet · New Zealand</span><span class="sh-ds-chip">paper</span></span>
  <code class="sh-ds-id">geonet_nz</code>
  <span class="sh-ds-desc">GeoNet earthquake catalog for New Zealand.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>epicenter</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>origin time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/earthquakes-stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Earthquakes · legacy split</span><span class="sh-ds-chip sh-ds-chip--muted">legacy</span></span>
  <code class="sh-ds-id">earthquakes</code>
  <span class="sh-ds-desc">The Neural-STPP earthquake benchmark split (USGS); some test windows overlap training windows in time.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>epicenter</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>origin time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/us_wildfires_stpp" target="_blank" rel="noopener">
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
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/covid-stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">COVID-19 · legacy split</span><span class="sh-ds-chip sh-ds-chip--muted">legacy</span></span>
  <code class="sh-ds-id">covid</code>
  <span class="sh-ds-desc">Reported COVID-19 cases over space and time (The New York Times); the Neural-STPP split, whose validation and test windows lie inside the training period.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>case location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>report date</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

## Beyond the map

<p class="sh-ds-eyebrow">space · not a map at all</p>

<div class="sh-ds-feature" markdown="0">
<a class="sh-ds-card sh-ds-card--abstract" href="https://huggingface.co/datasets/seahorse-stpp/bold5000-stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">BOLD5000</span><span class="sh-ds-chip sh-ds-chip--muted">non-2D</span><span class="sh-ds-chip sh-ds-chip--muted">not benchmarked</span></span>
  <code class="sh-ds-id">bold5000</code>
  <span class="sh-ds-desc">Eventized neural responses derived from BOLD5000 fMRI recordings — included to show STPP space need not be geographic.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>the brain</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>scan time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<div class="sh-ds-aside">
  <span class="sh-ds-aside-title">Supported — but not in the benchmark</span>
  <p>BOLD5000, US wildfires and US traffic accidents ship as fully supported datasets. The paper's comparison covers the thirteen <strong>paper</strong> datasets above and the three <strong>legacy</strong> Neural-STPP splits, under one evaluation protocol.</p>
</div>
</div>

## Synthetic controls

<p class="sh-ds-eyebrow">space · whatever you configure it to be</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--synth sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/seahorse_synthetic_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Seahorse synthetic controls</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--synth">synthetic</span></span>
  <code class="sh-ds-id">synthetic</code>
  <span class="sh-ds-desc">Three generators with analytic ground truth — <code>homogeneous_poisson</code>, <code>moving_seasonal_poisson</code>, <code>gamma_renewal</code> — selected with <code>config=</code>.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>unit square</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>known process</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

When you need known ground truth — to isolate one factor that real data
confounds — Seahorse also uses synthetic sequences from
[HawkesNest](https://github.com/YahyaAalaila/HawkesNest).

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--synth sh-ds-card--wide" href="https://github.com/YahyaAalaila/HawkesNest" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">HawkesNest · entanglement suite</span><span class="sh-ds-chip sh-ds-chip--synth">synthetic</span></span>
  <code class="sh-ds-id">github.com/YahyaAalaila/HawkesNest</code>
  <span class="sh-ds-desc">Spatio-temporal sequences generated with tunable <em>entanglement</em> between the space and time dimensions, under known ground truth — so a model's behaviour can be probed against a factor that real-world data cannot cleanly separate.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>configurable domain</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>configurable process</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

---

Want to add your own? See [Add Your Dataset](add-dataset.md) for the preparation
checklist and [Conversion Standard](conversion.md) for the JSONL format.
