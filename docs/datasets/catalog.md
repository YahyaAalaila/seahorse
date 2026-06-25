# Dataset Catalog

Thirteen ready-to-use spatio-temporal point-process datasets, hosted on the
[`seahorse-stpp`](https://huggingface.co/seahorse-stpp) Hugging Face organization
and loadable by id. They share one trait — events observed in **space and
time** — but the meaning of "space" stretches from a city block to the human
cortex. The catalog is ordered along that arc.

Every dataset loads the same way:

```python
from seahorse.data import load_dataset

splits = load_dataset("seahorse-stpp/citibike-stpp")  # {"train": [...], "val": [...], "test": [...]}
```

…or from the command line with `--dataset seahorse-stpp/<id>`. Three of them —
marked **core benchmark** below — are the 2D-spatial trio used for the headline
real-data comparison.

## Moving through the city

<p class="sh-ds-eyebrow">space · streets, docks, and pickup points</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/citibike-stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">NYC Citi Bike</span><span class="sh-ds-chip">core benchmark</span></span>
  <code class="sh-ds-id">seahorse-stpp/citibike-stpp</code>
  <span class="sh-ds-desc">Bike-share trips between docking stations across New York City.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>dock coordinates</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>ride start</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/uber_pickups_nyc_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Uber Pickups · NYC</span></span>
  <code class="sh-ds-id">seahorse-stpp/uber_pickups_nyc_stpp</code>
  <span class="sh-ds-desc">Ride-hailing pickup requests across New York City.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>pickup location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>request time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/us_accidents_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">US Traffic Accidents</span></span>
  <code class="sh-ds-id">seahorse-stpp/us_accidents_stpp</code>
  <span class="sh-ds-desc">Road traffic accident reports across the United States.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>accident location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>report time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

## Incidents &amp; safety

<p class="sh-ds-eyebrow">space · where something was reported</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/chicago_crime_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Chicago Crime</span></span>
  <code class="sh-ds-id">seahorse-stpp/chicago_crime_stpp</code>
  <span class="sh-ds-desc">Reported crime incidents across the city of Chicago.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>block location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>incident time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/la_crime_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">LA Crime</span></span>
  <code class="sh-ds-id">seahorse-stpp/la_crime_stpp</code>
  <span class="sh-ds-desc">Reported crime incidents across Los Angeles.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>report location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>incident time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/gtd_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Global Terrorism Database</span></span>
  <code class="sh-ds-id">seahorse-stpp/gtd_stpp</code>
  <span class="sh-ds-desc">Worldwide terrorism events from the GTD.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>event location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>event date</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/austin_311_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Austin 311</span></span>
  <code class="sh-ds-id">seahorse-stpp/austin_311_stpp</code>
  <span class="sh-ds-desc">Non-emergency city service requests in Austin, Texas.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>request location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>request time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

## Earth &amp; environment

<p class="sh-ds-eyebrow">space · latitude and longitude on the globe</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/earthquakes-stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Earthquakes</span><span class="sh-ds-chip">core benchmark</span></span>
  <code class="sh-ds-id">seahorse-stpp/earthquakes-stpp</code>
  <span class="sh-ds-desc">Seismic events with epicenter and origin time.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>epicenter</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>origin time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/us_wildfires_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">US Wildfires</span></span>
  <code class="sh-ds-id">seahorse-stpp/us_wildfires_stpp</code>
  <span class="sh-ds-desc">Wildfire ignition records across the United States.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>ignition location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>discovery date</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

## Population health

<p class="sh-ds-eyebrow">space · case geography</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--bench" href="https://huggingface.co/datasets/seahorse-stpp/covid-stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">COVID-19</span><span class="sh-ds-chip">core benchmark</span></span>
  <code class="sh-ds-id">seahorse-stpp/covid-stpp</code>
  <span class="sh-ds-desc">Reported COVID-19 cases over space and time.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>case location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>report date</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

## Social check-ins

<p class="sh-ds-eyebrow">space · where people check in</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/gowalla_checkins_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Gowalla</span></span>
  <code class="sh-ds-id">seahorse-stpp/gowalla_checkins_stpp</code>
  <span class="sh-ds-desc">Location-based social check-ins from the Gowalla network.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>venue location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>check-in time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
<a class="sh-ds-card" href="https://huggingface.co/datasets/seahorse-stpp/brightkite_checkins_stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">Brightkite</span></span>
  <code class="sh-ds-id">seahorse-stpp/brightkite_checkins_stpp</code>
  <span class="sh-ds-desc">Location-based social check-ins from the Brightkite network.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>venue location</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>check-in time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

## Beyond the map

<p class="sh-ds-eyebrow">space · not a map at all</p>

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--abstract" href="https://huggingface.co/datasets/seahorse-stpp/bold5000-stpp" target="_blank" rel="noopener">
  <span class="sh-ds-head"><span class="sh-ds-name">BOLD5000</span><span class="sh-ds-chip sh-ds-chip--muted">non-2D</span></span>
  <code class="sh-ds-id">seahorse-stpp/bold5000-stpp</code>
  <span class="sh-ds-desc">Eventized neural responses derived from BOLD5000 fMRI recordings — included to show STPP space need not be geographic.</span>
  <span class="sh-ds-axes"><span class="sh-ds-axis"><span class="sh-ds-k">space</span>the brain</span><span class="sh-ds-axis"><span class="sh-ds-k">time</span>scan time</span></span>
  <span class="sh-ds-go" aria-hidden="true">↗</span>
</a>
</div>

!!! note "BOLD5000 is supported, not part of the headline benchmark"
    BOLD5000 ships as a fully supported dataset, but the main real-data
    comparison stays on the 2D-spatial trio — **COVID, Earthquakes, and
    Citibike** — because several benchmark presets are built for
    two-dimensional spatial event domains.

## Synthetic benchmark suites

<p class="sh-ds-eyebrow">space · whatever you configure it to be</p>

When you need known ground truth — to isolate one factor that real data
confounds — Seahorse uses synthetic sequences from
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
