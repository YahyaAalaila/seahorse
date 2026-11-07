# Dataset cards

<p class="sh-ds-eyebrow">13 real datasets · HawkesNest entanglement suite · released extras</p>

The NeurIPS 2026 paper benchmarks 19 presets on **13 real datasets**: ten with chronological train/validation/test splits and the three datasets of the Neural STPP benchmark (COVID, Earthquakes, Citi Bike daily) under their original splits. The **HawkesNest entanglement suite** is the controlled synthetic stress test. All are on the Hugging Face Hub under [`seahorse-stpp`](https://huggingface.co/seahorse-stpp), pinned to a release revision and verified against the card hashes on load. Three repositories are script-only: they rebuild the data from the source instead of hosting it.

Each card shows where and when the events fall, the split sizes, the licence, how to load the data and the best presets of the paper.

## Real data, chronological splits

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--bench" href="cards/comcat_california/"><img class="sh-ds-thumb" src="../assets/datasets/comcat_california.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">ComCat · California</span><span class="sh-ds-chip">paper</span></span><code class="sh-ds-id">comcat_california</code><span class="sh-ds-desc">Earthquakes in California (USGS ANSS ComCat, as packaged by EarthquakeNPP).</span><span class="sh-ds-size">77,331 events · 14,260 sequences</span></a>
<a class="sh-ds-card sh-ds-card--bench" href="cards/geonet_nz/"><img class="sh-ds-thumb" src="../assets/datasets/geonet_nz.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">GeoNet · New Zealand</span><span class="sh-ds-chip">paper</span></span><code class="sh-ds-id">geonet_nz</code><span class="sh-ds-desc">Earthquakes in New Zealand (GeoNet catalogue).</span><span class="sh-ds-size">42,000 events · 730 sequences</span></a>
<a class="sh-ds-card sh-ds-card--bench" href="cards/citibike_nyc/"><span class="sh-ds-thumb sh-ds-thumb--empty">rebuild script</span><span class="sh-ds-head"><span class="sh-ds-name">Citi Bike · hourly</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">script-only</span></span><code class="sh-ds-id">citibike_nyc</code><span class="sh-ds-desc">Bike-share trip starts in New York City, hourly sequences.</span></a>
<a class="sh-ds-card sh-ds-card--bench" href="cards/austin311/"><img class="sh-ds-thumb" src="../assets/datasets/austin311.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">Austin 311</span><span class="sh-ds-chip">paper</span></span><code class="sh-ds-id">austin311</code><span class="sh-ds-desc">311 service requests in Austin, Texas.</span><span class="sh-ds-size">352,977 events · 365 sequences</span></a>
<a class="sh-ds-card sh-ds-card--bench" href="cards/seattle_fire911/"><span class="sh-ds-thumb sh-ds-thumb--empty">rebuild script</span><span class="sh-ds-head"><span class="sh-ds-name">Seattle fire 911</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">script-only</span></span><code class="sh-ds-id">seattle_fire911</code><span class="sh-ds-desc">Fire-department 911 dispatches in Seattle.</span></a>
<a class="sh-ds-card sh-ds-card--bench" href="cards/nyc311/"><img class="sh-ds-thumb" src="../assets/datasets/nyc311.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">NYC 311</span><span class="sh-ds-chip">paper</span></span><code class="sh-ds-id">nyc311</code><span class="sh-ds-desc">311 service requests in New York City.</span><span class="sh-ds-size">3,176,144 events · 365 sequences</span></a>
<a class="sh-ds-card sh-ds-card--bench" href="cards/nyc_collisions/"><img class="sh-ds-thumb" src="../assets/datasets/nyc_collisions.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">NYC collisions</span><span class="sh-ds-chip">paper</span></span><code class="sh-ds-id">nyc_collisions</code><span class="sh-ds-desc">Police-reported motor-vehicle collisions in New York City.</span><span class="sh-ds-size">89,014 events · 365 sequences</span></a>
<a class="sh-ds-card sh-ds-card--bench" href="cards/chicago_crime/"><img class="sh-ds-thumb" src="../assets/datasets/chicago_crime.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">Chicago crime</span><span class="sh-ds-chip">paper</span></span><code class="sh-ds-id">chicago_crime</code><span class="sh-ds-desc">Crime reports in Chicago.</span><span class="sh-ds-size">261,244 events · 365 sequences</span></a>
<a class="sh-ds-card sh-ds-card--bench" href="cards/la_crime/"><img class="sh-ds-thumb" src="../assets/datasets/la_crime.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">LA crime</span><span class="sh-ds-chip">paper</span></span><code class="sh-ds-id">la_crime</code><span class="sh-ds-desc">Crime reports in Los Angeles.</span><span class="sh-ds-size">232,341 events · 365 sequences</span></a>
<a class="sh-ds-card sh-ds-card--bench" href="cards/uber_nyc/"><img class="sh-ds-thumb" src="../assets/datasets/uber_nyc.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">Uber pickups · NYC</span><span class="sh-ds-chip">paper</span></span><code class="sh-ds-id">uber_nyc</code><span class="sh-ds-desc">Ride-hailing pickups in New York City.</span><span class="sh-ds-size">4,513,058 events · 4,391 sequences</span></a>
</div>

## Real data, Neural STPP benchmark splits

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--bench" href="cards/covid/"><img class="sh-ds-thumb" src="../assets/datasets/covid.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">COVID-19 · New Jersey</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">Neural STPP split</span></span><code class="sh-ds-id">covid</code><span class="sh-ds-desc">COVID-19 case reports in New Jersey counties (The New York Times).</span><span class="sh-ds-size">161,542 events · 1,650 sequences</span></a>
<a class="sh-ds-card sh-ds-card--bench" href="cards/earthquakes/"><img class="sh-ds-thumb" src="../assets/datasets/earthquakes.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">Earthquakes · Japan</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">Neural STPP split</span></span><code class="sh-ds-id">earthquakes</code><span class="sh-ds-desc">Earthquakes of magnitude ≥ 2.5 around Japan (USGS ComCat).</span><span class="sh-ds-size">79,840 events · 1,050 sequences</span></a>
<a class="sh-ds-card sh-ds-card--bench" href="cards/citibike_legacy/"><span class="sh-ds-thumb sh-ds-thumb--empty">rebuild script</span><span class="sh-ds-head"><span class="sh-ds-name">Citi Bike · daily</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--muted">Neural STPP split</span><span class="sh-ds-chip sh-ds-chip--muted">script-only</span></span><code class="sh-ds-id">citibike_legacy</code><span class="sh-ds-desc">Bike-share trip starts in New York City, daily sequences (thinned, Neural STPP construction).</span></a>
</div>

## Synthetic stress test

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card sh-ds-card--bench" href="cards/hawkesnest_entanglement/"><img class="sh-ds-thumb" src="../assets/datasets/hawkesnest_entanglement.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">HawkesNest entanglement suite</span><span class="sh-ds-chip">paper</span><span class="sh-ds-chip sh-ds-chip--synth">synthetic</span></span><code class="sh-ds-id">hawkesnest_entanglement</code><span class="sh-ds-desc">Spatiotemporal Hawkes process on the unit square with tunable space–time coupling, levels L0–L3.</span><span class="sh-ds-size">44,129 events · 110 sequences</span></a>
</div>

## Released, not benchmarked in the paper

<div class="sh-ds-grid" markdown="0">
<a class="sh-ds-card" href="cards/synthetic/"><img class="sh-ds-thumb" src="../assets/datasets/synthetic.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">Synthetic controls</span><span class="sh-ds-chip sh-ds-chip--synth">synthetic</span><span class="sh-ds-chip sh-ds-chip--muted">not benchmarked</span></span><code class="sh-ds-id">synthetic</code><span class="sh-ds-desc">Homogeneous Poisson, moving seasonal Poisson and gamma renewal processes on the unit square.</span><span class="sh-ds-size">61,198 events · 768 sequences</span></a>
<a class="sh-ds-card" href="cards/bold5000/"><img class="sh-ds-thumb" src="../assets/datasets/bold5000.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">BOLD5000</span><span class="sh-ds-chip sh-ds-chip--muted">not benchmarked</span></span><code class="sh-ds-id">bold5000</code><span class="sh-ds-desc">Eventized responses derived from BOLD5000 fMRI recordings (space is not geographic).</span><span class="sh-ds-size">13,138 events · 100 sequences</span></a>
<a class="sh-ds-card" href="cards/us_wildfires/"><img class="sh-ds-thumb" src="../assets/datasets/us_wildfires.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">US wildfires</span><span class="sh-ds-chip sh-ds-chip--muted">not benchmarked</span></span><code class="sh-ds-id">us_wildfires</code><span class="sh-ds-desc">Wildfire ignitions in the United States (USFS RDS-2013-0009.4).</span><span class="sh-ds-size">1,880,465 events · 18,806 sequences</span></a>
<a class="sh-ds-card" href="cards/us_accidents/"><img class="sh-ds-thumb" src="../assets/datasets/us_accidents.png" alt="" loading="lazy"><span class="sh-ds-head"><span class="sh-ds-name">US traffic accidents</span><span class="sh-ds-chip sh-ds-chip--muted">not benchmarked</span></span><code class="sh-ds-id">us_accidents</code><span class="sh-ds-desc">Road traffic accident reports in the United States (Moosavi et al.).</span><span class="sh-ds-size">6,985,228 events · 69,854 sequences</span></a>
</div>

## Summary table

| Dataset | Short name | Paper column | Period and splits | Sequences (train / val / test) | Events | Licence | Hosted |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| [ComCat · California](cards/comcat_california.md) | `comcat_california` | ComCat | 1981–2020; validation from 1998, test from 2007 | 6,209 / 3,287 / 4,764 | 77,331 | public domain (USGS ANSS ComCat) | yes |
| [GeoNet · New Zealand](cards/geonet_nz.md) | `geonet_nz` | GeoNet | 2018–2020; validation from 2019-01, test from 2019-07 | 365 / 181 / 184 | 42,000 | CC BY 3.0 NZ (GeoNet) | yes |
| [Citi Bike · hourly](cards/citibike_nyc.md) | `citibike_nyc` | Citi Bike (h) | 2019-04 – 2019-08; validation from 08-01, test from 08-16 | see card | see card | script MIT; data under the Citi Bike data licence, not hosted | script |
| [Austin 311](cards/austin311.md) | `austin311` | Austin | 2023; validation from 09-14, test from 10-20 | 256 / 36 / 73 | 352,977 | public domain (City of Austin) | yes |
| [Seattle fire 911](cards/seattle_fire911.md) | `seattle_fire911` | Seattle | 2023; validation from 09-14, test from 10-20 | see card | see card | script MIT; data public domain (City of Seattle), not hosted | script |
| [NYC 311](cards/nyc311.md) | `nyc311` | NYC 311 | 2023; validation from 09-14, test from 10-20 | 256 / 36 / 73 | 3,176,144 | NYC Open Data terms of use | yes |
| [NYC collisions](cards/nyc_collisions.md) | `nyc_collisions` | NYC coll. | 2023; validation from 09-14, test from 10-20 | 256 / 36 / 73 | 89,014 | NYC Open Data terms of use | yes |
| [Chicago crime](cards/chicago_crime.md) | `chicago_crime` | Chicago | 2023; validation from 09-14, test from 10-20 | 256 / 36 / 73 | 261,244 | City of Chicago data terms of use | yes |
| [LA crime](cards/la_crime.md) | `la_crime` | LA | 2023; validation from 09-14, test from 10-20 | 256 / 36 / 73 | 232,341 | CC0 1.0 (LAPD) | yes |
| [Uber pickups · NYC](cards/uber_nyc.md) | `uber_nyc` | Uber | 2014-04 – 2014-09; validation from 08-01, test from 09-01 | 2,928 / 744 / 719 | 4,513,058 | CC0 1.0 (FiveThirtyEight / NYC TLC FOIL) | yes |
| [COVID-19 · New Jersey](cards/covid.md) | `covid` | COVID | 2020-03 – 2020-07; original Neural STPP split | 1,450 / 100 / 100 | 161,542 | CC BY-NC 4.0 (The New York Times) | yes |
| [Earthquakes · Japan](cards/earthquakes.md) | `earthquakes` | Earthquakes | 1990–2019; original Neural STPP split | 950 / 50 / 50 | 79,840 | public domain (USGS) | yes |
| [Citi Bike · daily](cards/citibike_legacy.md) | `citibike_legacy` | Citi Bike (d) | 2019-04 – 2019-08; original Neural STPP split | see card | see card | script MIT; data under the Citi Bike data licence, not hosted | script |
| [HawkesNest entanglement suite](cards/hawkesnest_entanglement.md) | `hawkesnest_entanglement` | – | ten realizations of 8,000 events per level, cut into 20 windows each | 80 / 10 / 20 | 44,129 | CC BY 4.0 (Seahorse, HawkesNest) | yes |
| [Synthetic controls](cards/synthetic.md) | `synthetic` | – | independent windows on [0, 1) | 512 / 128 / 128 | 61,198 | CC BY 4.0 (Seahorse) | yes |
| [BOLD5000](cards/bold5000.md) | `bold5000` | – | see card | 70 / 10 / 20 | 13,138 | CC0 1.0 | yes |
| [US wildfires](cards/us_wildfires.md) | `us_wildfires` | – | see card | 13,164 / 2,821 / 2,821 | 1,880,465 | USFS RDS-2013-0009.4 (no permission required) | yes |
| [US traffic accidents](cards/us_accidents.md) | `us_accidents` | – | see card | 48,363 / 11,441 / 10,050 | 6,985,228 | CC BY-NC-SA 4.0 (Moosavi et al.) | yes |

Counts are for the default configuration; the HawkesNest row counts level L0. Regenerate this page and the cards with `python docs/_scripts/make_dataset_cards.py`.
