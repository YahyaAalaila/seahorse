"""Small mathematical and protocol checks for native statstpp models."""

import math
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from seahorse.models.statistical import build_statistical_model, model_from_state
from seahorse.models.statistical.components import BACKGROUND_REGISTRY, INFERENCE_REGISTRY, SPATIAL_KERNEL_REGISTRY, TEMPORAL_KERNEL_REGISTRY
from seahorse.models.statistical.core import contains, sequences
from seahorse.models.statistical.hawkes import HawkesSTPP, _cache_conditional_contexts, _ConditionalExpGaussianLikelihood, _IndependentLikelihood, _rectangular_gaussian_mass, _source_simulate_one_after
from seahorse.models.statistical._hawkes_source.hawkes_package.inference.likelihood import SpatioTemporalLogLikelihood


SUPPORT = (0.0, 2.0, 0.0, 2.0)


def test_statistical_source_components_are_name_registrable():
    assert set(BACKGROUND_REGISTRY) == {"constant"}
    assert {"exponential", "omori"} <= set(TEMPORAL_KERNEL_REGISTRY)
    assert {"gaussian", "pareto"} <= set(SPATIAL_KERNEL_REGISTRY)
    assert "mle" in INFERENCE_REGISTRY
    with pytest.raises(ValueError, match="component pairs"):
        HawkesSTPP(support=SUPPORT, temporal="exponential", spatial="pareto")


def test_vendored_hawkes_source_matches_its_frozen_manifest():
    root = Path(__file__).parents[1] / "seahorse/models/statistical/_hawkes_source"
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    rows = []
    for path in sorted((root / "hawkes_package").rglob("*.py")):
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n")
    digest = hashlib.sha256("".join(rows).encode()).hexdigest()
    assert len(rows) == manifest["file_count"]
    assert digest == manifest["ordered_sha256_manifest"]


def _seq(times=(0.5, 1.0, 1.8), locations=((0.2, 0.2), (0.4, 0.3), (1.5, 1.7))):
    return {"t0": 0.0, "t1": 2.0, "times": list(times), "locations": list(locations)}


def test_poisson_next_event_nll_has_survival_and_continuous_area_density():
    model = build_statistical_model("homogeneous_poisson", {"support": SUPPORT, "alpha": 0.0})
    model.fit([_seq()], seed=7)
    # n / (T*A) = 3/(2*4); joint density lambda; survival r*(1.0-0.5).
    actual = model.next_event_nll({"times": [0.5], "locations": [[0.2, 0.2]]}, 1.0, np.array([0.4, 0.3]))
    assert np.isclose(actual, -math.log(3 / 8) + (3 / 2) * .5)
    # Full window has all event density terms and terminal survival through t1.
    assert np.isclose(model.full_window_nll(_seq()), -3 * math.log(3 / 8) + 3)


def test_poisson_sampling_and_state_are_numpy_runner_contracts():
    model = build_statistical_model("seasonal_poisson", {"support": SUPPORT, "period": 2.0, "n_phases": 2, "grid": (2, 2)})
    model.fit([_seq()], seed=4)
    history = {"times": [0.5], "locations": [[0.2, 0.2]]}
    t, x = model.sample_next_event(history, n_samples=5, rng=np.random.default_rng(3))
    assert t.shape == (5,) and x.shape == (5, 2) and np.all(t > .5)
    assert np.all((x >= 0) & (x < 2))
    restored = model_from_state(model.to_state())
    assert np.isclose(restored.next_event_nll(history, 1.0, np.array([.4, .3])), model.next_event_nll(history, 1.0, np.array([.4, .3])))


def test_closed_spatial_support_preserves_inclusive_preprocessing_boundaries():
    upper = np.array([[2.0, 2.0]])
    assert bool(contains(SUPPORT, upper)[0])
    assert not bool(contains(SUPPORT, np.array([[2.0 + 1e-12, 2.0]]))[0])
    sequence = {"t0": 0., "t1": 1., "times": [.2, .7], "locations": [[0., 0.], [2., 2.]]}
    poisson = build_statistical_model("spatial_poisson", {"support": SUPPORT, "grid": (2, 2), "alpha": 0.})
    poisson.fit([sequence], seed=0)
    assert np.isfinite(poisson.full_window_nll(sequence))
    assert np.isfinite(poisson.next_event_nll({"t0": 0., "t1": .2, "times": [.2], "locations": [[0., 0.]]},
                                              .7, np.array([2., 2.])))
    hawkes = HawkesSTPP(support=SUPPORT, parameters=np.array([.25, .1, 2., .5]))
    assert np.isfinite(hawkes.full_window_nll(sequence))
    assert np.isfinite(hawkes.next_event_nll({"t0": 0., "t1": .2, "times": [.2], "locations": [[0., 0.]]},
                                             .7, np.array([2., 2.])))


def test_hawkes_source_formula_and_final_survival_are_preserved():
    model = HawkesSTPP(support=SUPPORT, temporal="exponential", spatial="gaussian")
    # background density, alpha, beta, sigma.  With alpha=0 the vendored source
    # likelihood reduces exactly to homogeneous Poisson and includes terminal survival.
    model.parameters = np.array([.25, 1e-6, 2.0, .5])
    assert np.isclose(model.full_window_nll(_seq()), -3 * math.log(.25) + 2.0)
    model.parameters = np.array([.25, .4, 2.0, .5])
    kernel = model._source_model().components.spatial.build(model.parameters[3:])
    assert np.isclose(kernel(np.array([0.0]))[0], 1 / (2 * math.pi * .25))
    assert np.isclose(model._source_model().branching(model.parameters[None, :])[0], .2)


def test_gaussian_rectangle_integral_removes_source_grid_visibility_floor():
    support = (-130.0, -110.0, 30.79, 44.45)
    # This is the analytic product-normal probability, including the hard case
    # that was previously rejected solely because 32 grid nodes imply 1.25 deg.
    centre = np.array([[-120.0, 37.0]])
    assert np.isclose(_rectangular_gaussian_mass(centre, .05, support)[0], 1.0, atol=1e-12)
    model = HawkesSTPP(support=support, quadrature_order=32, parameters=np.array([.01, .2, 1., .05]))
    assert bool(model._source_model().support(model.parameters))
    sequence = {"t0": 0., "t1": 1., "times": [.2, .6], "locations": [[-120., 37.], [-120.01, 37.01]]}
    assert np.isfinite(model.full_window_nll(sequence))


def test_conditional_h20_composite_fit_is_reiterable_and_explicitly_finite_history():
    support = SUPPORT
    base_times = np.linspace(.05, 1.0, 20)
    base_locations = np.column_stack([np.linspace(.1, 1.0, 20), np.linspace(.2, 1.1, 20)])
    rows = []
    for step in range(3):
        shift = .03 * step
        rows.append({
            "history": {"t0": 0., "t1": float(base_times[-1] + shift),
                        "times": (base_times + shift).tolist(), "locations": base_locations.tolist()},
            "next_time": float(1.2 + shift), "next_location": [1.2, 1.25],
        })
    model = HawkesSTPP(support=support, edge_correction="renormalise")
    diagnostics = model.fit_conditional(lambda: iter(rows), seed=0,
                                        inference={"history_events": 20, "max_iterations": 600})
    assert diagnostics["method"] == "source_hawkes_h20_conditional_composite_mle"
    assert diagnostics["objective_kind"] == "finite_history_conditional_composite_likelihood"
    assert diagnostics["n_contexts"] == len(rows)
    assert diagnostics["conditional_cache_bytes"] > 0
    assert np.isfinite(model.parameters).all()


def test_exact_gaussian_scalar_and_vectorized_conditional_density_agree_with_quiet_interval():
    times = np.linspace(.05, 1.0, 20)
    locations = np.column_stack([np.linspace(.1, 1.0, 20), np.linspace(.2, 1.1, 20)])
    history = {"t0": 0., "t1": 1.15, "times": times, "locations": locations}
    target_time, target_location = 1.4, np.array([1.2, 1.25])
    model = HawkesSTPP(support=SUPPORT, edge_correction="renormalise",
                       parameters=np.array([.25, .1, 2., .5]))
    chunks, _, count, _ = _cache_conditional_contexts(
        [{"history": history, "next_time": target_time, "next_location": target_location}], SUPPORT, 20,
        chunk_size=1, max_cache_bytes=10_000,
    )
    composite = _ConditionalExpGaussianLikelihood(model._source_model(), SUPPORT, chunks, 20)
    assert count == 1
    assert model.nll_kind == "exact" and model.nll_footnote == ""
    assert np.isclose(model.next_event_nll(history, target_time, target_location), -composite.total(model.parameters, None))
    assert np.isclose(
        model.score_conditional([{"history": history, "next_time": target_time, "next_location": target_location}])[0],
        model.next_event_nll(history, target_time, target_location),
    )


def test_hawkes_source_likelihood_sums_independent_windows_and_walkforward_prefixes():
    model = build_statistical_model("hawkes_exp_gaussian", {"support": SUPPORT})
    model.parameters = np.array([.25, .1, 2.0, .5])
    first, second = _seq(), _seq((0.2, 1.2), ((.1, .1), (1.8, 1.8)))
    source_model = model._source_model()
    histories = [model._source_history(seq) for seq in sequences([first, second], SUPPORT)]
    likelihoods = [SpatioTemporalLogLikelihood(source_model, backend="cached") for _ in histories]
    combined = _IndependentLikelihood(source_model, likelihoods, histories)
    assert np.isclose(-combined.total(model.parameters, histories[0]), model.full_window_nll(first) + model.full_window_nll(second))
    value = model.next_event_nll({"times": [.5], "locations": [[.2, .2]]}, 1.0, np.array([.4, .3]))
    assert np.isfinite(value)
    start = model._valid_start(1e5, .5, "omori", .1)
    omori = HawkesSTPP(support=SUPPORT, temporal="omori", spatial="pareto")._source_model()
    assert bool(omori.support(start))


def test_hawkes_empty_window_uses_exact_constant_background_exposure():
    model = HawkesSTPP(support=SUPPORT)
    model.parameters = np.array([.25, .1, 2.0, .5])
    empty = {"t0": 0.0, "t1": 2.0, "times": [], "locations": []}
    # The source hooks path has no event geometry to cache. Its full-window
    # likelihood is exactly the background compensator on an empty window.
    assert np.isclose(model.full_window_nll(empty), .25 * 4.0 * 2.0)


def test_hawkes_uses_vendored_source_simulator_with_explicit_spatial_budget():
    model = HawkesSTPP(support=SUPPORT, sampler_iterations=600)
    model.parameters = np.array([.25, .1, 2.0, .5])
    times, locations = model.sample_next_event({"t0": 0.0, "times": [.5], "locations": [[.2, .2]]},
                                               n_samples=1, rng=np.random.default_rng(2))
    assert times.shape == (1,) and locations.shape == (1, 2) and times[0] > .5
    with pytest.raises(ValueError, match="burn-in"):
        HawkesSTPP(support=SUPPORT, sampler_iterations=500, parameters=model.parameters)._process(
            {"t0": 0.0, "times": [.5], "locations": [[.2, .2]]}, np.random.default_rng(0)
        )


def test_gaussian_sampler_is_exact_truncated_mixture_at_a_narrow_boundary_kernel():
    model = HawkesSTPP(support=SUPPORT, sampler_iterations=600,
                       parameters=np.array([1e-6, .8, 1., .2]))
    history = {"t0": 0., "t1": .5, "times": [.5], "locations": [[.01, 1.]]}
    process, _ = model._process(history, np.random.default_rng(12))
    samples = np.asarray([process._draw_location(.6) for _ in range(2000)])
    assert model.sampling_method == "source_ogata_with_exact_truncated_gaussian_mixture"
    assert np.all((samples >= 0.) & (samples < 2.))
    # The parent component dominates the negligible background.  Its x mean is
    # the closed-form lower-truncated normal mean, so this catches accidental
    # rejection-to-box or untruncated Gaussian sampling near an edge.
    a, b = (-.01 / .2), ((2. - .01) / .2)
    expected_x = .01 + .2 * (np.exp(-.5 * a**2) - np.exp(-.5 * b**2)) / np.sqrt(2 * np.pi) / (ndtr(b) - ndtr(a))
    assert abs(samples[:, 0].mean() - expected_x) < .02
    repeat, _ = model._process(history, np.random.default_rng(12))
    assert np.array_equal(samples[:10], np.asarray([repeat._draw_location(.6) for _ in range(10)]))


def test_source_one_event_extraction_does_not_return_last_of_a_multi_event_horizon():
    class Rng:
        def __init__(self):
            self.values = iter([.2, .9, .3, .1])
        def exponential(self):
            return next(self.values)
        def uniform(self):
            return next(self.values)
    class Process:
        def __init__(self):
            self.rng, self.events, self.recorded = Rng(), np.empty((3, 0)), []
        def _bound_at(self, time):
            return 1.0
        def _integrated_intensity(self, time):
            return 0.0 if time < .3 else 1.0
        def _draw_location(self, time):
            return np.array([time, time])
        def _record(self, time, location):
            self.recorded.append((time, location))
            self.events = np.column_stack([self.events, np.r_[time, location]])
    process = Process()
    assert _source_simulate_one_after(process, 0.0)
    assert len(process.recorded) == 1
    assert np.isclose(process.recorded[0][0], .5)


def test_hawkes_respects_observed_quiet_interval_and_persists_nll_metadata():
    model = build_statistical_model("hawkes_exp_gaussian", {"support": SUPPORT, "sampler_iterations": 600})
    model.parameters = np.array([.25, 1e-6, 2.0, .5])
    short = {"t0": 0.0, "t1": .5, "times": [.5], "locations": [[.2, .2]]}
    quiet = {**short, "t1": 1.0}
    # With negligible alpha this isolates no-event survival: the observed quiet interval
    # moves the conditional origin from .5 to 1.0 without accessing future events.
    assert model.next_event_nll(short, 1.5, np.array([.4, .3])) - \
        model.next_event_nll(quiet, 1.5, np.array([.4, .3])) > .5
    sampled, _ = model.sample_next_event(quiet, n_samples=1, rng=np.random.default_rng(3))
    assert sampled[0] > 1.0
    restored = model_from_state(model.to_state())
    assert restored.metadata["nll_kind"] == "exact"
    with pytest.raises(ValueError, match="finite"):
        model.next_event_nll(quiet, float("nan"), np.array([.4, .3]))


def test_statistical_models_reject_chronology_and_test_support_inference():
    with pytest.raises(ValueError, match="support"):
        build_statistical_model("homogeneous_poisson", {})
    model = build_statistical_model("homogeneous_poisson", {"support": SUPPORT})
    with pytest.raises(ValueError, match="strictly chronological"):
        model.fit([_seq((1.0, .5), ((.2, .2), (.3, .3)))], seed=0)
    model.fit([_seq()], seed=0)
    assert math.isinf(model.next_event_nll({"times": [.5], "locations": [[.2, .2]]}, 1., np.array([3., 3.])))
