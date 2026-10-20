import numpy as np

from seahorse.experiments.synthetic_controls import (
    SyntheticRecipe, generate_splits, generate_windows, moving_hotspot_center,
    moving_seasonal_intensity, seasonal_rate,
)


def test_synthetic_splits_are_reproducible_separate_and_preserve_empty_windows():
    recipe = SyntheticRecipe("homogeneous_poisson", horizon=1., rate=.01)
    a = generate_splits(recipe, n_train=4, n_val=3, n_test=2, seed=7)
    b = generate_splits(recipe, n_train=4, n_val=3, n_test=2, seed=7)
    assert a == b and a["train"] != a["val"]
    assert all(w["support"] == [[0.,1.],[0.,1.]] and w["start"] == 0. and w["end"] == 1. for w in a["train"])
    assert any(len(w["times"]) == 0 for w in generate_windows(recipe, 100, seed=3))


def test_moving_poisson_has_analytic_rate_and_moving_hotspot():
    recipe = SyntheticRecipe("moving_seasonal_poisson", rate=10., seasonal_amplitude=.5, seasonal_period=1.)
    assert np.isclose(seasonal_rate(.25, recipe), 15.)
    assert np.allclose(moving_hotspot_center(0., recipe), [.75,.5])
    assert np.allclose(moving_hotspot_center(.25, recipe), [.5,.75], atol=1e-7)
    center = moving_hotspot_center(.25, recipe)
    assert moving_seasonal_intensity(.25, center, recipe) > moving_seasonal_intensity(.25, np.array([0.,0.]), recipe)


def test_gamma_renewal_resets_each_window_and_is_inhibitive_by_construction():
    recipe = SyntheticRecipe("gamma_renewal", rate=20., renewal_shape=2.)
    windows = generate_windows(recipe, 3, seed=2)
    for w in windows:
        times = np.asarray(w["times"])
        assert np.all(np.diff(times) > 0) and np.all((times > 0) & (times <= 1.))
