import math

import numpy as np

from seahorse.experiments.count_baselines import CalendarMonthPoisson, CalendarPoisson, SeasonalCalendarPoisson, SpatialCellPoisson, prepared_windows, select_spatial_grid


def _windows():
    # Bin [0,1), [1,2) are whole; [2,2.5) is intentionally excluded.
    return [{"start": 0., "end": 2.5, "times": [0.1, 1.1, 2.2], "locations": [[.1,.1],[.9,.9],[.5,.5]]}]


def test_calendar_poisson_fits_train_exposure_and_scores_zero_bins():
    model = CalendarPoisson.fit(_windows(), bin_width=1.)
    assert model.rate == 1.  # only the first two whole-bin events/exposure count
    score = model.score([{"start":0.,"end":3.,"times":[.1],"locations":[[.1,.1]]}])
    # Counts [1,0,0], lambda=1 each: 1 + 1 + 1 = 3 (lgamma(2)=0).
    assert np.isclose(score["coarse_count_nll"], 3.)


def test_seasonal_and_spatial_controls_are_train_fit_and_whole_bin_only():
    train=[{"start":0.,"end":4.,"times":[.1,.2,2.1],"locations":[[.1,.1],[.1,.1],[.9,.9]]}]
    seasonal=SeasonalCalendarPoisson.fit(train,bin_width=1.,period_bins=2)
    assert seasonal.rates == (1.5, 0.)
    spatial=SpatialCellPoisson.fit(train,rectangle=(0.,1.,0.,1.),grid=(2,2),bin_width=1.,alpha=.5)
    test=[{"start":0.,"end":1.,"times":[],"locations":[]}]
    score=spatial.score(test)
    # Four zero cells are all retained in the Poisson coarse count likelihood.
    assert np.isfinite(score["coarse_spatiotemporal_count_nll"])
    selected=select_spatial_grid(train,test,rectangle=(0.,1.,0.,1.),grids=[(1,1),(2,2)],bin_width=1.)
    assert selected.grid in {(1,1),(2,2)}


def test_prepared_windows_use_frozen_calendar_bounds_and_retain_empty_records():
    records = [{"sequence_id":"w0", "window_start":"2010-01-06 00:00:00", "window_end":"2010-01-13 00:00:00",
                "t0":0.,"t1":7.,"times":[0.,6.],"locations":[[-120.,35.],[-121.,36.]]},
               {"sequence_id":"w1", "window_start":"2010-01-13 00:00:00", "window_end":"2010-01-20 00:00:00",
                "t0":0.,"t1":7.,"times":[],"locations":[]}]
    windows=prepared_windows(records,time_unit="day")
    assert len(windows)==2 and windows[0]["end"]-windows[0]["start"]==7 and windows[1]["times"]==[]
    model=CalendarPoisson.fit(windows,bin_width=1.)
    assert model.train_exposure==14 and model.rate==2/14


def test_partial_window_has_explicit_unavailable_count_score_not_zero_likelihood():
    model=CalendarPoisson.fit([{"start":0.,"end":1.,"times":[],"locations":[]}],bin_width=1.)
    score=model.score([{"start":.1,"end":.9,"times":[],"locations":[]}])
    assert score["coarse_count_nll"] is None and score["n_scored_bins"] == 0
