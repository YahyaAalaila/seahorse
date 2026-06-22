from __future__ import annotations

import unittest
from unittest.mock import Mock

from seahorse.api import STPPEstimator
from seahorse.api.viz import STPPPlotter


class ApiPlottingTest(unittest.TestCase):
    def test_estimator_plotter_requires_fitted_model(self):
        estimator = STPPEstimator("AutoSTPP")
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            _ = estimator.plotter

    def test_plot_intensity_requires_run_directory(self):
        plotter = STPPPlotter(Mock(), run_dir=None)
        with self.assertRaisesRegex(RuntimeError, "run directory"):
            plotter.plot_intensity({"times": [0.0], "locations": [[0.0, 0.0]]})


if __name__ == "__main__":
    unittest.main()
