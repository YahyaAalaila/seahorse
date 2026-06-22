from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from seahorse.api import STPPEstimator


class ApiPersistenceTest(unittest.TestCase):
    def test_save_delegates_to_runner(self):
        estimator = STPPEstimator("AutoSTPP")
        runner = Mock()
        runner.save.return_value = Path("saved")
        estimator._runner = runner
        estimator._is_fitted = True

        self.assertEqual(estimator.save("saved"), Path("saved"))
        runner.save.assert_called_once_with("saved")

    def test_save_requires_fitted_model(self):
        estimator = STPPEstimator("AutoSTPP")
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            estimator.save("saved")

    @patch("seahorse.api.estimator.STPPRunner")
    def test_load_delegates_to_runner_load(self, runner_cls):
        runner = runner_cls.load.return_value
        runner.config.model.preset = "auto_stpp"

        estimator = STPPEstimator.load("saved")

        runner_cls.load.assert_called_once_with("saved")
        self.assertTrue(estimator._is_fitted)
        self.assertIs(estimator.runner, runner)
        self.assertEqual(estimator.preset, "auto_stpp")


if __name__ == "__main__":
    unittest.main()
