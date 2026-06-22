from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from seahorse.api import STPPEstimator, list_available_models, resolve_preset


class ApiEstimatorTest(unittest.TestCase):
    def test_model_names_resolve_through_registry(self):
        self.assertEqual(resolve_preset("AutoSTPP"), "auto_stpp")
        self.assertEqual(resolve_preset("auto_stpp"), "auto_stpp")
        self.assertIn("AutoSTPP", list_available_models())
        self.assertIn("auto_stpp", list_available_models())

    def test_unknown_model_name_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown model class"):
            resolve_preset("DefinitelyNotAModel")

    def test_fit_requires_validation_sequences(self):
        estimator = STPPEstimator("AutoSTPP")
        with self.assertRaisesRegex(ValueError, "val_seqs is required"):
            estimator.fit([{"times": [0.0], "locations": [[0.0, 0.0]]}])

    @patch("seahorse.api.estimator.STPPRunner")
    @patch("seahorse.api.estimator.STPPConfig")
    def test_fit_builds_config_and_delegates_to_runner(self, config_cls, runner_cls):
        config = Mock(name="config")
        config_cls.from_source.return_value = config
        runner = runner_cls.return_value
        runner.fit.return_value = Mock(name="run_result")

        train = [{"times": [0.0, 0.2], "locations": [[0.0, 0.0], [0.1, 0.2]]}]
        val = [{"times": [0.0, 0.3], "locations": [[0.0, 0.0], [0.2, 0.3]]}]
        test = [{"times": [0.0, 0.4], "locations": [[0.0, 0.0], [0.3, 0.4]]}]

        estimator = STPPEstimator(
            "AutoSTPP",
            config_overrides={"model": {"hidden_dim": 8}},
            device="cpu",
            seed=7,
        )
        returned = estimator.fit(
            train,
            val,
            test,
            epochs=2,
            lr=0.01,
            batch_size=4,
            dataset_id="unit",
        )

        self.assertIs(returned, estimator)
        config_cls.from_source.assert_called_once()
        _, kwargs = config_cls.from_source.call_args
        self.assertEqual(kwargs["preset"], "auto_stpp")
        cli_values = kwargs["cli_values"]
        self.assertEqual(cli_values["model"]["hidden_dim"], 8)
        self.assertEqual(cli_values["training"]["device"], "cpu")
        self.assertEqual(cli_values["training"]["seed"], 7)
        self.assertEqual(cli_values["training"]["n_epochs"], 2)
        self.assertEqual(cli_values["training"]["lr"], 0.01)
        self.assertEqual(cli_values["training"]["batch_size"], 4)
        self.assertEqual(cli_values["data"]["seed"], 7)
        self.assertEqual(cli_values["data"]["batch_size"], 4)
        runner_cls.assert_called_once_with(config)
        runner.fit.assert_called_once_with(train, val, test, dataset_id="unit")
        self.assertTrue(estimator._is_fitted)


if __name__ == "__main__":
    unittest.main()
