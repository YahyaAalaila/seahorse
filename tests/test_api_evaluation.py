from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from seahorse.api import STPPEstimator


def _fitted_estimator() -> STPPEstimator:
    estimator = STPPEstimator("AutoSTPP", device="cpu", seed=3)
    model = Mock()
    model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    model.event_model.capabilities = SimpleNamespace(has_native_sampler=False, nll_kind="exact")
    runner = Mock()
    runner.model = model
    estimator._runner = runner
    estimator._is_fitted = True
    return estimator


class ApiEvaluationTest(unittest.TestCase):
    @patch("seahorse.api.estimator.compute_seq_nlls")
    @patch("seahorse.api.estimator.compute_next_event_test_nll")
    def test_core_evaluate_delegates_to_existing_likelihood_helpers(self, next_nll, seq_nlls):
        next_nll.return_value = {"mean_nll": 1.25, "sampling_backend": "backend"}
        seq_nlls.return_value = np.asarray([1.0, 2.0], dtype=np.float32)
        estimator = _fitted_estimator()
        seqs = [{"times": [0.0, 1.0], "locations": [[0.0, 0.0], [1.0, 1.0]]}]

        result = estimator.evaluate(seqs)

        self.assertEqual(result["test_nll"], 1.25)
        self.assertEqual(result["mean_seq_nll"], 1.5)
        self.assertEqual(result["sampling_backend"], "backend")
        next_nll.assert_called_once()
        seq_nlls.assert_called_once()

    def test_unsupported_metric_raises(self):
        estimator = _fitted_estimator()
        with self.assertRaisesRegex(NotImplementedError, "Unsupported estimator metrics"):
            estimator.evaluate([], metrics=["temporal_crps"])


if __name__ == "__main__":
    unittest.main()
