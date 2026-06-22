from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from seahorse.api import STPPEstimator


def _estimator_with_caps(*, nll_kind: str, has_native_sampler: bool) -> STPPEstimator:
    estimator = STPPEstimator("AutoSTPP", device="cpu", seed=5)
    model = Mock()
    model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    model.event_model.capabilities = SimpleNamespace(
        nll_kind=nll_kind,
        has_native_sampler=has_native_sampler,
    )
    runner = Mock()
    runner.model = model
    estimator._runner = runner
    estimator._is_fitted = True
    return estimator


class ApiSamplingTest(unittest.TestCase):
    @patch("seahorse.api.estimator.compute_predictive_samples")
    def test_predict_next_delegates_for_exact_models(self, compute_samples):
        compute_samples.return_value = SimpleNamespace(
            next_times=np.ones((1, 3), dtype=np.float32),
            next_locs=np.ones((1, 3, 2), dtype=np.float32),
            true_next_times=np.asarray([1.0], dtype=np.float32),
            true_next_locs=np.ones((1, 2), dtype=np.float32),
            history_end_times=np.asarray([0.5], dtype=np.float32),
            sequence_index=np.asarray([0], dtype=np.int64),
            target_event_index=np.asarray([1], dtype=np.int64),
            history_length=np.asarray([1], dtype=np.int64),
            sampling_succeeded=np.asarray([True]),
            sampling_backend="exact_intensity_thinning",
        )
        estimator = _estimator_with_caps(nll_kind="exact", has_native_sampler=False)

        result = estimator.predict_next([{"times": [0.0, 1.0], "locations": [[0, 0], [1, 1]]}], n_samples=3)

        self.assertEqual(result["next_times"].shape, (1, 3))
        self.assertEqual(result["sampling_backend"], "exact_intensity_thinning")
        compute_samples.assert_called_once()

    def test_predict_next_rejects_unsupported_models(self):
        estimator = _estimator_with_caps(nll_kind="none", has_native_sampler=False)
        with self.assertRaisesRegex(NotImplementedError, "not available"):
            estimator.predict_next([])


if __name__ == "__main__":
    unittest.main()
