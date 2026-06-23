from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from unified_stpp.config import STPPConfig
from unified_stpp.models.configs import ConfigRegistry
from unified_stpp.runner import STPPRunner


def _toy_sequences(n: int = 6) -> list[dict]:
    base = []
    for i in range(n):
        offset = 0.015 * i
        base.append(
            {
                "times": [0.10 + offset, 0.28 + offset, 0.55 + offset, 0.91 + offset],
                "locations": [
                    [0.12 + 0.01 * i, 0.20],
                    [0.24 + 0.01 * i, 0.28],
                    [0.48, 0.50 + 0.01 * i],
                    [0.72, 0.69 + 0.01 * i],
                ],
            }
        )
    return base


class DemoGRUGaussianTest(unittest.TestCase):
    def test_preset_is_registered_and_forward_exposes_eventwise_terms(self):
        self.assertTrue(ConfigRegistry.is_registered("demo_gru_gaussian"))
        cfg = STPPConfig.from_source(
            preset="demo_gru_gaussian",
            cli_values={"model": {"hidden_dim": 8}, "training": {"device": "cpu"}},
        )
        model = cfg.model.build_model()

        times = torch.tensor([[0.1, 0.3, 0.6, 1.0]], dtype=torch.float32)
        locations = torch.tensor(
            [[[0.1, 0.2], [0.2, 0.3], [0.5, 0.5], [0.8, 0.7]]],
            dtype=torch.float32,
        )
        lengths = torch.tensor([4], dtype=torch.long)
        out = model(times=times, locations=locations, lengths=lengths)

        self.assertIn("nll", out)
        self.assertIn("nll_matrix", out)
        self.assertIn("next_event_mask", out)
        self.assertTrue(torch.isfinite(out["nll"]))
        self.assertEqual(tuple(out["nll_matrix"].shape), (1, 4))
        self.assertEqual(int(out["next_event_mask"].sum().item()), 3)

    def test_one_epoch_fit_reports_finite_test_nll(self):
        train = _toy_sequences(6)
        val = _toy_sequences(3)
        test = _toy_sequences(3)
        with tempfile.TemporaryDirectory() as tmp:
            runner = STPPRunner.from_config_source(
                "demo_gru_gaussian",
                None,
                cli_values={
                    "logging": {"out_dir": str(Path(tmp) / "runs")},
                    "model": {"hidden_dim": 8},
                    "training": {
                        "device": "cpu",
                        "n_epochs": 1,
                        "batch_size": 2,
                        "checkpoint_select": "best",
                        "test_nll_space": "native",
                    },
                    "data": {
                        "protocol": "raw",
                        "normalize": False,
                        "batch_size": 2,
                        "num_workers": 0,
                        "adapter_kwargs": {"training_view": "full_sequence"},
                    },
                },
            )
            result = runner.fit(train, val, test, dataset_id="demo_unit")

        self.assertTrue(math.isfinite(result.val_objective))
        self.assertTrue(math.isfinite(result.test_nll))
        self.assertEqual(result.preset, "demo_gru_gaussian")


if __name__ == "__main__":
    unittest.main()
