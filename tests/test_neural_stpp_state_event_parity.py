"""Regression checks for the canonical Neural STPP preset outputs."""

from __future__ import annotations

import unittest

import torch

from seahorse.registry import build_model


def _tiny_batch():
    times = torch.tensor(
        [
            [0.10, 0.30, 0.60, 1.00, 1.40],
            [0.10, 0.40, 0.70, 1.10, 1.30],
        ],
        dtype=torch.float32,
    )
    locations = torch.tensor(
        [
            [[0.00, 0.00], [0.25, -0.10], [0.15, 0.30], [-0.20, 0.10], [-0.10, 0.25]],
            [[0.00, 0.00], [0.10, 0.20], [0.30, -0.20], [0.35, 0.15], [0.20, 0.25]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([5, 4], dtype=torch.long)
    return times, locations, lengths


class TestNeuralSTPPStateEventOutputs(unittest.TestCase):
    def _assert_outputs(self, preset: str, forward_seed: int):
        torch.manual_seed(7)
        model = build_model(
            config={},
            preset=preset,
            spatial_dim=2,
            hidden_dim=16,
        )
        self.assertIsNotNone(model.state_model)
        self.assertIsNotNone(model.event_model)

        model.eval()
        times, locations, lengths = _tiny_batch()

        with torch.no_grad():
            torch.manual_seed(forward_seed)
            out = model(times=times, locations=locations, lengths=lengths)

        self.assertIn("nll", out)
        self.assertIn("nll_per_event", out)
        self.assertIn("total_events", out)
        self.assertIn("nll_matrix", out)
        self.assertIn("mask", out)
        self.assertIn("temporal_nll_matrix", out)
        self.assertIn("spatial_nll_matrix", out)
        self.assertIn("temporal_energy_reg", out)
        self.assertIn("spatial_reg", out)
        self.assertIn("regularization_total", out)
        self.assertTrue(torch.isfinite(out["nll"]))

        # nll = pure base NLL (no regularization); loss = nll + regularization
        torch.testing.assert_close(out["nll"], out["base_mean_nll"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            out["loss"],
            out["base_mean_nll"] + out["regularization_total"],
            rtol=1e-6,
            atol=1e-6,
        )
        # Breakdown scalars are present
        self.assertIn("temporal_nll", out)
        self.assertIn("spatial_nll", out)

    def test_neural_jumpcnf_outputs(self):
        self._assert_outputs("neural_jumpcnf", forward_seed=11)

    def test_neural_attncnf_outputs(self):
        self._assert_outputs("neural_attncnf", forward_seed=13)


if __name__ == "__main__":
    unittest.main()
