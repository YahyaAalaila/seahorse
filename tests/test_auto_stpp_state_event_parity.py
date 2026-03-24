"""Regression checks for auto_stpp coarse path outputs."""

from __future__ import annotations

import unittest

import torch

from unified_stpp.registry import build_model


def _tiny_batch():
    times = torch.tensor(
        [
            [0.00, 0.20, 0.50, 0.90, 1.30],
            [0.00, 0.30, 0.60, 1.00, 1.20],
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


class TestAutoSTPPStateEventOutputs(unittest.TestCase):
    def test_auto_stpp_outputs(self):
        torch.manual_seed(7)
        model = build_model(
            config={},
            preset="auto_stpp",
            spatial_dim=2,
            hidden_dim=16,
        )
        self.assertIsNotNone(model.state_model)
        self.assertIsNotNone(model.event_model)

        model.eval()
        times, locations, lengths = _tiny_batch()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)

        self.assertIn("nll", out)
        self.assertIn("nll_per_event", out)
        self.assertIn("total_events", out)
        self.assertIn("sll", out)
        self.assertIn("tll", out)
        self.assertIn("nll_matrix", out)
        self.assertIn("sll_matrix", out)
        self.assertIn("tll_matrix", out)
        self.assertIn("mask", out)
        self.assertIn("lambs_sum", out)
        self.assertIn("lamb_t", out)
        self.assertIn("lamb_ints", out)
        self.assertIn("background_rate", out)
        self.assertTrue(torch.isfinite(out["nll"]))

        torch.testing.assert_close(
            out["nll"],
            -(out["sll"] + out["tll"]),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            out["sll"],
            torch.zeros_like(out["sll"]),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
