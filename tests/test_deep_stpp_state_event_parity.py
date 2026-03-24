"""Regression checks for deep_stpp coarse path outputs."""

from __future__ import annotations

import unittest

import torch

from unified_stpp.registry import build_model


def _tiny_batch():
    times = torch.tensor(
        [[0.00, 0.20, 0.50, 0.90], [0.00, 0.10, 0.40, 0.80]],
        dtype=torch.float32,
    )
    locations = torch.tensor(
        [
            [[0.00, 0.00], [0.30, -0.10], [0.10, 0.20], [-0.20, 0.10]],
            [[0.00, 0.00], [0.20, 0.10], [0.20, -0.20], [0.40, 0.30]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([4, 4], dtype=torch.long)
    return times, locations, lengths


class TestDeepSTPPStateEventOutputs(unittest.TestCase):
    def _build_model(self, extra_cfg: dict | None = None):
        return build_model(
            config=dict(extra_cfg or {}),
            preset="deep_stpp",
            spatial_dim=2,
            hidden_dim=16,
        )

    def _assert_common(self, out):
        self.assertIn("nll", out)
        self.assertIn("nll_per_event", out)
        self.assertIn("total_events", out)
        self.assertIn("sll", out)
        self.assertIn("tll", out)
        self.assertIn("nll_matrix", out)
        self.assertIn("sll_matrix", out)
        self.assertIn("tll_matrix", out)
        self.assertIn("mask", out)
        self.assertTrue(torch.isfinite(out["nll"]))
        torch.testing.assert_close(
            out["nll"],
            -(out["sll"] + out["tll"]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_deep_stpp_outputs(self):
        torch.manual_seed(7)
        model = self._build_model(extra_cfg={})
        self.assertIsNotNone(model.state_model)
        self.assertIsNotNone(model.event_model)

        model.eval()
        times, locations, lengths = _tiny_batch()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)
        self._assert_common(out)

    def test_deep_stpp_vae_outputs_include_kl(self):
        torch.manual_seed(11)
        model = self._build_model(extra_cfg={"vae": True})
        model.eval()
        times, locations, lengths = _tiny_batch()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)
        self._assert_common(out)
        self.assertIn("kl_loss", out)
        self.assertTrue(torch.isfinite(out["kl_loss"]))


if __name__ == "__main__":
    unittest.main()
