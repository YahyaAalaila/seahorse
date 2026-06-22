"""Compatibility checks for active coarse-framework presets."""

from __future__ import annotations

import unittest

import torch

from seahorse.models.configs import ConfigRegistry
from seahorse.registry import build_model


_FORWARD_SMOKE_PRESETS = (
    "deep_stpp",
    "auto_stpp",
    "smash",
)

_PAPER_NEURAL_PRESETS = (
    "neural_attncnf",
    "neural_jumpcnf",
    "njsde",
)

_REMOVED_PRESETS = (
    "neural_stpp",
    "neural_stpp_jump",
    "deep_stpp_free",
    "dstpp",
)


def _tiny_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    times = torch.tensor(
        [[0.00, 0.20, 0.50, 0.90], [0.00, 0.10, 0.40, 0.80]],
        dtype=torch.float32,
    )
    locations = torch.tensor(
        [
            [[0.0, 0.0], [0.3, -0.1], [0.1, 0.2], [-0.2, 0.1]],
            [[0.0, 0.0], [0.2, 0.1], [0.2, -0.2], [0.4, 0.3]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([4, 4], dtype=torch.long)
    return times, locations, lengths


class TestStateEventCompat(unittest.TestCase):
    def test_registered_presets_build_across_statuses(self):
        for preset in _FORWARD_SMOKE_PRESETS + _PAPER_NEURAL_PRESETS:
            with self.subTest(preset=preset):
                model = build_model(
                    config={},
                    preset=preset,
                    spatial_dim=2,
                    hidden_dim=16,
                )
                self.assertIsNotNone(model.state_model)
                self.assertIsNotNone(model.event_model)

    def test_forward_smoke_presets_forward_finite(self):
        times, locations, lengths = _tiny_batch()
        for preset in _FORWARD_SMOKE_PRESETS:
            with self.subTest(preset=preset):
                self.assertEqual(ConfigRegistry.canonical_status(preset), "canonical")
                torch.manual_seed(7)
                model = build_model(
                    config={},
                    preset=preset,
                    spatial_dim=2,
                    hidden_dim=16,
                )
                model.eval()
                with torch.no_grad():
                    out = model(times=times, locations=locations, lengths=lengths)
                self.assertIn("nll", out)
                self.assertIn("nll_per_event", out)
                self.assertIn("total_events", out)
                self.assertTrue(torch.isfinite(out["nll"]))

    def test_paper_neural_presets_are_benchmark_supported(self):
        for preset in _PAPER_NEURAL_PRESETS:
            with self.subTest(preset=preset):
                self.assertEqual(ConfigRegistry.canonical_status(preset), "canonical")

    def test_removed_presets_rejected(self):
        for preset in _REMOVED_PRESETS:
            with self.subTest(preset=preset):
                with self.assertRaises(ValueError):
                    build_model(
                        config={},
                        preset=preset,
                        spatial_dim=2,
                        hidden_dim=16,
                    )


if __name__ == "__main__":
    unittest.main()
