"""Compatibility checks for active coarse-framework presets."""

from __future__ import annotations

import unittest

import torch

from unified_stpp.registry import build_model


_ACTIVE_PRESETS = (
    "neural_stpp_attn_sc",
    "neural_stpp_jump_sc",
    "deep_stpp",
    "auto_stpp",
    "auto_stpp_faithful",
    "smash",
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
    def test_active_presets_use_coarse_path(self):
        for preset in _ACTIVE_PRESETS:
            with self.subTest(preset=preset):
                model = build_model(
                    config={},
                    preset=preset,
                    spatial_dim=2,
                    hidden_dim=16,
                )
                self.assertIsNotNone(model.state_model)
                self.assertIsNotNone(model.event_model)

    def test_active_presets_forward_finite(self):
        times, locations, lengths = _tiny_batch()
        for preset in _ACTIVE_PRESETS:
            with self.subTest(preset=preset):
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
