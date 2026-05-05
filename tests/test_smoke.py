import unittest

import torch

from unified_stpp.models.configs import ConfigRegistry
from unified_stpp.registry import build_model


class SmokeTest(unittest.TestCase):
    def test_forward_smoke_preset_nll_is_finite(self):
        times = torch.tensor([[0.0, 0.2, 0.5, 0.9]], dtype=torch.float32)
        locations = torch.tensor(
            [[[0.0, 0.0], [0.3, -0.1], [0.1, 0.2], [-0.2, 0.1]]], dtype=torch.float32
        )
        lengths = torch.tensor([4], dtype=torch.long)

        for preset in (
            "deep_stpp",
            "auto_stpp",
            "smash",
        ):
            with self.subTest(preset=preset):
                self.assertEqual(ConfigRegistry.canonical_status(preset), "canonical")
                torch.manual_seed(0)
                model = build_model(
                    config={},
                    preset=preset,
                    spatial_dim=2,
                    hidden_dim=16,
                    event_cov_dim=0,
                    field_cov_dim=0,
                )
                model.eval()
                with torch.no_grad():
                    out = model(times=times, locations=locations, lengths=lengths)
                self.assertIn("nll", out)
                self.assertTrue(torch.isfinite(out["nll"]))


if __name__ == "__main__":
    unittest.main()
