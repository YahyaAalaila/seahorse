"""Compatibility coverage for the historical ``unified_stpp.registry`` shim."""

from __future__ import annotations

import unittest

from unified_stpp.registry import PRESETS, build_model


class TestRegistryCompat(unittest.TestCase):
    def test_presets_expose_registered_names(self):
        from unified_stpp.models.configs import ConfigRegistry

        self.assertEqual(set(PRESETS), set(ConfigRegistry.preset_names()))
        self.assertTrue(all(payload == {} for payload in PRESETS.values()))

    def test_build_model_supports_known_preset(self):
        from unified_stpp.models.unified_model import UnifiedSTPP

        model = build_model(
            config={},
            preset="poisson_gmm",
            spatial_dim=2,
            hidden_dim=16,
            event_cov_dim=0,
            field_cov_dim=0,
        )

        self.assertIsInstance(model, UnifiedSTPP)

    def test_build_model_rejects_unknown_preset(self):
        with self.assertRaisesRegex(ValueError, "Unknown preset"):
            build_model(
                config={},
                preset="not_a_real_preset",
                spatial_dim=2,
                hidden_dim=16,
                event_cov_dim=0,
                field_cov_dim=0,
            )


if __name__ == "__main__":
    unittest.main()
