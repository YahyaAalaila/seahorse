"""Compatibility coverage for the historical ``seahorse.registry`` shim."""

from __future__ import annotations

import unittest

from seahorse.config.schema import STPPConfig
from seahorse.registry import PRESETS, build_model


class TestRegistryCompat(unittest.TestCase):
    def test_presets_expose_canonical_public_names(self):
        from seahorse.models.configs import ConfigRegistry

        self.assertEqual(set(PRESETS), set(ConfigRegistry.canonical_preset_names()))
        self.assertIn("njsde", PRESETS)
        self.assertNotIn("neural_cond_gmm", PRESETS)
        self.assertTrue(all(payload == {} for payload in PRESETS.values()))

    def test_registry_exposes_canonical_and_status_metadata(self):
        from seahorse.models.configs import ConfigRegistry

        auto = ConfigRegistry.describe("auto_stpp")
        self.assertEqual(auto.canonical_name, "auto_stpp")
        self.assertEqual(auto.status, "canonical")
        self.assertFalse(auto.is_alias)

        neural = ConfigRegistry.describe("njsde")
        self.assertEqual(neural.canonical_name, "njsde")
        self.assertEqual(neural.status, "canonical")
        self.assertFalse(neural.is_alias)

        deprecated_neural = ConfigRegistry.describe("neural_cond_gmm")
        self.assertEqual(deprecated_neural.canonical_name, "njsde")
        self.assertEqual(deprecated_neural.status, "deprecated")
        self.assertEqual(deprecated_neural.canonical_status, "canonical")
        self.assertTrue(deprecated_neural.is_alias)

    def test_config_loading_supports_nsmpp_without_direct_module_import(self):
        cfg = STPPConfig.from_source(preset="nsmpp", config=None)
        self.assertEqual(cfg.model.preset, "nsmpp")

    def test_deprecated_neural_cond_gmm_alias_loads_njsde(self):
        cfg = STPPConfig.from_source(preset="neural_cond_gmm", config=None)
        self.assertEqual(cfg.model.preset, "njsde")

    def test_build_model_supports_known_preset(self):
        from seahorse.models.unified_model import UnifiedSTPP

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
