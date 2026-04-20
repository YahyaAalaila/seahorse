"""Compatibility coverage for the historical ``unified_stpp.registry`` shim."""

from __future__ import annotations

import unittest

from unified_stpp.config.schema import STPPConfig
from unified_stpp.registry import PRESETS, build_model


class TestRegistryCompat(unittest.TestCase):
    def test_presets_expose_canonical_public_names(self):
        from unified_stpp.models.configs import ConfigRegistry

        self.assertEqual(set(PRESETS), set(ConfigRegistry.canonical_preset_names()))
        self.assertTrue(all(payload == {} for payload in PRESETS.values()))

    def test_registry_exposes_canonical_and_status_metadata(self):
        from unified_stpp.models.configs import ConfigRegistry

        auto = ConfigRegistry.describe("auto_stpp")
        self.assertEqual(auto.canonical_name, "auto_stpp")
        self.assertEqual(auto.status, "canonical")
        self.assertFalse(auto.is_alias)

        legacy = ConfigRegistry.describe("auto_stpp_legacy")
        self.assertEqual(legacy.canonical_name, "auto_stpp_legacy")
        self.assertEqual(legacy.status, "legacy")
        self.assertFalse(legacy.is_alias)

        provisional = ConfigRegistry.describe("neural_cond_gmm")
        self.assertEqual(provisional.status, "provisional")

    def test_config_loading_supports_nsmpp_without_direct_module_import(self):
        cfg = STPPConfig.from_source(preset="nsmpp", config=None)
        self.assertEqual(cfg.model.preset, "nsmpp")

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
