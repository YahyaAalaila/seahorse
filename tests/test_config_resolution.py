"""Tests for centralized config resolution precedence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from unified_stpp.config.schema import STPPConfig
from unified_stpp.config.tuning import TuningConfig
from unified_stpp.runner import STPPRunner


class ConfigResolutionTest(unittest.TestCase):
    def _write_yaml(self, data: dict) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "config.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return path

    def test_from_source_applies_cli_values(self):
        path = self._write_yaml(
            {
                "data": {},
                "model": {"preset": "auto_stpp"},
                "training": {},
                "logging": {"out_dir": "yaml_out"},
            }
        )
        cfg = STPPConfig.from_source(
            config=str(path),
            cli_values={"logging": {"out_dir": "cli_out"}},
        )
        self.assertEqual(cfg.logging.out_dir, "cli_out")

    def test_from_source_applies_override_after_cli_values(self):
        path = self._write_yaml(
            {
                "data": {},
                "model": {"preset": "auto_stpp"},
                "training": {},
                "logging": {"out_dir": "yaml_out"},
            }
        )
        cfg = STPPConfig.from_source(
            config=str(path),
            cli_values={"logging": {"out_dir": "cli_out"}},
            override_list=["logging.out_dir=override_out"],
        )
        self.assertEqual(cfg.logging.out_dir, "override_out")

    def test_from_source_leaves_yaml_value_when_cli_absent(self):
        path = self._write_yaml(
            {
                "data": {},
                "model": {"preset": "auto_stpp"},
                "training": {},
                "logging": {"out_dir": "yaml_out"},
            }
        )
        cfg = STPPConfig.from_source(config=str(path))
        self.assertEqual(cfg.logging.out_dir, "yaml_out")

    def test_raw_source_dict_preserves_search_space_syntax(self):
        path = self._write_yaml(
            {
                "data": {},
                "model": {"preset": "auto_stpp"},
                "training": {"lr": {"min": 1e-4, "max": 1e-2, "default": 1e-3}},
            }
        )
        raw = STPPConfig.raw_source_dict(config=str(path))
        self.assertEqual(raw["training"]["lr"]["min"], 1e-4)
        self.assertEqual(raw["training"]["lr"]["max"], 1e-2)
        self.assertEqual(raw["training"]["lr"]["default"], 1e-3)

    def test_runner_from_config_source_delegates_to_config_resolution(self):
        path = self._write_yaml(
            {
                "data": {},
                "model": {"preset": "auto_stpp"},
                "training": {},
                "logging": {"out_dir": "yaml_out"},
            }
        )
        runner = STPPRunner.from_config_source(
            preset=None,
            config=str(path),
            cli_values={"logging": {"out_dir": "cli_out"}},
            override_list=["logging.out_dir=override_out"],
        )
        self.assertEqual(runner.config.logging.out_dir, "override_out")

    def test_tuning_from_sources_uses_yaml_base_and_cli_override(self):
        tuning = TuningConfig.from_sources(
            yaml_tuning={
                "n_trials": 12,
                "search_alg": "random",
                "scheduler": "none",
                "fail_fast": False,
            },
            cli_values={"n_trials": 5, "fail_fast": True},
        )
        self.assertEqual(tuning.n_trials, 5)
        self.assertTrue(tuning.fail_fast)
        self.assertEqual(tuning.search_alg, "random")
        self.assertEqual(tuning.scheduler, "none")


if __name__ == "__main__":
    unittest.main()
