"""Tests for centralized config resolution precedence."""

from __future__ import annotations

import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from unified_stpp.data.registry import DataBundle
from unified_stpp.benchmark.hpo import RayTuneValidationReportCallback, run_hpo
from unified_stpp.config.schema import STPPConfig
from unified_stpp.config.tuning import TuningConfig
from unified_stpp.runner import STPPRunner
from unified_stpp.training.data_module import STPPDataModule


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

    def test_runner_uses_dataset_spatial_dim_for_model_construction(self):
        cfg = STPPConfig.from_source(
            preset="njsde",
            override_list=["model.spatial_dim=2"],
        )
        runner = STPPRunner(cfg)
        dataset = types.SimpleNamespace(
            sequences=[
                {
                    "times": [0.0, 1.0],
                    "locations": [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]],
                }
            ]
        )
        dm = STPPDataModule(
            DataBundle(
                train_dataset=dataset,
                val_dataset=dataset,
                test_dataset=None,
                collate_fn=lambda batch: batch,
                train_batch_sampler=None,
            )
        )

        runner._sync_model_spatial_dim_from_data(dm)

        self.assertEqual(runner.config.model.spatial_dim, 3)

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

    def test_bundled_hpo_configs_use_raw_data_and_non_deprecated_metric(self):
        config_dir = Path("unified_stpp/configs")
        for name in (
            "auto_stpp_hpo.yaml",
            "njsde_hpo.yaml",
            "neural_attncnf_hpo.yaml",
            "neural_jumpcnf_hpo.yaml",
            "nsmpp_hpo.yaml",
        ):
            raw = STPPConfig.raw_source_dict(config=str(config_dir / name))
            cfg_dict, raw_tuning = STPPConfig.split_tuning_dict(raw)
            tuning = TuningConfig.from_sources(yaml_tuning=raw_tuning)

            self.assertFalse(
                cfg_dict["data"]["normalize"],
                f"{name} should keep HPO on raw data to match the benchmark path",
            )
            self.assertEqual(
                tuning.metric,
                "val_objective",
                f"{name} should not rely on the deprecated val_nll label",
            )

    def test_all_bundled_hpo_configs_use_current_objective_label(self):
        config_dir = Path("unified_stpp/configs")
        for path in sorted(config_dir.glob("*_hpo.yaml")):
            raw = STPPConfig.raw_source_dict(config=str(path))
            cfg_dict, raw_tuning = STPPConfig.split_tuning_dict(raw)
            tuning = TuningConfig.from_sources(yaml_tuning=raw_tuning)

            self.assertIn("model", cfg_dict, path.name)
            self.assertIn("preset", cfg_dict["model"], path.name)
            self.assertEqual(
                tuning.metric,
                "val_objective",
                f"{path.name} should not rely on the deprecated val_nll label",
            )

    def test_slow_neural_hpo_configs_are_gpu_bounded(self):
        config_dir = Path("unified_stpp/configs")
        for name in (
            "njsde_hpo.yaml",
            "neural_attncnf_hpo.yaml",
            "neural_jumpcnf_hpo.yaml",
        ):
            raw = STPPConfig.raw_source_dict(config=str(config_dir / name))
            cfg_dict, raw_tuning = STPPConfig.split_tuning_dict(raw)
            tuning = TuningConfig.from_sources(yaml_tuning=raw_tuning)

            self.assertEqual(cfg_dict["training"]["device"], "cuda")
            self.assertEqual(tuning.n_gpus_per_trial, 1)
            self.assertEqual(tuning.max_concurrent_trials, 1)

    def test_neural_jumpcnf_hpo_keeps_adjoint_disabled(self):
        raw = STPPConfig.raw_source_dict(config="unified_stpp/configs/neural_jumpcnf_hpo.yaml")
        cfg_dict, _raw_tuning = STPPConfig.split_tuning_dict(raw)

        spatial_cfg = cfg_dict["model"]["decoder"]["spatial"]
        self.assertTrue(spatial_cfg["solve_reverse"])
        self.assertFalse(spatial_cfg["use_adjoint"])

    def test_auto_stpp_configs_use_paper_sliding_window_training_view(self):
        config_dir = Path("unified_stpp/configs")
        for name in (
            "auto_stpp.yaml",
            "auto_stpp_hpo.yaml",
        ):
            raw = STPPConfig.raw_source_dict(config=str(config_dir / name))
            cfg_dict, raw_tuning = STPPConfig.split_tuning_dict(raw)

            adapter_kwargs = cfg_dict["data"]["adapter_kwargs"]
            self.assertFalse(cfg_dict["data"]["normalize"])
            self.assertEqual(adapter_kwargs["training_view"], "sliding_window")
            self.assertEqual(adapter_kwargs["lookback"], 20)
            self.assertEqual(adapter_kwargs["lookahead"], 1)
            self.assertEqual(cfg_dict["training"]["batch_size"], 128)

            if raw_tuning:
                tuning = TuningConfig.from_sources(yaml_tuning=raw_tuning)
                self.assertEqual(tuning.n_trials, 30)
                self.assertEqual(
                    cfg_dict["model"]["decoder"]["n_prodnet"],
                    [2, 6, 10],
                )

    def test_temporal_gmm_yaml_fields_are_forwarded_directly(self):
        for preset in ("rmtpp_gmm", "thp_gmm"):
            cfg = STPPConfig.from_preset(preset)
            overrides = cfg.model.build_overrides

            self.assertNotIn("build_overrides", overrides)
            self.assertEqual(overrides["hidden_size"], 64)
            self.assertEqual(overrides["sigma_prior"], 1.0)
            self.assertEqual(overrides["sigma_kernel"], 0.5)
            self.assertEqual(overrides["tau"], 1.0)

    def test_hpo_does_not_pass_seed_to_legacy_tune_run(self):
        captured_kwargs = {}

        fake_tune = types.ModuleType("ray.tune")
        fake_tune.choice = lambda values: ("choice", values)
        fake_tune.uniform = lambda lo, hi: ("uniform", lo, hi)
        fake_tune.loguniform = lambda lo, hi: ("loguniform", lo, hi)
        fake_tune.randint = lambda lo, hi: ("randint", lo, hi)
        fake_tune.report = lambda metrics: None

        def fake_run(*args, **kwargs):
            del args
            captured_kwargs.update(kwargs)
            if "seed" in kwargs:
                raise TypeError("run() got an unexpected keyword argument 'seed'")
            return types.SimpleNamespace(best_config={"training.lr": 1.0e-3})

        fake_tune.run = fake_run
        fake_ray = types.ModuleType("ray")
        fake_ray.tune = fake_tune
        fake_ray.is_initialized = lambda: False
        fake_ray.init = lambda **kwargs: None

        raw = {
            "data": {"seed": 0},
            "model": {"preset": "poisson_gmm", "spatial_dim": 2},
            "training": {"lr": [1.0e-3, 2.0e-3]},
            "logging": {"out_dir": "runs"},
        }
        tuning = TuningConfig(
            n_trials=1,
            scheduler="none",
            search_alg="random",
            seed=123,
        )

        with patch.dict(sys.modules, {"ray": fake_ray, "ray.tune": fake_tune}):
            best = run_hpo(
                raw,
                tuning,
                train_seqs=[],
                val_seqs=[],
            )

        self.assertEqual(best.training.lr, 1.0e-3)
        self.assertNotIn("seed", captured_kwargs)

    def test_ray_tune_validation_callback_reports_intermediate_val_objective(self):
        reports = []
        callback = RayTuneValidationReportCallback(report_fn=reports.append)
        trainer = types.SimpleNamespace(
            callback_metrics={"val/nll": 1.25},
            current_epoch=3,
            is_global_zero=True,
        )
        lightning_module = types.SimpleNamespace(val_monitor_key="val/nll")

        callback.on_validation_epoch_end(trainer, lightning_module)

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["val_objective"], 1.25)
        self.assertEqual(reports[0]["val_metric_key"], "nll")
        self.assertEqual(reports[0]["epoch"], 3)
        self.assertIn("mem_rss_mb", reports[0])
        self.assertIn("mem_peak_rss_mb", reports[0])

    def test_ray_tune_validation_callback_skips_non_global_rank(self):
        reports = []
        callback = RayTuneValidationReportCallback(report_fn=reports.append)
        trainer = types.SimpleNamespace(
            callback_metrics={"val/nll": 1.25},
            current_epoch=0,
            is_global_zero=False,
        )
        lightning_module = types.SimpleNamespace(val_monitor_key="val/nll")

        callback.on_validation_epoch_end(trainer, lightning_module)

        self.assertEqual(reports, [])


if __name__ == "__main__":
    unittest.main()
