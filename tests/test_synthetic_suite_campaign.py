from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from unified_stpp.training.callbacks import CURVE_FIELDNAMES


def _load_campaign_module():
    path = Path("scripts/synthetic_suite_campaign.py").resolve()
    spec = importlib.util.spec_from_file_location("synthetic_suite_campaign_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SyntheticSuiteCampaignTest(unittest.TestCase):
    def _write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "times": [0.0, 1.0, 2.0],
            "locations": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        }
        with open(path, "w") as f:
            f.write(json.dumps(payload) + "\n")

    def _make_suite(self, root: Path) -> Path:
        suite = root / "suite4_heterogeneity"
        for config_id in ("H1", "H0"):
            self._write_jsonl(suite / "jsonl" / config_id / "train.jsonl")
            self._write_jsonl(suite / "jsonl" / config_id / "val.jsonl")
            self._write_jsonl(suite / "jsonl" / config_id / "test.jsonl")
        (suite / "ground_truth").mkdir(parents=True, exist_ok=True)
        (suite / "sequences").mkdir(parents=True, exist_ok=True)
        with open(suite / "metadata.json", "w") as f:
            json.dump({"suite": "suite4_heterogeneity"}, f)
        return suite

    def test_discovery_finds_configs_and_anchor(self):
        module = _load_campaign_module()
        with tempfile.TemporaryDirectory() as td:
            suite = self._make_suite(Path(td))
            configs = module._discover_suite_configs(suite)
            self.assertEqual([cfg.config_id for cfg in configs], ["H0", "H1"])
            self.assertEqual(module._anchor_config(configs).config_id, "H0")

    def test_no_hpo_suite_preset_uses_bundled_yaml(self):
        module = _load_campaign_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite = self._make_suite(root)
            anchor = module._anchor_config(module._discover_suite_configs(suite))
            out_root = root / "campaign_out"
            with patch.object(
                module,
                "_run_tune_subprocess",
                side_effect=AssertionError("no-HPO preset should not launch tune subprocess"),
            ):
                module._run_tune_stage(
                    suite_name="suite4_heterogeneity",
                    anchor=anchor,
                    presets=["smash"],
                    hpo_config_dir=root / "unused_hpo",
                    out_root=out_root,
                    hpo_seed=42,
                    resume=False,
                    no_hpo_suite_presets=module._no_hpo_suite_presets("skip-heavy"),
                )
            best_yaml = out_root / "tune" / "smash_best.yaml"
            self.assertTrue(best_yaml.exists())
            payload = yaml.safe_load(best_yaml.read_text())
            self.assertEqual(payload["model"]["preset"], "smash")
            self.assertIn("encoder", payload["model"])

    def test_generative_thin_policy_runs_smash_hpo_instead_of_bundled_yaml(self):
        module = _load_campaign_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite = self._make_suite(root)
            anchor = module._anchor_config(module._discover_suite_configs(suite))
            out_root = root / "campaign_out"
            hpo_dir = root / "hpo"
            hpo_dir.mkdir(parents=True, exist_ok=True)
            (hpo_dir / "smash_hpo.yaml").write_text("model:\n  preset: smash\n")

            calls = []

            def fake_tune(**kwargs):
                calls.append(kwargs["preset"])
                kwargs["out_path"].parent.mkdir(parents=True, exist_ok=True)
                kwargs["out_path"].write_text("model:\n  preset: smash\n")

            with patch.object(module, "_run_tune_subprocess", side_effect=fake_tune):
                module._run_tune_stage(
                    suite_name="suite4_heterogeneity",
                    anchor=anchor,
                    presets=["smash"],
                    hpo_config_dir=hpo_dir,
                    out_root=out_root,
                    hpo_seed=42,
                    resume=False,
                    no_hpo_suite_presets=module._no_hpo_suite_presets("generative-thin"),
                )
            self.assertEqual(calls, ["smash"])

    def test_campaign_all_stage_writes_outputs_and_resume_skips_completed_runs(self):
        module = _load_campaign_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite = self._make_suite(root)
            out_root = root / "campaign_out"
            hpo_dir = root / "hpo"
            hpo_dir.mkdir(parents=True, exist_ok=True)
            (hpo_dir / "dummy_preset_hpo.yaml").write_text("model:\n  preset: dummy_preset\n")

            def fake_tune(*, preset, hpo_yaml, train_path, val_path, hpo_seed, out_path):
                del hpo_yaml, train_path, val_path, hpo_seed
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(f"model:\n  preset: {preset}\n")
                out_path.with_suffix(".trials.json").write_text("[]\n")
                out_path.with_suffix(".trials.csv").write_text("trial_id,val_objective\n")
                out_path.with_suffix(".hpo_manifest.json").write_text("{}\n")

            def fake_fit(*, suite_name, config, preset, seed, best_yaml, out_root, curve_step, device, run_batch_size):
                del best_yaml, curve_step, device, run_batch_size
                run_dir = out_root / "fit" / suite_name / config.config_id / preset / f"seed_{seed}" / "run_0"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "run_result.json").write_text(
                    json.dumps(
                        {
                            "preset": preset,
                            "dataset_id": config.config_id,
                            "seed": seed,
                            "test_nll": 1.0,
                            "run_dir": str(run_dir),
                        }
                    )
                )
                curve_rows = [
                    {
                        "suite": suite_name,
                        "config_id": config.config_id,
                        "preset": preset,
                        "seed": seed,
                        "epoch": 1,
                        "train_progress_fraction": 0.1,
                        "train_progress_percent": 10.0,
                        "test_nll": 1.0 + (0.5 if config.config_id == "H1" else 0.0),
                        "nll_kind": "exact",
                        "nll_report_space": "raw",
                        "test_nll_method": "exact_next_event_from_eventwise_terms",
                        "test_nll_contexts": 2,
                        "test_nll_scored_contexts": 2,
                        "test_nll_missing_contexts": 0,
                        "wall_time_sec": 0.1,
                    },
                    {
                        "suite": suite_name,
                        "config_id": config.config_id,
                        "preset": preset,
                        "seed": seed,
                        "epoch": 10,
                        "train_progress_fraction": 1.0,
                        "train_progress_percent": 100.0,
                        "test_nll": 0.8 + (0.5 if config.config_id == "H1" else 0.0),
                        "nll_kind": "exact",
                        "nll_report_space": "raw",
                        "test_nll_method": "exact_next_event_from_eventwise_terms",
                        "test_nll_contexts": 2,
                        "test_nll_scored_contexts": 2,
                        "test_nll_missing_contexts": 0,
                        "wall_time_sec": 1.0,
                    },
                ]
                with open(run_dir / "test_nll_curve.jsonl", "w") as f:
                    for row in curve_rows:
                        f.write(json.dumps(row) + "\n")
                with open(run_dir / "test_nll_curve.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=CURVE_FIELDNAMES)
                    writer.writeheader()
                    for row in curve_rows:
                        writer.writerow(row)
                return module.RunIndexRecord(
                    suite=suite_name,
                    config_id=config.config_id,
                    preset=preset,
                    seed=seed,
                    run_dir=run_dir,
                    run_result_path=run_dir / "run_result.json",
                    curve_jsonl_path=run_dir / "test_nll_curve.jsonl",
                    curve_csv_path=run_dir / "test_nll_curve.csv",
                    train_path=config.train_path,
                    val_path=config.val_path,
                    test_path=config.test_path,
                )

            argv = [
                "--suite-path",
                str(suite),
                "--presets",
                "dummy_preset",
                "--out",
                str(out_root),
                "--hpo-config-dir",
                str(hpo_dir),
                "--stage",
                "all",
            ]

            with patch.object(module, "_run_tune_subprocess", side_effect=fake_tune), patch.object(
                module, "_fit_campaign_run", side_effect=fake_fit
            ):
                rc = module.main(argv)
            self.assertEqual(rc, 0)

            self.assertTrue((out_root / "tune" / "dummy_preset_best.yaml").exists())
            self.assertTrue((out_root / "manifests" / "campaign_manifest.json").exists())
            self.assertTrue((out_root / "manifests" / "run_index.jsonl").exists())
            self.assertTrue((out_root / "tables" / "test_nll_curve_long.csv").exists())
            self.assertTrue((out_root / "tables" / "test_nll_curve_summary.csv").exists())
            self.assertTrue((out_root / "tables" / "evaluate_targets.csv").exists())
            self.assertTrue(
                (out_root / "plots" / "suite4_heterogeneity__dummy_preset__test_nll_curve.png").exists()
            )

            with open(out_root / "tables" / "evaluate_targets.csv", newline="") as f:
                targets = list(csv.DictReader(f))
            self.assertEqual(len(targets), 2)
            self.assertEqual(sorted(row["config_id"] for row in targets), ["H0", "H1"])
            self.assertTrue(all(Path(row["test_path"]).exists() for row in targets))

            manifest = json.loads((out_root / "manifests" / "campaign_manifest.json").read_text())
            self.assertEqual(manifest["hpo_policy"], "skip-heavy")

            with patch.object(module, "_run_tune_subprocess", side_effect=AssertionError("tune should skip")), patch.object(
                module,
                "_fit_campaign_run",
                side_effect=AssertionError("run stage should skip completed runs"),
            ):
                rc = module.main(argv)
            self.assertEqual(rc, 0)

    def test_fit_campaign_run_loads_tuned_yaml_without_hpo_sanitization(self):
        module = _load_campaign_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite = self._make_suite(root)
            config = module._discover_suite_configs(suite)[0]
            best_yaml = root / "poisson_gmm_best.yaml"
            yaml.safe_dump(
                {
                    "data": {
                        "protocol": "raw",
                        "normalize": True,
                        "paper_split_ratio": [8, 1, 1],
                    },
                    "model": {
                        "preset": "poisson_gmm",
                        "hidden_dim": 128,
                        "spatial_dim": 2,
                        "sigma_prior": 1.0,
                        "sigma_kernel": 0.5,
                        "tau": 1.0,
                    },
                    "training": {
                        "lr": 1.0e-3,
                        "batch_size": 32,
                        "n_epochs": 5,
                        "patience": 2,
                        "test_nll_space": "raw",
                    },
                    "logging": {"out_dir": "artifacts"},
                },
                best_yaml.open("w"),
                sort_keys=False,
            )

            captured = {}

            def fake_fit(self, train_seqs, val_seqs, test_seqs, data_module=None, dataset_id="unknown", extra_callbacks=None):
                del train_seqs, val_seqs, test_seqs, data_module, dataset_id
                captured["config"] = self.config
                captured["callbacks"] = list(extra_callbacks or [])
                run_dir = root / "run_dir"
                run_dir.mkdir(parents=True, exist_ok=True)
                return SimpleNamespace(run_dir=run_dir)

            with patch.object(module.STPPRunner, "fit", new=fake_fit):
                record = module._fit_campaign_run(
                    suite_name="suite4_heterogeneity",
                    config=config,
                    preset="poisson_gmm",
                    seed=42,
                    best_yaml=best_yaml,
                    out_root=root / "campaign_out",
                    curve_step=0.1,
                    device="cpu",
                    run_batch_size=64,
                )

            cfg = captured["config"]
            self.assertEqual(cfg.data.paper_split_ratio, (8, 1, 1))
            self.assertEqual(cfg.data.train_path, str(config.train_path))
            self.assertEqual(cfg.data.val_path, str(config.val_path))
            self.assertEqual(cfg.data.test_path, str(config.test_path))
            self.assertEqual(cfg.training.patience, None)
            self.assertEqual(cfg.training.device, "cpu")
            self.assertEqual(cfg.training.batch_size, 64)
            self.assertEqual(cfg.training.checkpoint_select, "best")
            self.assertEqual(cfg.training.test_nll_space, "raw")
            self.assertEqual(cfg.logging.experiment_name, "suite4_heterogeneity/H0/poisson_gmm/seed_42")
            self.assertEqual(len(captured["callbacks"]), 1)
            self.assertEqual(record.config_id, "H0")
