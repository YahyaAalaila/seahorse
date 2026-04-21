from __future__ import annotations

import csv
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from unified_stpp.training.callbacks import (
    CURVE_FIELDNAMES,
    PeriodicTestNLLCallback,
    _progress_milestones,
)


class _DummyModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self


class PeriodicTestNLLCallbackTest(unittest.TestCase):
    def test_progress_milestones_cover_full_budget_and_dedupe(self):
        self.assertEqual(_progress_milestones(10, 0.1), list(range(1, 11)))
        self.assertEqual(_progress_milestones(7, 0.25), [2, 4, 6, 7])

    def test_callback_writes_curve_row_and_restores_training_mode(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            config = types.SimpleNamespace(
                training=types.SimpleNamespace(
                    n_epochs=10,
                    predictive_test_nll_samples=128,
                ),
                data=types.SimpleNamespace(normalize=False),
            )
            callback = PeriodicTestNLLCallback(
                suite="suite4_heterogeneity",
                config_id="H0",
                preset="deep_stpp",
                seed=42,
                curve_step=0.1,
                config=config,
            )
            callback.bind_run_context(run_dir=run_dir)

            test_dataset = types.SimpleNamespace(
                sequences=[
                    {
                        "times": np.asarray([0.0, 1.0], dtype=np.float32),
                        "locations": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                    }
                ]
            )
            datamodule = types.SimpleNamespace(
                _bundle=types.SimpleNamespace(test_dataset=test_dataset),
                get_norm_stats=lambda normalize: {"normalize": normalize},
            )
            pl_module = types.SimpleNamespace(
                model=_DummyModel(),
                device=torch.device("cpu"),
            )
            trainer = types.SimpleNamespace(
                is_global_zero=True,
                sanity_checking=False,
                current_epoch=0,
                datamodule=datamodule,
            )

            captured = {}

            def _fake_compute(runner, seqs, *, device, predictive_samples, seed):
                captured["runner_config"] = runner.config
                captured["norm_stats"] = runner.norm_stats
                captured["seqs"] = seqs
                captured["device"] = device
                captured["predictive_samples"] = predictive_samples
                captured["seed"] = seed
                pl_module.model.eval()
                return {
                    "mean_nll": 1.25,
                    "method": "exact_next_event_from_eventwise_terms",
                    "kind": "exact",
                    "report_space": "raw",
                    "n_contexts": 4,
                    "n_scored_contexts": 4,
                    "n_missing_contexts": 0,
                }

            callback.on_fit_start(trainer, pl_module)
            with patch(
                "unified_stpp.training.callbacks.compute_next_event_test_nll",
                side_effect=_fake_compute,
            ):
                callback.on_train_epoch_end(trainer, pl_module)

            self.assertIs(captured["runner_config"], config)
            self.assertEqual(captured["norm_stats"], {"normalize": False})
            self.assertEqual(len(captured["seqs"]), 1)
            self.assertEqual(captured["device"], torch.device("cpu"))
            self.assertEqual(captured["predictive_samples"], 128)
            self.assertEqual(captured["seed"], 42)
            self.assertTrue(pl_module.model.training)

            jsonl_path = run_dir / "test_nll_curve.jsonl"
            csv_path = run_dir / "test_nll_curve.csv"
            self.assertTrue(jsonl_path.exists())
            self.assertTrue(csv_path.exists())

            with open(jsonl_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["suite"], "suite4_heterogeneity")
            self.assertEqual(rows[0]["config_id"], "H0")
            self.assertEqual(rows[0]["preset"], "deep_stpp")
            self.assertEqual(rows[0]["seed"], 42)
            self.assertEqual(rows[0]["epoch"], 1)
            self.assertAlmostEqual(rows[0]["train_progress_percent"], 10.0, places=6)
            self.assertEqual(rows[0]["test_nll"], 1.25)
            self.assertEqual(rows[0]["nll_report_space"], "raw")

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, CURVE_FIELDNAMES)
                csv_rows = list(reader)
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]["test_nll_method"], "exact_next_event_from_eventwise_terms")
