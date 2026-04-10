"""
Unit tests for TrainingConfig validation and scheduler selection.

These are fast, dependency-light tests that do not require a full trainer or
real data — they verify config-level correctness only.
"""

import unittest
import warnings
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
import torch.nn as nn

from unified_stpp.training.lightning_module import STPPLightningModule
from unified_stpp.config.schema import STPPConfig, TrainingConfig
from unified_stpp.models.unified_model import LossResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lm(
    lr_schedule: str,
    lr_step_size=None,
    lr_final=None,
    lr_warmup_epochs=0,
    optimizer="adamw",
) -> STPPLightningModule:
    """Build a minimal LightningModule with a stub model."""
    tc = TrainingConfig(
        lr_schedule=lr_schedule,
        lr_step_size=lr_step_size,
        lr_step_gamma=0.5,
        lr_final=lr_final,
        lr_warmup_epochs=lr_warmup_epochs,
        optimizer=optimizer,
    )
    return STPPLightningModule(model=nn.Linear(4, 4), tc=tc)


def _scheduler_from(lm: STPPLightningModule):
    """Call configure_optimizers and extract the scheduler object."""
    result = lm.configure_optimizers()
    return result["lr_scheduler"]["scheduler"]


# ---------------------------------------------------------------------------
# Scheduler selection tests
# ---------------------------------------------------------------------------

class TestSchedulerSelection(unittest.TestCase):

    def test_checkpoint_select_defaults_to_best(self):
        tc = TrainingConfig()
        self.assertEqual(tc.checkpoint_select, "best")

    def test_test_nll_space_defaults_to_raw(self):
        tc = TrainingConfig()
        self.assertEqual(tc.test_nll_space, "raw")

    def test_test_nll_space_accepts_native(self):
        tc = TrainingConfig(test_nll_space="native")
        self.assertEqual(tc.test_nll_space, "native")

    def test_checkpoint_select_accepts_last(self):
        tc = TrainingConfig(checkpoint_select="last")
        self.assertEqual(tc.checkpoint_select, "last")

    def test_lightning_runtime_fields_forward_to_trainer(self):
        tc = TrainingConfig(
            devices=2,
            strategy="ddp",
            precision="16-mixed",
            num_nodes=3,
        )
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            with patch("pytorch_lightning.Trainer") as trainer_cls:
                tc.build_trainer(run_dir, accelerator="gpu", loggers=[], monitor_key="val/nll")
        kwargs = trainer_cls.call_args.kwargs
        self.assertEqual(kwargs["accelerator"], "gpu")
        self.assertEqual(kwargs["devices"], 2)
        self.assertEqual(kwargs["strategy"], "ddp")
        self.assertEqual(kwargs["precision"], "16-mixed")
        self.assertEqual(kwargs["num_nodes"], 3)

    def test_constant_uses_lambda_lr(self):
        sched = _scheduler_from(_make_lm("constant"))
        self.assertIsInstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_constant_is_truly_flat(self):
        sched = _scheduler_from(_make_lm("constant"))
        # LambdaLR stores the lambda(s) as lr_lambdas
        fn = sched.lr_lambdas[0]
        self.assertAlmostEqual(fn(0),   1.0, places=9)
        self.assertAlmostEqual(fn(50),  1.0, places=9)
        self.assertAlmostEqual(fn(200), 1.0, places=9)

    def test_step_uses_step_lr(self):
        sched = _scheduler_from(_make_lm("step", lr_step_size=10))
        self.assertIsInstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_optimizer_adam_builds_adam(self):
        lm = _make_lm("constant", optimizer="adam")
        opt = lm.configure_optimizers()["optimizer"]
        self.assertIsInstance(opt, torch.optim.Adam)

    def test_optimizer_adamw_builds_adamw(self):
        lm = _make_lm("constant", optimizer="adamw")
        opt = lm.configure_optimizers()["optimizer"]
        self.assertIsInstance(opt, torch.optim.AdamW)

    def test_optimizer_adadelta_builds_adadelta(self):
        lm = _make_lm("constant", optimizer="adadelta")
        opt = lm.configure_optimizers()["optimizer"]
        self.assertIsInstance(opt, torch.optim.Adadelta)

    def test_unknown_optimizer_raises(self):
        lm = _make_lm("constant", optimizer="banana")
        with self.assertRaises(ValueError, msg="Unknown optimizer should raise"):
            lm.configure_optimizers()

    def test_linear_decay_uses_lambda_lr(self):
        sched = _scheduler_from(_make_lm("linear_decay", lr_final=5e-5, lr_warmup_epochs=5))
        self.assertIsInstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_step_without_step_size_raises(self):
        lm = _make_lm("step", lr_step_size=None)
        with self.assertRaises(ValueError):
            lm.configure_optimizers()

    def test_reduce_on_plateau_uses_correct_class(self):
        sched = _scheduler_from(_make_lm("reduce_on_plateau"))
        self.assertIsInstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_unknown_schedule_raises(self):
        lm = _make_lm("banana")
        with self.assertRaises(ValueError, msg="Unknown lr_schedule should raise"):
            lm.configure_optimizers()

    def test_legacy_lr_step_size_without_schedule_name_still_works(self):
        # Backward compat: lr_step_size set with default lr_schedule="constant"
        # should still produce StepLR, not a constant schedule.
        sched = _scheduler_from(_make_lm("constant", lr_step_size=5))
        self.assertIsInstance(sched, torch.optim.lr_scheduler.StepLR)


# ---------------------------------------------------------------------------
# TrainingConfig field-rename warning tests
# ---------------------------------------------------------------------------

class TestTrainingConfigWarnings(unittest.TestCase):

    def test_max_epochs_emits_warning(self):
        from unified_stpp.config.schema import TrainingConfig
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TrainingConfig(max_epochs=50)
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(
            any("max_epochs" in m and "n_epochs" in m for m in messages),
            f"Expected deprecation warning for 'max_epochs', got: {messages}",
        )

    def test_early_stopping_patience_emits_warning(self):
        from unified_stpp.config.schema import TrainingConfig
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TrainingConfig(early_stopping_patience=10)
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(
            any("early_stopping_patience" in m and "patience" in m for m in messages),
            f"Expected deprecation warning for 'early_stopping_patience', got: {messages}",
        )

    def test_valid_fields_emit_no_warning(self):
        from unified_stpp.config.schema import TrainingConfig
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TrainingConfig(n_epochs=50, patience=5)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(user_warnings, [], f"Unexpected warnings: {user_warnings}")


class TestYamlConfigCompatibility(unittest.TestCase):
    def test_from_yaml_accepts_legacy_python_tuple_tag(self):
        raw = """
data:
  protocol: raw
model:
  preset: auto_stpp_faithful
training:
  batch_size: 128
  checkpoint_select: best
extra_tuple: !!python/tuple [1, 2, 3]
"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy.yaml"
            path.write_text(raw)
            cfg = STPPConfig.from_yaml(path, sanitize=False)
        self.assertEqual(cfg.model.preset, "auto_stpp")


class _DummyEvalModel:
    def __init__(self, result: LossResult):
        self._result = result
        self.event_model = SimpleNamespace(capabilities=SimpleNamespace(metric_key="nll"))

    def eval_forward(self, **kwargs):
        del kwargs
        return {"dummy": True}

    def compute_loss(self, output):
        del output
        return self._result


class TestTestNLLReporting(unittest.TestCase):
    def _dummy_batch(self):
        return {
            "times": torch.zeros(1, 1),
            "locations": torch.zeros(1, 1, 2),
            "lengths": torch.ones(1, dtype=torch.long),
        }

    def test_test_step_prefers_raw_reporting_when_available(self):
        result = LossResult(
            loss=torch.tensor(1.0),
            nll=torch.tensor(1.0),
            total_events=torch.tensor(4.0),
            kl=None,
            aux_terms={},
            temporal_nll=0.4,
            spatial_nll=0.6,
            extra_metrics={
                "raw_space_nll": 1.5,
                "raw_space_temporal_nll": 0.7,
                "raw_space_spatial_nll": 0.8,
            },
        )
        lm = STPPLightningModule(model=_DummyEvalModel(result), tc=TrainingConfig())
        captured = {}
        lm.log = lambda name, value, **kwargs: captured.setdefault(
            name,
            float(value.detach().reshape(()).item()) if isinstance(value, torch.Tensor) else float(value),
        )

        lm.test_step(self._dummy_batch(), 0)

        self.assertAlmostEqual(captured["test/nll"], 1.5, places=6)
        self.assertAlmostEqual(captured["test/temporal_nll"], 0.7, places=6)
        self.assertAlmostEqual(captured["test/spatial_nll"], 0.8, places=6)
        self.assertAlmostEqual(captured["test/native_nll"], 1.0, places=6)
        self.assertAlmostEqual(captured["test/native_temporal_nll"], 0.4, places=6)
        self.assertAlmostEqual(captured["test/native_spatial_nll"], 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
