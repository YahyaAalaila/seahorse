"""
Unit tests for TrainingConfig validation and scheduler selection.

These are fast, dependency-light tests that do not require a full trainer or
real data — they verify config-level correctness only.
"""

import unittest
import warnings

import torch
import torch.nn as nn

from unified_stpp.training.lightning_module import STPPLightningModule
from unified_stpp.config.schema import TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lm(lr_schedule: str, lr_step_size=None) -> STPPLightningModule:
    """Build a minimal LightningModule with a stub model."""
    tc = TrainingConfig(
        lr_schedule=lr_schedule,
        lr_step_size=lr_step_size,
        lr_step_gamma=0.5,
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


if __name__ == "__main__":
    unittest.main()
