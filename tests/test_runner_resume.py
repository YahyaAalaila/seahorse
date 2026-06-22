from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from seahorse.config import STPPConfig
from seahorse.runner import STPPRunner


class RunnerResumeCheckpointTest(unittest.TestCase):
    def test_fit_forwards_resume_checkpoint_to_lightning(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_path = tmp_path / "last.ckpt"
            ckpt_path.write_bytes(b"checkpoint")
            cfg = STPPConfig.from_source(
                preset="poisson_gmm",
                override_list=[
                    f"logging.out_dir={tmp_path / 'runs'}",
                    f"training.resume_from_checkpoint={ckpt_path}",
                ],
            )
            runner = STPPRunner(cfg)
            dm = SimpleNamespace()
            model = torch.nn.Linear(1, 1)
            lm = SimpleNamespace()
            trainer = SimpleNamespace(fit=Mock())

            runner._prepare_data_module = Mock(return_value=dm)
            runner._sync_model_spatial_dim_from_data = Mock()
            runner._build_training_stack = Mock(return_value=(model, lm, trainer))
            runner._finalize_fit = Mock(return_value="done")

            result = runner.fit([], [], dataset_id="toy")

            self.assertEqual(result, "done")
            trainer.fit.assert_called_once_with(
                lm,
                datamodule=dm,
                ckpt_path=str(ckpt_path.resolve()),
            )

    def test_fit_fails_fast_when_resume_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.ckpt"
            cfg = STPPConfig.from_source(
                preset="poisson_gmm",
                override_list=[f"training.resume_from_checkpoint={missing}"],
            )
            runner = STPPRunner(cfg)
            runner._prepare_data_module = Mock(return_value=SimpleNamespace())
            runner._sync_model_spatial_dim_from_data = Mock()

            with self.assertRaises(FileNotFoundError):
                runner.fit([], [], dataset_id="toy")


if __name__ == "__main__":
    unittest.main()
