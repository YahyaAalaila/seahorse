import tempfile
import unittest
from pathlib import Path

import torch

from unified_stpp.runner.artifacts import checkpoint_file, load_state_dict


class TestCheckpointSelection(unittest.TestCase):
    def test_checkpoint_file_resolves_best_and_last(self):
        run_dir = Path("/tmp/example_run")
        self.assertEqual(checkpoint_file(run_dir, "best"), run_dir / "checkpoints" / "best.ckpt")
        self.assertEqual(checkpoint_file(run_dir, "last"), run_dir / "checkpoints" / "last.ckpt")

    def test_load_state_dict_prefers_selected_lightning_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir()

            torch.save({"state_dict": {"model.weight": torch.tensor([1.0])}}, ckpt_dir / "best.ckpt")
            torch.save({"state_dict": {"model.weight": torch.tensor([2.0])}}, ckpt_dir / "last.ckpt")

            best_state = load_state_dict(run_dir, selection="best")
            last_state = load_state_dict(run_dir, selection="last")

            self.assertEqual(float(best_state["weight"].item()), 1.0)
            self.assertEqual(float(last_state["weight"].item()), 2.0)

    def test_load_state_dict_falls_back_to_plain_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            torch.save({"weight": torch.tensor([3.0])}, run_dir / "model.ckpt")

            state = load_state_dict(run_dir, selection="best")
            self.assertEqual(float(state["weight"].item()), 3.0)


if __name__ == "__main__":
    unittest.main()
