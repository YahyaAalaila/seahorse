import numpy as np
import pytest

from seahorse.experiments.campaign_progress import NLLProgress


def test_resume_replays_only_uncommitted_targets_and_retains_target_identity(tmp_path):
    binding = {"manifest": "frozen", "checkpoint": "weights", "seed": 42}
    progress = NLLProgress(tmp_path / "progress", binding=binding, size=6)
    progress.arrays["nll"][:2] = [1.5, 2.5]
    progress.arrays["sequence_index"][:2] = [0, 0]
    progress.arrays["target_event_index"][:2] = [20, 21]
    progress.commit(2)
    progress.arrays["nll"][2:4] = 999  # interrupted work, never committed
    del progress
    resumed = NLLProgress(tmp_path / "progress", binding=binding, size=6)
    assert resumed.position == 2
    resumed.arrays["nll"][2:] = [3.5, 4.5, 5.5, 6.5]
    resumed.arrays["sequence_index"][2:] = [1, 1, 1, 1]
    resumed.arrays["target_event_index"][2:] = [22, 23, 24, 25]
    resumed.commit(6)
    resumed.finalize(tmp_path / "target_nll.npz")
    with np.load(tmp_path / "target_nll.npz") as scores:
        np.testing.assert_array_equal(scores["nll"], np.arange(1.5, 7.5))
        np.testing.assert_array_equal(scores["target_event_index"], np.arange(20, 26))


def test_changed_binding_and_incomplete_coverage_cannot_be_reported(tmp_path):
    progress = NLLProgress(tmp_path / "progress", binding={"checkpoint": "A"}, size=3)
    with pytest.raises(ValueError, match="incomplete"):
        progress.finalize(tmp_path / "target_nll.npz")
    with pytest.raises(ValueError, match="different checkpoint"):
        NLLProgress(tmp_path / "progress", binding={"checkpoint": "B"}, size=3)
