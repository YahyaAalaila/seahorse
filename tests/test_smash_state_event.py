"""SMASH coarse-path regression tests."""

from __future__ import annotations

import unittest

import torch

from unified_stpp.registry import build_model


def _tiny_unmarked_batch():
    times = torch.tensor(
        [[0.10, 0.30, 0.60, 1.00], [0.05, 0.25, 0.55, 0.95]],
        dtype=torch.float32,
    )
    locations = torch.tensor(
        [
            [[0.0, 0.0], [0.2, -0.1], [0.1, 0.3], [-0.2, 0.2]],
            [[0.1, 0.0], [0.3, -0.2], [0.2, 0.1], [0.0, 0.25]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([4, 4], dtype=torch.long)
    return times, locations, lengths


def _tiny_marked_batch():
    times = torch.tensor(
        [[0.10, 0.40, 0.70, 1.10], [0.05, 0.35, 0.65, 1.05]],
        dtype=torch.float32,
    )
    locations = torch.tensor(
        [
            [[0.0, 0.0], [0.1, 0.2], [0.3, -0.1], [0.2, 0.1]],
            [[0.2, -0.1], [0.1, 0.3], [0.0, 0.2], [0.3, 0.0]],
        ],
        dtype=torch.float32,
    )
    # Unified contract marks: 0..K-1 (state model shifts internally).
    marks = torch.tensor(
        [[0, 1, 2, 1], [2, 0, 1, 2]],
        dtype=torch.long,
    )
    lengths = torch.tensor([4, 4], dtype=torch.long)
    return times, locations, lengths, marks


class TestSMASHStateEvent(unittest.TestCase):
    def test_smash_unmarked_forward(self):
        torch.manual_seed(7)
        model = build_model(
            config={},
            preset="smash",
            spatial_dim=2,
            hidden_dim=64,
            n_marks=0,
        )

        caps = model.event_model.capabilities
        self.assertEqual(caps.training_objective, "score_matching")
        self.assertFalse(caps.has_eval_nll)
        self.assertTrue(caps.has_score)
        self.assertTrue(caps.has_native_sampler)

        times, locations, lengths = _tiny_unmarked_batch()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)

        self.assertIn("loss", out)
        self.assertIn("nll", out)
        self.assertIn("objective", out)
        self.assertIn("total_events", out)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertGreater(int(out["total_events"].item()), 0)

    def test_smash_marked_forward_and_sampling(self):
        torch.manual_seed(11)
        model = build_model(
            config={},
            preset="smash",
            spatial_dim=2,
            hidden_dim=64,
            n_marks=3,
        )

        times, locations, lengths, marks = _tiny_marked_batch()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths, marks=marks)

        self.assertIn("loss", out)
        self.assertIn("total_events", out)
        self.assertTrue(torch.isfinite(out["loss"]))

        state_ctx = model.state_model.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
            marks=marks,
        )

        samples = model.event_model.sample_native(
            state=state_ctx,
            step=2,
            is_last=False,
            n_samples=16,
        )
        self.assertIn("samples", samples)
        self.assertEqual(samples["samples"].shape, (times.shape[0], 16, 3))
        self.assertTrue(torch.isfinite(samples["samples"]).all())
        self.assertIn("marks", samples)
        self.assertEqual(samples["marks"].shape, (times.shape[0], 16))
        self.assertGreaterEqual(int(samples["marks"].min().item()), 1)
        self.assertLessEqual(int(samples["marks"].max().item()), 3)

    def test_smash_eval_nll_unsupported(self):
        model = build_model(
            config={},
            preset="smash",
            spatial_dim=2,
            hidden_dim=32,
            n_marks=0,
        )
        times, locations, lengths = _tiny_unmarked_batch()
        state_ctx = model.state_model.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
        )

        with self.assertRaises(NotImplementedError):
            model.event_model.eval_nll(
                times=times,
                locations=locations,
                lengths=lengths,
                state=state_ctx,
            )


if __name__ == "__main__":
    unittest.main()
