"""SMASH coarse-path regression tests."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from unified_stpp.registry import build_model
from unified_stpp.models.configs.smash import SMASHConfig
from unified_stpp.models.history_encoders.smash_upstream_transformer import (
    SMASHUpstreamTransformerST,
)


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
    def test_smash_uses_upstream_transformer(self):
        model = build_model(
            config={},
            preset="smash",
            spatial_dim=2,
            hidden_dim=64,
            n_marks=0,
        )
        self.assertIsInstance(model.state_model.transformer, SMASHUpstreamTransformerST)

    def test_smash_data_init_overrides_compute_global_stats(self):
        class _Dataset:
            def __init__(self, sequences, *, normalize_time, normalize_space, time_mean, time_std, loc_mean, loc_std):
                self.sequences = sequences
                self.normalize_time = normalize_time
                self.normalize_space = normalize_space
                self.time_mean = time_mean
                self.time_std = time_std
                self.loc_mean = np.asarray(loc_mean, dtype=np.float32)
                self.loc_std = np.asarray(loc_std, dtype=np.float32)

        class _Bundle:
            def __init__(self, train_dataset, val_dataset, test_dataset):
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset
                self.test_dataset = test_dataset

        class _DM:
            def __init__(self, bundle):
                self._bundle = bundle

        train = _Dataset(
            [
                {
                    "times": np.array([1.0, 3.0, 6.0], dtype=np.float32),
                    "locations": np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
                    "marks": np.array([0, 1, 2], dtype=np.int64),
                }
            ],
            normalize_time=True,
            normalize_space=True,
            time_mean=2.0,
            time_std=3.0,
            loc_mean=[10.0, 20.0],
            loc_std=[4.0, 5.0],
        )
        val = _Dataset(
            [
                {
                    "times": np.array([2.0, 2.5], dtype=np.float32),
                    "locations": np.array([[-1.0, 0.5], [1.0, 7.0]], dtype=np.float32),
                    "marks": np.array([2, 1], dtype=np.int64),
                }
            ],
            normalize_time=True,
            normalize_space=True,
            time_mean=2.0,
            time_std=3.0,
            loc_mean=[10.0, 20.0],
            loc_std=[4.0, 5.0],
        )
        dm = _DM(_Bundle(train, val, None))

        overrides = SMASHConfig.data_init_overrides(dm)
        self.assertTrue(overrides["input_time_normalized"])
        self.assertTrue(overrides["input_space_normalized"])
        self.assertEqual(overrides["decoder"]["num_types"], 3)
        self.assertEqual(overrides["token_loc_min"], (-1.0, 0.5))
        self.assertEqual(overrides["token_loc_range"], (5.0, 6.5))
        self.assertAlmostEqual(overrides["token_time_min_raw"], 0.5, places=6)
        self.assertAlmostEqual(overrides["token_time_range_raw"], 2.5, places=6)
        self.assertAlmostEqual(overrides["token_time_min_log"], math.log(0.5), places=6)
        self.assertAlmostEqual(overrides["token_time_range_log"], math.log(3.0) - math.log(0.5), places=6)

    def test_smash_state_recovers_raw_inputs_and_global_minmax(self):
        raw_times = torch.tensor([[1.0, 3.0, 6.0]], dtype=torch.float32)
        raw_locations = torch.tensor(
            [[[10.0, 20.0], [12.0, 28.0], [14.0, 36.0]]],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3], dtype=torch.long)

        time_mean = 2.0
        time_std = 4.0
        loc_mean = torch.tensor([8.0, 16.0], dtype=torch.float32)
        loc_std = torch.tensor([2.0, 4.0], dtype=torch.float32)

        times = (raw_times - time_mean) / time_std
        locations = (raw_locations - loc_mean) / loc_std

        time_min_log = math.log(1.0)
        time_range_log = math.log(3.0) - math.log(1.0)

        model = build_model(
            config={
                "input_time_normalized": True,
                "input_space_normalized": True,
                "input_time_mean": time_mean,
                "input_time_std": time_std,
                "input_loc_mean": [8.0, 16.0],
                "input_loc_std": [2.0, 4.0],
                "token_time_min_raw": 1.0,
                "token_time_range_raw": 5.0,
                "token_time_min_log": time_min_log,
                "token_time_range_log": time_range_log,
                "token_loc_min": [10.0, 20.0],
                "token_loc_range": [4.0, 16.0],
                "decoder": {
                    "log_normalization": True,
                    "minmax_normalize_time": True,
                    "minmax_normalize_loc": True,
                },
            },
            preset="smash",
            spatial_dim=2,
            hidden_dim=64,
            n_marks=0,
        )

        state_ctx = model.state_model.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
        )
        payload = state_ctx.payload
        self.assertTrue(torch.allclose(payload["event_time_origin"][:, :3], raw_times, atol=1e-6))
        self.assertTrue(torch.allclose(payload["locations"][:, :3], raw_locations, atol=1e-6))

        expected_event_time = torch.tensor(
            [[0.0, (math.log(2.0) - time_min_log) / time_range_log, 1.0]],
            dtype=torch.float32,
        )
        expected_event_loc = torch.tensor(
            [[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(payload["event_time"][:, :3], expected_event_time, atol=1e-5))
        self.assertTrue(torch.allclose(payload["event_loc"][:, :3], expected_event_loc, atol=1e-5))

        smash_img = payload["smash_img"]
        self.assertEqual(smash_img.shape, (2, 1, 3))
        self.assertTrue(torch.allclose(smash_img[:, 0, 1:], expected_event_loc[:, 1:, :].reshape(2, 2), atol=1e-5))

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
        self.assertTrue(caps.has_eval_nll)           # P1-B: derived from nll_kind != "none"
        self.assertEqual(caps.nll_kind, "approx")
        self.assertIn("non-upstream", caps.nll_description)
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

        flattened = model.event_model.sample_upstream_flattened(
            state=state_ctx,
            per_step=2,
            total_steps=4,
            n_samples=8,
        )
        total_events = int(state_ctx.payload["smash_total_events"].item())
        self.assertEqual(flattened["samples"].shape, (total_events, 8, 3))
        self.assertIn("marks", flattened)
        self.assertEqual(flattened["marks"].shape, (total_events, 8))

    def test_smash_eval_nll(self):
        """eval_nll returns finite approx NLL with temporal+spatial breakdown (P1-B)."""
        torch.manual_seed(42)
        model = build_model(
            config={},
            preset="smash",
            spatial_dim=2,
            hidden_dim=32,
            n_marks=0,
        )
        times, locations, lengths = _tiny_unmarked_batch()

        # Get eval_nll output via UnifiedSTPP.eval_forward
        with torch.no_grad():
            out_train = model(times=times, locations=locations, lengths=lengths)
            out_eval = model.eval_forward(times=times, locations=locations, lengths=lengths)

        # Required keys
        for key in ("nll", "loss", "temporal_nll", "spatial_nll", "total_events"):
            self.assertIn(key, out_eval, f"missing key {key!r}")

        # Finite values
        self.assertTrue(torch.isfinite(out_eval["nll"]), "eval nll not finite")
        self.assertFalse(torch.isnan(torch.tensor(out_eval["temporal_nll"])))
        self.assertFalse(torch.isnan(torch.tensor(out_eval["spatial_nll"])))

        # Temporal + spatial ≈ total (within float tolerance)
        t_nll = out_eval["temporal_nll"]
        s_nll = out_eval["spatial_nll"]
        total = float(out_eval["nll"].item())
        self.assertAlmostEqual(t_nll + s_nll, total, places=4)

        # eval_nll and training_loss produce DIFFERENT quantities
        train_nll = float(out_train["nll"].item())
        eval_nll  = float(out_eval["nll"].item())
        self.assertNotAlmostEqual(train_nll, eval_nll, places=3)


if __name__ == "__main__":
    unittest.main()
