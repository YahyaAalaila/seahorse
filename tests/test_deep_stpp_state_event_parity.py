"""Regression checks for deep_stpp coarse path outputs."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import torch

from unified_stpp.data.dataset import STPPDataset
from unified_stpp.models.configs.deep_stpp import DeepSTPPConfig
from unified_stpp.registry import build_model


def _tiny_batch():
    times = torch.tensor(
        [[0.00, 0.20, 0.50, 0.90], [0.00, 0.10, 0.40, 0.80]],
        dtype=torch.float32,
    )
    locations = torch.tensor(
        [
            [[0.00, 0.00], [0.30, -0.10], [0.10, 0.20], [-0.20, 0.10]],
            [[0.00, 0.00], [0.20, 0.10], [0.20, -0.20], [0.40, 0.30]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([4, 4], dtype=torch.long)
    return times, locations, lengths


class TestDeepSTPPStateEventOutputs(unittest.TestCase):
    def _build_model(self, extra_cfg: dict | None = None):
        return build_model(
            config=dict(extra_cfg or {}),
            preset="deep_stpp",
            spatial_dim=2,
            hidden_dim=16,
        )

    def _assert_common(self, out):
        self.assertIn("nll", out)
        self.assertIn("nll_per_event", out)
        self.assertIn("total_events", out)
        self.assertIn("sll", out)
        self.assertIn("tll", out)
        self.assertIn("nll_matrix", out)
        self.assertIn("sll_matrix", out)
        self.assertIn("tll_matrix", out)
        self.assertIn("mask", out)
        self.assertTrue(torch.isfinite(out["nll"]))
        torch.testing.assert_close(
            out["nll"],
            -(out["sll"] + out["tll"]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_deep_stpp_outputs(self):
        torch.manual_seed(7)
        model = self._build_model(extra_cfg={"decoder": {"seq_len": 2}})
        self.assertIsNotNone(model.state_model)
        self.assertIsNotNone(model.event_model)

        model.eval()
        times, locations, lengths = _tiny_batch()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)
        self._assert_common(out)

    def test_deep_stpp_vae_outputs_include_kl(self):
        torch.manual_seed(11)
        model = self._build_model(extra_cfg={"vae": True, "decoder": {"seq_len": 2}})
        model.eval()
        times, locations, lengths = _tiny_batch()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)
        self._assert_common(out)
        self.assertIn("kl_loss", out)
        self.assertTrue(torch.isfinite(out["kl_loss"]))

    def test_default_seq_len_skips_short_sequences_without_error(self):
        torch.manual_seed(13)
        model = self._build_model(extra_cfg={})
        model.eval()
        times, locations, lengths = _tiny_batch()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)
        self.assertEqual(float(out["total_events"].item()), 0.0)
        self.assertTrue(torch.isfinite(out["nll"]))

    def test_data_init_overrides_restore_zscore_inputs_then_apply_train_paper_stats(self):
        train_seqs = [
            {
                "times": np.array([1.0, 3.0, 6.0], dtype=np.float32),
                "locations": np.array(
                    [[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]],
                    dtype=np.float32,
                ),
            }
        ]
        val_seqs = [
            {
                "times": np.array([2.0, 5.0, 9.0], dtype=np.float32),
                "locations": np.array(
                    [[-1.0, 2.0], [1.0, 4.0], [3.0, 6.0]],
                    dtype=np.float32,
                ),
            }
        ]
        test_seqs = [
            {
                "times": np.array([4.0, 8.0, 13.0], dtype=np.float32),
                "locations": np.array(
                    [[-2.0, -2.0], [0.0, 3.0], [6.0, 12.0]],
                    dtype=np.float32,
                ),
            }
        ]

        train_ds = STPPDataset(train_seqs, normalize_time=True, normalize_space=True, min_length=1)
        val_ds = STPPDataset(
            val_seqs,
            normalize_time=True,
            normalize_space=True,
            min_length=1,
            cov_mean=train_ds.cov_mean,
            cov_std=train_ds.cov_std,
        )
        test_ds = STPPDataset(
            test_seqs,
            normalize_time=True,
            normalize_space=True,
            min_length=1,
            cov_mean=train_ds.cov_mean,
            cov_std=train_ds.cov_std,
        )
        for ds in (val_ds, test_ds):
            ds.time_mean = train_ds.time_mean
            ds.time_std = train_ds.time_std
            ds.loc_mean = train_ds.loc_mean
            ds.loc_std = train_ds.loc_std

        dm_like = SimpleNamespace(
            _bundle=SimpleNamespace(
                train_dataset=train_ds,
                val_dataset=val_ds,
                test_dataset=test_ds,
            )
        )

        overrides = DeepSTPPConfig.data_init_overrides(dm_like)
        self.assertTrue(overrides["input_normalized"])
        self.assertAlmostEqual(overrides["paper_dt_min"], 1.0)
        self.assertAlmostEqual(overrides["paper_dt_range"], 2.0)
        self.assertEqual(overrides["paper_loc_min"], (0.0, 0.0))
        self.assertEqual(overrides["paper_loc_range"], (4.0, 8.0))

        model = self._build_model(
            extra_cfg={
                **overrides,
                "decoder": {"seq_len": 2},
            }
        )
        val_item = val_ds[0]
        state_ctx = model.state_model.encode_history(
            times=val_item["times"].unsqueeze(0),
            locations=val_item["locations"].unsqueeze(0),
            lengths=torch.tensor([val_item["length"]], dtype=torch.long),
        )
        hist = state_ctx.payload["paper_st_x"]
        target = state_ctx.payload["paper_st_y"]
        expected_hist = torch.tensor(
            [[[-0.25, 0.25, 0.5], [0.25, 0.5, 1.0]]],
            dtype=torch.float32,
        )
        expected_target = torch.tensor(
            [[[0.75, 0.75, 1.5]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(hist, expected_hist, atol=1e-5), (hist, expected_hist))
        self.assertTrue(torch.allclose(target, expected_target, atol=1e-5), (target, expected_target))

    def test_sampling_state_matches_full_query_state(self):
        torch.manual_seed(17)
        model = self._build_model(extra_cfg={"decoder": {"seq_len": 2}})
        model.eval()
        times, locations, lengths = _tiny_batch()
        times = times[:1]
        locations = locations[:1]
        lengths = lengths[:1]

        full_state = model.state_model.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
        )
        sampling_state = model.state_model.encode_sampling_history(
            times=times,
            locations=locations,
            lengths=lengths,
        )

        self.assertNotIn("paper_st_x", sampling_state.payload)
        torch.testing.assert_close(
            sampling_state.payload["query_st_x"],
            full_state.payload["query_st_x"],
        )
        torch.testing.assert_close(
            sampling_state.payload["query_last_time_raw"],
            full_state.payload["query_last_time_raw"],
        )
        torch.testing.assert_close(
            sampling_state.payload["query_z"],
            full_state.payload["query_z"],
        )

        query_times = torch.tensor([1.2, 1.2], dtype=torch.float32)
        query_locs = torch.tensor([[0.1, 0.2], [0.3, -0.1]], dtype=torch.float32)
        out_full = model.event_model.intensity(
            state=full_state,
            query_times=query_times,
            query_locations=query_locs,
        )
        out_sampling = model.event_model.intensity(
            state=sampling_state,
            query_times=query_times,
            query_locations=query_locs,
        )
        torch.testing.assert_close(out_sampling, out_full)

    def test_sampling_state_append_matches_reencode(self):
        torch.manual_seed(19)
        model = self._build_model(extra_cfg={"decoder": {"seq_len": 2}})
        model.eval()

        times = torch.tensor([[0.00, 0.20, 0.50]], dtype=torch.float32)
        locations = torch.tensor(
            [[[0.00, 0.00], [0.30, -0.10], [0.10, 0.20]]],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3], dtype=torch.long)
        sampling_state = model.state_model.encode_sampling_history(
            times=times,
            locations=locations,
            lengths=lengths,
        )

        appended = model.state_model.append_sampling_event(
            sampling_state,
            event_time_raw=torch.tensor([0.90], dtype=torch.float32),
            event_location_raw=torch.tensor([[-0.20, 0.10]], dtype=torch.float32),
        )
        reencoded = model.state_model.encode_sampling_history(
            times=torch.tensor([[0.00, 0.20, 0.50, 0.90]], dtype=torch.float32),
            locations=torch.tensor(
                [[[0.00, 0.00], [0.30, -0.10], [0.10, 0.20], [-0.20, 0.10]]],
                dtype=torch.float32,
            ),
            lengths=torch.tensor([4], dtype=torch.long),
        )

        torch.testing.assert_close(
            appended.payload["query_st_x"],
            reencoded.payload["query_st_x"],
        )
        torch.testing.assert_close(
            appended.payload["query_last_time_raw"],
            reencoded.payload["query_last_time_raw"],
        )
        torch.testing.assert_close(
            appended.payload["query_z"],
            reencoded.payload["query_z"],
        )


if __name__ == "__main__":
    unittest.main()
