"""Regression checks for the canonical AutoSTPP preset."""

from __future__ import annotations

import types
import unittest

import numpy as np
import torch

from unified_stpp.config.schema import STPPConfig
from unified_stpp.models.configs.auto_stpp import AutoSTPPConfig
from unified_stpp.models.event_models.auto_stpp_kernel import AutoSTPPCuboid
from unified_stpp.registry import build_model


def _tiny_batch():
    times = torch.tensor(
        [
            [0.05, 0.20, 0.50, 0.90, 1.30],
            [0.10, 0.30, 0.60, 1.00, 1.20],
        ],
        dtype=torch.float32,
    )
    locations = torch.tensor(
        [
            [[0.00, 0.10], [0.25, 0.00], [0.15, 0.30], [0.05, 0.10], [0.10, 0.25]],
            [[0.05, 0.05], [0.10, 0.20], [0.30, 0.10], [0.35, 0.15], [0.20, 0.25]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([5, 4], dtype=torch.long)
    return times, locations, lengths


class TestAutoSTPPDataInit(unittest.TestCase):
    def test_data_init_overrides_uses_train_only_minmax(self):
        seqs = [
            {
                "times": np.array([0.2, 0.5, 1.0], dtype=np.float32),
                "locations": np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32),
            },
            {
                "times": np.array([0.1, 0.4], dtype=np.float32),
                "locations": np.array([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32),
            },
        ]
        train_ds = types.SimpleNamespace(
            sequences=seqs,
            normalize_time=True,
            normalize_space=True,
            time_mean=0.7,
            time_std=0.4,
            loc_mean=np.array([2.0, 5.0], dtype=np.float32),
            loc_std=np.array([1.5, 2.0], dtype=np.float32),
        )
        dm = types.SimpleNamespace(_bundle=types.SimpleNamespace(train_dataset=train_ds))

        overrides = AutoSTPPConfig.data_init_overrides(dm)
        self.assertAlmostEqual(overrides["paper_dt_min"], 0.1, places=6)
        self.assertAlmostEqual(overrides["paper_dt_range"], 0.4, places=6)
        self.assertEqual(overrides["paper_loc_min"], (0.0, 2.0))
        self.assertEqual(overrides["paper_loc_range"], (4.0, 6.0))
        self.assertTrue(overrides["input_normalized"])


class TestAutoSTPPForward(unittest.TestCase):
    def test_outputs_include_exact_terms_and_orig_space_metrics(self):
        torch.manual_seed(7)
        model = build_model(
            config={
                "decoder": {
                    "lookback": 3,
                    "lookahead": 1,
                    "n_prodnet": 2,
                    "hidden_size": 8,
                    "num_layers": 1,
                    "activation": "tanh",
                    "trunc": False,
                },
                "paper_dt_min": 0.0,
                "paper_dt_range": 1.5,
                "paper_loc_min": [0.0, 0.0],
                "paper_loc_range": [1.0, 1.0],
            },
            preset="auto_stpp",
            spatial_dim=2,
            hidden_dim=8,
        )
        model.eval()
        times, locations, lengths = _tiny_batch()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)

        self.assertIn("nll", out)
        self.assertIn("nll_per_event", out)
        self.assertIn("total_events", out)
        self.assertIn("sll", out)
        self.assertIn("tll", out)
        self.assertIn("nll_matrix", out)
        self.assertIn("sll_matrix", out)
        self.assertIn("tll_matrix", out)
        self.assertIn("mask", out)
        self.assertIn("lambs_sum", out)
        self.assertIn("lamb_t", out)
        self.assertIn("lamb_ints", out)
        self.assertIn("background_rate", out)
        self.assertIn("extra_metrics", out)
        self.assertTrue(torch.isfinite(out["nll"]))
        self.assertTrue(torch.isfinite(out["background_rate"]))
        torch.testing.assert_close(
            out["nll"],
            -(out["sll"] + out["tll"]),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertIn("orig_space_nll", out["extra_metrics"])
        self.assertIn("orig_space_spatial_nll", out["extra_metrics"])
        self.assertIn("orig_space_temporal_nll", out["extra_metrics"])
        self.assertIn("raw_space_nll", out["extra_metrics"])
        self.assertIn("raw_space_spatial_nll", out["extra_metrics"])
        self.assertIn("raw_space_temporal_nll", out["extra_metrics"])
        self.assertAlmostEqual(
            out["extra_metrics"]["orig_space_nll"],
            out["extra_metrics"]["raw_space_nll"],
            places=6,
        )

    def test_surface_query_contract(self):
        torch.manual_seed(0)
        model = build_model(
            config={
                "paper_dt_min": 0.0,
                "paper_dt_range": 1.0,
                "paper_loc_min": [0.0, 0.0],
                "paper_loc_range": [1.0, 1.0],
            },
            preset="auto_stpp",
            spatial_dim=2,
            hidden_dim=8,
        )
        model.eval()
        times, locations, lengths = _tiny_batch()
        with torch.no_grad():
            state = model.state_model.encode_history(
                times=times[:1],
                locations=locations[:1],
                lengths=lengths[:1],
            )
            out = model.event_model.query_surface(
                state=state,
                grid_times=torch.zeros(4, dtype=torch.float32),
                grid_locs=torch.tensor(
                    [[0.0, 0.0], [0.2, 0.1], [0.4, 0.3], [0.1, 0.5]],
                    dtype=torch.float32,
                ),
            )
        self.assertEqual(out.shape, (4,))
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.all(out >= 0))

    def test_sampling_state_matches_full_history_intensity(self):
        torch.manual_seed(23)
        model = build_model(
            config={
                "paper_dt_min": 0.0,
                "paper_dt_range": 1.0,
                "paper_loc_min": [0.0, 0.0],
                "paper_loc_range": [1.0, 1.0],
            },
            preset="auto_stpp",
            spatial_dim=2,
            hidden_dim=8,
        )
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
            sampling_state.payload["paper_history_scaled"],
            full_state.payload["paper_history_scaled"],
        )
        torch.testing.assert_close(
            sampling_state.payload["times_raw"],
            full_state.payload["times_raw"],
        )
        torch.testing.assert_close(
            sampling_state.payload["lengths"],
            full_state.payload["lengths"],
        )

        query_times = torch.tensor([1.4, 1.4], dtype=torch.float32)
        query_locs = torch.tensor([[0.1, 0.2], [0.25, 0.1]], dtype=torch.float32)
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
        torch.manual_seed(29)
        model = build_model(
            config={
                "paper_dt_min": 0.0,
                "paper_dt_range": 1.0,
                "paper_loc_min": [0.0, 0.0],
                "paper_loc_range": [1.0, 1.0],
            },
            preset="auto_stpp",
            spatial_dim=2,
            hidden_dim=8,
        )
        model.eval()

        times = torch.tensor([[0.05, 0.20, 0.50, 0.90]], dtype=torch.float32)
        locations = torch.tensor(
            [[[0.00, 0.10], [0.25, 0.00], [0.15, 0.30], [0.05, 0.10]]],
            dtype=torch.float32,
        )
        lengths = torch.tensor([4], dtype=torch.long)
        sampling_state = model.state_model.encode_sampling_history(
            times=times,
            locations=locations,
            lengths=lengths,
        )

        appended = model.state_model.append_sampling_event(
            sampling_state,
            event_time_raw=torch.tensor([1.30], dtype=torch.float32),
            event_location_raw=torch.tensor([[0.10, 0.25]], dtype=torch.float32),
        )
        reencoded = model.state_model.encode_sampling_history(
            times=torch.tensor([[0.05, 0.20, 0.50, 0.90, 1.30]], dtype=torch.float32),
            locations=torch.tensor(
                [[[0.00, 0.10], [0.25, 0.00], [0.15, 0.30], [0.05, 0.10], [0.10, 0.25]]],
                dtype=torch.float32,
            ),
            lengths=torch.tensor([5], dtype=torch.long),
        )

        torch.testing.assert_close(
            appended.payload["paper_history_scaled"],
            reencoded.payload["paper_history_scaled"],
        )
        torch.testing.assert_close(
            appended.payload["times_raw"],
            reencoded.payload["times_raw"],
        )
        torch.testing.assert_close(
            appended.payload["lengths"],
            reencoded.payload["lengths"],
        )


class TestAutoSTPPKernel(unittest.TestCase):
    def test_compensator_monotone_and_matches_temporal_derivative(self):
        torch.manual_seed(3)
        kernel = AutoSTPPCuboid(
            n_prodnet=2,
            hidden_size=8,
            num_layers=1,
            activation="tanh",
            bias=True,
        )
        kernel.project()
        s = torch.tensor([[0.3, 0.4]], dtype=torch.float32)
        ta = torch.tensor([[0.1]], dtype=torch.float32)
        tb = torch.tensor([[0.5]], dtype=torch.float32)
        eps = 1.0e-4

        with torch.no_grad():
            base = kernel.int_lamb_stpp(s, ta, tb)
            plus = kernel.int_lamb_stpp(s, ta, tb + eps)
            lamb_t = kernel.lamb_t_stpp(s, tb)

        self.assertTrue(torch.all(base >= 0))
        self.assertTrue(torch.all(plus >= base))
        fd = (plus - base) / eps
        torch.testing.assert_close(fd, lamb_t, rtol=5e-2, atol=5e-2)


class TestAutoSTPPSmoke(unittest.TestCase):
    def test_bundled_yaml_loads_from_canonical_name(self):
        cfg = STPPConfig.from_preset("auto_stpp")
        self.assertEqual(cfg.model.preset, "auto_stpp")
        self.assertEqual(cfg.training.optimizer, "adam")
        self.assertEqual(cfg.data.protocol, "raw")


if __name__ == "__main__":
    unittest.main()
