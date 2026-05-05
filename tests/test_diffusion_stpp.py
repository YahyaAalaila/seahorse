"""Diffusion STPP integration smoke tests."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from unified_stpp.data.dataset import STPPDataset
from unified_stpp.models.configs.diffusion_stpp import DiffusionSTPPConfig
from unified_stpp.registry import build_model


def _tiny_batch():
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


def _batch_to_sequences(times: torch.Tensor, locations: torch.Tensor, lengths: torch.Tensor) -> list[dict]:
    seqs: list[dict] = []
    for b in range(times.shape[0]):
        L = int(lengths[b].item())
        seqs.append(
            {
                "times": times[b, :L].cpu().numpy(),
                "locations": locations[b, :L].cpu().numpy(),
            }
        )
    return seqs


def _token_overrides_from_sequences(
    sequences: list[dict],
    *,
    input_normalized: bool = False,
    input_time_mean: float = 0.0,
    input_time_std: float = 1.0,
    input_loc_mean=(0.0, 0.0),
    input_loc_std=(1.0, 1.0),
) -> dict:
    delta_chunks = []
    loc_chunks = []
    for seq in sequences:
        times = torch.as_tensor(seq["times"], dtype=torch.float32)
        locs = torch.as_tensor(seq["locations"], dtype=torch.float32)
        if times.numel() > 1:
            delta_chunks.append(times[1:] - times[:-1])
        if locs.numel() > 0:
            loc_chunks.append(locs)

    if delta_chunks:
        delta_all = torch.cat(delta_chunks, dim=0)
        delta_min = float(delta_all.min().item())
        delta_range = float(max((delta_all.max() - delta_all.min()).item(), 1e-8))
    else:
        delta_min = 0.0
        delta_range = 1.0

    if loc_chunks:
        loc_all = torch.cat(loc_chunks, dim=0)
        loc_min = tuple(float(x) for x in loc_all.min(dim=0).values.tolist())
        loc_range = tuple(
            float(max(v, 1e-8))
            for v in (loc_all.max(dim=0).values - loc_all.min(dim=0).values).tolist()
        )
    else:
        loc_min = (0.0, 0.0)
        loc_range = (1.0, 1.0)

    return {
        "input_normalized": input_normalized,
        "input_time_mean": float(input_time_mean),
        "input_time_std": float(input_time_std),
        "input_loc_mean": tuple(float(x) for x in input_loc_mean),
        "input_loc_std": tuple(float(x) for x in input_loc_std),
        "token_delta_t_min": delta_min,
        "token_delta_t_range": delta_range,
        "token_loc_min": loc_min,
        "token_loc_range": loc_range,
    }


def _default_token_overrides() -> dict:
    times, locations, lengths = _tiny_batch()
    return _token_overrides_from_sequences(_batch_to_sequences(times, locations, lengths))


def _build(timesteps=4, sampling_timesteps=2, config_overrides=None, **extra):
    """Build a tiny diffusion_stpp model suitable for unit tests."""
    config = _default_token_overrides()
    if config_overrides:
        config.update(config_overrides)
    config["decoder"] = {
        "type": "diffusion_stpp",
        "hidden_units": 16,
        "timesteps": timesteps,
        "sampling_timesteps": sampling_timesteps,
        **extra,
    }
    return build_model(
        config=config,
        preset="diffusion_stpp",
        spatial_dim=2,
        hidden_dim=32,
        n_marks=0,
    )


class TestDiffusionSTPPConfig(unittest.TestCase):
    """Build / config smoke tests."""

    def test_build_succeeds(self):
        torch.manual_seed(0)
        model = _build()
        self.assertIsNotNone(model)

    def test_capabilities(self):
        model = _build()
        caps = model.event_model.capabilities
        self.assertEqual(caps.training_objective, "elbo")
        self.assertEqual(caps.nll_kind, "approx")
        self.assertTrue(caps.has_eval_nll)
        self.assertTrue(caps.has_native_sampler)
        self.assertFalse(caps.has_intensity)
        self.assertFalse(caps.has_density)
        self.assertFalse(caps.has_score)

    def test_marked_rejected(self):
        with self.assertRaises(ValueError):
            build_model(
                config={},
                preset="diffusion_stpp",
                spatial_dim=2,
                hidden_dim=32,
                n_marks=3,
            )

    def test_non_2d_spatial_rejected(self):
        with self.assertRaises(ValueError):
            build_model(
                config={},
                preset="diffusion_stpp",
                spatial_dim=3,
                hidden_dim=32,
                n_marks=0,
            )

    def test_beta_schedule_cosine_and_linear(self):
        for sched in ("cosine", "linear"):
            _build(beta_schedule=sched)

    def test_objective_pred_noise(self):
        model = _build(objective="pred_noise")
        times, locations, lengths = _tiny_batch()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)
        self.assertIn("loss", out)
        self.assertTrue(torch.isfinite(out["loss"]))


class TestDiffusionSTPPStateEvent(unittest.TestCase):
    """State model / event model smoke tests."""

    def test_training_loss_forward(self):
        torch.manual_seed(1)
        model = _build()
        times, locations, lengths = _tiny_batch()

        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)

        self.assertIn("loss", out)
        self.assertIn("nll", out)
        self.assertIn("total_events", out)
        self.assertIn("objective_name", out)
        self.assertEqual(out["objective_name"], "diffusion_elbo")
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertGreater(int(out["total_events"].item()), 0)

    def test_state_payload_keys(self):
        torch.manual_seed(2)
        model = _build()
        times, locations, lengths = _tiny_batch()

        state_ctx = model.state_model.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
        )

        for key in ("diff_img", "diff_cond", "diff_cond_last", "diff_total_events"):
            self.assertIn(key, state_ctx.payload, f"Missing payload key: {key}")

        img = state_ctx.payload["diff_img"]
        cond = state_ctx.payload["diff_cond"]
        cond_last = state_ctx.payload["diff_cond_last"]

        # seq_length = 1 (time) + 2 (loc) = 3
        self.assertEqual(img.shape[-1], 3)
        # img and cond must have matching leading dim (N_flat)
        self.assertEqual(img.shape[0], cond.shape[0])
        # cond_last must have one entry per sequence
        self.assertEqual(cond_last.shape[0], times.shape[0])

    def test_empty_sequence_handled(self):
        """A batch where all sequences have length <= 1 should not crash."""
        torch.manual_seed(3)
        model = _build()
        times = torch.tensor([[0.5], [0.3]], dtype=torch.float32)
        locations = torch.tensor([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=torch.float32)
        lengths = torch.tensor([1, 1], dtype=torch.long)

        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)

        self.assertIn("loss", out)
        self.assertTrue(torch.isfinite(out["loss"]))


class TestDiffusionSTPPSampling(unittest.TestCase):
    """Native sampling smoke tests."""

    def test_sample_native_shape(self):
        torch.manual_seed(4)
        model = _build(sampling_timesteps=2)
        times, locations, lengths = _tiny_batch()

        state_ctx = model.state_model.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
        )

        samples_out = model.event_model.sample_native(state=state_ctx)
        self.assertIn("samples", samples_out)
        samples = samples_out["samples"]

        # (B, 1, seq_length) — one sample per sequence in [0, 1]
        B = times.shape[0]
        self.assertEqual(samples.shape[0], B)
        self.assertEqual(samples.shape[1], 1)
        self.assertEqual(samples.shape[2], 3)  # 1 + spatial_dim

    def test_sample_native_finite(self):
        torch.manual_seed(5)
        model = _build(sampling_timesteps=2)
        times, locations, lengths = _tiny_batch()

        state_ctx = model.state_model.encode_history(
            times=times, locations=locations, lengths=lengths,
        )
        out = model.event_model.sample_native(state=state_ctx)
        self.assertTrue(torch.isfinite(out["samples"]).all())

    def test_sample_native_custom_batch_size(self):
        torch.manual_seed(6)
        model = _build(sampling_timesteps=2)
        times, locations, lengths = _tiny_batch()

        state_ctx = model.state_model.encode_history(
            times=times[:1], locations=locations[:1], lengths=lengths[:1],
        )
        out = model.event_model.sample_native(state=state_ctx, batch_size=4)
        self.assertEqual(out["samples"].shape[0], 4)

    def test_ddpm_sampling(self):
        """sampling_timesteps == timesteps forces full DDPM loop."""
        torch.manual_seed(7)
        model = _build(timesteps=4, sampling_timesteps=4)
        times, locations, lengths = _tiny_batch()
        state_ctx = model.state_model.encode_history(
            times=times, locations=locations, lengths=lengths,
        )
        out = model.event_model.sample_native(state=state_ctx)
        self.assertTrue(torch.isfinite(out["samples"]).all())


class TestDiffusionSTPPApproxNLL(unittest.TestCase):
    """Approximate NLL (eval_nll / NLL_cal) smoke tests."""

    def test_eval_nll_returns_finite(self):
        torch.manual_seed(8)
        # Use very small timesteps so the loop is fast in tests.
        model = _build(timesteps=4, sampling_timesteps=2)
        times, locations, lengths = _tiny_batch()

        state_ctx = model.state_model.encode_history(
            times=times, locations=locations, lengths=lengths,
        )
        nll_out = model.event_model.eval_nll(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state_ctx,
        )

        self.assertIn("nll", nll_out)
        self.assertIn("total_events", nll_out)
        self.assertTrue(torch.isfinite(nll_out["nll"]))
        self.assertGreater(float(nll_out["total_events"].item()), 0)

    def test_eval_nll_temporal_spatial_keys(self):
        torch.manual_seed(9)
        model = _build(timesteps=4, sampling_timesteps=2)
        times, locations, lengths = _tiny_batch()
        state_ctx = model.state_model.encode_history(
            times=times, locations=locations, lengths=lengths,
        )
        nll_out = model.event_model.eval_nll(
            times=times, locations=locations, lengths=lengths, state=state_ctx,
        )
        self.assertIn("temporal_nll", nll_out)
        self.assertIn("spatial_nll", nll_out)
        self.assertIn("extra_metrics", nll_out)
        self.assertIn("test_nll_upstream_per_dim", nll_out["extra_metrics"])
        self.assertIn("temporal_nll_upstream_per_dim", nll_out["extra_metrics"])
        self.assertIn("spatial_nll_upstream_per_dim", nll_out["extra_metrics"])
        self.assertTrue(torch.isfinite(nll_out["temporal_nll"]))
        self.assertTrue(torch.isfinite(nll_out["spatial_nll"]))
        self.assertTrue(torch.isfinite(nll_out["extra_metrics"]["test_nll_upstream_per_dim"]))
        self.assertTrue(torch.isfinite(nll_out["extra_metrics"]["temporal_nll_upstream_per_dim"]))
        self.assertTrue(torch.isfinite(nll_out["extra_metrics"]["spatial_nll_upstream_per_dim"]))

    def test_eval_nll_reports_benchmark_and_per_dim_units(self):
        torch.manual_seed(9)
        model = _build(timesteps=4, sampling_timesteps=2)
        times, locations, lengths = _tiny_batch()
        state_ctx = model.state_model.encode_history(
            times=times, locations=locations, lengths=lengths,
        )
        nll_out = model.event_model.eval_nll(
            times=times, locations=locations, lengths=lengths, state=state_ctx,
        )
        extra = nll_out["extra_metrics"]
        total = float(nll_out["nll"].item())
        temporal = float(nll_out["temporal_nll"].item())
        spatial = float(nll_out["spatial_nll"].item())
        total_per_dim = float(extra["test_nll_upstream_per_dim"].item())
        temporal_per_dim = float(extra["temporal_nll_upstream_per_dim"].item())
        spatial_per_dim = float(extra["spatial_nll_upstream_per_dim"].item())

        self.assertAlmostEqual(total, total_per_dim * 3.0, places=5)
        self.assertAlmostEqual(temporal, temporal_per_dim, places=5)
        self.assertAlmostEqual(spatial, spatial_per_dim * 2.0, places=5)
        self.assertAlmostEqual(total, temporal + spatial, places=5)

    def test_eval_nll_empty_sequences(self):
        """eval_nll with all-length-1 sequences must not crash."""
        torch.manual_seed(10)
        model = _build(timesteps=4, sampling_timesteps=2)
        times = torch.tensor([[0.5], [0.3]], dtype=torch.float32)
        locations = torch.tensor([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=torch.float32)
        lengths = torch.tensor([1, 1], dtype=torch.long)
        state_ctx = model.state_model.encode_history(
            times=times, locations=locations, lengths=lengths,
        )
        nll_out = model.event_model.eval_nll(
            times=times, locations=locations, lengths=lengths, state=state_ctx,
        )
        self.assertIn("nll", nll_out)
        self.assertTrue(torch.isfinite(nll_out["nll"]))


class TestVBDecoder(unittest.TestCase):
    """Verify that the VB decoder NLL at t=0 does not explode.

    Root cause of explosion: posterior_log_variance_clipped[0] ≈ -46 because
    posterior_variance[0] = betas[0] * (1 - alphas_bar_prev[0]) / (1 - alphas_bar[0]) = 0
    (alphas_bar_prev[0] = 1 by construction).  exp(-46) < 1e-8, so the 1e-8 clamp in
    dec_nll = diff² / (exp(log_var) + 1e-8) takes over → diff² / 1e-8 → millions.

    Fix: use betas[t] as decoder variance — betas[0] >> 1e-8 for any schedule.
    """

    def test_dec_nll_finite_and_bounded(self):
        """eval_nll with T=4 must not explode from the t=0 decoder term."""
        torch.manual_seed(11)
        model = _build(timesteps=4, sampling_timesteps=2)
        times, locations, lengths = _tiny_batch()
        state_ctx = model.state_model.encode_history(
            times=times, locations=locations, lengths=lengths,
        )
        nll_out = model.event_model.eval_nll(
            times=times, locations=locations, lengths=lengths, state=state_ctx,
        )
        nll = float(nll_out["nll"].item())
        self.assertTrue(torch.isfinite(nll_out["nll"]),
                        "eval_nll returned non-finite value")
        # With fix (betas[t] variance): random T=4 model gives ~tens–hundreds nats.
        # With bug (posterior_log_var[0] ≈ -46 → clamp 1e-8): ~37 million nats.
        self.assertLess(nll, 1e4,
                        f"eval_nll = {nll:.1f} nats is too large — "
                        "likely posterior_log_var[0] clamp explosion at t=0")


class TestDiffusionImgNormalization(unittest.TestCase):
    """Verify that diff_img is always in [0, 1] before entering GaussianDiffusionST.

    This invariant is required for NLL_cal: the DDPM variance schedule is calibrated
    for x_0 ∈ [-1, 1] (after normalize_to_neg_one_to_one maps [0,1] → [-1,1]).
    z-scored locations (range ≈ [-3, 3]) cause huge bpd values (~10⁸) if not normalized.
    """

    def test_diff_img_in_unit_range(self):
        """diff_img must be in [0, 1] regardless of input coordinate range."""
        torch.manual_seed(42)
        model = _build(timesteps=4, sampling_timesteps=2)

        # Use z-scored locations (the typical data module output), NOT [0,1]
        times = torch.tensor([[0.1, 0.5, 0.9]], dtype=torch.float32)
        locations = torch.tensor([[[-2.5, 1.8], [0.3, -1.7], [2.1, 0.5]]], dtype=torch.float32)
        lengths = torch.tensor([3], dtype=torch.long)

        state_ctx = model.state_model.encode_history(
            times=times, locations=locations, lengths=lengths,
        )
        img = state_ctx.payload["diff_img"]  # (N, 1, 1+spatial_dim)
        self.assertGreaterEqual(float(img.min().item()), 0.0 - 1e-6,
            f"diff_img has values below 0: min={img.min().item():.4f}")
        self.assertLessEqual(float(img.max().item()), 1.0 + 1e-6,
            f"diff_img has values above 1: max={img.max().item():.4f}")


class TestDiffusionFixedTokenStats(unittest.TestCase):
    """Regression coverage for the fixed global diffusion token space."""

    def test_diff_img_is_invariant_to_batch_composition(self):
        base_times = torch.tensor([[0.10, 0.30, 0.60, 1.00]], dtype=torch.float32)
        base_locs = torch.tensor(
            [[[0.0, 0.0], [0.2, -0.1], [0.1, 0.3], [-0.2, 0.2]]],
            dtype=torch.float32,
        )
        base_lengths = torch.tensor([4], dtype=torch.long)

        extra_times = torch.tensor([[0.0, 5.0, 11.0, 18.0]], dtype=torch.float32)
        extra_locs = torch.tensor(
            [[[-3.0, -4.0], [2.0, 6.0], [4.0, 8.0], [7.0, 10.0]]],
            dtype=torch.float32,
        )
        extra_lengths = torch.tensor([4], dtype=torch.long)

        full_sequences = _batch_to_sequences(
            torch.cat([base_times, extra_times], dim=0),
            torch.cat([base_locs, extra_locs], dim=0),
            torch.cat([base_lengths, extra_lengths], dim=0),
        )
        model = _build(config_overrides=_token_overrides_from_sequences(full_sequences))

        state_single = model.state_model.encode_history(
            times=base_times,
            locations=base_locs,
            lengths=base_lengths,
        )
        state_mixed = model.state_model.encode_history(
            times=torch.cat([base_times, extra_times], dim=0),
            locations=torch.cat([base_locs, extra_locs], dim=0),
            lengths=torch.cat([base_lengths, extra_lengths], dim=0),
        )

        self.assertTrue(
            torch.allclose(
                state_single.payload["diff_img"],
                state_mixed.payload["diff_img"][: state_single.payload["diff_img"].shape[0]],
                atol=1e-6,
            )
        )

    def test_data_init_overrides_use_all_splits_and_invert_zscore_inputs(self):
        train_seqs = [
            {
                "times": np.array([0.0, 2.0, 5.0], dtype=np.float32),
                "locations": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
            }
        ]
        val_seqs = [
            {
                "times": np.array([0.0, 1.0, 3.0], dtype=np.float32),
                "locations": np.array([[-1.0, -2.0], [0.0, 0.0], [3.0, 4.0]], dtype=np.float32),
            }
        ]
        test_seqs = [
            {
                "times": np.array([0.0, 4.0, 10.0], dtype=np.float32),
                "locations": np.array([[-2.0, 5.0], [1.0, -3.0], [4.0, 6.0]], dtype=np.float32),
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

        overrides = DiffusionSTPPConfig.data_init_overrides(dm_like)
        self.assertAlmostEqual(overrides["token_delta_t_min"], 1.0)
        self.assertAlmostEqual(overrides["token_delta_t_range"], 5.0)
        self.assertEqual(overrides["token_loc_min"], (-2.0, -3.0))
        self.assertEqual(overrides["token_loc_range"], (6.0, 9.0))
        self.assertTrue(overrides["input_normalized"])

        model = _build(config_overrides=overrides)
        val_item = val_ds[0]
        state_ctx = model.state_model.encode_history(
            times=val_item["times"].unsqueeze(0),
            locations=val_item["locations"].unsqueeze(0),
            lengths=torch.tensor([val_item["length"]], dtype=torch.long),
        )
        img = state_ctx.payload["diff_img"].squeeze(1)
        expected = torch.tensor(
            [
                [0.0, 2.0 / 6.0, 3.0 / 9.0],
                [0.2, 5.0 / 6.0, 7.0 / 9.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(img, expected, atol=1e-5), (img, expected))

    def test_state_dict_roundtrip_preserves_fixed_token_stats(self):
        sequences = _batch_to_sequences(*_tiny_batch())
        model = _build(config_overrides=_token_overrides_from_sequences(sequences))
        restored = _build(
            config_overrides={
                "input_normalized": True,
                "input_time_mean": 5.0,
                "input_time_std": 2.0,
                "input_loc_mean": (-9.0, 4.0),
                "input_loc_std": (3.0, 7.0),
                "token_delta_t_min": 99.0,
                "token_delta_t_range": 42.0,
                "token_loc_min": (-8.0, -6.0),
                "token_loc_range": (11.0, 13.0),
            }
        )
        restored.load_state_dict(model.state_dict())

        self.assertTrue(torch.allclose(
            model.state_model.token_delta_t_min,
            restored.state_model.token_delta_t_min,
        ))
        self.assertTrue(torch.allclose(
            model.state_model.token_delta_t_range,
            restored.state_model.token_delta_t_range,
        ))
        self.assertTrue(torch.allclose(
            model.state_model.token_loc_min,
            restored.state_model.token_loc_min,
        ))
        self.assertTrue(torch.allclose(
            model.state_model.token_loc_range,
            restored.state_model.token_loc_range,
        ))
        self.assertTrue(torch.allclose(
            model.state_model.input_time_mean,
            restored.state_model.input_time_mean,
        ))
        self.assertTrue(torch.allclose(
            model.state_model.input_time_std,
            restored.state_model.input_time_std,
        ))
        self.assertTrue(torch.allclose(
            model.state_model.input_loc_mean,
            restored.state_model.input_loc_mean,
        ))
        self.assertTrue(torch.allclose(
            model.state_model.input_loc_std,
            restored.state_model.input_loc_std,
        ))

        times, locations, lengths = _tiny_batch()
        img_ref = model.state_model.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
        ).payload["diff_img"]
        img_restored = restored.state_model.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
        ).payload["diff_img"]
        self.assertTrue(torch.allclose(img_ref, img_restored, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
