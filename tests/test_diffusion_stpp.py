"""Diffusion STPP integration smoke tests."""

from __future__ import annotations

import unittest

import torch

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


def _build(timesteps=4, sampling_timesteps=2, **extra):
    """Build a tiny diffusion_stpp model suitable for unit tests."""
    return build_model(
        config={
            "decoder": {
                "type": "diffusion_stpp",
                "hidden_units": 16,
                "timesteps": timesteps,
                "sampling_timesteps": sampling_timesteps,
                **extra,
            }
        },
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
        self.assertEqual(caps.training_objective, "approx_nll")
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
        self.assertIn("nll_temporal_bpd", nll_out)
        self.assertIn("nll_spatial_bpd", nll_out)
        self.assertTrue(torch.isfinite(nll_out["nll_temporal_bpd"]))
        self.assertTrue(torch.isfinite(nll_out["nll_spatial_bpd"]))

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


if __name__ == "__main__":
    unittest.main()
