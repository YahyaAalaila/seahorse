"""Faithful ConditionalGMM checks on top of the shared Neural STPP backbone."""

from __future__ import annotations

import math
import unittest

import torch

from unified_stpp.config.schema import STPPConfig
from unified_stpp.models.spatial_models.njsde import ConditionalGMMSpatial


class TestConditionalGMMSpatial(unittest.TestCase):
    def test_parameter_shapes_and_aux_width(self):
        decoder = ConditionalGMMSpatial(
            spatial_dim=2,
            hidden_dim=32,
            spatial_aux_dim=16,
            hidden_dims=[8],
            n_mixtures=3,
        )
        aux = torch.randn(2, 4, 16)
        params = decoder._params_from_aux(aux)
        self.assertTrue(decoder.USES_NEURAL_AUX_STATE)
        self.assertEqual(decoder.aux_hidden_dim, 16)
        self.assertEqual(params.shape, (2, 4, 2 * 3 * 3))

    def test_exact_sequence_nll_matches_manual_standard_normal(self):
        decoder = ConditionalGMMSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=4,
            hidden_dims=[],
            n_mixtures=2,
        )
        for param in decoder.parameters():
            param.data.zero_()

        z_seq = torch.zeros(1, 3, 4)
        s_seq = torch.tensor([[[0.0, 0.0], [1.0, -1.0], [2.0, 2.0]]], dtype=torch.float32)
        t_seq = torch.zeros(1, 3, 1)
        t_prev_seq = torch.zeros(1, 3, 1)
        lengths = torch.tensor([2], dtype=torch.long)
        mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

        nll = decoder.sequence_nll(
            z_seq=z_seq,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        )

        log2pi = math.log(2.0 * math.pi)
        expected = torch.tensor(
            [[
                0.5 * (0.0**2 + log2pi) + 0.5 * (0.0**2 + log2pi),
                0.5 * (1.0**2 + log2pi) + 0.5 * ((-1.0)**2 + log2pi),
                0.0,
            ]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(nll, expected, rtol=1e-6, atol=1e-6)

    def test_conditional_logprob_and_sampling_shapes(self):
        decoder = ConditionalGMMSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=4,
            hidden_dims=[],
            n_mixtures=2,
        )
        for param in decoder.parameters():
            param.data.zero_()

        z_aug = torch.zeros(3, 4)
        logprob_fn = decoder.conditional_logprob_fn(
            t_query=1.5,
            event_times=torch.tensor([0.2, 0.8]),
            event_locs=torch.tensor([[0.0, 0.0], [0.1, -0.2]]),
            z_aug=z_aug,
        )
        s_query = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
        logprob = logprob_fn(s_query)
        self.assertEqual(logprob.shape, (2,))
        log2pi = math.log(2.0 * math.pi)
        expected = torch.tensor(
            [
                -0.5 * (0.0**2 + log2pi) - 0.5 * (0.0**2 + log2pi),
                -0.5 * (1.0**2 + log2pi) - 0.5 * ((-1.0)**2 + log2pi),
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(logprob, expected, rtol=1e-6, atol=1e-6)

        samples = decoder.sample_spatial(
            nsamples=5,
            event_times=torch.tensor([[0.2, 0.8, 1.1]], dtype=torch.float32),
            spatial_locations=torch.zeros(1, 3, 2),
            input_mask=torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
            aux_state=torch.zeros(1, 3, 4),
        )
        self.assertEqual(samples.shape, (5, 1, 3, 2))
        torch.testing.assert_close(samples[:, :, 2], torch.zeros(5, 1, 2), rtol=0.0, atol=0.0)


class TestConditionalGMMPreset(unittest.TestCase):
    def test_preset_loads_from_yaml_and_builds_model(self):
        cfg = STPPConfig.from_source(preset="njsde")
        self.assertEqual(cfg.model.preset, "njsde")
        self.assertEqual(cfg.data.adapter_kwargs.get("max_events"), 4000)
        model = cfg.model.build_model()

        self.assertEqual(type(model.event_model.spatial_decoder).__name__, "ConditionalGMMSpatial")
        self.assertTrue(model.event_model.spatial_decoder.USES_NEURAL_AUX_STATE)
        self.assertEqual(
            model.event_model.spatial_decoder.aux_hidden_dim,
            model.state_model.spatial_aux_dim,
        )

        times = torch.tensor([[0.1, 0.4, 1.0], [0.2, 0.6, 0.6]], dtype=torch.float32)
        locations = torch.tensor(
            [
                [[0.0, 0.0], [0.2, -0.1], [0.4, 0.3]],
                [[-0.1, 0.2], [0.5, 0.1], [9.0, 9.0]],
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3, 2], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)
        self.assertIn("nll", out)
        self.assertTrue(torch.isfinite(out["nll"]))

    def test_fixed_time_query_terms_match_joint_intensity_factorization(self):
        cfg = STPPConfig.from_source(preset="njsde")
        model = cfg.model.build_model()
        model.eval()

        times = torch.tensor([[0.1, 0.4, 1.0], [0.2, 0.6, 0.6]], dtype=torch.float32)
        locations = torch.tensor(
            [
                [[0.0, 0.0], [0.2, -0.1], [0.4, 0.3]],
                [[-0.1, 0.2], [0.5, 0.1], [9.0, 9.0]],
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3, 2], dtype=torch.long)

        state = model.state_model.encode_history(
            times=times,
            locations=locations,
            lengths=lengths,
        )
        query_time = torch.tensor(1.3, dtype=torch.float32)
        query_locations = torch.tensor(
            [[0.0, 0.0], [0.5, -0.2], [-0.1, 0.4]],
            dtype=torch.float32,
        )

        terms = model.event_model.fixed_time_query_terms(
            state=state,
            query_time=query_time,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(torch.as_tensor(terms["lambda_t"])))
        logprob = terms["logprob_fn"](query_locations)
        joint = model.event_model.intensity(
            state=state,
            query_times=query_time.repeat(query_locations.shape[0]),
            query_locations=query_locations,
        )
        expected = torch.as_tensor(terms["lambda_t"]).to(logprob) * torch.exp(logprob)
        torch.testing.assert_close(joint, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
