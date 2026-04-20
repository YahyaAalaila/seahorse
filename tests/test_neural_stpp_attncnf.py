"""Faithful attentive CNF checks on top of the shared Neural STPP backbone."""

from __future__ import annotations

import math
import unittest

import torch

from unified_stpp.config.schema import STPPConfig
from unified_stpp.models.spatial_models.neural_attncnf import (
    L2MultiheadAttention,
    MultiheadAttention,
    NeuralAttnCNFSpatial,
)


class TestNeuralAttnCNFSpatial(unittest.TestCase):
    def test_parameter_shapes_attention_path_and_aux_tail_semantics(self):
        decoder = NeuralAttnCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=3,
            hidden_dims="8-8-8",
            l2_attn=True,
            naive_hutch=False,
            nblocks=2,
            num_heads=4,
        )
        self.assertFalse(bool(getattr(decoder, "USES_NEURAL_AUX_STATE", False)))
        self.assertEqual(decoder.aux_hidden_dim, 3)
        self.assertEqual(decoder.t_embedding_dim, 8)
        self.assertEqual(decoder.base_dist_params[0].in_features, 3 + 8)
        self.assertIsInstance(decoder.odefunc.self_attns[0], L2MultiheadAttention)
        self.assertTrue(decoder.lowvar_trace)
        self.assertTrue(decoder.cnf.nonself_connections)

        z_full = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]]],
            dtype=torch.float32,
        )
        expected_aux = torch.tensor(
            [[[3.0, 4.0, 5.0], [30.0, 40.0, 50.0]]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(decoder._select_aux_tail(z_full), expected_aux)

    def test_naive_hutch_disables_lowvar_trace(self):
        decoder = NeuralAttnCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=4,
            hidden_dims="8-8-8",
            l2_attn=False,
            naive_hutch=True,
            nblocks=2,
            num_heads=4,
        )
        self.assertTrue(decoder.naive_hutch)
        self.assertFalse(decoder.lowvar_trace)
        self.assertFalse(decoder.cnf.nonself_connections)
        self.assertIsInstance(decoder.odefunc.self_attns[0], MultiheadAttention)

    def test_exact_single_event_nll_matches_standard_normal_when_flows_zero(self):
        decoder = NeuralAttnCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=3,
            hidden_dims="8-8-8",
            l2_attn=True,
            naive_hutch=False,
            nblocks=2,
            num_heads=4,
        )
        for param in decoder.parameters():
            param.data.zero_()
        decoder.eval()

        z_seq = torch.zeros(1, 1, 6)
        t_seq = torch.tensor([[[0.5]]], dtype=torch.float32)
        s_seq = torch.tensor([[[1.0, -1.0]]], dtype=torch.float32)
        t_prev_seq = torch.zeros(1, 1, 1)
        lengths = torch.tensor([1], dtype=torch.long)
        mask = torch.tensor([[1.0]], dtype=torch.float32)

        nll = decoder.sequence_nll(
            z_seq=z_seq,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        )
        expected = torch.tensor(
            [[
                0.5 * (1.0**2 + math.log(2.0 * math.pi))
                + 0.5 * ((-1.0) ** 2 + math.log(2.0 * math.pi))
            ]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(nll, expected, rtol=1e-5, atol=1e-5)

    def test_conditional_logprob_fn_shape(self):
        decoder = NeuralAttnCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=3,
            hidden_dims="8-8-8",
            l2_attn=True,
            naive_hutch=False,
            nblocks=2,
            num_heads=4,
        )
        for param in decoder.parameters():
            param.data.zero_()
        decoder.eval()

        z_aug = torch.zeros(3, 6)
        logprob_fn = decoder.conditional_logprob_fn(
            t_query=1.5,
            event_times=torch.tensor([0.2, 0.8]),
            event_locs=torch.tensor([[0.0, 0.0], [0.1, -0.2]]),
            z_aug=z_aug,
        )
        s_query = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
        logprob = logprob_fn(s_query)
        self.assertEqual(logprob.shape, (2,))


class TestNeuralAttnCNFPreset(unittest.TestCase):
    def test_preset_loads_builds_and_wires_attentive_decoder(self):
        cfg = STPPConfig.from_source(preset="neural_attncnf")
        self.assertEqual(cfg.model.preset, "neural_attncnf")
        self.assertEqual(cfg.data.adapter_kwargs.get("max_events"), 4000)
        model = cfg.model.build_model()

        decoder = model.event_model.spatial_decoder
        self.assertEqual(type(decoder).__name__, "NeuralAttnCNFSpatial")
        self.assertTrue(decoder.l2_attn)
        self.assertTrue(decoder.lowvar_trace)
        self.assertFalse(bool(getattr(decoder, "USES_NEURAL_AUX_STATE", False)))
        self.assertEqual(
            decoder.aux_hidden_dim,
            model.state_model.spatial_aux_dim,
        )

        times = torch.tensor([[0.1, 0.4], [0.2, 0.2]], dtype=torch.float32)
        locations = torch.tensor(
            [
                [[0.0, 0.0], [0.2, -0.1]],
                [[-0.1, 0.2], [9.0, 9.0]],
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([2, 1], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            out = model(times=times, locations=locations, lengths=lengths)
        self.assertIn("nll", out)
        self.assertTrue(torch.isfinite(out["nll"]))

    def test_fixed_time_query_terms_match_joint_intensity_factorization(self):
        cfg = STPPConfig.from_source(preset="neural_attncnf")
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

    def test_conditional_logprob_fn_matches_last_sequence_term(self):
        cfg = STPPConfig.from_source(preset="neural_attncnf")
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
        logprob = terms["logprob_fn"](query_locations)

        t_hist = int(state.payload["lengths"][0].item())
        event_times = state.payload["times_raw"][0, :t_hist]
        event_locs = state.payload["locations_norm"][0, :t_hist]
        z_seq = model.event_model._spatial_sequence_inputs(state)[0, :t_hist]
        query_time_raw = float(terms["query_time_raw"])
        h_query, _ = state.payload["_h_at_query_raw"](
            torch.tensor([[query_time_raw]], dtype=torch.float32)
        )
        if bool(getattr(model.event_model.spatial_decoder, "USES_NEURAL_AUX_STATE", False)):
            aux_dim = int(state.payload.get("spatial_aux_dim", 0))
            h_query_for_spatial = h_query[:, -aux_dim:] if aux_dim > 0 else h_query[:, :0]
        else:
            h_query_for_spatial = h_query
        z_aug = torch.cat([z_seq, h_query_for_spatial], dim=0)

        bsz, dim = query_locations.shape
        bsz_event_times = event_times.unsqueeze(0).expand(bsz, t_hist)
        bsz_event_times = torch.cat(
            [
                bsz_event_times,
                torch.full((bsz, 1), query_time_raw, dtype=bsz_event_times.dtype),
            ],
            dim=1,
        )
        bsz_event_locs = event_locs.unsqueeze(0).expand(bsz, t_hist, dim)
        bsz_event_locs = torch.cat([bsz_event_locs, query_locations.reshape(bsz, 1, dim)], dim=1)
        bsz_aux_state = z_aug.reshape(1, t_hist + 1, -1).expand(bsz, -1, -1)
        full_logprob = model.event_model.spatial_decoder.sequence_logprob(
            event_times=bsz_event_times,
            spatial_locations=bsz_event_locs,
            input_mask=None,
            aux_state=bsz_aux_state,
        )
        torch.testing.assert_close(logprob, full_logprob[:, -1], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
