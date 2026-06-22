"""Faithful JumpCNF checks on top of the shared Neural STPP backbone."""

from __future__ import annotations

import math
import unittest

import torch

from seahorse.config.schema import STPPConfig
from seahorse.models.configs.neural_stpp import NeuralSTPPSharedJumpCNFConfig
from seahorse.models.spatial_models import neural_jumpcnf as neural_jumpcnf_module
from seahorse.models.spatial_models.neural_jumpcnf import NeuralJumpCNFSpatial


class TestNeuralJumpCNFSpatial(unittest.TestCase):
    def test_parameter_shapes_and_aux_slice_semantics(self):
        decoder = NeuralJumpCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=3,
            hidden_dims="8-8",
            solve_reverse=False,
            n_flows=4,
        )
        self.assertEqual(decoder.inst_flow.hypernet[0].in_features, 1 + 2 + 3)
        self.assertEqual(tuple(decoder.z_mean.shape), (1, 2))
        self.assertEqual(tuple(decoder.z_logstd.shape), (1, 2))
        self.assertTrue(decoder.z_mean.requires_grad)
        self.assertTrue(decoder.z_logstd.requires_grad)

        z_full = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]]],
            dtype=torch.float32,
        )
        expected_aux = torch.tensor([[[3.0, 4.0, 5.0], [30.0, 40.0, 50.0]]], dtype=torch.float32)
        torch.testing.assert_close(decoder._select_aux(z_full), expected_aux)

    def test_exact_single_event_nll_matches_standard_normal_when_flows_zero(self):
        decoder = NeuralJumpCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=2,
            hidden_dims=[],
            solve_reverse=False,
            n_flows=4,
        )
        for param in decoder.parameters():
            param.data.zero_()

        z_seq = torch.zeros(1, 1, 4)
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
            [[0.5 * (1.0**2 + math.log(2.0 * math.pi)) + 0.5 * ((-1.0) ** 2 + math.log(2.0 * math.pi))]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(nll, expected, rtol=1e-5, atol=1e-5)

    def test_learnable_base_distribution_changes_nll(self):
        decoder = NeuralJumpCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=2,
            hidden_dims=[],
            solve_reverse=False,
            n_flows=4,
        )
        for param in decoder.parameters():
            param.data.zero_()

        z_seq = torch.zeros(1, 1, 4)
        t_seq = torch.tensor([[[0.5]]], dtype=torch.float32)
        s_seq = torch.zeros(1, 1, 2)
        t_prev_seq = torch.zeros(1, 1, 1)
        lengths = torch.tensor([1], dtype=torch.long)
        mask = torch.tensor([[1.0]], dtype=torch.float32)

        base_nll = decoder.sequence_nll(
            z_seq=z_seq,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        )
        decoder.z_mean.data.fill_(1.0)
        shifted_nll = decoder.sequence_nll(
            z_seq=z_seq,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        )
        torch.testing.assert_close(shifted_nll, base_nll + 1.0, rtol=1e-5, atol=1e-5)

    def test_masked_padded_intervals_never_reach_cnf_with_zero_dt(self):
        decoder = NeuralJumpCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=2,
            hidden_dims=[],
            solve_reverse=False,
            n_flows=2,
        )
        for param in decoder.parameters():
            param.data.zero_()

        seen_dt: list[torch.Tensor] = []

        def _recording_integrate(t0, t1, x, logpx, **kwargs):
            del kwargs
            dt = (t1 - t0).detach().cpu()
            seen_dt.append(dt)
            self.assertTrue(torch.all(dt.abs() > 0.0), f"zero dt reached spatial CNF: {dt}")
            return x, logpx, x.new_tensor(0.0)

        decoder.cnf.integrate = _recording_integrate  # type: ignore[method-assign]

        z_seq = torch.zeros(2, 3, 4)
        t_seq = torch.tensor(
            [[[0.4], [0.8], [1.2]], [[0.3], [0.0], [0.0]]],
            dtype=torch.float32,
        )
        s_seq = torch.tensor(
            [
                [[0.0, 0.0], [0.2, -0.1], [0.4, 0.3]],
                [[-0.1, 0.2], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        t_prev_seq = torch.zeros(2, 3, 1)
        lengths = torch.tensor([3, 1], dtype=torch.long)
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float32)

        nll = decoder.sequence_nll(
            z_seq=z_seq,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        )
        self.assertTrue(torch.isfinite(nll).all())
        self.assertTrue(seen_dt)

    def test_tiny_intervals_never_reach_cnf_solver(self):
        decoder = NeuralJumpCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=2,
            hidden_dims=[],
            solve_reverse=False,
            n_flows=2,
        )
        for param in decoder.parameters():
            param.data.zero_()

        seen_dt: list[torch.Tensor] = []

        def _recording_integrate(t0, t1, x, logpx, **kwargs):
            del kwargs
            dt = (t1 - t0).detach().cpu()
            seen_dt.append(dt)
            scale = torch.maximum(torch.maximum(t0.abs(), t1.abs()), torch.ones_like(t0)).detach().cpu()
            tiny_threshold = torch.finfo(t0.dtype).eps * scale
            self.assertTrue(torch.all(dt.abs() > tiny_threshold), f"tiny dt reached spatial CNF: {dt}")
            return x, logpx, x.new_tensor(0.0)

        decoder.cnf.integrate = _recording_integrate  # type: ignore[method-assign]

        z_seq = torch.zeros(1, 3, 4)
        t_seq = torch.tensor([[[0.3], [0.30000005], [0.8]]], dtype=torch.float32)
        s_seq = torch.tensor([[[0.0, 0.0], [0.2, -0.1], [0.4, 0.3]]], dtype=torch.float32)
        t_prev_seq = torch.zeros(1, 3, 1)
        lengths = torch.tensor([3], dtype=torch.long)
        mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)

        nll = decoder.sequence_nll(
            z_seq=z_seq,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        )
        self.assertTrue(torch.isfinite(nll).all())
        self.assertTrue(seen_dt)

    def test_backward_with_masked_padding_stays_finite(self):
        decoder = NeuralJumpCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=2,
            hidden_dims=[],
            solve_reverse=False,
            n_flows=2,
        )
        for param in decoder.parameters():
            param.data.zero_()

        decoder.train()
        z_seq = torch.zeros(2, 3, 4)
        t_seq = torch.tensor(
            [[[0.4], [0.8], [1.2]], [[0.3], [0.0], [0.0]]],
            dtype=torch.float32,
        )
        s_seq = torch.tensor(
            [
                [[0.0, 0.0], [0.2, -0.1], [0.4, 0.3]],
                [[-0.1, 0.2], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        t_prev_seq = torch.zeros(2, 3, 1)
        lengths = torch.tensor([3, 1], dtype=torch.long)
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float32)

        loss = decoder.sequence_nll(
            z_seq=z_seq,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        ).sum()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for param in decoder.parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_underflow_error_re_raises_with_jumpcnf_diagnostics(self):
        decoder = NeuralJumpCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=2,
            hidden_dims=[],
            solve_reverse=False,
            use_adjoint=True,
            n_flows=2,
        )

        original_odeint_adj = neural_jumpcnf_module._odeint_adj

        def _raise_underflow(*args, **kwargs):
            del args, kwargs
            raise AssertionError("underflow in dt 0.0")

        neural_jumpcnf_module._odeint_adj = _raise_underflow
        try:
            with self.assertRaisesRegex(RuntimeError, "JumpCNF spatial CNF underflow in dt") as ctx:
                decoder.cnf.integrate(
                    torch.tensor([2.8], dtype=torch.float32),
                    torch.tensor([2.3], dtype=torch.float32),
                    torch.zeros(1, 4, dtype=torch.float32),
                    torch.zeros(1, dtype=torch.float32),
                )
        finally:
            neural_jumpcnf_module._odeint_adj = original_odeint_adj

        msg = str(ctx.exception)
        self.assertIn("min_abs_dt=", msg)
        self.assertIn("max_abs_dt=", msg)
        self.assertIn("mean_abs_dt=", msg)
        self.assertIn("tiny_interval_count=", msg)
        self.assertIn("use_adjoint=True", msg)
        self.assertIn("solve_reverse=False", msg)
        self.assertIn("training=True", msg)
        self.assertIn("nfe=", msg)

    def test_adjoint_solver_uses_float64_time_dtype(self):
        decoder = NeuralJumpCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=2,
            hidden_dims=[],
            solve_reverse=False,
            use_adjoint=True,
            n_flows=2,
        )

        original_odeint_adj = neural_jumpcnf_module._odeint_adj
        captured = {}

        def _capture_solver(func, init_state, tt, **kwargs):
            del func, tt
            captured.update(kwargs)
            return tuple(torch.stack([s, s], dim=0) for s in init_state)

        neural_jumpcnf_module._odeint_adj = _capture_solver
        try:
            decoder.cnf.integrate(
                torch.tensor([2.8], dtype=torch.float32),
                torch.tensor([2.3], dtype=torch.float32),
                torch.zeros(1, 4, dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32),
            )
        finally:
            neural_jumpcnf_module._odeint_adj = original_odeint_adj

        self.assertEqual(captured["options"]["dtype"], torch.float64)
        self.assertEqual(captured["adjoint_options"]["dtype"], torch.float64)

    def test_integrate_repairs_reverse_endpoint_before_solver(self):
        decoder = NeuralJumpCNFSpatial(
            spatial_dim=2,
            hidden_dim=8,
            spatial_aux_dim=2,
            hidden_dims=[],
            solve_reverse=False,
            use_adjoint=False,
            n_flows=2,
        )

        original_odeint_std = neural_jumpcnf_module._odeint_std
        captured: dict[str, torch.Tensor] = {}

        def _capture_solver(func, init_state, tt, **kwargs):
            del func, tt, kwargs
            captured["t0"] = init_state[0].detach().clone()
            captured["t1"] = init_state[1].detach().clone()
            return tuple(torch.stack([s, s], dim=0) for s in init_state)

        neural_jumpcnf_module._odeint_std = _capture_solver
        try:
            decoder.cnf.integrate(
                torch.tensor([3.0], dtype=torch.float32),
                torch.tensor([3.0], dtype=torch.float32),
                torch.zeros(1, 4, dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32),
            )
        finally:
            neural_jumpcnf_module._odeint_std = original_odeint_std

        self.assertIn("t0", captured)
        self.assertIn("t1", captured)
        self.assertTrue(bool((captured["t1"] < captured["t0"]).all()))


class TestNeuralJumpCNFPreset(unittest.TestCase):
    def test_preset_loads_builds_and_wires_solve_reverse(self):
        cfg = STPPConfig.from_source(preset="neural_jumpcnf")
        self.assertEqual(cfg.model.preset, "neural_jumpcnf")
        self.assertEqual(cfg.data.adapter_kwargs.get("max_events"), 4000)
        model = cfg.model.build_model()

        decoder = model.event_model.spatial_decoder
        self.assertEqual(type(decoder).__name__, "NeuralJumpCNFSpatial")
        self.assertTrue(decoder.solve_reverse)
        self.assertFalse(decoder.cnf.use_adjoint)
        self.assertIs(decoder.aux_odefunc, model.state_model.temporal_core.hidden_state_dynamics)

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

    def test_config_can_disable_adjoint_without_runner_changes(self):
        cfg = NeuralSTPPSharedJumpCNFConfig.from_dict(
            {"decoder": {"spatial": {"use_adjoint": False}}},
            hidden_dim=32,
            spatial_dim=2,
        )
        model = cfg.build_model()
        decoder = model.event_model.spatial_decoder
        self.assertFalse(decoder.cnf.use_adjoint)

    def test_fixed_time_query_terms_match_joint_intensity_factorization(self):
        cfg = STPPConfig.from_source(preset="neural_jumpcnf")
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
        cfg = STPPConfig.from_source(preset="neural_jumpcnf")
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
