"""Shared-foundation invariants for the Neural STPP family.

These tests lock the shared contract before any faithful spatial variant ports
land. They verify:
  - local raw-time reconstruction while standardized space stays untouched
  - hidden-state partition semantics (intensity slice vs. spatial aux slice)
  - what the temporal backbone consumes
  - what spatial decoders will receive from the event-side contract
  - upstream-style integrate_lambda output shapes and masking behavior
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

import unified_stpp.models.temporal_models.neural_point_process as neural_point_process_module
from unified_stpp.models.base import Decoder
from unified_stpp.models.configs.neural_stpp import NeuralSTPPConfig
from unified_stpp.models.event_models.neural_stpp_event import NeuralSTPPEventModel
from unified_stpp.models.model_registry import register_spatial
from unified_stpp.models.state_models.neural_stpp_state import NeuralSTPPStateModel
from unified_stpp.models.temporal_models.neural_point_process import ActNorm, NeuralPointProcess, TimeVariableODE


class _CaptureTemporalCore(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._init_state = nn.Parameter(torch.zeros(hidden_dim))
        self.captured_event_times = None
        self.captured_spatial_location = None
        self.captured_input_mask = None
        self.captured_t0 = None
        self.captured_t1 = None

    def sequence_nll_and_states(self, event_times, spatial_location, input_mask, *, t0=None, t1=None):
        self.captured_event_times = event_times.detach().clone()
        self.captured_spatial_location = spatial_location.detach().clone()
        self.captured_input_mask = input_mask.detach().clone()
        self.captured_t0 = t0.detach().clone() if isinstance(t0, torch.Tensor) else t0
        self.captured_t1 = t1.detach().clone() if isinstance(t1, torch.Tensor) else t1

        bsz, steps = event_times.shape
        device = event_times.device
        dtype = event_times.dtype
        temporal_nll = torch.arange(steps, device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1)
        basis = torch.arange(self.hidden_dim, device=device, dtype=dtype).view(1, 1, -1)
        batch_offset = (100.0 * torch.arange(bsz, device=device, dtype=dtype)).view(bsz, 1, 1)
        step_offset = (10.0 * torch.arange(steps, device=device, dtype=dtype)).view(1, steps, 1)
        h_seq = basis + batch_offset + step_offset
        energy_reg = torch.tensor(0.25, device=device, dtype=dtype)
        h_final = torch.arange(bsz * self.hidden_dim, device=device, dtype=dtype).view(bsz, self.hidden_dim)
        return temporal_nll, h_seq, energy_reg, h_final

    def integrate_hidden(self, h_prev, t_prev, t_curr):
        del t_prev, t_curr
        zeros = torch.zeros(h_prev.shape[0], device=h_prev.device, dtype=h_prev.dtype)
        return h_prev + 1.0, zeros, zeros.sum()

    def get_intensity(self, state):
        return state[:, :1] + 1.0


@register_spatial("test_neural_shared_full")
class _FullCaptureDecoder(Decoder):
    SEQUENCE_COUPLED = True

    def __init__(self, hidden_dim: int, spatial_dim: int, **kwargs):
        super().__init__(hidden_dim=hidden_dim, spatial_dim=spatial_dim, **kwargs)
        self.last_call = None

    def log_prob(self, z, t, s, t_prev, x_field=None):
        raise NotImplementedError

    def nll(self, z, t, s, t_prev, x_field=None):
        raise NotImplementedError

    def sequence_nll(self, z_seq, t_seq, s_seq, t_prev_seq, lengths, mask, **kwargs):
        self.last_call = {
            "z_seq": z_seq.detach().clone(),
            "t_seq": t_seq.detach().clone(),
            "s_seq": s_seq.detach().clone(),
            "t_prev_seq": t_prev_seq.detach().clone(),
            "lengths": lengths.detach().clone(),
            "mask": mask.detach().clone(),
        }
        return z_seq.new_full(mask.shape, 0.5)


@register_spatial("test_neural_shared_aux")
class _AuxCaptureDecoder(_FullCaptureDecoder):
    USES_NEURAL_AUX_STATE = True


def _normalized_batch():
    times_raw = torch.tensor(
        [
            [10.0, 12.0, 16.0],
            [10.0, 14.0, 14.0],
        ],
        dtype=torch.float32,
    )
    time_mean = 10.0
    time_std = 2.0
    times_norm = (times_raw - time_mean) / time_std
    locations_norm = torch.tensor(
        [
            [[-1.0, 0.5], [0.2, -0.3], [0.8, 1.0]],
            [[0.1, -0.4], [0.5, 0.6], [9.9, 9.9]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([3, 2], dtype=torch.long)
    return times_norm, times_raw, locations_norm, lengths, time_mean, time_std


class TestNeuralSTPPSharedFoundation(unittest.TestCase):
    def test_temporal_actnorm_singleton_init_stays_finite(self):
        layer = ActNorm(num_features=4)
        x = torch.tensor([[0.0, 1.0, -2.0, 3.0]], dtype=torch.float32)
        y = layer(x)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.isfinite(layer.weight).all())
        self.assertTrue(torch.isfinite(layer.bias).all())

    def test_temporal_ode_underflow_re_raises_with_diagnostics(self):
        class _ZeroDrift(nn.Module):
            def forward(self, t, state):
                del t
                lambda_state, hidden_state = state
                return torch.zeros_like(lambda_state), torch.zeros_like(hidden_state)

        solver = TimeVariableODE(_ZeroDrift(), use_adjoint=False)
        original_odeint_std = neural_point_process_module._odeint_std

        def _raise_underflow(*args, **kwargs):
            del args, kwargs
            raise AssertionError("underflow in dt 0.0")

        neural_point_process_module._odeint_std = _raise_underflow
        try:
            with self.assertRaisesRegex(RuntimeError, "Neural STPP temporal ODE underflow in dt") as ctx:
                solver.integrate(
                    torch.tensor([1.0], dtype=torch.float32),
                    torch.tensor([1.0 + 1e-8], dtype=torch.float32),
                    (
                        torch.zeros(1, dtype=torch.float32),
                        torch.zeros(1, 4, dtype=torch.float32),
                    ),
                )
        finally:
            neural_point_process_module._odeint_std = original_odeint_std

        msg = str(ctx.exception)
        self.assertIn("min_raw_dt=", msg)
        self.assertIn("max_raw_dt=", msg)
        self.assertIn("mean_raw_dt=", msg)
        self.assertIn("non_increasing_count=", msg)
        self.assertIn("tiny_interval_count=", msg)
        self.assertIn("use_adjoint=False", msg)
        self.assertIn("method=dopri5", msg)
        self.assertIn("nfe=", msg)

    def test_temporal_adjoint_solver_uses_float64_time_dtype(self):
        class _ZeroDrift(nn.Module):
            def forward(self, t, state):
                del t
                lambda_state, hidden_state = state
                return torch.zeros_like(lambda_state), torch.zeros_like(hidden_state)

        solver = TimeVariableODE(_ZeroDrift(), use_adjoint=True)
        original_odeint_adj = neural_point_process_module._odeint_adj
        captured = {}

        def _capture_solver(func, init_state, eval_grid, **kwargs):
            del func, eval_grid
            captured.update(kwargs)
            return tuple(torch.stack([s, s], dim=0) for s in init_state)

        neural_point_process_module._odeint_adj = _capture_solver
        try:
            solver.integrate(
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([1.5], dtype=torch.float32),
                (
                    torch.zeros(1, dtype=torch.float32),
                    torch.zeros(1, 4, dtype=torch.float32),
                ),
            )
        finally:
            neural_point_process_module._odeint_adj = original_odeint_adj

        self.assertEqual(captured["options"]["dtype"], torch.float64)
        self.assertEqual(captured["adjoint_options"]["dtype"], torch.float64)

    def test_config_data_init_overrides_and_hidden_dim_parsing(self):
        dm = SimpleNamespace(
            train_dataset=SimpleNamespace(
                normalize_time=True,
                normalize_space=True,
                time_mean=3.5,
                time_std=1.25,
            )
        )
        overrides = NeuralSTPPConfig.data_init_overrides(dm)
        self.assertEqual(
            overrides,
            {
                "backbone": {
                    "normalize_time_inputs": True,
                    "normalize_space_inputs": True,
                    "time_mean": 3.5,
                    "time_std": 1.25,
                }
            },
        )

        cfg = NeuralSTPPConfig.from_dict(
            {
                "backbone": {"tpp_hidden_dims": "12-8"},
                "decoder": {"spatial": {"type": "jump_cnf"}},
            },
            hidden_dim=16,
            spatial_dim=2,
        )
        self.assertEqual(cfg.backbone_cfg["tpp_hidden_dims"], [12, 8])
        self.assertEqual(cfg._resolved_temporal_hidden_dim(), 12)
        self.assertEqual(cfg._resolved_temporal_hdim(), 6)

    def test_state_model_reconstructs_raw_time_and_preserves_normalized_space(self):
        times_norm, times_raw, locations_norm, lengths, time_mean, time_std = _normalized_batch()
        model = NeuralSTPPStateModel(
            hidden_dim=6,
            spatial_dim=2,
            tpp_hidden_dims=[6, 6],
            normalize_time_inputs=True,
            normalize_space_inputs=True,
            time_mean=time_mean,
            time_std=time_std,
        )
        capture = _CaptureTemporalCore(hidden_dim=6)
        model.temporal_core = capture

        state = model.encode_history(times=times_norm, locations=locations_norm, lengths=lengths)

        torch.testing.assert_close(capture.captured_event_times, times_raw)
        torch.testing.assert_close(capture.captured_spatial_location, locations_norm)
        self.assertTrue(torch.equal(capture.captured_input_mask, torch.tensor([[True, True, True], [True, True, False]])))

        payload = state.payload
        torch.testing.assert_close(payload["times_raw"], times_raw)
        torch.testing.assert_close(payload["locations_norm"], locations_norm)
        self.assertEqual(payload["temporal_hidden_seq"].shape, (2, 3, 6))
        self.assertEqual(payload["temporal_intensity_hidden_seq"].shape, (2, 3, 3))
        self.assertEqual(payload["spatial_aux_seq"].shape, (2, 3, 3))
        torch.testing.assert_close(
            payload["temporal_intensity_hidden_seq"],
            payload["temporal_hidden_seq"][..., :3],
        )
        torch.testing.assert_close(
            payload["spatial_aux_seq"],
            payload["temporal_hidden_seq"][..., 3:],
        )

    def test_state_model_repairs_denormalized_time_collapse_before_temporal_core(self):
        tiny = torch.nextafter(torch.tensor(0.0, dtype=torch.float32), torch.tensor(float("inf")))
        times_norm = torch.tensor([[0.0, tiny.item(), 1.0]], dtype=torch.float32)
        locations = torch.zeros(1, 3, 2, dtype=torch.float32)
        lengths = torch.tensor([3], dtype=torch.long)
        time_mean = 5.009698
        time_std = 1.0

        model = NeuralSTPPStateModel(
            hidden_dim=6,
            spatial_dim=2,
            tpp_hidden_dims=[6, 6],
            normalize_time_inputs=True,
            normalize_space_inputs=False,
            time_mean=time_mean,
            time_std=time_std,
        )
        capture = _CaptureTemporalCore(hidden_dim=6)
        model.temporal_core = capture

        naive_raw = times_norm * time_std + time_mean
        self.assertFalse(bool((naive_raw[:, 1] > naive_raw[:, 0]).all()))

        state = model.encode_history(times=times_norm, locations=locations, lengths=lengths)
        repaired_raw = capture.captured_event_times

        self.assertTrue(bool((repaired_raw[:, 1] > repaired_raw[:, 0]).all()))
        self.assertTrue(bool((repaired_raw[:, 2] > repaired_raw[:, 1]).all()))
        self.assertTrue(bool((state.payload["times_raw"][:, 1] > state.payload["times_raw"][:, 0]).all()))

    def test_temporal_intensity_depends_only_on_intensity_slice(self):
        model = NeuralPointProcess(
            cond_dim=2,
            hidden_dims=[6, 6],
            cond=True,
            hdim=2,
            separate=1,
        )
        state = torch.tensor(
            [
                [0.2, -0.1, 1.0, 2.0, 3.0, 4.0],
                [0.0, 0.5, -1.0, -2.0, -3.0, -4.0],
            ],
            dtype=torch.float32,
        )
        state_changed_aux = state.clone()
        state_changed_aux[:, 2:] = torch.tensor([[100.0, 101.0, 102.0, 103.0], [-100.0, -101.0, -102.0, -103.0]])
        torch.testing.assert_close(
            model.get_intensity(state),
            model.get_intensity(state_changed_aux),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_integrate_lambda_shapes_and_masking_follow_upstream_contract(self):
        torch.manual_seed(0)
        model = NeuralPointProcess(
            cond_dim=2,
            hidden_dims=[6, 6],
            cond=True,
            hdim=3,
            separate=1,
        )
        event_times = torch.tensor(
            [
                [0.2, 0.6, 1.0, 1.4],
                [0.1, 0.4, 0.4, 0.4],
            ],
            dtype=torch.float32,
        )
        locations = torch.tensor(
            [
                [[0.0, 0.0], [0.1, 0.2], [0.2, -0.1], [0.3, 0.1]],
                [[-0.2, 0.1], [0.4, 0.3], [9.0, 9.0], [9.0, 9.0]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

        intensities, _, hidden_states, details = model.integrate_lambda(
            event_times,
            locations,
            mask,
            t0=0.0,
            t1=None,
            nlinspace=1,
            return_details=True,
        )

        self.assertEqual(intensities.shape, (2, 4))
        self.assertEqual(hidden_states.shape, (2, 5, 6))
        self.assertEqual(details["cumulative_lambdas"].shape, (2, 4))
        torch.testing.assert_close(intensities[1, 2:], torch.ones(2))
        torch.testing.assert_close(details["cumulative_lambdas"][1, 2:], torch.zeros(2))
        torch.testing.assert_close(hidden_states[1, 3:], torch.zeros(2, 6), atol=1e-6, rtol=1e-6)

        temporal_nll, event_hidden, reg, h_final = model.sequence_nll_and_states(
            event_times,
            locations,
            mask,
            t0=0.0,
            t1=None,
        )
        self.assertEqual(temporal_nll.shape, (2, 4))
        self.assertEqual(event_hidden.shape, (2, 4, 6))
        self.assertEqual(h_final.shape, (2, 6))
        self.assertTrue(torch.isfinite(reg))

    def test_integrate_lambda_raises_on_non_increasing_active_event_times(self):
        model = NeuralPointProcess(
            cond_dim=2,
            hidden_dims=[6, 6],
            cond=True,
            hdim=3,
            separate=1,
        )
        event_times = torch.tensor([[0.2, 0.6, 0.6]], dtype=torch.float32)
        locations = torch.tensor([[[0.0, 0.0], [0.1, 0.2], [0.3, -0.1]]], dtype=torch.float32)
        mask = torch.tensor([[True, True, True]])

        with self.assertRaisesRegex(
            RuntimeError,
            "Neural STPP temporal integrate_lambda received non-increasing active event times",
        ) as ctx:
            model.integrate_lambda(
                event_times,
                locations,
                mask,
                t0=0.0,
                t1=None,
                nlinspace=1,
            )

        msg = str(ctx.exception)
        self.assertIn("event_index=2", msg)
        self.assertIn("bad_count=1", msg)
        self.assertIn("min_raw_dt=0.000000e+00", msg)

    def test_event_contract_passes_full_hidden_and_raw_time(self):
        times_norm, _, locations_norm, lengths, time_mean, time_std = _normalized_batch()
        state_model = NeuralSTPPStateModel(
            hidden_dim=6,
            spatial_dim=2,
            tpp_hidden_dims=[6, 6],
            normalize_time_inputs=True,
            normalize_space_inputs=True,
            time_mean=time_mean,
            time_std=time_std,
        )
        capture = _CaptureTemporalCore(hidden_dim=6)
        state_model.temporal_core = capture
        state = state_model.encode_history(times=times_norm, locations=locations_norm, lengths=lengths)

        event_model = NeuralSTPPEventModel(
            hidden_dim=6,
            spatial_dim=2,
            spatial_type="test_neural_shared_full",
        )
        out = event_model.training_loss(
            times=times_norm,
            locations=locations_norm,
            lengths=lengths,
            state=state,
        )

        call = event_model.spatial_decoder.last_call
        self.assertIsNotNone(call)
        torch.testing.assert_close(call["z_seq"], state.payload["temporal_hidden_seq"])
        torch.testing.assert_close(call["t_seq"].squeeze(-1), state.payload["times_raw"])
        torch.testing.assert_close(call["s_seq"], state.payload["locations_norm"])
        torch.testing.assert_close(
            call["t_prev_seq"].squeeze(-1),
            torch.tensor([[0.0, 10.0, 12.0], [0.0, 10.0, 14.0]], dtype=torch.float32),
        )
        self.assertTrue(torch.equal(call["mask"], torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])))
        self.assertEqual(out["nll_matrix"].shape, (2, 3))

    def test_event_contract_can_route_aux_slice_for_future_spatial_variants(self):
        times_norm, _, locations_norm, lengths, time_mean, time_std = _normalized_batch()
        state_model = NeuralSTPPStateModel(
            hidden_dim=6,
            spatial_dim=2,
            tpp_hidden_dims=[6, 6],
            normalize_time_inputs=True,
            normalize_space_inputs=True,
            time_mean=time_mean,
            time_std=time_std,
        )
        capture = _CaptureTemporalCore(hidden_dim=6)
        state_model.temporal_core = capture
        state = state_model.encode_history(times=times_norm, locations=locations_norm, lengths=lengths)

        event_model = NeuralSTPPEventModel(
            hidden_dim=3,
            spatial_dim=2,
            spatial_type="test_neural_shared_aux",
        )
        event_model.training_loss(
            times=times_norm,
            locations=locations_norm,
            lengths=lengths,
            state=state,
        )

        call = event_model.spatial_decoder.last_call
        self.assertIsNotNone(call)
        torch.testing.assert_close(call["z_seq"], state.payload["spatial_aux_seq"])


if __name__ == "__main__":
    unittest.main()
