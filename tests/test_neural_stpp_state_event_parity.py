"""
Parity checks for Stage-2 Neural STPP migration to StateModel/EventModel.
"""

from __future__ import annotations

import copy
import unittest

import torch

from unified_stpp.registry import build_model


def _tiny_batch():
    times = torch.tensor(
        [
            [0.00, 0.20, 0.50, 0.90, 1.30],
            [0.00, 0.30, 0.60, 1.00, 1.20],
        ],
        dtype=torch.float32,
    )
    locations = torch.tensor(
        [
            [[0.00, 0.00], [0.25, -0.10], [0.15, 0.30], [-0.20, 0.10], [-0.10, 0.25]],
            [[0.00, 0.00], [0.10, 0.20], [0.30, -0.20], [0.35, 0.15], [0.20, 0.25]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([5, 4], dtype=torch.long)
    return times, locations, lengths


def _build_neural_stpp_model(preset: str, use_state_event_path: bool):
    cfg = {"use_state_event_path": use_state_event_path}
    return build_model(
        config=copy.deepcopy(cfg),
        preset=preset,
        spatial_dim=2,
        hidden_dim=16,
    )


class TestNeuralSTPPStateEventParity(unittest.TestCase):
    def _assert_parity(self, preset: str, forward_seed: int = 11):
        torch.manual_seed(7)
        model_new = _build_neural_stpp_model(preset, use_state_event_path=True)
        torch.manual_seed(7)
        model_fallback = _build_neural_stpp_model(preset, use_state_event_path=False)
        model_fallback.load_state_dict(model_new.state_dict())

        self.assertTrue(model_new.use_state_event_path)
        self.assertFalse(model_fallback.use_state_event_path)

        model_new.eval()
        model_fallback.eval()
        times, locations, lengths = _tiny_batch()

        with torch.no_grad():
            torch.manual_seed(forward_seed)
            out_new = model_new(times=times, locations=locations, lengths=lengths)
            torch.manual_seed(forward_seed)
            out_fallback = model_fallback(times=times, locations=locations, lengths=lengths)

        torch.testing.assert_close(out_new["nll"], out_fallback["nll"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            out_new["nll_per_event"], out_fallback["nll_per_event"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            out_new["total_events"], out_fallback["total_events"], rtol=0.0, atol=0.0
        )

        # Stage-2 path keeps eventwise and regularization terms explicit.
        self.assertIn("nll_matrix", out_new)
        self.assertIn("mask", out_new)
        self.assertIn("temporal_nll_matrix", out_new)
        self.assertIn("spatial_nll_matrix", out_new)
        self.assertIn("temporal_energy_reg", out_new)
        self.assertIn("spatial_reg", out_new)
        self.assertIn("regularization_total", out_new)

    def test_neural_stpp_jump_sc_parity(self):
        self._assert_parity("neural_stpp_jump_sc")

    def test_neural_stpp_attn_sc_parity(self):
        # SelfAttentiveCNFSpatial samples internal noise during forward;
        # reset RNG before each forward for strict parity.
        self._assert_parity("neural_stpp_attn_sc", forward_seed=13)


if __name__ == "__main__":
    unittest.main()
