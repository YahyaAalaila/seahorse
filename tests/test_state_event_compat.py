"""
Parity tests for Stage-1 StateModel/EventModel compatibility adapters.
"""

import copy
import unittest

import torch

from unified_stpp.registry import build_model


def _tiny_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    times = torch.tensor(
        [[0.00, 0.20, 0.50, 0.90], [0.00, 0.10, 0.40, 0.80]],
        dtype=torch.float32,
    )
    locations = torch.tensor(
        [
            [[0.0, 0.0], [0.3, -0.1], [0.1, 0.2], [-0.2, 0.1]],
            [[0.0, 0.0], [0.2, 0.1], [0.2, -0.2], [0.4, 0.3]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([4, 4], dtype=torch.long)
    return times, locations, lengths


class TestStateEventAdapterParity(unittest.TestCase):
    def _assert_parity(self, *, preset=None, config=None):
        cfg_legacy = copy.deepcopy(config or {})
        cfg_adapter = copy.deepcopy(config or {})

        torch.manual_seed(7)
        legacy_model = build_model(
            config=cfg_legacy,
            preset=preset,
            spatial_dim=2,
            hidden_dim=16,
        )
        torch.manual_seed(7)
        adapter_model = build_model(
            config=cfg_adapter,
            preset=preset,
            spatial_dim=2,
            hidden_dim=16,
        )
        adapter_model.load_state_dict(legacy_model.state_dict())

        self.assertIsNotNone(adapter_model.state_model)
        self.assertIsNotNone(adapter_model.event_model)
        self.assertFalse(legacy_model.use_state_event_path)
        adapter_model.use_state_event_path = True

        legacy_model.eval()
        adapter_model.eval()

        times, locations, lengths = _tiny_batch()
        with torch.no_grad():
            legacy_out = legacy_model(times=times, locations=locations, lengths=lengths)
            adapter_out = adapter_model(times=times, locations=locations, lengths=lengths)

        torch.testing.assert_close(legacy_out["nll"], adapter_out["nll"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            legacy_out["nll_per_event"], adapter_out["nll_per_event"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            legacy_out["total_events"], adapter_out["total_events"], rtol=0.0, atol=0.0
        )

    def test_identity_path_parity(self):
        self._assert_parity(preset="deep_stpp")

    def test_non_identity_path_parity(self):
        config = {
            "encoder": {"type": "gru", "num_layers": 1},
            "dynamics": {
                "type": "neural_ode",
                "solver": "euler",
                "n_steps": 8,
                "atol": 1e-5,
                "rtol": 1e-5,
                "augmented": False,
                "use_adjoint": False,
            },
            "updater": {"type": "gru_jump"},
            "decoder": {
                "type": "factorized",
                "temporal": {"type": "lognormal_mixture", "n_components": 4},
                "spatial": {"type": "gaussian_mixture", "n_components": 4},
            },
        }
        self._assert_parity(config=config)


if __name__ == "__main__":
    unittest.main()
