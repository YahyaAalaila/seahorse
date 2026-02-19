import math
import unittest

import torch

from unified_stpp.config_utils import resolve_optimizer_hparams, resolve_t_end
from unified_stpp.models.decoders.spatial import CNFSpatial
from unified_stpp.training.trainer import Trainer


class NeuralSTPPBugfixTests(unittest.TestCase):
    def test_cnf_log_prob_subtracts_log_det(self):
        spatial = CNFSpatial(spatial_dim=2, hidden_dim=4)

        # Force a deterministic "ODE solution" so this test only verifies the
        # change-of-variables sign in log_prob().
        spatial._run_odeint = (
            lambda t_span, s_aug: torch.tensor(
                [[1.0, -2.0, 0.7]], dtype=s_aug.dtype, device=s_aug.device
            )
        )

        z = torch.zeros(1, 4)
        t = torch.ones(1, 1)
        s = torch.zeros(1, 2)
        t_prev = torch.zeros(1, 1)

        got = spatial.log_prob(z, t, s, t_prev).item()
        log_p0 = -0.5 * (2.0 * math.log(2.0 * math.pi) + (1.0**2 + (-2.0) ** 2))
        expected = log_p0 - 0.7
        self.assertAlmostEqual(got, expected, places=6)

    def test_resolve_t_end_supports_legacy_T_key(self):
        cfg = {"data": {"T": 1.0}}
        t_end = resolve_t_end(cfg["data"], fallback_t_end=5.0)
        self.assertEqual(t_end, 1.0)

    def test_optimizer_hparams_flow_into_trainer(self):
        lr, weight_decay, grad_clip = resolve_optimizer_hparams(
            {"lr": 1e-3, "weight_decay": 0.123, "grad_clip": 0.456},
            lr_default=1e-4,
            weight_decay_default=1e-5,
            grad_clip_default=5.0,
        )

        model = torch.nn.Linear(2, 1)
        trainer = Trainer(
            model,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            device="cpu",
        )

        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["weight_decay"], 0.123, places=12)
        self.assertAlmostEqual(trainer.grad_clip, 0.456, places=12)


if __name__ == "__main__":
    unittest.main()
