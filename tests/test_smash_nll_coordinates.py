"""SMASH surrogate NLL: the network is queried in its training coordinates y = 2u - 1."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from seahorse.models.event_models.smash import ScoreMatchingProcess
from seahorse.registry import build_model


class _ConstantIntensityNet(nn.Module):
    """Stub score network: constant intensity, zero location score; records its inputs."""

    self_condition = False

    def __init__(self, rate: float):
        super().__init__()
        self.rate = rate
        self.time_queries: list[torch.Tensor] = []
        self.loc_queries: list[torch.Tensor] = []

    def get_intensity(self, t, cond):
        self.time_queries.append(t.detach().clone())
        return torch.full_like(t, self.rate)

    def get_score_loc(self, x, cond):
        self.loc_queries.append(x.detach().clone())
        return torch.zeros_like(x[:, :, 1:])


def _process(net):
    return ScoreMatchingProcess(model=net, sigma=(0.05, 0.05), seq_length=3)


def test_constant_intensity_gives_the_exponential_density_of_the_token():
    # lambda(y) = c on y in [-1, 1] is an exponential law with rate 2c on u = (y + 1) / 2.
    rate = 3.0
    u = torch.tensor([[[0.1, 0.2, 0.3]], [[0.5, 0.9, 0.1]], [[0.8, 0.0, 1.0]]])
    cond = torch.zeros(3, 1, 4)
    net = _ConstantIntensityNet(rate)
    terms = _process(net).nll_terms(u, cond)
    expected = -(math.log(2 * rate) - 2 * rate * u[:, 0, 0])
    torch.testing.assert_close(terms["temporal"], expected, rtol=1e-6, atol=1e-6)


def test_network_is_queried_in_the_training_range():
    net = _ConstantIntensityNet(1.0)
    u = torch.tensor([[[0.0, 0.0, 0.0]], [[0.25, 0.5, 1.0]], [[1.0, 1.0, 0.75]]])
    _process(net).nll_terms(u, torch.zeros(3, 1, 4))
    grid, at_event = net.time_queries
    assert float(grid.min()) == -1.0                      # compensator starts at u = 0, i.e. y = -1
    torch.testing.assert_close(at_event.view(-1), 2 * u[:, 0, 0] - 1)
    torch.testing.assert_close(net.loc_queries[0], 2 * u - 1)


def test_nll_cal_sums_the_per_event_terms_and_eval_nll_averages_them():
    torch.manual_seed(0)
    model = build_model(config={}, preset="smash", spatial_dim=2, hidden_dim=32, n_marks=0)
    sp = model.event_model.score_matching
    u = torch.rand(5, 1, 3)
    cond = torch.randn(5, 1, sp.model.cond_dim * 3)
    terms = sp.nll_terms(u, cond)
    total, temporal, spatial = sp.NLL_cal(u, cond)
    assert math.isclose(temporal, float(terms["temporal"].sum()), rel_tol=1e-5)
    assert math.isclose(spatial, float(terms["spatial_surrogate"].sum()), rel_tol=1e-5)
    assert math.isclose(total, temporal + spatial, rel_tol=1e-6)
    # The spatial surrogate is the Tweedie constant plus a non-negative data term.
    const = math.log(2 * math.pi * 0.05 ** 2)
    assert bool((terms["spatial_surrogate"] >= const - 1e-5).all())
