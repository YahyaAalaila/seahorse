"""RMTPP waiting-time density must integrate to one in float32.

The compensator was computed as ``(a / b) * (1 - exp(-b * dt))``.  With ``b`` at
its 1e-6 floor, ``exp(-b * dt)`` rounds to 1 in float32 and the compensator
vanishes, so the waiting-time density ``a * exp(-b dt - Lambda(dt))`` integrated
to far more than one and training could exploit it.  The fix uses ``-expm1``.
"""
import math

import torch

from seahorse.models.temporal_models.neural_temporal import RMTPPTemporalProcess


def _fixed_head(a_value: float, b_raw: float) -> RMTPPTemporalProcess:
    torch.manual_seed(0)
    model = RMTPPTemporalProcess(hidden_size=8)
    last = model.mlp[-1]
    with torch.no_grad():
        last.weight.zero_()
        # softplus^{-1}(a) so that a = a_value exactly; b_raw << 0 puts b at the 1e-6 floor.
        last.bias.copy_(torch.tensor([math.log(math.expm1(a_value)), b_raw]))
    return model.eval()


def _waiting_density_mass(model: RMTPPTemporalProcess, upper: float, n: int = 20001) -> float:
    tau = torch.linspace(0.0, upper, n, dtype=torch.float32)
    event_times = tau.reshape(-1, 1)
    mask = torch.ones_like(event_times)
    with torch.no_grad():
        # One event per row: first gap from t=0 uses h_0, so (a, b) come from the fixed head.
        event_ll, survival = model._event_terms(event_times, mask, event_times[:, 0])
    assert torch.all(survival == 0)
    density = torch.exp(event_ll[:, 0].double())
    return float(torch.trapezoid(density, tau.double()))


def test_b_is_at_floor():
    model = _fixed_head(100.0, -40.0)
    b = torch.nn.functional.softplus(model.mlp[-1].bias[1]).clamp(min=1e-6)
    assert float(b) == torch.tensor(1e-6, dtype=torch.float32).item()


def test_waiting_density_integrates_to_one_at_b_floor():
    a = 100.0
    model = _fixed_head(a, -40.0)
    # With b at the floor the process is ~Poisson(a); [0, 20/a] carries 1 - e^-20 of the mass.
    mass = _waiting_density_mass(model, 20.0 / a)
    assert abs(mass - 1.0) < 1e-3, mass


def test_waiting_density_integrates_to_one_moderate_b():
    model = _fixed_head(2.0, 0.5)
    b = float(torch.nn.functional.softplus(torch.tensor(0.5)))
    a = 2.0
    upper = 40.0
    expected = 1.0 - math.exp(-(a / b) * (1.0 - math.exp(-b * upper)))
    mass = _waiting_density_mass(model, upper, n=200001)
    assert abs(mass - expected) < 1e-3, (mass, expected)
