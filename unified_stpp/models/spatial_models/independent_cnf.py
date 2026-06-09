"""Independent CNF spatial model for factorized STPP baselines.

Each event's spatial distribution is modelled independently via a continuous
normalizing flow conditioned only on the event time — no cross-event conditioning.

Two variants controlled by `squash_time`:
  squash_time=True  (cnf):   all events share integration window [0, time_offset]
  squash_time=False (tvcnf): each event integrates over [t_i + time_offset, 0]

Interface (matches GaussianMixtureSpatialModel):

    logprob(event_times, locations, input_mask) -> (B, T)

Returns per-event spatial log-probs; 0.0 at padding positions.

Implementation note
-------------------
TimeVariableCNF and the supporting diffeq layer classes are inlined here
(originally from facebook/neural_stpp/models/spatial/cnf.py and
neural_stpp/diffeq_layers/). The math is identical to the original; only
package structure has changed so we carry no dependency on neural_stpp internals.
"""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torchdiffeq import odeint_adjoint as _odeint_adjoint
    HAS_ODEINT_ADJOINT = True
except ImportError:
    HAS_ODEINT_ADJOINT = False


# ============================================================================
# Local divergence helper (same math as cnf_spatial.divergence_bf)
# ============================================================================

def _divergence_bf(f: Tensor, y: Tensor, training: bool) -> Tensor:
    """Exact trace via brute-force Jacobian diagonal. O(d) backward passes."""
    sum_diag = 0.0
    for i in range(f.shape[1]):
        retain = training or (i < f.shape[1] - 1)
        grad_i = torch.autograd.grad(
            f[:, i].sum(), y, create_graph=training, retain_graph=retain
        )[0].contiguous()
        sum_diag = sum_diag + grad_i[:, i].contiguous()
    return sum_diag  # type: ignore[return-value]


# ============================================================================
# ODE-compatible linear layer primitives
# (inlined from neural_stpp/diffeq_layers/basic.py — math unchanged)
# ============================================================================

class _ConcatLinear_v2(nn.Module):
    """forward(t, x) = Linear(x) + HyperBias(t)  — per-element time bias."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self._layer      = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        return self._layer(x) + self._hyper_bias(t.reshape(-1, 1))


class _ConcatSquashLinear(nn.Module):
    """forward(t, x) = Linear(x) * σ(HyperGate(t)) + HyperBias(t)."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self._layer      = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        t1 = t.reshape(-1, 1)
        return (
            self._layer(x) * torch.sigmoid(self._hyper_gate(t1))
            + self._hyper_bias(t1)
        )


class _TimeDependentSwish(nn.Module):
    """forward(t, x) = x * σ(x * β(t))  where β is a small learned MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.beta = nn.Sequential(
            nn.Linear(1, min(64, dim * 4)),
            nn.Softplus(),
            nn.Linear(min(64, dim * 4), dim),
            nn.Softplus(),
        )

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        beta = self.beta(t.reshape(-1, 1))
        return x * torch.sigmoid(x * beta)


class _SequentialDiffEq(nn.Module):
    """Sequential ODE layers.  Each layer has forward(t, x) or forward(x).

    The `_is_time_dep` flag is computed once at construction to avoid
    hot-path signature inspection on every ODE function evaluation.
    """

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self._is_time_dep = [
            len(inspect.signature(L.forward).parameters) >= 2
            for L in layers
        ]

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        for layer, td in zip(self.layers, self._is_time_dep):
            x = layer(t, x) if td else layer(x)
        return x


_LAYERTYPES = {
    "concat":       _ConcatLinear_v2,
    "concatsquash": _ConcatSquashLinear,
}
_ACTFNS = {
    "softplus": lambda dim: nn.Softplus(),
    "swish":    lambda dim: _TimeDependentSwish(dim),
}


def _build_fc_odefunc(
    dim: int,
    hidden_dims: list[int],
    actfn: str = "softplus",
    layer_type: str = "concat",
    zero_init: bool = True,
) -> _SequentialDiffEq:
    """Build a fully-connected ODE velocity field v_θ(t, x).

    Internalized from neural_stpp/models/spatial/cnf.py::build_fc_odefunc.
    Math is identical to the original; only the layer classes come from
    local definitions instead of the neural_stpp.diffeq_layers package.
    """
    if layer_type not in _LAYERTYPES:
        raise ValueError(f"layer_type must be one of {list(_LAYERTYPES)} got {layer_type!r}")
    if actfn not in _ACTFNS:
        raise ValueError(f"actfn must be one of {list(_ACTFNS)} got {actfn!r}")

    layer_fn = _LAYERTYPES[layer_type]
    dims = [dim] + list(hidden_dims)
    layers: list[nn.Module] = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers.append(layer_fn(d_in, d_out))
        layers.append(_ACTFNS[actfn](d_out))
    layers.append(layer_fn(hidden_dims[-1], dim))

    if zero_init:
        for m in layers[-1].modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    return _SequentialDiffEq(*layers)


# ============================================================================
# TimeVariableCNF
# (inlined from neural_stpp/models/spatial/cnf.py — math unchanged)
# ============================================================================

class _TimeVariableCNF(nn.Module):
    """Per-sample time-variable CNF integrator.

    Integrates from individual (t0[i], t1[i]) limits by mapping to a shared
    dummy time domain [start_time=0, end_time=1].

    At dummy step s:
        t_actual[i] = t0[i] + (t1[i] - t0[i]) * s
        dx_dummy[i] = v(t_actual[i], x[i]) * (t1[i] - t0[i])

    Inlined from neural_stpp/models/spatial/cnf.py::TimeVariableCNF.
    Only change: `odeint_adjoint` receives `options={"dtype": t0.dtype}`
    instead of the original's hardcoded float32.
    """

    start_time = 0.0
    end_time   = 1.0

    def __init__(
        self,
        func: nn.Module,
        dim: int,
        tol: float = 1e-5,
        method: str = "dopri5",
        energy_regularization: float = 0.0,
        jacnorm_regularization: float = 0.0,
    ):
        super().__init__()
        self.func                   = func
        self.dim                    = dim
        self.tol                    = tol
        self.method                 = method
        self.energy_regularization  = energy_regularization
        self.jacnorm_regularization = jacnorm_regularization
        self.nfe                    = 0

    def integrate(
        self,
        t0: Tensor,     # (N,)
        t1: Tensor,     # (N,)
        x: Tensor,      # (N, D)
        logpx: Tensor,  # (N,)
    ) -> tuple[Tensor, Tensor]:
        """Integrate the CNF from t0 to t1 per sample.

        Returns:
            z:         (N, D) — transformed samples
            delta_logp: (N,) — accumulated log-det Jacobian (add to logpz to get logpx)
        """
        self.nfe = 0
        e      = torch.randn_like(x)[:, : self.dim]
        energy  = torch.zeros(1, device=x.device, dtype=x.dtype)
        jacnorm = torch.zeros(1, device=x.device, dtype=x.dtype)
        tt = torch.tensor(
            [self.start_time, self.end_time], device=t0.device, dtype=t0.dtype
        )
        initial_state = (t0, t1, e, x, logpx, energy, jacnorm)

        if HAS_ODEINT_ADJOINT:
            solution = _odeint_adjoint(
                self, initial_state, tt,
                rtol=self.tol, atol=self.tol, method=self.method,
                options={"dtype": t0.dtype},
            )
            _, _, _, y, logpy, energy_out, jacnorm_out = tuple(s[-1] for s in solution)
        else:
            # Euler fallback (low accuracy — only for quick smoke tests)
            state = initial_state
            n_steps = 50
            dt_dummy = (self.end_time - self.start_time) / n_steps
            for k in range(n_steps):
                s = self.start_time + k * dt_dummy
                s_tensor = torch.tensor(s, dtype=t0.dtype, device=t0.device)
                d_state = self.forward(s_tensor, state)
                state = tuple(
                    sv + dt_dummy * dv for sv, dv in zip(state, d_state)
                )
            _, _, _, y, logpy, energy_out, jacnorm_out = state

        regularization = (
            self.energy_regularization  * (energy_out  - energy_out.detach())
            + self.jacnorm_regularization * (jacnorm_out - jacnorm_out.detach())
        )
        return y, logpy + regularization

    def forward(self, s: Tensor, state: tuple) -> tuple:
        """ODE RHS in dummy time.  Called by torchdiffeq at each solver step."""
        self.nfe += 1
        t0, t1, e, x, logpx, _energy, _jacnorm = state

        ratio = (t1 - t0) / (self.end_time - self.start_time)          # (N,)
        t     = (s - self.start_time) * ratio + t0                      # (N,) actual time

        vjp = None
        with torch.enable_grad():
            x = x.requires_grad_(True)

            dx     = self.func(t, x)
            dx     = dx * ratio.reshape(-1, *([1] * (x.ndim - 1)))      # scaled to dummy
            dx_div = dx  # nonself_connections not used for IndependentCNF

            if not self.training:
                div = _divergence_bf(dx_div[:, : self.dim], x, self.training)
            else:
                vjp = torch.autograd.grad(
                    dx_div[:, : self.dim], x, e,
                    create_graph=True, retain_graph=True,
                )[0]
                vjp  = vjp[:, : self.dim]
                div  = torch.sum(vjp * e, dim=1)

        if not self.training:
            dx  = dx.detach()
            div = div.detach()

        d_energy  = torch.sum(dx * dx).reshape(1) / x.shape[0]
        d_jacnorm = (
            torch.sum(vjp * vjp).reshape(1) / x.shape[0]
            if self.training and vjp is not None
            else torch.zeros(1, device=x.device, dtype=x.dtype)
        )

        return (
            torch.zeros_like(t0),
            torch.zeros_like(t1),
            torch.zeros_like(e),
            dx,
            -div,
            d_energy,
            d_jacnorm,
        )

    def extra_repr(self) -> str:
        return (
            f"method={self.method}, tol={self.tol}, "
            f"energy={self.energy_regularization}, "
            f"jacnorm={self.jacnorm_regularization}"
        )


# ============================================================================
# Gaussian log-likelihood helper
# (inlined from neural_stpp/models/spatial/independent.py — math unchanged)
# ============================================================================

def _gaussian_loglik(z: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    """Diagonal Gaussian log-likelihood.  Returns (N, D); sum over D for total."""
    c   = torch.tensor([math.log(2.0 * math.pi)], device=z.device, dtype=z.dtype)
    inv = torch.exp(-log_std)
    tmp = (z - mean) * inv
    return -0.5 * (tmp * tmp + 2.0 * log_std + c)


# ============================================================================
# IndependentCNF — public API
# ============================================================================

class IndependentCNF(nn.Module):
    """Independent CNF spatial density for factorized STPP baselines.

    Models each event's spatial distribution independently (no cross-event
    conditioning) via a continuous normalizing flow conditioned on event time.

    squash_time=True  (cnf):   all events share integration window [0, time_offset]
    squash_time=False (tvcnf): each event integrates over [t_i + time_offset, 0]

    Parameters
    ----------
    dim           : spatial dimension
    hidden_dims   : hidden layer widths for the velocity-field MLP
    layer_type    : "concat" (ConcatLinear_v2) or "concatsquash" (ConcatSquashLinear)
    actfn         : "softplus" or "swish"
    zero_init     : initialise output layer of velocity field to zero (recommended)
    tol           : ODE solver tolerance (rtol = atol = tol)
    otreg_strength: OT regularisation coefficient (0 = off)
    squash_time   : True → cnf, False → tvcnf (see module docstring)
    """

    time_offset: float = 2.0

    def __init__(
        self,
        dim: int = 2,
        hidden_dims: tuple[int, ...] = (64, 64, 64),
        layer_type: str = "concat",
        actfn: str = "softplus",
        zero_init: bool = True,
        tol: float = 1e-5,
        otreg_strength: float = 0.0,
        squash_time: bool = True,
    ):
        super().__init__()
        self.squash_time = squash_time

        func = _build_fc_odefunc(
            dim=dim,
            hidden_dims=list(hidden_dims),
            actfn=actfn,
            layer_type=layer_type,
            zero_init=zero_init,
        )
        self.cnf = _TimeVariableCNF(
            func, dim,
            tol=tol,
            energy_regularization=otreg_strength,
            jacnorm_regularization=otreg_strength,
        )
        self.z_mean   = nn.Parameter(torch.zeros(1, dim))
        self.z_logstd = nn.Parameter(torch.zeros(1, dim))

    def logprob(
        self,
        event_times: Tensor,  # (B, T)
        locations:   Tensor,  # (B, T, D)
        input_mask:  Tensor,  # (B, T) float — 1 for valid events, 0 for padding
    ) -> Tensor:              # (B, T) per-event log-prob; 0.0 at padding
        """Compute per-event spatial log-probabilities.

        Args:
            event_times : (B, T)    — event times (sequence-relative, non-negative)
            locations   : (B, T, D) — spatial event locations
            input_mask  : (B, T)    float — 1 for valid, 0 for padding
        Returns:
            (B, T) — per-event log-prob; 0.0 at padding positions
        """
        B, T, D = locations.shape

        times_flat = event_times.reshape(B * T)
        locs_flat  = locations.reshape(B * T, D)

        if self.squash_time:
            # cnf: time-invariant — all events share [0, time_offset]
            t0 = torch.zeros_like(times_flat)
            t1 = torch.full_like(times_flat, self.time_offset)
        else:
            # tvcnf: time-varying — each event integrates [t_i + time_offset, 0]
            t0 = times_flat + self.time_offset
            t1 = torch.zeros_like(times_flat)

        self.cnf.nfe = 0
        z, delta_logp = self.cnf.integrate(
            t0, t1, locs_flat, torch.zeros_like(times_flat)
        )

        logpz = _gaussian_loglik(z, self.z_mean, self.z_logstd).sum(-1)  # (B*T,)
        logpx = logpz - delta_logp                                         # (B*T,)
        return logpx.reshape(B, T) * input_mask

    def log_spatial_density_at(
        self,
        t_query: Tensor,
        s_query: Tensor,
        history_times: Tensor = None,
        history_locs: Tensor = None,
        history_mask: Tensor = None,
    ) -> Tensor:
        """Log spatial density at arbitrary query points.

        The IndependentCNF conditions only on the event time, not on history
        (each event's distribution is independent given its time). History
        arguments are accepted for interface compatibility but are ignored.

        Args:
            t_query      : (M,) query times (used by the time-varying variant)
            s_query      : (M, D) query spatial locations
            history_times: ignored
            history_locs : ignored
            history_mask : ignored
        Returns:
            (M,) log p(s_query | t_query)
        """
        del history_times, history_locs, history_mask
        M = t_query.shape[0]
        ones_mask = torch.ones(M, 1, device=t_query.device, dtype=t_query.dtype)
        log_prob = self.logprob(
            t_query.unsqueeze(1),   # (M, 1)
            s_query.unsqueeze(1),   # (M, 1, D)
            ones_mask,              # (M, 1)
        )  # (M, 1)
        return log_prob.squeeze(1)  # (M,)

    def extra_repr(self) -> str:
        return f"squash_time={self.squash_time}, time_offset={self.time_offset}"
