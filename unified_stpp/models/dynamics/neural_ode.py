"""
Neural ODE Dynamics — dz/dt = f_θ(z, t, X_field) for NeuralSTPP.

Evolves the latent state continuously between events using an ODE solver.
Optionally conditions the dynamics on field covariates.

Optimizations:
- use_adjoint=True: uses odeint_adjoint for O(1) memory backprop.
- augmented=True + intensity_fn: jointly integrates [z(t), Λ(t)] in one ODE
  solve, caching Λ on self._cached_Lambda for the temporal decoder to reuse
  (avoids a second quadrature pass).
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, TYPE_CHECKING

from ..base import Dynamics

try:
    from torchdiffeq import odeint as _odeint_std
    from torchdiffeq import odeint_adjoint as _odeint_adj
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

class ODEFunc(nn.Module):
    """The right-hand side f(z, t) of the ODE dz/dt = f(z, t).

    Output is bounded via tanh, preventing exploding ODE trajectories.
    Matches the original NeuralSTPP design (SimpleHiddenStateODEFunc applies
    torch.tanh to the network output). The last linear layer is zero-initialized
    so that dz/dt ≈ 0 at init, following the original's zero_init=True flag.
    """

    def __init__(self, hidden_dim: int, field_cov_dim: int = 0):
        super().__init__()
        input_dim = hidden_dim + 1  # +1 for time
        if field_cov_dim > 0:
            input_dim += field_cov_dim
        # Zero-init the last linear so dz/dt ≈ 0 at init (matches original).
        last_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.Tanh(),
            last_linear,
        )
        self.field_cov_dim = field_cov_dim
        self._x_field_fn = None  # Set externally before each ODE solve

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """
        Args:
            t: scalar tensor — current time
            z: (B, h)
        Returns:
            dz/dt: (B, h) — bounded in (-1, 1) via tanh
        """
        t_expand = t.expand(z.shape[0], 1)  # (B, 1)
        inp = [z, t_expand]
        if self.field_cov_dim > 0 and self._x_field_fn is not None:
            x_f = self._x_field_fn(t)  # (B, r)
            inp.append(x_f)
        # tanh bounds velocity to (-1, 1) preventing ODE blow-up
        return torch.tanh(self.net(torch.cat(inp, dim=-1)))


class AugmentedODEFunc(nn.Module):
    """
    Augmented ODE right-hand side: d[z, Λ]/dt = [f(z,t), λ*(z,t)].

    Jointly integrates state dynamics and cumulative hazard in one ODE solve,
    eliminating the need for a separate quadrature pass in the temporal decoder.

    The last dimension of the state is the Λ accumulator (starts at 0).
    """

    def __init__(
        self,
        ode_func: ODEFunc,
        intensity_fn: Callable,
        intensity_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.ode_func = ode_func
        self.intensity_fn = intensity_fn  # (z, elapsed) → (B,)
        # Register as a submodule so .cpu()/.to() moves its parameters too.
        # This is critical for the MPS→CPU fallback in _run_odeint.
        if intensity_module is not None:
            self.intensity_module = intensity_module

    def forward(self, t: Tensor, z_aug: Tensor) -> Tensor:
        """
        Args:
            t: scalar tensor — current time
            z_aug: (B, h+1) — last dim is the Λ accumulator
        Returns:
            dz_aug/dt: (B, h+1)
        """
        z = z_aug[:, :-1]   # (B, h)
        dz = self.ode_func(t, z)  # (B, h)

        B = z.shape[0]
        # t is a 0-dim scalar from torchdiffeq; reshape to (B, 1) safely.
        elapsed = t.reshape(1, 1).expand(B, 1)  # (B, 1)
        lam = self.intensity_fn(z, elapsed)  # (B,)

        return torch.cat([dz, lam.unsqueeze(-1)], dim=-1)  # (B, h+1)


def euler_solve(func, z0, t_span, n_steps=50):
    """Simple Euler solver as fallback when torchdiffeq is unavailable."""
    dt = (t_span[-1] - t_span[0]) / n_steps
    z = z0
    t = t_span[0]
    trajectory = [z0]
    target_times = t_span[1:]  # skip initial
    target_idx = 0
    for step in range(n_steps):
        z = z + dt * func(t, z)
        t = t + dt
        # Check if we've passed a target time
        while target_idx < len(target_times) and t >= target_times[target_idx] - 1e-6:
            trajectory.append(z)
            target_idx += 1
    # Ensure we have all target times
    while len(trajectory) < len(t_span):
        trajectory.append(z)
    return torch.stack(trajectory, dim=0)  # (T, B, h)


class NeuralODEDynamics(Dynamics):
    """
    Continuous-time state dynamics via Neural ODE.

    Parameters
    ----------
    use_adjoint : bool
        If True (default), use odeint_adjoint for O(1)-memory backprop.
        Set False for debugging or when adjoint is not needed.
    augmented : bool
        If True and intensity_fn is provided, jointly integrates [z(t), Λ(t)]
        in a single ODE solve. After forward(), Λ is cached on
        self._cached_Lambda (shape: B×M) so the temporal decoder can skip
        its quadrature pass. Default: False.
    intensity_fn : callable or None
        Required when augmented=True. Signature: (z, elapsed) → (B,).
        Typically CumulativeHazardTemporal._intensity.
    n_steps : int
        For the 'euler' solver: number of fixed steps over the integration
        interval. 0 (default) means the solver chooses adaptively.
    """

    def __init__(
        self,
        hidden_dim: int,
        field_cov_dim: int = 0,
        solver: str = "dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        use_adjoint: bool = True,
        augmented: bool = False,
        intensity_fn: Optional[Callable] = None,
        n_steps: int = 0,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim)
        self.func = ODEFunc(hidden_dim, field_cov_dim)
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.use_adjoint = use_adjoint
        self.augmented = augmented
        self.intensity_fn = intensity_fn
        self.n_steps = n_steps
        self._cached_Lambda: Optional[Tensor] = None  # side channel for temporal decoder

        # Build augmented func once; can also be set after construction via
        # build_model() when the intensity_fn comes from the temporal decoder.
        if augmented and intensity_fn is not None:
            self.aug_func = AugmentedODEFunc(self.func, intensity_fn)
        else:
            self.aug_func = None

    def _run_odeint(self, func: nn.Module, y0: Tensor, t_span: Tensor) -> Tensor:
        """Run ODE integration, dispatching between adjoint and standard."""
        if not HAS_TORCHDIFFEQ:
            fallback_steps = self.n_steps if self.n_steps > 0 else 100
            return euler_solve(func, y0, t_span, n_steps=fallback_steps)

        # torchdiffeq's adaptive solvers default to dtype=float64 for all
        # time-like tensors (rtol, atol, step sizes).  MPS (Apple Silicon)
        # does not support float64, so we override via options to stay in
        # y0's dtype (float32 on MPS; preserves float64 on CPU/CUDA if needed).
        options: dict = {"dtype": y0.dtype}

        if self.solver == "euler" and self.n_steps > 0:
            dt_total = abs((t_span[-1] - t_span[0]).item())
            if dt_total > 1e-8:
                options["step_size"] = dt_total / self.n_steps

        kwargs: dict = {
            "method": self.solver,
            "atol": self.atol,
            "rtol": self.rtol,
            "options": options,
        }

        if self.use_adjoint:
            return _odeint_adj(func, y0, t_span, **kwargs)
        else:
            return _odeint_std(func, y0, t_span, **kwargs)

    def forward(
        self,
        z_n: Tensor,
        dt: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            z_n: (B, h)
            dt: (B, M) — query time offsets from t_n. In the sequential loop
                M=1; the first batch element's dt values define the time grid.
            x_field: (B, M, r) optional — field covariates at query times.
        Returns:
            z_t: (B, M, h)
        """
        B, M = dt.shape

        # Build time grid: [0, dt_sorted_0, ..., dt_sorted_{M-1}]
        dt_sorted, sort_idx = dt[0].sort()  # (M,) — use first batch element
        t_span = torch.cat(
            [torch.zeros(1, device=dt.device), dt_sorted.clamp(min=1e-6)]
        )

        # Wire up field covariate interpolation
        if x_field is not None:
            self.func._x_field_fn = lambda t_q: _interp_field(t_q, dt[0], x_field)
        else:
            self.func._x_field_fn = None

        # ------------------------------------------------------------------ #
        # Standard (non-augmented) solve
        # ------------------------------------------------------------------ #
        if not (self.augmented and self.aug_func is not None):
            self._cached_Lambda = None
            z_traj = self._run_odeint(self.func, z_n, t_span)  # (T, B, h)
            z_at_dt = z_traj[1:].permute(1, 0, 2)              # (B, M, h)
            _, unsort_idx = sort_idx.sort()
            return z_at_dt[:, unsort_idx, :]

        # ------------------------------------------------------------------ #
        # Augmented solve: [z(t), Λ(t)] jointly
        # ------------------------------------------------------------------ #
        h = z_n.shape[1]
        Lambda_0 = torch.zeros(B, 1, device=z_n.device)
        z_aug_0 = torch.cat([z_n, Lambda_0], dim=-1)  # (B, h+1)

        z_aug_traj = self._run_odeint(self.aug_func, z_aug_0, t_span)  # (T, B, h+1)

        z_at_dt_aug  = z_aug_traj[1:]                # (M, B, h+1)
        z_traj_part  = z_at_dt_aug[:, :, :h]         # (M, B, h)
        lam_traj_part = z_at_dt_aug[:, :, h]         # (M, B)

        z_at_dt    = z_traj_part.permute(1, 0, 2)    # (B, M, h)
        lambda_at_dt = lam_traj_part.permute(1, 0)   # (B, M)

        _, unsort_idx = sort_idx.sort()
        z_at_dt    = z_at_dt[:, unsort_idx, :]
        lambda_at_dt = lambda_at_dt[:, unsort_idx]

        self._cached_Lambda = lambda_at_dt            # side channel: (B, M)
        return z_at_dt


def _interp_field(t_query: Tensor, dt_grid: Tensor, x_field: Tensor) -> Tensor:
    """Nearest-neighbor interpolation of field covariates."""
    # t_query: scalar, dt_grid: (M,), x_field: (B, M, r)
    idx = (dt_grid - t_query).abs().argmin()
    return x_field[:, idx, :]  # (B, r)
