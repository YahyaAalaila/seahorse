"""Shared temporal backbone for the Neural STPP family.

This module implements the neural temporal ODE structure used by the shared
NJSDE, Neural JumpCNF, and Neural AttnCNF presets while keeping an explicit
state-side contract for downstream spatial decoders.

Key semantics:
  - raw event times drive the temporal ODE
  - standardized spatial locations are the optional jump-update conditioning
  - hidden-state partition semantics:
      intensity consumes the first ``hdim`` channels
      spatial models consume the last ``aux_dim`` channels
  - full-sequence per-event outputs (event 0 is kept; no drop-first shortcut)
"""

from __future__ import annotations

import math
from inspect import signature
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torchdiffeq import odeint as _odeint_std
    from torchdiffeq import odeint_adjoint as _odeint_adj

    HAS_TORCHDIFFEQ = True
except ImportError:  # pragma: no cover - fallback is covered instead
    HAS_TORCHDIFFEQ = False


def _normalize_hidden_dims(
    hidden_dims: Sequence[int] | int | str | None,
    *,
    fallback: int,
) -> list[int]:
    """Accept common ``tpp_hidden_dims`` shapes."""
    if hidden_dims is None:
        return [fallback, fallback]
    if isinstance(hidden_dims, int):
        dims = [hidden_dims]
    elif isinstance(hidden_dims, str):
        raw = hidden_dims.strip().strip("[]()")
        if not raw:
            raise ValueError("tpp_hidden_dims string cannot be empty")
        if "-" in raw:
            parts = [p.strip() for p in raw.split("-")]
        elif "," in raw:
            parts = [p.strip() for p in raw.split(",")]
        else:
            parts = [raw]
        dims = [int(p) for p in parts if p]
    else:
        dims = [int(d) for d in hidden_dims]
    if not dims:
        raise ValueError("tpp_hidden_dims must contain at least one width")
    if any(d <= 0 for d in dims):
        raise ValueError(f"tpp_hidden_dims must be positive; got {dims}")
    return dims


def _euler_tuple_solve(func, y0, t_span, n_steps: int = 50):
    """Lightweight tuple-valued Euler fallback when torchdiffeq is unavailable."""
    dt = (t_span[-1] - t_span[0]) / max(n_steps, 1)
    state = tuple(y.clone() for y in y0)
    traj = [[y.clone()] for y in state]
    t = t_span[0]
    for _ in range(n_steps):
        deriv = func(t, state)
        state = tuple(y + dt * dy for y, dy in zip(state, deriv))
        t = t + dt
    for bucket, y in zip(traj, state):
        bucket.append(y)
    return tuple(torch.stack(bucket, dim=0) for bucket in traj)


def _repair_strictly_increasing_endpoint(t0: Tensor, t1: Tensor) -> Tensor:
    if hasattr(torch, "nextafter"):
        next_up = torch.nextafter(t0, torch.full_like(t0, float("inf")))
    else:  # pragma: no cover - torch.nextafter exists in supported runtimes
        scale = torch.maximum(torch.maximum(t0.abs(), t1.abs()), torch.ones_like(t0))
        next_up = t0 + torch.finfo(t0.dtype).eps * scale
    return torch.where(t1 > t0, t1, next_up)


class DiffEqWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        n_params = len(signature(self.module.forward).parameters)
        if n_params == 1:
            return self.module(y)
        if n_params == 2:
            return self.module(t, y)
        raise ValueError("Differential equation layers must take (y,) or (t, y).")


class SequentialDiffEq(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([DiffEqWrapper(layer) for layer in layers])

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(t, x)
        return x


class IgnoreLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        del t
        return self._layer(x)


class ConcatLinearV2(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        return self._layer(x) + self._hyper_bias(t.reshape(-1, 1))


class Swish(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5] * dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x * F.softplus(self.beta))


class GatedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer_f = nn.Linear(in_features, out_features)
        self.layer_g = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer_f(x) * torch.sigmoid(self.layer_g(x))


class ActNorm(nn.Module):
    def __init__(self, num_features: int, init_scale: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.init_scale = init_scale
        self.weight = nn.Parameter(torch.zeros(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("initialized", torch.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        if not bool(self.initialized.item()):
            with torch.no_grad():
                x_ = x.reshape(-1, x.shape[-1])
                batch_mean = torch.mean(x_, dim=0)
                if x_.shape[0] <= 1:
                    # torch.var(...) is undefined for a singleton init batch
                    # under PyTorch's default correction and yields NaNs.
                    # Guard only that singleton initialization edge case.
                    batch_var = torch.full_like(batch_mean, 0.2)
                else:
                    batch_var = torch.var(x_, dim=0)
                    batch_var = torch.max(batch_var, torch.tensor(0.2, device=x.device, dtype=x.dtype))
                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var) + math.log(self.init_scale))
                self.initialized.fill_(1)
        bias = self.bias.expand_as(x)
        weight = self.weight.expand_as(x)
        return (x + bias) * F.softplus(weight)


ACTFNS = {
    "softplus": lambda dim: nn.Softplus(),
    "swish": lambda dim: Swish(dim),
    "celu": lambda dim: nn.CELU(),
    "relu": lambda dim: nn.ReLU(inplace=True),
}


def construct_diffeqnet(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    time_dependent: bool = False,
    actfn: str = "softplus",
    zero_init: bool = False,
    gated: bool = False,
) -> SequentialDiffEq:
    linear_fn = IgnoreLinear if time_dependent else ConcatLinearV2
    if gated:
        linear_fn = GatedLinear

    layers: list[nn.Module] = []
    hidden_dims = list(hidden_dims)
    if hidden_dims:
        dims = [input_dim] + hidden_dims
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(linear_fn(d_in, d_out))
            layers.append(ActNorm(d_out))
            if not gated:
                layers.append(ACTFNS[actfn](d_out))
        layers.append(linear_fn(hidden_dims[-1], output_dim))
    else:
        layers.append(linear_fn(input_dim, output_dim))

    if zero_init:
        for m in layers[-1].modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    return SequentialDiffEq(*layers)


class IntensityODEFunc(nn.Module):
    def __init__(self, hdim: int, dstate_fn: nn.Module, intensity_fn: nn.Module):
        super().__init__()
        self.hdim = hdim
        self.dstate_fn = dstate_fn
        self.intensity_fn = intensity_fn

    def forward(self, t: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        Lambda, tpp_state = state
        intensity = self.get_intensity(tpp_state).reshape(-1)
        return intensity, self.dstate_fn(t, tpp_state)

    def get_intensity(self, tpp_state: Tensor) -> Tensor:
        return torch.sigmoid(self.intensity_fn(tpp_state[..., : self.hdim]) - 2.0) * 50.0


class SplitHiddenStateODEFunc(nn.Module):
    def __init__(self, dstate_net: nn.Module, update_net: nn.Module):
        super().__init__()
        self.dstate_net = dstate_net
        self.update_net = update_net

    def forward(self, t: Tensor, tpp_state: Tensor) -> Tensor:
        dstate = self.dstate_net(t, tpp_state)
        c, h = torch.split(tpp_state, tpp_state.shape[1] // 2, dim=1)
        dcdt, dhdt = torch.split(dstate, tpp_state.shape[1] // 2, dim=1)
        dcdt = dcdt - (dcdt * c).sum(dim=-1, keepdim=True) / (c * c).sum(dim=-1, keepdim=True) * c
        dhdt = -F.softplus(dhdt) * h
        return torch.cat([dcdt, dhdt], dim=1)

    def update_state(self, t: Tensor, tpp_state: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        del t
        if cond is not None:
            inputs = torch.cat([tpp_state, cond], dim=1)
        else:
            inputs = tpp_state
        upd_c, upd_h = torch.split(self.update_net(inputs), tpp_state.shape[1] // 2, dim=1)
        update = torch.cat([torch.zeros_like(upd_c), upd_h], dim=1)
        return tpp_state + update


class SimpleHiddenStateODEFunc(nn.Module):
    def __init__(self, dstate_net: nn.Module, update_net: nn.Module):
        super().__init__()
        self.dstate_net = dstate_net
        self.update_net = update_net

    def forward(self, t: Tensor, tpp_state: Tensor) -> Tensor:
        return torch.tanh(self.dstate_net(t, tpp_state))

    def update_state(self, t: Tensor, tpp_state: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        del t
        if cond is not None:
            inputs = torch.cat([tpp_state, cond], dim=1)
        else:
            inputs = tpp_state
        return self.update_net(inputs)


class GRUHiddenStateODEFunc(nn.Module):
    def __init__(self, dstate_net: nn.Module, update_net: nn.GRUCell):
        super().__init__()
        self.dstate_net = dstate_net
        self.update_net = update_net

    def forward(self, t: Tensor, tpp_state: Tensor) -> Tensor:
        return self.dstate_net(t, tpp_state)

    def update_state(self, t: Tensor, tpp_state: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        del t
        if cond is None:
            cond = torch.zeros(tpp_state.shape[0], 0, device=tpp_state.device, dtype=tpp_state.dtype)
        return self.update_net(cond, tpp_state)


class HiddenStateODEFuncList(nn.Module):
    def __init__(self, *odefuncs: nn.Module):
        super().__init__()
        self.odefuncs = nn.ModuleList(odefuncs)

    def forward(self, t: Tensor, tpp_state: Tensor) -> Tensor:
        states = torch.split(tpp_state, tpp_state.shape[-1] // len(self.odefuncs), dim=-1)
        ds = [func(t, s) for s, func in zip(states, self.odefuncs)]
        return torch.cat(ds, dim=-1)

    def update_state(self, t: Tensor, tpp_state: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        states = torch.split(tpp_state, tpp_state.shape[-1] // len(self.odefuncs), dim=-1)
        updates = [func.update_state(t, s, cond) for s, func in zip(states, self.odefuncs)]
        return torch.cat(updates, dim=-1)


class TimeVariableODE(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(
        self,
        func: nn.Module,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        method: str = "dopri5",
        energy_regularization: float = 0.0,
        use_adjoint: bool = False,
    ):
        super().__init__()
        self.func = func
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.energy_regularization = energy_regularization
        self.use_adjoint = use_adjoint
        self.nfe = 0

    def integrate(
        self,
        t0: Tensor,
        t1: Tensor,
        x0: tuple[Tensor, ...],
        *,
        nlinspace: int = 1,
        method: Optional[str] = None,
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        assert nlinspace > 0
        method = method or self.method
        t0 = t0.reshape(-1)
        t1 = t1.reshape(-1)
        raw_dt = t1 - t0
        dt_abs = raw_dt.abs()
        dt_scale = torch.maximum(torch.maximum(t0.abs(), t1.abs()), torch.ones_like(t0))
        tiny_threshold = torch.finfo(t0.dtype).eps * dt_scale
        non_increasing = raw_dt <= 0
        t1 = _repair_strictly_increasing_endpoint(t0, t1)
        if not bool(torch.all(t1 > t0).item()):
            raise RuntimeError(
                "Neural STPP temporal ODE interval repair failed; "
                f"non_increasing_count={int(non_increasing.sum().item())}, dtype={t0.dtype}"
            )
        energy0 = torch.zeros(1, device=x0[0].device, dtype=x0[0].dtype)
        init_state = (t0, t1, energy0, *x0)
        eval_grid = torch.linspace(
            self.start_time,
            self.end_time,
            nlinspace + 1,
            device=t0.device,
            dtype=t0.dtype,
        )

        if HAS_TORCHDIFFEQ:
            odeint_fn = _odeint_adj if self.use_adjoint else _odeint_std
            # Keep adaptive time bookkeeping in float64, matching torchdiffeq defaults.
            solver_options = {"dtype": torch.float64}
            solver_kwargs = {
                "rtol": self.rtol,
                "atol": self.atol,
                "method": method,
                "options": solver_options,
            }
            if self.use_adjoint:
                solver_kwargs["adjoint_options"] = solver_options.copy()
            try:
                solution = odeint_fn(
                    self,
                    init_state,
                    eval_grid,
                    **solver_kwargs,
                )
            except AssertionError as exc:
                if "underflow in dt" not in str(exc):
                    raise
                min_raw_dt = float(raw_dt.min().item()) if raw_dt.numel() else 0.0
                max_raw_dt = float(raw_dt.max().item()) if raw_dt.numel() else 0.0
                mean_raw_dt = float(raw_dt.mean().item()) if raw_dt.numel() else 0.0
                tiny_count = int((dt_abs <= tiny_threshold).sum().item()) if dt_abs.numel() else 0
                raise RuntimeError(
                    "Neural STPP temporal ODE underflow in dt; "
                    f"min_raw_dt={min_raw_dt:.6e}, max_raw_dt={max_raw_dt:.6e}, mean_raw_dt={mean_raw_dt:.6e}, "
                    f"non_increasing_count={int(non_increasing.sum().item())}, tiny_interval_count={tiny_count}, "
                    f"use_adjoint={self.use_adjoint}, training={self.training}, method={method}, "
                    f"rtol={self.rtol:.1e}, atol={self.atol:.1e}, nfe={self.nfe}"
                ) from exc
        else:  # pragma: no cover - exercised only without torchdiffeq
            solution = _euler_tuple_solve(self, init_state, eval_grid, n_steps=max(50, nlinspace * 10))

        _, _, energy, *xs = solution
        reg = energy[-1] * self.energy_regularization
        return tuple(xs), reg.reshape(() if reg.numel() == 1 else reg.shape)

    def forward(self, s: Tensor, state):
        self.nfe += 1
        t0, t1, _, *x = state

        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0

        with torch.enable_grad():
            x = tuple(x_.requires_grad_(True) for x_ in x)
            dx = self.func(t, x)
            dx = tuple(dx_ * ratio.reshape(-1, *([1] * (dx_.ndim - 1))) for dx_ in dx)
            d_energy = sum(torch.sum(dx_ * dx_) for dx_ in dx) / sum(x_.numel() for x_ in x)

        if not self.training:
            dx = tuple(dx_.detach() for dx_ in dx)

        return (torch.zeros_like(t0), torch.zeros_like(t1), d_energy, *dx)


class NeuralPointProcess(nn.Module):
    """Shared neural temporal backbone for the Neural STPP family."""

    dynamics_dict = {
        "split": SplitHiddenStateODEFunc,
        "simple": SimpleHiddenStateODEFunc,
        "gru": GRUHiddenStateODEFunc,
    }

    def __init__(
        self,
        *,
        cond_dim: int = 0,
        hidden_dims: Sequence[int] | int | str | None = None,
        cond: bool = True,
        style: str = "gru",
        actfn: str = "softplus",
        hdim: Optional[int] = None,
        separate: int = 1,
        tol: float = 1e-6,
        otreg_strength: float = 0.0,
        method: str = "dopri5",
        use_adjoint: bool = False,
    ):
        super().__init__()
        hidden_dims = _normalize_hidden_dims(hidden_dims, fallback=32)
        if not cond:
            cond_dim = 0
        self.cond = cond
        self.cond_dim = cond_dim
        self.hidden_dims = list(hidden_dims)
        self.hidden_dim = int(hidden_dims[0])
        self.hdim = self.hidden_dim if hdim is None else int(hdim)
        if self.hdim <= 0 or self.hdim > self.hidden_dim:
            raise ValueError(
                f"hdim must be in [1, hidden_dim]; got hdim={self.hdim}, hidden_dim={self.hidden_dim}"
            )
        self.aux_dim = max(0, self.hidden_dim - self.hdim)
        self._init_state = nn.Parameter(torch.randn(self.hidden_dim) / math.sqrt(self.hidden_dim))

        dynamics = []
        if self.hidden_dim % max(separate, 1) != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by separate ({separate})"
            )
        for _ in range(separate):
            seg_dim = self.hidden_dim // separate
            dstate_net = construct_diffeqnet(
                seg_dim,
                self.hidden_dims[1:],
                seg_dim,
                time_dependent=False,
                actfn=actfn,
                zero_init=True,
            )
            if style in {"split", "simple"}:
                update_net = construct_diffeqnet(
                    seg_dim + cond_dim,
                    self.hidden_dims[1:],
                    seg_dim,
                    time_dependent=False,
                    actfn="celu",
                    gated=True,
                    zero_init=False,
                )
            elif style == "gru":
                update_net = nn.GRUCell(cond_dim, seg_dim)
            else:
                raise ValueError(f"Unknown Neural STPP style {style!r}")
            dynamics.append(self.dynamics_dict[style](dstate_net, update_net))

        self.hidden_state_dynamics = HiddenStateODEFuncList(*dynamics)

        intensity_net = nn.Sequential(
            nn.Linear(self.hdim, self.hdim * 4),
            nn.Softplus(),
            nn.Linear(self.hdim * 4, 1),
        )
        self.ode_solver = TimeVariableODE(
            IntensityODEFunc(self.hdim, self.hidden_state_dynamics, intensity_net),
            atol=tol,
            rtol=tol,
            method=method,
            energy_regularization=otreg_strength,
            use_adjoint=use_adjoint,
        )

    def get_intensity(self, state: Tensor) -> Tensor:
        return self.ode_solver.func.get_intensity(state)

    def integrate_hidden(
        self,
        h_prev: Tensor,
        t_prev: Tensor,
        t_curr: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        zeros = torch.zeros(h_prev.shape[0], device=h_prev.device, dtype=h_prev.dtype)
        state_traj, reg = self.ode_solver.integrate(
            t_prev.reshape(-1),
            t_curr.reshape(-1),
            (zeros, h_prev),
            nlinspace=1,
            method="dopri5" if self.training else "dopri5",
        )
        Lambda_traj, hidden_traj = state_traj
        return hidden_traj[-1], Lambda_traj[-1], reg

    def integrate_lambda(
        self,
        event_times: Tensor,
        spatial_location: Optional[Tensor],
        input_mask: Optional[Tensor],
        t0: Tensor | float,
        t1: Optional[Tensor | float],
        *,
        nlinspace: int = 1,
        return_details: bool = False,
    ):
        N, T = event_times.shape
        if not self.cond:
            spatial_location = None

        if input_mask is None:
            input_mask = torch.ones_like(event_times)
        input_mask = input_mask.bool()

        init_state = self._init_state[None].expand(N, -1)
        state = (
            torch.zeros(N, device=init_state.device, dtype=init_state.dtype),
            init_state,
        )
        t0_tensor = t0 if torch.is_tensor(t0) else torch.tensor(t0, device=event_times.device, dtype=event_times.dtype)
        t0_tensor = t0_tensor.expand(N).to(event_times)

        intensities: list[Tensor] = []
        cumulative_lambdas: list[Tensor] = []
        prejump_hidden_states: list[Tensor] = []
        regularization_terms: list[Tensor] = []
        final_postjump_state = init_state

        for i in range(T):
            active = input_mask[:, i]
            if bool(active.any().item()):
                raw_dt_i = event_times[:, i][active] - t0_tensor[active]
                if bool((raw_dt_i <= 0).any().item()):
                    raise RuntimeError(
                        "Neural STPP temporal integrate_lambda received non-increasing active event times; "
                        f"event_index={i}, bad_count={int((raw_dt_i <= 0).sum().item())}, "
                        f"min_raw_dt={float(raw_dt_i.min().item()):.6e}, "
                        f"prev_t_min={float(t0_tensor[active].min().item()):.6e}, "
                        f"prev_t_max={float(t0_tensor[active].max().item()):.6e}, "
                        f"curr_t_min={float(event_times[:, i][active].min().item()):.6e}, "
                        f"curr_t_max={float(event_times[:, i][active].max().item()):.6e}"
                    )
            t1_i = torch.where(input_mask[:, i], event_times[:, i], t0_tensor)
            state_traj, reg_i = self.ode_solver.integrate(
                t0_tensor,
                t1_i,
                state,
                nlinspace=nlinspace,
                method="dopri5" if self.training else "dopri5",
            )
            Lambda_traj, hidden_traj = state_traj
            hiddens = hidden_traj
            if i > 0:
                hiddens = hiddens[1:]
            hiddens = torch.where(
                input_mask[:, i].reshape(1, -1, 1).expand_as(hiddens),
                hiddens,
                torch.zeros_like(hiddens),
            )
            prejump_hidden_states.append(hiddens)
            regularization_terms.append(torch.as_tensor(reg_i, device=event_times.device, dtype=event_times.dtype))

            state = tuple(s[-1] for s in state_traj)
            Lambda, tpp_state = state
            intensity = self.get_intensity(tpp_state).reshape(-1)
            intensity = torch.where(input_mask[:, i], intensity, torch.ones_like(intensity))
            intensities.append(intensity)
            cumulative_lambdas.append(torch.where(input_mask[:, i], Lambda, torch.zeros_like(Lambda)))

            cond = spatial_location[:, i] if spatial_location is not None else None
            updated_tpp_state = self.hidden_state_dynamics.update_state(
                event_times[:, i],
                tpp_state,
                cond=cond,
            )
            postjump_tpp_state = torch.where(
                input_mask[:, i].reshape(-1, 1).expand_as(tpp_state),
                updated_tpp_state,
                tpp_state,
            )
            final_postjump_state = postjump_tpp_state
            if i < T - 1 or t1 is not None:
                state = (Lambda, postjump_tpp_state)

            t0_tensor = torch.where(input_mask[:, i], event_times[:, i], t0_tensor)

        if t1 is not None:
            t1_tensor = t1 if torch.is_tensor(t1) else torch.tensor(t1, device=event_times.device, dtype=event_times.dtype)
            t1_tensor = t1_tensor.expand(N).to(event_times)
            raw_dt_tail = t1_tensor - t0_tensor
            if bool((raw_dt_tail <= 0).any().item()):
                raise RuntimeError(
                    "Neural STPP temporal integrate_lambda received a non-increasing tail interval; "
                    f"bad_count={int((raw_dt_tail <= 0).sum().item())}, "
                    f"min_raw_dt={float(raw_dt_tail.min().item()):.6e}, "
                    f"t_last_min={float(t0_tensor.min().item()):.6e}, "
                    f"t_last_max={float(t0_tensor.max().item()):.6e}, "
                    f"t1_min={float(t1_tensor.min().item()):.6e}, "
                    f"t1_max={float(t1_tensor.max().item()):.6e}"
                )
            state_traj, reg_tail = self.ode_solver.integrate(
                t0_tensor,
                t1_tensor,
                state,
                nlinspace=nlinspace,
                method="dopri5" if self.training else "dopri5",
            )
            _, hidden_traj = state_traj
            prejump_hidden_states.append(hidden_traj[1:])
            regularization_terms.append(torch.as_tensor(reg_tail, device=event_times.device, dtype=event_times.dtype))
            state = tuple(s[-1] for s in state_traj)

        intensities_t = (
            torch.stack(intensities, dim=1)
            if intensities
            else torch.zeros(N, 0, device=event_times.device, dtype=event_times.dtype)
        )
        cumulative_t = (
            torch.stack(cumulative_lambdas, dim=1)
            if cumulative_lambdas
            else torch.zeros(N, 0, device=event_times.device, dtype=event_times.dtype)
        )
        if prejump_hidden_states:
            prejump_t = torch.cat(prejump_hidden_states, dim=0).transpose(0, 1)
        else:
            prejump_t = torch.zeros(
                N,
                0,
                self.hidden_dim,
                device=event_times.device,
                dtype=event_times.dtype,
            )
        reg_total = (
            torch.stack([r.reshape(()) for r in regularization_terms]).sum()
            if regularization_terms
            else torch.tensor(0.0, device=event_times.device, dtype=event_times.dtype)
        )
        Lambda_final, final_prejump_state = state

        if return_details:
            details = {
                "cumulative_lambdas": cumulative_t,
                "regularization": reg_total,
                "final_prejump_state": final_prejump_state,
                "final_postjump_state": final_postjump_state,
                "t_last_event": t0_tensor,
            }
            return intensities_t, Lambda_final, prejump_t, details
        return intensities_t, Lambda_final, prejump_t

    def sequence_nll_and_states(
        self,
        event_times: Tensor,
        spatial_location: Tensor,
        input_mask: Optional[Tensor],
        *,
        t0: Optional[Tensor | float] = None,
        t1: Optional[Tensor | float] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if input_mask is None:
            input_mask = torch.ones_like(event_times)
        t0 = 0.0 if t0 is None else t0
        intensities, _, prejump_hidden_states, details = self.integrate_lambda(
            event_times,
            spatial_location,
            input_mask,
            t0=t0,
            t1=t1,
            nlinspace=1,
            return_details=True,
        )
        cumulative = details["cumulative_lambdas"]
        prev_cumulative = torch.cat(
            [torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]],
            dim=1,
        )
        mask_bool = input_mask.bool()
        increments = cumulative - prev_cumulative
        temporal_nll = -torch.log(intensities + 1e-8) + increments
        temporal_nll = torch.where(mask_bool, temporal_nll, torch.zeros_like(temporal_nll))
        # Upstream integrate_lambda exposes a trace that includes the initial
        # state at t=t0 and, when requested, an extra tail state at t=t1.
        # Spatial variants consume event-aligned pre-jump states only.
        event_aligned_prehidden = prejump_hidden_states[:, 1 : event_times.shape[1] + 1, :]
        return (
            temporal_nll,
            event_aligned_prehidden,
            details["regularization"],
            details["final_postjump_state"],
        )


__all__ = ["NeuralPointProcess", "_normalize_hidden_dims"]
