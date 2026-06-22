"""Faithful JumpCNF spatial decoder for the shared Neural STPP family."""

from __future__ import annotations

import math
from inspect import signature
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import Decoder
from ..model_registry import register_spatial

try:
    from torchdiffeq import odeint as _odeint_std
    from torchdiffeq import odeint_adjoint as _odeint_adj

    HAS_TORCHDIFFEQ = True
except ImportError:  # pragma: no cover
    HAS_TORCHDIFFEQ = False


def _normalize_hidden_dims(hidden_dims: Sequence[int] | int | str | None) -> list[int]:
    if hidden_dims is None:
        return [64, 64, 64]
    if isinstance(hidden_dims, int):
        return [hidden_dims]
    if isinstance(hidden_dims, str):
        raw = hidden_dims.strip().strip("[]()")
        if not raw:
            raise ValueError("JumpCNF hidden_dims string cannot be empty")
        if "-" in raw:
            parts = [p.strip() for p in raw.split("-")]
        elif "," in raw:
            parts = [p.strip() for p in raw.split(",")]
        else:
            parts = [raw]
        dims = [int(p) for p in parts if p]
    else:
        dims = [int(h) for h in hidden_dims]
    return dims


class _DiffEqWrapper(nn.Module):
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


class _SequentialDiffEq(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([_DiffEqWrapper(layer) for layer in layers])

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(t, x)
        return x


class _ConcatLinearV2(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        return self._layer(x) + self._hyper_bias(t.reshape(-1, 1))


class _ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        tt = t.reshape(-1, 1)
        return self._layer(x) * torch.sigmoid(self._hyper_gate(tt)) + self._hyper_bias(tt)


class _TimeDependentSwish(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.beta = nn.Sequential(
            nn.Linear(1, min(64, dim * 4)),
            nn.Softplus(),
            nn.Linear(min(64, dim * 4), dim),
            nn.Softplus(),
        )

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        beta = self.beta(t.reshape(-1, 1))
        return x * torch.sigmoid(x * beta)


_ACTFNS = {
    "softplus": lambda dim: nn.Softplus(),
    "swish": lambda dim: _TimeDependentSwish(dim),
}
_LAYERTYPES = {
    "concat": _ConcatLinearV2,
    "concatsquash": _ConcatSquashLinear,
}


def _build_fc_odefunc(
    dim: int,
    hidden_dims: Sequence[int],
    *,
    out_dim: Optional[int] = None,
    nonzero_dim: Optional[int] = None,
    actfn: str = "softplus",
    layer_type: str = "concat",
    zero_init: bool = True,
) -> _SequentialDiffEq:
    if layer_type not in _LAYERTYPES:
        raise ValueError(f"Unknown JumpCNF layer_type {layer_type!r}")
    if actfn not in _ACTFNS:
        raise ValueError(f"Unknown JumpCNF actfn {actfn!r}")
    nonzero_dim = dim if nonzero_dim is None else nonzero_dim
    out_dim = dim if out_dim is None else out_dim
    layer_fn = _LAYERTYPES[layer_type]

    hidden_dims = list(hidden_dims)
    layers: list[nn.Module] = []
    if hidden_dims:
        dims = [dim] + hidden_dims
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(layer_fn(d_in, d_out))
            layers.append(_ACTFNS[actfn](d_out))
        layers.append(layer_fn(hidden_dims[-1], out_dim))
    else:
        layers.append(layer_fn(dim, out_dim))

    first = layers[0]
    linear = getattr(first, "_layer", None)
    if isinstance(linear, nn.Linear) and nonzero_dim < dim:
        linear.weight.data[:, nonzero_dim:].fill_(0.0)

    if zero_init:
        for m in layers[-1].modules():
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0.0)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    return _SequentialDiffEq(*layers)


def _divergence_bf(f: Tensor, y: Tensor, training: bool) -> Tensor:
    sum_diag = 0.0
    for i in range(f.shape[1]):
        retain = training or i < (f.shape[1] - 1)
        grad_i = torch.autograd.grad(
            f[:, i].sum(),
            y,
            create_graph=training,
            retain_graph=retain,
        )[0]
        sum_diag = sum_diag + grad_i[:, i]
    return sum_diag


def _euler_cnf_solve(func, y0, t_span, n_steps: int = 50):
    dt = (t_span[-1] - t_span[0]) / max(n_steps, 1)
    state = tuple(y.clone() if isinstance(y, Tensor) else y for y in y0)
    traj = [[y.clone() if isinstance(y, Tensor) else y] for y in state]
    t = t_span[0]
    for _ in range(n_steps):
        deriv = func(t, state)
        state = tuple(y + dt * dy if isinstance(y, Tensor) else y for y, dy in zip(state, deriv))
        t = t + dt
    for bucket, y in zip(traj, state):
        bucket.append(y.clone() if isinstance(y, Tensor) else y)
    return tuple(torch.stack(bucket, dim=0) for bucket in traj)


def _repair_strictly_decreasing_endpoint(t0: Tensor, t1: Tensor) -> Tensor:
    if hasattr(torch, "nextafter"):
        next_down = torch.nextafter(t0, torch.full_like(t0, float("-inf")))
    else:  # pragma: no cover - torch.nextafter exists in supported runtimes
        scale = torch.maximum(torch.maximum(t0.abs(), t1.abs()), torch.ones_like(t0))
        next_down = t0 - torch.finfo(t0.dtype).eps * scale
    return torch.where(t1 < t0, t1, next_down)


class TimeVariableCNF(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(
        self,
        func: nn.Module,
        dim: int,
        *,
        tol: float = 1e-4,
        method: str = "dopri5",
        energy_regularization: float = 0.0,
        jacnorm_regularization: float = 0.0,
        use_adjoint: bool = True,
    ):
        super().__init__()
        self.func = func
        self.dim = dim
        self.tol = tol
        self.method = method
        self.energy_regularization = energy_regularization
        self.jacnorm_regularization = jacnorm_regularization
        self.use_adjoint = use_adjoint
        self.nfe = 0

    def integrate(
        self,
        t0: Tensor,
        t1: Tensor,
        x: Tensor,
        logpx: Tensor,
        *,
        tol: Optional[float] = None,
        method: Optional[str] = None,
        norm=None,
        intermediate_states: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del norm
        self.nfe = 0
        tol = self.tol if tol is None else tol
        method = self.method if method is None else method
        t0 = t0.reshape(-1)
        t1 = t1.reshape(-1)
        raw_dt = t1 - t0
        dt_abs = (t1 - t0).abs()
        dt_scale = torch.maximum(torch.maximum(t0.abs(), t1.abs()), torch.ones_like(t0))
        tiny_threshold = torch.finfo(t0.dtype).eps * dt_scale
        non_decreasing = raw_dt >= 0
        t1 = _repair_strictly_decreasing_endpoint(t0, t1)
        if not bool(torch.all(t1 < t0).item()):
            raise RuntimeError(
                "JumpCNF spatial CNF interval repair failed; "
                f"non_decreasing_count={int(non_decreasing.sum().item())}, dtype={t0.dtype}"
            )
        e = torch.randn_like(x)[:, : self.dim]
        energy = torch.zeros(1, device=x.device, dtype=x.dtype)
        jacnorm = torch.zeros(1, device=x.device, dtype=x.dtype)
        initial_state = (t0, t1, e, x, logpx, energy, jacnorm)

        if intermediate_states > 1:
            tt = torch.linspace(self.start_time, self.end_time, intermediate_states, device=t0.device, dtype=t0.dtype)
        else:
            tt = torch.tensor([self.start_time, self.end_time], device=t0.device, dtype=t0.dtype)

        if HAS_TORCHDIFFEQ:
            odeint_fn = _odeint_adj if self.use_adjoint else _odeint_std
            # Keep adaptive time bookkeeping in float64, matching torchdiffeq defaults.
            solver_options = {"dtype": torch.float64}
            solver_kwargs = {
                "rtol": tol,
                "atol": tol,
                "method": method,
                "options": solver_options,
            }
            if self.use_adjoint:
                solver_kwargs["adjoint_options"] = solver_options.copy()
            try:
                solution = odeint_fn(
                    self,
                    initial_state,
                    tt,
                    **solver_kwargs,
                )
            except AssertionError as exc:
                if "underflow in dt" not in str(exc):
                    raise
                min_dt = float(dt_abs.min().item()) if dt_abs.numel() else 0.0
                max_dt = float(dt_abs.max().item()) if dt_abs.numel() else 0.0
                mean_dt = float(dt_abs.mean().item()) if dt_abs.numel() else 0.0
                tiny_count = int((dt_abs <= tiny_threshold).sum().item()) if dt_abs.numel() else 0
                solve_reverse = getattr(self, "debug_solve_reverse", None)
                raise RuntimeError(
                    "JumpCNF spatial CNF underflow in dt; "
                    f"min_abs_dt={min_dt:.6e}, max_abs_dt={max_dt:.6e}, mean_abs_dt={mean_dt:.6e}, "
                    f"non_decreasing_count={int(non_decreasing.sum().item())}, tiny_interval_count={tiny_count}, "
                    f"use_adjoint={self.use_adjoint}, solve_reverse={solve_reverse}, "
                    f"training={self.training}, method={method}, tol={tol:.1e}, nfe={self.nfe}"
                ) from exc
        else:  # pragma: no cover
            solution = _euler_cnf_solve(self, initial_state, tt, n_steps=max(50, intermediate_states * 10 or 50))

        if intermediate_states > 1:
            y = solution[3]
            _, _, _, _, logpy, energy, jacnorm = tuple(s[-1] for s in solution)
        else:
            _, _, _, y, logpy, energy, jacnorm = tuple(s[-1] for s in solution)

        regularization = (
            self.energy_regularization * (energy - energy.detach())
            + self.jacnorm_regularization * (jacnorm - jacnorm.detach())
        )
        return y, logpy, regularization.reshape(() if regularization.numel() == 1 else regularization.shape)

    def forward(self, s: Tensor, state):
        self.nfe += 1
        t0, t1, e, x, logpx, _, _ = state
        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0

        vjp = None
        with torch.enable_grad():
            x = x.requires_grad_(True)
            dx = self.func(t, x)
            dx = dx * ratio.reshape(-1, *([1] * (x.ndim - 1)))

            if not self.training:
                div = _divergence_bf(dx[:, : self.dim], x, self.training)
            else:
                vjp = torch.autograd.grad(
                    dx[:, : self.dim],
                    x,
                    e,
                    create_graph=self.training,
                    retain_graph=self.training,
                )[0]
                vjp = vjp[:, : self.dim]
                div = torch.sum(vjp * e, dim=1)

        if not self.training:
            dx = dx.detach()
            div = div.detach()

        d_energy = torch.sum(dx * dx).reshape(1) / x.shape[0]
        if self.training and vjp is not None:
            d_jacnorm = torch.sum(vjp * vjp).reshape(1) / x.shape[0]
        else:
            d_jacnorm = torch.zeros(1, device=x.device, dtype=x.dtype)
        return (
            torch.zeros_like(t0),
            torch.zeros_like(t1),
            torch.zeros_like(e),
            dx,
            -div,
            d_energy,
            d_jacnorm,
        )


def _gaussian_loglik(z: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    c = torch.tensor(math.log(2.0 * math.pi), device=z.device, dtype=z.dtype)
    inv_sigma = torch.exp(-log_std)
    delta = (z - mean) * inv_sigma
    return -0.5 * (delta * delta + 2.0 * log_std + c)


class _RadialFlow(nn.Module):
    def __init__(self, nd: int, *, hypernet: bool = False):
        super().__init__()
        self.nd = nd
        self.hypernet = hypernet
        if not hypernet:
            self.z0 = nn.Parameter(torch.zeros(1, nd))
            self.log_alpha = nn.Parameter(torch.zeros(1, 1))
            self._beta = nn.Parameter(torch.zeros(1, 1))

    def forward(
        self,
        x: Tensor,
        logpx: Optional[Tensor] = None,
        *,
        reverse: bool = False,
        z0: Optional[Tensor] = None,
        log_alpha: Optional[Tensor] = None,
        beta: Optional[Tensor] = None,
    ):
        del reverse
        if self.hypernet:
            if z0 is None or log_alpha is None or beta is None:
                raise ValueError("Hypernetwork radial flow requires z0/log_alpha/beta.")
            beta = -torch.exp(log_alpha) + F.softplus(beta)
        else:
            z0 = self.z0
            log_alpha = self.log_alpha
            beta = -torch.exp(log_alpha) + F.softplus(self._beta)

        z0 = z0.expand_as(x)
        r = torch.norm(x - z0, dim=-1, keepdim=True)
        h = 1.0 / (torch.exp(log_alpha) + r)
        y = x + beta * h * (x - z0)

        if logpx is None:
            return y
        logdetgrad = (self.nd - 1) * torch.log(1.0 + beta * h) + torch.log(
            1.0 + beta * h - beta * r / (torch.exp(log_alpha) + r) ** 2
        )
        return y, logpx - logdetgrad.reshape(-1)


class HypernetworkRadialFlow(nn.Module):
    def __init__(self, nd: int, cond_dim: int, *, nflows: int = 4):
        super().__init__()
        self.nd = nd
        self.nflows = nflows
        self.radial_flows = nn.ModuleList([_RadialFlow(nd, hypernet=True) for _ in range(nflows)])
        self.hypernet = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, (self.nd + 2) * nflows),
        )

    def forward(
        self,
        x: Tensor,
        *,
        logpx: Optional[Tensor] = None,
        reverse: bool = False,
        cond: Optional[Tensor] = None,
        **kwargs,
    ):
        del kwargs
        if cond is None:
            raise ValueError("HypernetworkRadialFlow requires cond.")
        hyper_out = self.hypernet(cond)
        out = (x, logpx)
        for i in range(self.nflows):
            start = (self.nd + 2) * i
            z0 = hyper_out[:, start : start + self.nd]
            log_alpha = hyper_out[:, start + self.nd : start + self.nd + 1] - 6.0
            beta = hyper_out[:, start + self.nd + 1 : start + self.nd + 2] - 6.0
            out = self.radial_flows[i](
                *out,
                reverse=reverse,
                z0=z0,
                log_alpha=log_alpha,
                beta=beta,
            )
        return out


def _zero_diffeq(t: Tensor, h: Tensor) -> Tensor:
    del t
    return torch.zeros_like(h)


class AuxODEFunc(nn.Module):
    def __init__(
        self,
        func: nn.Module,
        *,
        dim: int,
        aux_dim: int,
        aux_odefunc,
        time_offset: float,
    ):
        super().__init__()
        self.func = func
        self.dim = dim
        self.aux_dim = aux_dim
        self.time_offset = time_offset
        object.__setattr__(self, "_aux_odefunc_ref", aux_odefunc)

    @property
    def aux_odefunc(self):
        return self._aux_odefunc_ref

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        x = state[:, : self.dim]
        h = state[:, self.dim :]
        aux = h[:, -self.aux_dim :] if self.aux_dim > 0 else h[:, :0]
        dx = self.func(t, torch.cat([x, aux], dim=1))
        dh = self._aux_odefunc_ref(t - self.time_offset, h)
        return torch.cat([dx, dh], dim=1)


@register_spatial("neural_jumpcnf")
class NeuralJumpCNFSpatial(Decoder):
    """Faithful shared-hidden JumpCNF port for Neural STPP."""

    SEQUENCE_COUPLED = True

    time_offset = 2.0

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        *,
        spatial_aux_dim: Optional[int] = None,
        hidden_dims: Sequence[int] | int | str | None = None,
        layer_type: str = "concat",
        actfn: str = "swish",
        zero_init: bool = True,
        tol: float = 1e-4,
        otreg_strength: float = 1e-4,
        use_adjoint: bool = True,
        solve_reverse: bool = True,
        aux_odefunc=None,
        n_flows: int = 4,
        field_cov_dim: int = 0,
        **kwargs,
    ):
        del field_cov_dim, kwargs
        super().__init__(hidden_dim=hidden_dim, spatial_dim=spatial_dim)
        self.hidden_dims = _normalize_hidden_dims(hidden_dims)
        self.aux_hidden_dim = int(spatial_aux_dim or hidden_dim // 2)
        if self.aux_hidden_dim <= 0:
            raise ValueError("NeuralJumpCNFSpatial requires spatial_aux_dim > 0")
        self.solve_reverse = bool(solve_reverse)
        self.n_flows = int(n_flows)
        if self.n_flows <= 0:
            raise ValueError("NeuralJumpCNFSpatial requires n_flows > 0")
        if self.solve_reverse and aux_odefunc is None:
            raise ValueError("solve_reverse=True requires aux_odefunc from the temporal backbone.")
        object.__setattr__(
            self,
            "_aux_odefunc_ref",
            aux_odefunc if aux_odefunc is not None else _zero_diffeq,
        )
        self._energy_reg: Tensor | float = 0.0

        base_func = _build_fc_odefunc(
            spatial_dim + self.aux_hidden_dim,
            self.hidden_dims,
            out_dim=spatial_dim,
            nonzero_dim=spatial_dim,
            layer_type=layer_type,
            actfn=actfn,
            zero_init=zero_init,
        )
        odefunc = AuxODEFunc(
            base_func,
            dim=spatial_dim,
            aux_dim=self.aux_hidden_dim,
            aux_odefunc=self._aux_odefunc_ref,
            time_offset=self.time_offset,
        )
        self.cnf = TimeVariableCNF(
            odefunc,
            spatial_dim,
            tol=tol,
            method="dopri5",
            energy_regularization=otreg_strength,
            jacnorm_regularization=otreg_strength,
            use_adjoint=bool(use_adjoint),
        )
        self.cnf.debug_solve_reverse = self.solve_reverse
        self.inst_flow = HypernetworkRadialFlow(
            spatial_dim,
            cond_dim=1 + spatial_dim + self.aux_hidden_dim,
            nflows=self.n_flows,
        )
        self.z_mean = nn.Parameter(torch.zeros(1, spatial_dim))
        self.z_logstd = nn.Parameter(torch.zeros(1, spatial_dim))

    @property
    def aux_odefunc(self):
        return self._aux_odefunc_ref

    def _select_aux(self, h: Tensor) -> Tensor:
        if h.shape[-1] < self.aux_hidden_dim:
            raise ValueError(
                f"NeuralJumpCNFSpatial expected hidden width >= {self.aux_hidden_dim}, got {h.shape[-1]}"
            )
        return h[..., -self.aux_hidden_dim :]

    def log_prob(self, z, t, s, t_prev, x_field=None):
        raise NotImplementedError("NeuralJumpCNFSpatial.SEQUENCE_COUPLED=True; use sequence_nll().")

    def nll(self, z, t, s, t_prev, x_field=None):
        raise NotImplementedError("NeuralJumpCNFSpatial.SEQUENCE_COUPLED=True; use sequence_nll().")

    def sequence_logprob(
        self,
        *,
        event_times: Tensor,
        spatial_locations: Tensor,
        input_mask: Optional[Tensor] = None,
        aux_state: Optional[Tensor] = None,
    ) -> Tensor:
        if input_mask is None:
            input_mask = torch.ones_like(event_times)

        if event_times.shape != input_mask.shape:
            raise ValueError("JumpCNF event_times and input_mask must have the same shape.")
        if event_times.shape[:2] != spatial_locations.shape[:2]:
            raise ValueError("JumpCNF event_times and spatial_locations must align.")
        if aux_state is not None and event_times.shape[:2] != aux_state.shape[:2]:
            raise ValueError("JumpCNF aux_state must align with event_times.")

        n_batch, n_steps, dim = spatial_locations.shape
        if dim != self.spatial_dim:
            raise ValueError(f"Expected spatial_dim={self.spatial_dim}, got {dim}")

        self.cnf.nfe = 0
        self._energy_reg = spatial_locations.new_tensor(0.0)
        input_mask_bool = input_mask.bool()

        shifted_event_times = self.time_offset + event_times
        shifted_event_times = torch.cat(
            [torch.zeros(n_batch, 1, device=event_times.device, dtype=event_times.dtype), shifted_event_times],
            dim=1,
        )
        extended_mask = torch.cat(
            [torch.ones(n_batch, 1, device=input_mask.device, dtype=input_mask.dtype), input_mask],
            dim=1,
        ).bool()

        xs = None
        dlogps = None
        reg_total = spatial_locations.new_tensor(0.0)

        for i in range(n_steps):
            # Only real event intervals should reach the CNF solve. Masked/padded
            # rows must behave like identity updates, otherwise torchdiffeq's
            # adjoint backward can hit dt=0 underflow on the spatial side.
            interval_active = extended_mask[:, -i - 1] & extended_mask[:, -i - 2]
            interval_active_flat = (
                interval_active.reshape(n_batch, 1).expand(n_batch, i + 1).reshape(-1)
            )
            t0 = (
                shifted_event_times[:, -i - 1]
                .reshape(n_batch, 1)
                .expand(n_batch, i + 1)
                .reshape(-1)
            )
            t1 = (
                shifted_event_times[:, -i - 2]
                .reshape(n_batch, 1)
                .expand(n_batch, i + 1)
                .reshape(-1)
            )
            dt_abs = (t1 - t0).abs()
            dt_scale = torch.maximum(torch.maximum(t0.abs(), t1.abs()), torch.ones_like(t0))
            tiny_threshold = torch.finfo(t0.dtype).eps * dt_scale
            tiny_interval_flat = interval_active_flat & (dt_abs <= tiny_threshold)
            cnf_active_flat = interval_active_flat & ~tiny_interval_flat

            if i == 0:
                xs = spatial_locations[:, -1].reshape(n_batch, 1, dim)
                dlogps = torch.zeros(n_batch, 1, device=xs.device, dtype=xs.dtype)
            else:
                xs = torch.cat([spatial_locations[:, -i - 1].reshape(n_batch, 1, dim), xs], dim=1)
                dlogps = torch.cat([torch.zeros(n_batch, 1, device=xs.device, dtype=xs.dtype), dlogps], dim=1)

            xs_flat = xs.reshape(-1, dim)
            dlogps_flat = dlogps.reshape(-1)

            if aux_state is not None:
                hidden_width = aux_state.shape[-1]
                auxs = aux_state[:, -i - 1 :, :].expand(n_batch, i + 1, hidden_width).reshape(-1, hidden_width)
                xs_in = torch.cat([xs_flat, auxs], dim=1)
            else:
                xs_in = xs_flat
                auxs = None

            xs_out = xs_in
            dlogps_after_cnf = dlogps_flat
            reg_i = xs_flat.new_tensor(0.0)
            if cnf_active_flat.any():
                active = cnf_active_flat
                xs_active, dlogps_active, reg_i = self.cnf.integrate(
                    t0[active],
                    t1[active],
                    xs_in[active],
                    dlogps_flat[active],
                    method="dopri5" if i < n_steps - 1 and self.training else "dopri5",
                )
                if active.all():
                    xs_out = xs_active
                    dlogps_after_cnf = dlogps_active
                else:
                    xs_out = xs_in.clone()
                    dlogps_after_cnf = dlogps_flat.clone()
                    xs_out[active] = xs_active
                    dlogps_after_cnf[active] = dlogps_active
            reg_total = reg_total + torch.as_tensor(
                reg_i,
                device=xs_flat.device,
                dtype=xs_flat.dtype,
            ).reshape(())

            if aux_state is not None:
                xs_flat = xs_out[:, :dim]
                auxs = xs_out[:, dim:]
            else:
                xs_flat = xs_out
            dlogps_flat = dlogps_after_cnf

            if i < n_steps - 1 and interval_active_flat.any():
                obs_x = spatial_locations[:, -i - 2].reshape(n_batch, 1, dim).expand(n_batch, i + 1, dim).reshape(-1, dim)
                obs_t = shifted_event_times[:, -i - 2].reshape(n_batch, 1).expand(n_batch, i + 1).reshape(-1, 1)
                aux_cond = self._select_aux(auxs) if auxs is not None else xs_flat[:, :0]
                cond = torch.cat([obs_t, obs_x, aux_cond], dim=1)
                active = interval_active_flat
                if active.all():
                    xs_flat, dlogps_flat = self.inst_flow(xs_flat, logpx=dlogps_flat, cond=cond)
                else:
                    xs_next = xs_flat.clone()
                    dlogps_next = dlogps_flat.clone()
                    xs_active, dlogps_active = self.inst_flow(
                        xs_flat[active],
                        logpx=dlogps_flat[active],
                        cond=cond[active],
                    )
                    xs_next[active] = xs_active
                    dlogps_next[active] = dlogps_active
                    xs_flat, dlogps_flat = xs_next, dlogps_next

            xs = xs_flat.reshape(n_batch, i + 1, dim)
            dlogps = dlogps_flat.reshape(n_batch, i + 1)
            dlogps = torch.where(extended_mask[:, -i - 1 :], dlogps, torch.zeros_like(dlogps))

        self._energy_reg = reg_total
        logpz = _gaussian_loglik(
            xs,
            self.z_mean.expand_as(xs),
            self.z_logstd.expand_as(xs),
        ).sum(dim=-1)
        logpx = logpz - dlogps
        return torch.where(extended_mask[:, 1:], logpx, torch.zeros_like(logpx))

    def sequence_nll(
        self,
        z_seq: Tensor,
        t_seq: Tensor,
        s_seq: Tensor,
        t_prev_seq: Tensor,
        lengths: Tensor,
        mask: Tensor,
        **kwargs,
    ) -> Tensor:
        del t_prev_seq, lengths, kwargs
        event_times = t_seq.squeeze(-1)
        logpx = self.sequence_logprob(
            event_times=event_times,
            spatial_locations=s_seq,
            input_mask=mask,
            aux_state=z_seq,
        )
        return -logpx

    def conditional_logprob_fn(
        self,
        t_query: float,
        event_times: Tensor,
        event_locs: Tensor,
        z_aug: Tensor,
    ) -> Callable[[Tensor], Tensor]:
        t_query_val = float(t_query)
        n_steps, dim = event_locs.shape

        def loglikelihood_fn(s: Tensor) -> Tensor:
            batch_size = s.shape[0]
            bsz_event_times = event_times.unsqueeze(0).expand(batch_size, n_steps)
            bsz_event_times = torch.cat(
                [
                    bsz_event_times,
                    torch.full((batch_size, 1), t_query_val, device=bsz_event_times.device, dtype=bsz_event_times.dtype),
                ],
                dim=1,
            )
            bsz_spatial_locations = event_locs.unsqueeze(0).expand(batch_size, n_steps, dim)
            bsz_spatial_locations = torch.cat([bsz_spatial_locations, s.reshape(batch_size, 1, dim)], dim=1)
            bsz_aux_state = z_aug.reshape(1, n_steps + 1, -1).expand(batch_size, -1, -1)
            return self.sequence_logprob(
                event_times=bsz_event_times,
                spatial_locations=bsz_spatial_locations,
                input_mask=None,
                aux_state=bsz_aux_state,
            )[:, -1]

        return loglikelihood_fn

    def vector_field_fn(
        self,
        t_query: float,
        event_times: Tensor,
        spatial_locations: Tensor,
        aux_state: Optional[Tensor] = None,
    ) -> Callable[[Tensor], Tensor]:
        del event_times, spatial_locations
        t_query_tensor = torch.tensor(t_query)

        def vecfield_fn(s: Tensor) -> Tensor:
            xs = s.reshape(s.shape[0], self.spatial_dim)
            if aux_state is not None:
                aux_full = aux_state[-1, :].reshape(1, -1).expand(s.shape[0], -1)
                xs = torch.cat([xs, aux_full], dim=1)
            return self.cnf.func(t_query_tensor.to(xs), xs)[:, : self.spatial_dim]

        return vecfield_fn


__all__ = ["NeuralJumpCNFSpatial"]
