"""Upstream-faithful AutoSTPP Cuboid / ProdNet kernel helpers.

This is a compact reimplementation of the active upstream AutoSTPP kernel path:

- each ProdNet factorizes over x / y / t using three independent 1D MLPs
- L uses a terminal sign flip (`neg=True`), M does not
- the joint intensity is the mixed third derivative of M-L
- temporal intensity and compensator are exact via separability and
  inclusion-exclusion on the unit square

The implementation keeps the upstream non-negativity constraint by clamping all
linear weights after every optimizer step.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _activation_forward(name: str, x: Tensor) -> Tensor:
    name = name.lower()
    if name == "tanh":
        return torch.tanh(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "relu":
        return F.relu(x)
    if name == "elu":
        return F.elu(x)
    raise ValueError(
        f"Unsupported AutoSTPP faithful activation '{name}'. "
        "Supported: tanh, sigmoid, relu, elu."
    )


def _activation_grad(name: str, pre: Tensor, post: Tensor) -> Tensor:
    name = name.lower()
    if name == "tanh":
        return 1.0 - post.pow(2)
    if name == "sigmoid":
        return post * (1.0 - post)
    if name == "relu":
        return (pre > 0).to(dtype=pre.dtype)
    if name == "elu":
        return torch.where(pre > 0, torch.ones_like(pre), torch.exp(pre))
    raise ValueError(
        f"Unsupported AutoSTPP faithful activation '{name}'. "
        "Supported: tanh, sigmoid, relu, elu."
    )


class _AxisMLP(nn.Module):
    """1D monotone MLP with exact first derivative via manual chain rule."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        activation: str,
        bias: bool,
        neg: bool,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        self.activation = str(activation)
        self.neg = bool(neg)
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(1, hidden_size, bias=bias))
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
        self.out = nn.Linear(hidden_size, 1, bias=bias)

    def forward_with_grad(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        value = x
        grad = torch.ones_like(value)

        for layer in self.hidden_layers:
            pre = F.linear(value, layer.weight, layer.bias)
            grad = grad @ layer.weight.t()
            value = _activation_forward(self.activation, pre)
            grad = grad * _activation_grad(self.activation, pre, value)

        value = F.linear(value, self.out.weight, self.out.bias)
        grad = grad @ self.out.weight.t()
        if self.neg:
            value = -value
            grad = -grad
        return value, grad

    def forward(self, x: Tensor) -> Tensor:
        value, _ = self.forward_with_grad(x)
        return value

    def project(self) -> None:
        with torch.no_grad():
            for layer in list(self.hidden_layers) + [self.out]:
                layer.weight.clamp_(min=0.0)


class AutoSTPPFaithfulProdNet(nn.Module):
    """Separable ProdNet used by upstream AutoSTPP."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        activation: str,
        bias: bool,
        neg: bool = False,
    ):
        super().__init__()
        self.x_net = _AxisMLP(
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            bias=bias,
            neg=neg,
        )
        self.y_net = _AxisMLP(
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            bias=bias,
            neg=neg,
        )
        self.t_net = _AxisMLP(
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            bias=bias,
            neg=neg,
        )

    def forward(self, st_diff: Tensor) -> Tensor:
        _, dx = self.x_net.forward_with_grad(st_diff[:, 0:1])
        _, dy = self.y_net.forward_with_grad(st_diff[:, 1:2])
        _, dt = self.t_net.forward_with_grad(st_diff[:, 2:3])
        return dx * dy * dt

    def lamb_t(
        self,
        xa: Tensor,
        xb: Tensor,
        ya: Tensor,
        yb: Tensor,
        t: Tensor,
    ) -> Tensor:
        xa = xa.reshape(-1, 1)
        xb = xb.reshape(-1, 1)
        ya = ya.reshape(-1, 1)
        yb = yb.reshape(-1, 1)
        t = t.reshape(-1, 1)
        fx_a = self.x_net(xa)
        fx_b = self.x_net(xb)
        fy_a = self.y_net(ya)
        fy_b = self.y_net(yb)
        _, ft = self.t_net.forward_with_grad(t)
        return (fx_b - fx_a) * (fy_b - fy_a) * ft

    def int_lamb(
        self,
        xa: Tensor,
        xb: Tensor,
        ya: Tensor,
        yb: Tensor,
        ta: Tensor,
        tb: Tensor,
    ) -> Tensor:
        xa = xa.reshape(-1, 1)
        xb = xb.reshape(-1, 1)
        ya = ya.reshape(-1, 1)
        yb = yb.reshape(-1, 1)
        ta = ta.reshape(-1, 1)
        tb = tb.reshape(-1, 1)
        fx_a = self.x_net(xa)
        fx_b = self.x_net(xb)
        fy_a = self.y_net(ya)
        fy_b = self.y_net(yb)
        ft_a = self.t_net(ta)
        ft_b = self.t_net(tb)
        return (fx_b - fx_a) * (fy_b - fy_a) * (ft_b - ft_a)

    def project(self) -> None:
        self.x_net.project()
        self.y_net.project()
        self.t_net.project()


class _SumProdNet(nn.Module):
    def __init__(self, nets: Iterable[AutoSTPPFaithfulProdNet]):
        super().__init__()
        self.nets = nn.ModuleList(list(nets))
        if not self.nets:
            raise ValueError("AutoSTPP faithful SumProdNet requires at least one ProdNet.")

    def forward(self, st_diff: Tensor) -> Tensor:
        out = self.nets[0].forward(st_diff)
        for net in self.nets[1:]:
            out = out + net.forward(st_diff)
        return out

    def lamb_t(
        self,
        xa: Tensor,
        xb: Tensor,
        ya: Tensor,
        yb: Tensor,
        t: Tensor,
    ) -> Tensor:
        out = self.nets[0].lamb_t(xa, xb, ya, yb, t)
        for net in self.nets[1:]:
            out = out + net.lamb_t(xa, xb, ya, yb, t)
        return out

    def int_lamb(
        self,
        xa: Tensor,
        xb: Tensor,
        ya: Tensor,
        yb: Tensor,
        ta: Tensor,
        tb: Tensor,
    ) -> Tensor:
        out = self.nets[0].int_lamb(xa, xb, ya, yb, ta, tb)
        for net in self.nets[1:]:
            out = out + net.int_lamb(xa, xb, ya, yb, ta, tb)
        return out

    def project(self) -> None:
        for net in self.nets:
            net.project()


class AutoSTPPFaithfulCuboid(nn.Module):
    """Exact Cuboid kernel used by upstream AutoSTPP."""

    def __init__(
        self,
        *,
        n_prodnet: int,
        hidden_size: int,
        num_layers: int,
        activation: str,
        bias: bool,
    ):
        super().__init__()
        self.L = _SumProdNet(
            AutoSTPPFaithfulProdNet(
                hidden_size=hidden_size,
                num_layers=num_layers,
                activation=activation,
                bias=bias,
                neg=True,
            )
            for _ in range(n_prodnet)
        )
        self.M = _SumProdNet(
            AutoSTPPFaithfulProdNet(
                hidden_size=hidden_size,
                num_layers=num_layers,
                activation=activation,
                bias=bias,
                neg=False,
            )
            for _ in range(n_prodnet)
        )

    def forward(self, st_diff: Tensor) -> Tensor:
        return 3.0 * (self.M.forward(st_diff) - self.L.forward(st_diff))

    def lamb_t(
        self,
        xa: Tensor,
        xb: Tensor,
        ya: Tensor,
        yb: Tensor,
        t: Tensor,
    ) -> Tensor:
        return 3.0 * (
            self.M.lamb_t(xa, xb, ya, yb, t) - self.L.lamb_t(xa, xb, ya, yb, t)
        )

    def lamb_t_stpp(self, s: Tensor, t: Tensor) -> Tensor:
        x = s[:, 0]
        y = s[:, 1]
        return self.lamb_t(
            -x,
            1.0 - x,
            -y,
            1.0 - y,
            t.reshape(-1),
        )

    def int_lamb(
        self,
        xa: Tensor,
        xb: Tensor,
        ya: Tensor,
        yb: Tensor,
        ta: Tensor,
        tb: Tensor,
    ) -> Tensor:
        return 3.0 * (
            self.M.int_lamb(xa, xb, ya, yb, ta, tb)
            - self.L.int_lamb(xa, xb, ya, yb, ta, tb)
        )

    def int_lamb_stpp(self, s: Tensor, ta: Tensor, tb: Tensor) -> Tensor:
        x = s[:, 0]
        y = s[:, 1]
        return self.int_lamb(
            -x,
            1.0 - x,
            -y,
            1.0 - y,
            ta.reshape(-1),
            tb.reshape(-1),
        )

    def project(self) -> None:
        self.M.project()
        self.L.project()
