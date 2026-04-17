"""Upstream-style low-rank DeepBasis kernel for the provisional NSMPP port."""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn


class DeepNetworkBasis(nn.Module):
    """Neural basis map used by the upstream DeepBasis kernel."""

    def __init__(
        self,
        *,
        data_dim: int,
        basis_dim: int,
        init_gain: float = 5e-1,
        init_bias: float = 1e-3,
        nn_width: int = 5,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, nn_width),
            nn.Softplus(),
            nn.Linear(nn_width, nn_width),
            nn.Softplus(),
            nn.Linear(nn_width, nn_width),
            nn.Softplus(),
            nn.Linear(nn_width, basis_dim),
            nn.Sigmoid(),
        )
        self._init_gain = float(init_gain)
        self._init_bias = float(init_bias)
        self.net.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=self._init_gain)
            module.bias.data.fill_(self._init_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) * 2.0 - 1.0


class DeepBasisKernel(nn.Module):
    """Finite-rank neural basis kernel from the upstream implementation."""

    def __init__(
        self,
        *,
        n_basis: int,
        data_dim: int,
        basis_dim: int,
        init_gain: float = 5e-1,
        init_bias: float = 1e-3,
        init_std: float = 1.0,
        init_weight_mean: float = -3.0,
        nn_width: int = 5,
    ) -> None:
        super().__init__()
        self.n_basis = int(n_basis)
        self.data_dim = int(data_dim)
        self.basis_dim = int(basis_dim)

        self.x_basis = nn.ModuleList(
            [
                DeepNetworkBasis(
                    data_dim=data_dim,
                    basis_dim=basis_dim,
                    init_gain=init_gain,
                    init_bias=init_bias,
                    nn_width=nn_width,
                )
                for _ in range(self.n_basis)
            ]
        )
        self.y_basis = nn.ModuleList(
            [
                DeepNetworkBasis(
                    data_dim=data_dim,
                    basis_dim=basis_dim,
                    init_gain=init_gain,
                    init_bias=init_bias,
                    nn_width=nn_width,
                )
                for _ in range(self.n_basis)
            ]
        )
        # Initialize in softplus-inverse space with a negative mean so excitation
        # starts near zero — the standard Hawkes prior (background-only at init).
        # init_weight_mean controls the initial softplus(raw_weight) scale:
        #   -3 → ~0.05,  -1 → ~0.31,  0 → ~0.69
        self.raw_weights = nn.Parameter(
            torch.empty(self.n_basis).normal_(mean=float(init_weight_mean), std=init_std)
        )

    def positive_weights(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_weights)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.shape != y.shape:
            raise ValueError(
                f"DeepBasisKernel.forward expects matched shapes, got {tuple(x.shape)} and {tuple(y.shape)}."
            )
        total = x.new_zeros(x.shape[:-1])
        for weight, x_basis, y_basis in zip(
            self.positive_weights(),
            self.x_basis,
            self.y_basis,
        ):
            q = x_basis(x)
            h = y_basis(y)
            total = total + weight * (q * h).sum(dim=-1)
        return total

    def sum_over_history(self, query: Tensor, history: Tensor, history_mask: Tensor) -> Tensor:
        """Return ``sum_j K(query_i, history_j)`` under an explicit history mask."""
        if query.ndim != 3 or history.ndim != 3:
            raise ValueError("sum_over_history expects query/history with shape (B, N, D)/(B, H, D).")
        if query.shape[0] != history.shape[0] or query.shape[-1] != history.shape[-1]:
            raise ValueError(
                f"Incompatible query/history shapes {tuple(query.shape)} vs {tuple(history.shape)}."
            )
        if history_mask.shape != (query.shape[0], query.shape[1], history.shape[1]):
            raise ValueError(
                "history_mask must have shape (B, N_query, N_history), "
                f"got {tuple(history_mask.shape)}."
            )

        total = query.new_zeros(query.shape[:2])
        mask = history_mask.to(dtype=query.dtype)
        for weight, x_basis, y_basis in zip(
            self.positive_weights(),
            self.x_basis,
            self.y_basis,
        ):
            q_feat = x_basis(query.reshape(-1, query.shape[-1])).reshape(
                query.shape[0], query.shape[1], self.basis_dim
            )
            h_feat = y_basis(history.reshape(-1, history.shape[-1])).reshape(
                history.shape[0], history.shape[1], self.basis_dim
            )
            pairwise = torch.einsum("bqd,bhd->bqh", q_feat, h_feat)
            total = total + weight * (pairwise * mask).sum(dim=-1)
        return total
