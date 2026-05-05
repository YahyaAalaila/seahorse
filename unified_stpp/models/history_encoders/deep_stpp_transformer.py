"""DeepSTPP Transformer encoder over fixed paper windows."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _subsequent_mask(size: int, *, device, dtype) -> Tensor:
    mask = torch.triu(
        torch.ones(size, size, device=device, dtype=torch.bool),
        diagonal=1,
    )
    out = torch.zeros(size, size, device=device, dtype=dtype)
    return out.masked_fill(mask, float("-inf"))


class DeepSTPPPositionalEncoding(nn.Module):
    """Sinusoidal time encoding used by DeepSTPP."""

    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.d_model = int(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = int(max_len)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(*t.shape, self.d_model, device=x.device, dtype=x.dtype)
        args = t.unsqueeze(-1) * div_term
        pe[..., 0::2] = torch.sin(args)
        pe[..., 1::2] = torch.cos(args)
        return self.dropout(x + pe)


class DeepSTPPTransformerEncoder(nn.Module):
    """Transformer encoder that maps one paper window to one latent Gaussian."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        seq_len: int,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.seq_len = int(seq_len)
        self.input_proj = nn.Linear(3, self.hidden_dim, bias=False)
        self.pos_encoder = DeepSTPPPositionalEncoding(
            self.hidden_dim,
            dropout=dropout,
            max_len=self.seq_len,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=dropout,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        self.input_proj.weight.data.uniform_(-initrange, initrange)
        self.output_proj.bias.data.zero_()
        self.output_proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, windows: Tensor) -> tuple[Tensor, Tensor]:
        if windows.ndim != 3 or windows.shape[-1] != 3:
            raise ValueError(
                "DeepSTPPTransformerEncoder expects windows with shape (B, T, 3)."
            )

        batch, seq_len, _ = windows.shape
        if batch == 0:
            empty = windows.new_zeros(0, self.hidden_dim)
            return empty, empty

        x = windows.transpose(0, 1)
        x_mask = _subsequent_mask(seq_len, device=windows.device, dtype=windows.dtype)
        t = torch.cumsum(x[..., -1], dim=0)
        x = self.input_proj(x) * math.sqrt(self.hidden_dim)
        x = self.pos_encoder(x, t)
        output = self.layers(x, x_mask)
        output = self.output_proj(output)[-1]
        mu, var_raw = torch.chunk(output, chunks=2, dim=-1)
        var = F.softplus(var_raw) + 1e-5
        return mu, var
