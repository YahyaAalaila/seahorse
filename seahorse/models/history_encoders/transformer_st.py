"""SMASH transformer encoder stack adapted to seahorse contracts."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


PAD_VALUE = 0.0


def get_non_pad_mask(seq: Tensor) -> Tensor:
    """Return non-padding mask with shape (B, N, 1)."""
    if seq.dim() != 2:
        raise ValueError(f"Expected seq shape (B, N), got {tuple(seq.shape)}")
    return seq.ne(PAD_VALUE).to(dtype=torch.float32).unsqueeze(-1)


def non_pad_mask_from_lengths(lengths: Tensor, max_len: int) -> Tensor:
    """Build non-pad mask from sequence lengths."""
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).to(dtype=torch.float32).unsqueeze(-1)


class RNNLayers(nn.Module):
    """Optional LSTM projection layer used in the original SMASH code."""

    def __init__(self, d_model: int, d_rnn: int):
        super().__init__()
        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data: Tensor, non_pad_mask: Tensor) -> Tensor:
        lengths = non_pad_mask.squeeze(-1).long().sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            data,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        out_packed, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        out = self.projection(out)
        return out


class EncoderST(nn.Module):
    """Faithful SMASH-style temporal/spatial/joint encoder stacks."""

    def __init__(
        self,
        *,
        d_model: int,
        d_inner: int,
        n_layers: int,
        n_head: int,
        d_k: int,
        d_v: int,
        dropout: float,
        device,
        loc_dim: int,
        CosSin: bool = False,
        num_types: int = 1,
    ):
        del d_k, d_v, CosSin
        super().__init__()

        self.d_model = int(d_model)
        self.loc_dim = int(loc_dim)

        self.register_buffer(
            "position_vec",
            torch.tensor(
                [
                    math.pow(10000.0, 2.0 * (i // 2) / float(d_model))
                    for i in range(d_model)
                ],
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.event_emb_loc = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        if self.loc_dim == 3:
            self.event_emb_mark = nn.Embedding(num_types + 1, d_model, padding_idx=0)

        def _make_layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_inner,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )

        self.layer_stack = nn.ModuleList([_make_layer() for _ in range(n_layers)])
        self.layer_stack_loc = nn.ModuleList([_make_layer() for _ in range(n_layers)])
        self.layer_stack_temporal = nn.ModuleList([_make_layer() for _ in range(n_layers)])

        if self.loc_dim == 3:
            self.layer_stack_mark = nn.ModuleList([_make_layer() for _ in range(n_layers)])

    def temporal_enc(self, time: Tensor, non_pad_mask: Tensor) -> Tensor:
        """Sin/cos positional encoding over event_time_origin."""
        pos = self.position_vec.to(device=time.device, dtype=time.dtype)
        result = time.unsqueeze(-1) / pos
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(
        self,
        event_loc: Tensor,
        event_time: Tensor,
        non_pad_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        if self.loc_dim == 3:
            event_mark = event_loc[:, :, 0]
            event_loc_xy = event_loc[:, :, 1:]
        else:
            event_mark = None
            event_loc_xy = event_loc

        B, N, _ = event_loc_xy.shape
        key_pad_mask = non_pad_mask.squeeze(-1).eq(0)
        causal_mask = torch.triu(
            torch.ones(N, N, device=event_loc.device, dtype=torch.bool), diagonal=1
        )

        enc_output_temporal = self.temporal_enc(event_time, non_pad_mask)
        enc_output_loc = self.event_emb_loc(event_loc_xy) * non_pad_mask

        if self.loc_dim == 3:
            enc_output_mark = self.event_emb_mark(event_mark.long()) * non_pad_mask
            enc_output = enc_output_temporal + enc_output_loc + enc_output_mark
        else:
            enc_output_mark = None
            enc_output = enc_output_temporal + enc_output_loc

        for idx in range(len(self.layer_stack)):
            enc_output_loc = self.layer_stack_loc[idx](
                enc_output_loc,
                src_mask=causal_mask,
                src_key_padding_mask=key_pad_mask,
            )
            enc_output_loc = enc_output_loc * non_pad_mask

            enc_output_temporal = self.layer_stack_temporal[idx](
                enc_output_temporal,
                src_mask=causal_mask,
                src_key_padding_mask=key_pad_mask,
            )
            enc_output_temporal = enc_output_temporal * non_pad_mask

            enc_output = self.layer_stack[idx](
                enc_output,
                src_mask=causal_mask,
                src_key_padding_mask=key_pad_mask,
            )
            enc_output = enc_output * non_pad_mask

            if self.loc_dim == 3:
                enc_output_mark = self.layer_stack_mark[idx](
                    enc_output_mark,
                    src_mask=causal_mask,
                    src_key_padding_mask=key_pad_mask,
                )
                enc_output_mark = enc_output_mark * non_pad_mask

        return enc_output, enc_output_temporal, enc_output_loc, enc_output_mark


class TransformerST(nn.Module):
    """SMASH conditioning transformer with temporal/spatial/joint outputs."""

    def __init__(
        self,
        d_model: int = 256,
        d_rnn: int = 128,
        d_inner: int = 1024,
        n_layers: int = 4,
        n_head: int = 4,
        d_k: int = 64,
        d_v: int = 64,
        dropout: float = 0.1,
        device=None,
        loc_dim: int = 2,
        CosSin: bool = False,
        num_types: int = 1,
    ):
        super().__init__()

        self.encoder = EncoderST(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim=loc_dim,
            CosSin=CosSin,
            num_types=num_types,
        )

        self.alpha = nn.Parameter(torch.tensor(-0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))

        self.rnn = RNNLayers(d_model, d_rnn)
        self.rnn_temporal = RNNLayers(d_model, d_rnn)
        self.rnn_spatial = RNNLayers(d_model, d_rnn)

        if loc_dim == 3:
            self.rnn_mark = RNNLayers(d_model, d_rnn)

    def forward(
        self,
        event_loc: Tensor,
        event_time: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return concatenated conditioning and non-pad mask."""
        if lengths is None:
            non_pad_mask = get_non_pad_mask(event_time)
        else:
            non_pad_mask = non_pad_mask_from_lengths(lengths, event_time.shape[1])
            non_pad_mask = non_pad_mask.to(device=event_time.device, dtype=event_time.dtype)

        enc_output, enc_temporal, enc_loc, enc_mark = self.encoder(
            event_loc,
            event_time,
            non_pad_mask,
        )

        enc_output = self.rnn(enc_output, non_pad_mask)
        enc_temporal = self.rnn_temporal(enc_temporal, non_pad_mask)
        enc_loc = self.rnn_spatial(enc_loc, non_pad_mask)

        if enc_mark is not None:
            enc_mark = self.rnn_mark(enc_mark, non_pad_mask)
            enc_output_all = torch.cat((enc_temporal, enc_loc, enc_output, enc_mark), dim=-1)
        else:
            enc_output_all = torch.cat((enc_temporal, enc_loc, enc_output), dim=-1)

        return enc_output_all, non_pad_mask
