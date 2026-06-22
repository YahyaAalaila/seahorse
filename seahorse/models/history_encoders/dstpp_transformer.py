"""DSTPP Transformer_ST stack for diffusion_stpp.

This module implements the DSTPP transformer stack for the seahorse batch
contract. Padding masks can be built from explicit sequence lengths instead of
relying solely on zero-valued absolute times.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


PAD_VALUE = 0.0


def get_non_pad_mask(seq: Tensor) -> Tensor:
    if seq.dim() != 2:
        raise ValueError(f"Expected seq shape (B, N), got {tuple(seq.shape)}")
    return seq.ne(PAD_VALUE).to(dtype=torch.float32).unsqueeze(-1)


def non_pad_mask_from_lengths(lengths: Tensor, max_len: int) -> Tensor:
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).to(dtype=torch.float32).unsqueeze(-1)


def get_subsequent_mask(seq: Tensor, dim: int = 2) -> Tensor:
    sz_b, len_s = seq.size()[:2]
    subsequent_mask = torch.triu(
        torch.ones((dim, len_s, len_s), device=seq.device, dtype=torch.uint8),
        diagonal=1,
    ).permute(1, 2, 0)
    return subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1, -1)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, attn_dropout: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        normalize_before: bool = False,
    ):
        super().__init__()
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(
            temperature=d_k ** 0.5,
            attn_dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = output + residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hid: int,
        dropout: float = 0.1,
        normalize_before: bool = False,
    ):
        super().__init__()
        self.normalize_before = normalize_before
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual
        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_head: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        normalize_before: bool = False,
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model,
            d_inner,
            dropout=dropout,
            normalize_before=normalize_before,
        )

    def forward(
        self,
        enc_input: Tensor,
        non_pad_mask: Optional[Tensor] = None,
        slf_attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input,
            enc_input,
            enc_input,
            mask=slf_attn_mask,
        )
        if non_pad_mask is not None:
            enc_output = enc_output * non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output = enc_output * non_pad_mask
        return enc_output, enc_slf_attn


class EncoderST(nn.Module):
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
    ):
        del CosSin
        super().__init__()
        self.d_model = int(d_model)
        self.loc_dim = int(loc_dim)
        self.register_buffer(
            "position_vec",
            torch.tensor(
                [math.pow(10000.0, 2.0 * (i // 2) / float(d_model)) for i in range(d_model)],
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.event_emb_loc = nn.Sequential(
            nn.Linear(self.loc_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    dropout=dropout,
                    normalize_before=False,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_stack_loc = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    dropout=dropout,
                    normalize_before=False,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_stack_temporal = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    dropout=dropout,
                    normalize_before=False,
                )
                for _ in range(n_layers)
            ]
        )

    def temporal_enc(self, time: Tensor, non_pad_mask: Tensor) -> Tensor:
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
    ) -> Tuple[Tensor, Tensor, Tensor]:
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim)
        key_pad_mask = non_pad_mask.squeeze(-1).eq(0)
        slf_attn_mask_keypad = key_pad_mask.unsqueeze(1).unsqueeze(-1).expand(
            -1, event_loc.size(1), -1, self.loc_dim
        )
        slf_attn_mask = (slf_attn_mask_keypad.to(dtype=slf_attn_mask_subseq.dtype) + slf_attn_mask_subseq).gt(0)

        enc_output_temporal = self.temporal_enc(event_time, non_pad_mask)
        enc_output_loc = self.event_emb_loc(event_loc)
        enc_output = enc_output_temporal + enc_output_loc

        slf_attn_mask = slf_attn_mask[:, :, :, 0]

        for index in range(len(self.layer_stack)):
            enc_output_loc, _ = self.layer_stack_loc[index](
                enc_output_loc,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
            )
            enc_output_temporal, _ = self.layer_stack_temporal[index](
                enc_output_temporal,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
            )
            enc_output, _ = self.layer_stack[index](
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
            )
        return enc_output, enc_output_temporal, enc_output_loc


class RNNLayers(nn.Module):
    def __init__(self, d_model: int, d_rnn: int):
        super().__init__()
        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data: Tensor, non_pad_mask: Tensor) -> Tensor:
        lengths = non_pad_mask.squeeze(2).long().sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            data,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out = self.rnn(packed)[0]
        out = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)[0]
        return self.projection(out)


class DSTPPTransformerST(nn.Module):
    """Transformer_ST for diffusion conditioning."""

    def __init__(
        self,
        d_model: int = 64,
        d_rnn: int = 256,
        d_inner: int = 128,
        n_layers: int = 4,
        n_head: int = 4,
        d_k: int = 16,
        d_v: int = 16,
        dropout: float = 0.1,
        device=None,
        loc_dim: int = 2,
        CosSin: bool = False,
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
        )
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.rnn = RNNLayers(d_model, d_rnn)
        self.rnn_temporal = RNNLayers(d_model, d_rnn)
        self.rnn_spatial = RNNLayers(d_model, d_rnn)

    def forward(
        self,
        event_loc: Tensor,
        event_time: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if lengths is None:
            non_pad_mask = get_non_pad_mask(event_time)
        else:
            non_pad_mask = non_pad_mask_from_lengths(lengths, event_time.shape[1])
            non_pad_mask = non_pad_mask.to(device=event_time.device, dtype=event_time.dtype)

        enc_output, enc_output_temporal, enc_output_loc = self.encoder(
            event_loc,
            event_time,
            non_pad_mask,
        )
        enc_output = self.rnn(enc_output, non_pad_mask)
        enc_output_temporal = self.rnn_temporal(enc_output_temporal, non_pad_mask)
        enc_output_loc = self.rnn_spatial(enc_output_loc, non_pad_mask)
        enc_output_all = torch.cat((enc_output_temporal, enc_output_loc, enc_output), dim=-1)
        return enc_output_all, non_pad_mask
