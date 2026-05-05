"""
Neural temporal point process models for use in the factorized baseline family.

Two supported models:

  RMTPPTemporalProcess  — GRU encoder + closed-form exponential intensity
  THPTemporalProcess    — Transformer encoder + learned decay + fixed MC compensator

Both implement the same 2-method interface as parametric_processes.py:

    logprob(event_times, locations, input_mask, t0, t1) -> (B,)
    intensity_at(t_query, history_times, history_mask)  -> (M,)

Implementation notes:
  - event_times are already sequence-relative shifted (first event ≈ 0); delta_t
    is derived via diff with a zero prepended rather than from absolute timestamps.
  - Per-batch t1 (B,) is used for the survival compensator; no global t_end needed.
  - THPTemporalProcess: fixed 30-sample Monte Carlo compensator.
  - intensity_at assumes all M rows of history_times are identical (single sequence
    expanded to M query points) — consistent with how FactorizedEventModel calls it.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# RMTPP
# ---------------------------------------------------------------------------

class RMTPPTemporalProcess(nn.Module):
    """
    Recurrent Marked Temporal Point Process (RMTPP) temporal model.

    Intensity: λ(t) = a_n · exp(−b_n · (t − t_n))
    where (a_n, b_n) = softplus(MLP(h_n)) and h_n = GRU hidden after event n.

    Compensator is closed-form:
        ∫_{t_n}^{t_{n+1}} λ dt = (a_n / b_n) · (1 − exp(−b_n · Δt))

    Reference: Du et al., "Recurrent Marked Temporal Point Processes", KDD 2016.
    Architecture follows Rose-STL-Lab/AutoSTPP src/models/rmtpp.py.
    """

    def __init__(self, *, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=1, hidden_size=hidden_size, batch_first=True, num_layers=1
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def _encode(self, event_times: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor]:
        """GRU-encode inter-event times.

        Returns:
            hidden  : (B, T+1, H)  — [h_0=zeros, h_1, ..., h_N]
            delta_t : (B, T)       — inter-event gaps (first gap from t=0)
        """
        B, T = event_times.shape
        device = event_times.device

        zeros = torch.zeros(B, 1, device=device)
        prepended = torch.cat([zeros, event_times[:, :-1]], dim=1)  # (B, T)
        delta_t = (event_times - prepended).clamp(min=0.0)          # (B, T)

        packed = pack_padded_sequence(
            delta_t.unsqueeze(-1), seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=T)  # (B, T, H)

        h0 = torch.zeros(B, 1, self.hidden_size, device=device)
        hidden = torch.cat([h0, gru_out], dim=1)  # (B, T+1, H)
        return hidden, delta_t

    def logprob(
        self,
        event_times: Tensor,
        locations: Tensor,
        input_mask: Tensor,
        t0: Tensor,
        t1: Tensor,
    ) -> Tensor:
        """
        Args:
            event_times : (B, T)   shifted event times (first event ≈ 0), padded
            locations   : (B, T, D) — unused
            input_mask  : (B, T)   float — 1 for valid, 0 for padding
            t0          : (B,)     observation window start
            t1          : (B,)     observation window end
        Returns:
            (B,) per-sequence temporal log-prob
        """
        B = event_times.shape[0]
        device = event_times.device
        seq_lens = input_mask.sum(dim=1).long().clamp(min=1)

        hidden, delta_t = self._encode(event_times, seq_lens)  # (B, T+1, H), (B, T)

        # Compute (a, b) from h_prev = h_0 .. h_{T-1}
        h_prev = hidden[:, :-1, :]                               # (B, T, H)
        ab = self.mlp(h_prev)                                    # (B, T, 2)
        a = F.softplus(ab[..., 0])                               # (B, T)
        b = F.softplus(ab[..., 1]).clamp(min=1e-6)              # (B, T)

        # Log-intensity at each event: log λ(t_i) = log(a_i) − b_i · Δt_i
        log_lambda = torch.log(a + 1e-20) - b * delta_t         # (B, T)
        tll = (log_lambda * input_mask).sum(dim=1)               # (B,)

        # Compensator over each inter-event interval (closed form)
        comp_per = (a / b) * (1.0 - torch.exp(-b * delta_t))   # (B, T)
        comp = (comp_per * input_mask).sum(dim=1)               # (B,)

        # Survival compensator from last event to t1 using h_N (after last event)
        arange_B = torch.arange(B, device=device)
        h_last = hidden[arange_B, seq_lens, :]                  # (B, H)
        ab_last = self.mlp(h_last)
        a_last = F.softplus(ab_last[:, 0])
        b_last = F.softplus(ab_last[:, 1]).clamp(min=1e-6)
        t_last = event_times[arange_B, (seq_lens - 1).clamp(min=0)]
        dt_surv = (t1 - t_last).clamp(min=0.0)
        survival = (a_last / b_last) * (1.0 - torch.exp(-b_last * dt_surv))  # (B,)

        return tll - comp - survival

    def intensity_at(
        self,
        t_query: Tensor,
        history_times: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            t_query      : (M,)    query times (shifted, ≥ 0)
            history_times: (M, T)  all rows identical (single history expanded to M)
            history_mask : (M, T)  float mask
        Returns:
            (M,) intensity values
        """
        device = t_query.device
        T = history_times.shape[1]
        seq_len_1 = history_mask[0:1].sum(dim=1).long().clamp(min=1)  # (1,)

        # Encode the single unique history
        hidden_1, _ = self._encode(history_times[0:1], seq_len_1)  # (1, T+1, H)
        hidden_1 = hidden_1.squeeze(0)                              # (T+1, H)

        ht = history_times[0]   # (T,)
        hm = history_mask[0]    # (T,)

        # k_m = number of events strictly before t_query[m]
        k_m = (
            (ht.unsqueeze(0) < t_query.unsqueeze(1)).float() * hm.unsqueeze(0)
        ).sum(dim=1).long().clamp(max=T)  # (M,)

        h_select = hidden_1[k_m]  # (M, H)

        # Time of the last event before query (0 if no event yet)
        ht_ext = torch.cat([torch.zeros(1, device=device), ht], dim=0)  # (T+1,)
        t_last = ht_ext[k_m]                                             # (M,)

        dt = (t_query - t_last).clamp(min=0.0)    # (M,)
        ab = self.mlp(h_select)                    # (M, 2)
        a = F.softplus(ab[:, 0])
        b = F.softplus(ab[:, 1]).clamp(min=1e-6)
        return a * torch.exp(-b * dt)


# ---------------------------------------------------------------------------
# THP — Transformer Hawkes Process
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Time-based sinusoidal positional encoding (batch-first: B, S, D).

    Based on the standard THP sinusoidal time encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x : (B, S, D) — embedded tokens
            t : (B, S, 1) — absolute event times for positional encoding
        Returns:
            (B, S, D)
        """
        B, S, D = x.shape
        device = x.device
        div_term = torch.exp(
            torch.arange(0, D, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / D)
        )  # (D/2,)
        pe = torch.zeros(B, S, D, device=device)
        pe[..., 0::2] = torch.sin(t * div_term)
        pe[..., 1::2] = torch.cos(t * div_term)
        return self.dropout(x + pe)


def _subsequent_mask(sz: int, device: torch.device) -> Tensor:
    """Causal boolean mask: True = ignore (future positions).

    Both src mask and src_key_padding_mask use bool type to avoid the
    PyTorch deprecation warning about mismatched mask types.
    """
    return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)


def _key_padding_mask(seq_lens: Tensor, max_len: int, device: torch.device) -> Tensor:
    """src_key_padding_mask: True = pad (ignore), shape (B, T)."""
    idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
    return idx >= seq_lens.unsqueeze(1)                       # (B, T)


class THPTemporalProcess(nn.Module):
    """
    Transformer Hawkes Process (THP) temporal model.

    Intensity: λ(t) = softplus(f(h_n) + β · (t − t_n))
    where h_n = Transformer output encoding history through event n−1.

    Compensator: fixed 30-sample Monte Carlo over each inter-event interval
    and the final survival interval [t_N, t1].

    Reference: Zuo et al., "Transformer Hawkes Process", ICML 2020.
    Uses a fixed ``MC_SAMPLES`` Monte Carlo compensator.
    """

    MC_SAMPLES: int = 30

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(1, hidden_size, bias=False)
        self.pos_enc = _PositionalEncoding(hidden_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        # Disable nested-tensor fast path: MPS currently misses
        # aten::_nested_tensor_from_mask_left_aligned.
        try:
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.f = nn.Linear(hidden_size, 1)
        self.beta = nn.Parameter(torch.full((1,), 0.15))

    def _encode(self, event_times: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor]:
        """Transformer-encode sequences.

        Returns:
            hidden  : (B, T+1, H)  — [h_0=zeros, h_1, ..., h_N]
            delta_t : (B, T)
        """
        B, T = event_times.shape
        device = event_times.device

        zeros = torch.zeros(B, 1, device=device)
        prepended = torch.cat([zeros, event_times[:, :-1]], dim=1)
        delta_t = (event_times - prepended).clamp(min=0.0)  # (B, T)

        # Transformer uses batch-first: (B, T, *)
        x = delta_t.unsqueeze(-1)            # (B, T, 1)  — token: Δt
        t = event_times.unsqueeze(-1)        # (B, T, 1)  — PE: shifted absolute time

        x_emb = self.input_proj(x) * math.sqrt(self.hidden_size)  # (B, T, H)
        x_emb = self.pos_enc(x_emb, t)                             # (B, T, H)

        causal = _subsequent_mask(T, device)           # (T, T)
        pad_m = _key_padding_mask(seq_lens, T, device) # (B, T)

        out = self.transformer(
            x_emb, mask=causal, src_key_padding_mask=pad_m
        )  # (B, T, H)

        h0 = torch.zeros(B, 1, self.hidden_size, device=device)
        hidden = torch.cat([h0, out], dim=1)  # (B, T+1, H)
        return hidden, delta_t

    def logprob(
        self,
        event_times: Tensor,
        locations: Tensor,
        input_mask: Tensor,
        t0: Tensor,
        t1: Tensor,
    ) -> Tensor:
        """
        Args:
            event_times : (B, T)   shifted event times, padded
            locations   : (B, T, D) — unused
            input_mask  : (B, T)   float
            t0          : (B,)
            t1          : (B,)
        Returns:
            (B,) per-sequence temporal log-prob
        """
        B = event_times.shape[0]
        device = event_times.device
        seq_lens = input_mask.sum(dim=1).long().clamp(min=1)

        hidden, delta_t = self._encode(event_times, seq_lens)  # (B, T+1, H), (B, T)

        # befores[b, i] = f(h_i), scalar projection of hidden before event i
        befores = self.f(hidden).squeeze(-1)        # (B, T+1)
        bef_ev = befores[:, :-1]                    # (B, T) — h_0..h_{T-1}

        # Log-intensity: log softplus(f(h_i) + β·Δt_i)
        log_lambda = torch.log(
            F.softplus(bef_ev + self.beta * delta_t) + 1e-20
        )  # (B, T)
        tll = (log_lambda * input_mask).sum(dim=1)  # (B,)

        # MC compensator over observed intervals
        S = self.MC_SAMPLES
        rand_dt = torch.rand(S, B, delta_t.shape[1], device=device) * delta_t.unsqueeze(0)
        lambda_s = F.softplus(bef_ev.unsqueeze(0) + self.beta * rand_dt)  # (S, B, T)
        comp_per = (lambda_s * delta_t.unsqueeze(0)).mean(dim=0)           # (B, T)
        comp = (comp_per * input_mask).sum(dim=1)                          # (B,)

        # Survival compensator from last event to t1
        arange_B = torch.arange(B, device=device)
        bef_last = befores[arange_B, seq_lens]                             # (B,)
        t_last = event_times[arange_B, (seq_lens - 1).clamp(min=0)]
        dt_surv = (t1 - t_last).clamp(min=0.0)                            # (B,)
        rand_surv = torch.rand(S, B, device=device) * dt_surv.unsqueeze(0)  # (S, B)
        lambda_surv = F.softplus(bef_last.unsqueeze(0) + self.beta * rand_surv)  # (S, B)
        survival = (lambda_surv * dt_surv.unsqueeze(0)).mean(dim=0)        # (B,)

        return tll - comp - survival

    def intensity_at(
        self,
        t_query: Tensor,
        history_times: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            t_query      : (M,)    query times
            history_times: (M, T)  all rows identical (single history expanded to M)
            history_mask : (M, T)  float mask
        Returns:
            (M,) intensity values
        """
        device = t_query.device
        T = history_times.shape[1]
        seq_len_1 = history_mask[0:1].sum(dim=1).long().clamp(min=1)

        hidden_1, _ = self._encode(history_times[0:1], seq_len_1)  # (1, T+1, H)
        hidden_1 = hidden_1.squeeze(0)                              # (T+1, H)

        ht = history_times[0]
        hm = history_mask[0]

        k_m = (
            (ht.unsqueeze(0) < t_query.unsqueeze(1)).float() * hm.unsqueeze(0)
        ).sum(dim=1).long().clamp(max=T)  # (M,)

        h_select = hidden_1[k_m]                    # (M, H)
        f_select = self.f(h_select).squeeze(-1)     # (M,)

        ht_ext = torch.cat([torch.zeros(1, device=device), ht], dim=0)
        t_last = ht_ext[k_m]                        # (M,)
        dt = (t_query - t_last).clamp(min=0.0)

        return F.softplus(f_select + self.beta.squeeze() * dt)
