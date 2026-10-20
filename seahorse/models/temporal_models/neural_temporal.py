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

Marks
-----
Both models are natively *marked* in their source papers; this module restores that.
Passing ``n_marks=M`` (M > 0) adds a mark head and a third method::

    marked_logprob(event_times, locations, input_mask, t0, t1)
        -> (tll: (B,), mark_log_probs: (B, T, M))

``tll`` is the log-likelihood of the *ground* process (all marks pooled) and
``mark_log_probs[b, i, k] = log p(k | t_i, H_i)``, exactly normalised over k.
The joint marked log-likelihood is the sum of the two::

    Σ_i [ log λ(t_i) + log p(k_i | t_i, H_i) ] − ∫ λ(u) du

Each model uses the mark mechanism of its own paper (see the class docstrings).

``n_marks=0`` (the default) constructs **no** mark parameters at all, so both the
module's random-initialisation stream and its forward numerics are bit-identical
to the unmarked model. ``logprob``/``intensity_at`` are never touched by the mark
path — the marked likelihood is computed by ``marked_logprob`` alone.
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

    Marks (n_marks > 0): a softmax head off the recurrent hidden state,

        p(k | t_{n+1}, H) = softmax(V_y · h_n + b_y)_k

    which is exactly the marker model of Du et al. (2016, §3.2): RMTPP factorises
    the joint density as f*(t) · P(y | h_n), with the mark distribution depending
    on the history only (constant between events). Because the ground intensity
    λ(t) is left untouched, its compensator is unchanged and the joint likelihood
    is *exact* for this model class.

    Reference: Du et al., "Recurrent Marked Temporal Point Processes", KDD 2016.
    Architecture follows Rose-STL-Lab/AutoSTPP src/models/rmtpp.py.
    """

    #: The inter-event compensator ∫λ is available in closed form.
    COMPENSATOR: str = "closed_form"

    def __init__(self, *, hidden_size: int = 64, n_marks: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_marks = int(n_marks)
        self.gru = nn.GRU(
            input_size=1, hidden_size=hidden_size, batch_first=True, num_layers=1
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )
        # Mark head is created LAST and only when marked, so an unmarked model
        # draws exactly the same random initialisation as before this feature.
        if self.n_marks > 0:
            self.mark_head = nn.Linear(hidden_size, self.n_marks)
        else:
            self.mark_head = None

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

    def _event_terms(
        self,
        event_times: Tensor,
        input_mask: Tensor,
        t1: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Shared source of truth for the temporal likelihood.

        Returns:
            event_ll : (B, T) per-event log-density contributions,
                       ``log λ(t_i) − ∫_{t_{i-1}}^{t_i} λ``, already masked.
            survival : (B,) tail compensator ``∫_{t_N}^{t1} λ`` (zero under the
                       last-event convention, where ``t1`` is the last event time).

        ``logprob`` and ``event_logprob_matrix`` are both derived from this so they
        cannot drift apart: ``logprob == event_ll.sum(dim=1) − survival``.
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

        # Compensator over each inter-event interval (closed form)
        comp_per = (a / b) * (-torch.expm1(-b * delta_t))   # (B, T)

        event_ll = (log_lambda - comp_per) * input_mask          # (B, T)

        # Survival compensator from last event to t1 using h_N (after last event)
        arange_B = torch.arange(B, device=device)
        h_last = hidden[arange_B, seq_lens, :]                  # (B, H)
        ab_last = self.mlp(h_last)
        a_last = F.softplus(ab_last[:, 0])
        b_last = F.softplus(ab_last[:, 1]).clamp(min=1e-6)
        t_last = event_times[arange_B, (seq_lens - 1).clamp(min=0)]
        dt_surv = (t1 - t_last).clamp(min=0.0)
        survival = (a_last / b_last) * (-torch.expm1(-b_last * dt_surv))  # (B,)

        return event_ll, survival

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
        del locations, t0
        event_ll, survival = self._event_terms(event_times, input_mask, t1)
        return event_ll.sum(dim=1) - survival

    def event_logprob_matrix(
        self,
        event_times: Tensor,
        locations: Tensor,
        input_mask: Tensor,
        t0: Tensor,
        t1: Tensor,
    ) -> Tensor:
        """Per-event temporal log-prob contributions over successive intervals.

        Same contract as the parametric processes: entry ``[b, i]`` is
        ``log λ(t_i) − ∫_{t_{i-1}}^{t_i} λ``, masked to valid events. Without this
        method ``FactorizedEventModel`` has no per-event temporal term to report and
        the benchmark-facing ``nll_matrix`` degenerates to the spatial term alone.
        """
        del locations, t0
        event_ll, _survival = self._event_terms(event_times, input_mask, t1)
        return event_ll

    def marked_logprob(
        self,
        event_times: Tensor,
        locations: Tensor,
        input_mask: Tensor,
        t0: Tensor,
        t1: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Ground-process log-prob and per-event mark log-probabilities.

        RMTPP's mark model is a softmax off the recurrent state, and the joint
        density factorises as f*(t) · P(y | h). The mark factor therefore leaves
        the ground intensity — and its compensator — completely unchanged, so the
        temporal term here is *literally* ``self.logprob(...)``: marked and unmarked
        RMTPP share one temporal likelihood.

        Returns:
            tll            : (B,)      ground-process log-prob (identical to logprob)
            mark_log_probs : (B, T, M) log p(k | H_i), exactly normalised over k
        """
        if self.mark_head is None:
            raise RuntimeError(
                "RMTPPTemporalProcess.marked_logprob() requires n_marks > 0; "
                "this instance was built unmarked."
            )
        tll = self.logprob(event_times, locations, input_mask, t0, t1)  # (B,)

        seq_lens = input_mask.sum(dim=1).long().clamp(min=1)
        hidden, _ = self._encode(event_times, seq_lens)  # (B, T+1, H)
        # h_{i-1} encodes the history strictly before event i — the correct
        # conditioning state for p(k_i | H_i).
        h_prev = hidden[:, :-1, :]                        # (B, T, H)
        logits = self.mark_head(h_prev)                   # (B, T, M)
        mark_log_probs = F.log_softmax(logits, dim=-1)    # (B, T, M)
        return tll, mark_log_probs

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

    Marks (n_marks > 0): one intensity *per mark type*, exactly as in the THP
    paper (Zuo et al., 2020, Eq. 6), which defines a type-specific intensity
    λ_k(t) and pools them into the ground intensity λ(t) = Σ_k λ_k(t):

        λ_k(t) = softplus(f_k(h_n) + β_k · (t − t_n))
        p(k | t, H) = λ_k(t) / Σ_l λ_l(t)

    This is the normalise-the-per-mark-intensities recipe (also used by SMASH,
    Li et al., ICML 2024). Unlike RMTPP's softmax head, the resulting mark
    distribution is *time-varying* between events, since each mark carries its
    own decay β_k.

    The factorisation Σ_i log λ_{k_i}(t_i) − ∫Σ_k λ_k = [ground term] + [mark
    term] is algebraically exact; the joint likelihood is nevertheless
    **approximate** overall because the compensator ∫λ is Monte Carlo, exactly as
    in the unmarked model.

    Reference: Zuo et al., "Transformer Hawkes Process", ICML 2020.
    Uses a fixed ``MC_SAMPLES`` Monte Carlo compensator.
    """

    MC_SAMPLES: int = 30

    #: The compensator ∫λ is estimated by Monte Carlo, never in closed form.
    COMPENSATOR: str = "monte_carlo"

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        n_marks: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_marks = int(n_marks)

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
        # Unmarked: a single pooled intensity head, exactly as before.
        # Marked: one head per mark type; the pooled intensity is then Σ_k λ_k, so
        # the single head would be dead weight and is not created at all.
        if self.n_marks > 0:
            self.f = None
            self.f_marks = nn.Linear(hidden_size, self.n_marks)
            self.beta_marks = nn.Parameter(torch.full((self.n_marks,), 0.15))
        else:
            self.f = nn.Linear(hidden_size, 1)
            self.beta = nn.Parameter(torch.full((1,), 0.15))
            self.f_marks = None

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

    def _quadrature_fractions(self, shape: Tuple[int, ...], device: torch.device) -> Tensor:
        """Nodes in [0, 1) for the ``MC_SAMPLES``-point compensator quadrature.

        Training keeps the documented uniform Monte Carlo draws. Evaluation uses the
        deterministic composite-midpoint nodes ``(k + 0.5) / S`` instead, for two
        reasons: a reported likelihood must be reproducible (a resampled estimator
        makes ``nll_matrix`` and the scalar NLL disagree by pure noise, and makes the
        same checkpoint score differently on re-run), and for a fixed budget the
        midpoint rule is a strictly lower-variance estimator of the same integral.
        """
        S = self.MC_SAMPLES
        if self.training:
            return torch.rand((S,) + shape, device=device)
        nodes = (torch.arange(S, device=device, dtype=torch.float32) + 0.5) / float(S)
        return nodes.view((S,) + (1,) * len(shape)).expand((S,) + shape)

    def _event_terms(
        self,
        event_times: Tensor,
        input_mask: Tensor,
        t1: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Shared source of truth for the temporal likelihood.

        Returns:
            event_ll : (B, T) per-event log-density contributions,
                       ``log λ(t_i) − ∫_{t_{i-1}}^{t_i} λ``, already masked.
            survival : (B,) tail compensator ``∫_{t_N}^{t1} λ`` (zero under the
                       last-event convention).

        ``logprob`` and ``event_logprob_matrix`` are both derived from this, so a
        single set of quadrature nodes backs both and they cannot disagree.
        """
        if self.n_marks > 0:
            # Marked: the ground process is the mark-pooled one. Delegating keeps a
            # single source of truth, exactly as the unmarked branch does below.
            event_ll, survival, _marks = self._marked_event_terms(
                event_times, input_mask, t1
            )
            return event_ll, survival

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

        # Compensator over observed intervals
        frac = self._quadrature_fractions(tuple(delta_t.shape), device)  # (S, B, T)
        node_dt = frac * delta_t.unsqueeze(0)
        lambda_s = F.softplus(bef_ev.unsqueeze(0) + self.beta * node_dt)  # (S, B, T)
        comp_per = (lambda_s * delta_t.unsqueeze(0)).mean(dim=0)           # (B, T)

        event_ll = (log_lambda - comp_per) * input_mask                    # (B, T)

        # Survival compensator from last event to t1
        arange_B = torch.arange(B, device=device)
        bef_last = befores[arange_B, seq_lens]                             # (B,)
        t_last = event_times[arange_B, (seq_lens - 1).clamp(min=0)]
        dt_surv = (t1 - t_last).clamp(min=0.0)                            # (B,)
        frac_surv = self._quadrature_fractions((B,), device)                # (S, B)
        node_surv = frac_surv * dt_surv.unsqueeze(0)
        lambda_surv = F.softplus(bef_last.unsqueeze(0) + self.beta * node_surv)  # (S, B)
        survival = (lambda_surv * dt_surv.unsqueeze(0)).mean(dim=0)        # (B,)

        return event_ll, survival

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

        When marked, the ground intensity is the mark-pooled λ(t) = Σ_k λ_k(t), so
        this delegates to :meth:`marked_logprob` — every entry point (logprob,
        intensity_at, marked_logprob) then describes one and the same process.
        """
        if self.n_marks > 0:
            return self.marked_logprob(event_times, locations, input_mask, t0, t1)[0]

        del locations, t0
        event_ll, survival = self._event_terms(event_times, input_mask, t1)
        return event_ll.sum(dim=1) - survival

    def event_logprob_matrix(
        self,
        event_times: Tensor,
        locations: Tensor,
        input_mask: Tensor,
        t0: Tensor,
        t1: Tensor,
    ) -> Tensor:
        """Per-event temporal log-prob contributions over successive intervals.

        Same contract as the parametric processes: entry ``[b, i]`` is
        ``log λ(t_i) − ∫_{t_{i-1}}^{t_i} λ``, masked to valid events. Without this
        method ``FactorizedEventModel`` has no per-event temporal term to report and
        the benchmark-facing ``nll_matrix`` degenerates to the spatial term alone.
        """
        del locations, t0
        event_ll, _survival = self._event_terms(event_times, input_mask, t1)
        return event_ll

    def marked_logprob(
        self,
        event_times: Tensor,
        locations: Tensor,
        input_mask: Tensor,
        t0: Tensor,
        t1: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Ground-process log-prob and per-event mark log-probabilities.

        Mirrors :meth:`logprob` but with one intensity per mark type, pooled into
        the ground intensity λ(t) = Σ_k λ_k(t) (Zuo et al., 2020, Eq. 6). The
        encoder runs **once** so the ground term and the mark term are computed
        from the same hidden states (and, in train mode, the same dropout draw).

        Returns:
            tll            : (B,)      ground-process log-prob, λ = Σ_k λ_k
            mark_log_probs : (B, T, M) log p(k | t_i, H_i), exactly normalised
        """
        del locations, t0
        event_ll, survival, mark_log_probs = self._marked_event_terms(
            event_times, input_mask, t1
        )
        return event_ll.sum(dim=1) - survival, mark_log_probs

    def _marked_event_terms(
        self,
        event_times: Tensor,
        input_mask: Tensor,
        t1: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Marked counterpart of :meth:`_event_terms`, plus the mark factor.

        Returns:
            event_ll       : (B, T)    ``log λ(t_i) − ∫_{t_{i-1}}^{t_i} λ``, masked
            survival       : (B,)      ``∫_{t_N}^{t1} λ``
            mark_log_probs : (B, T, M) ``log p(k | t_i, H_i)``, exactly normalised

        The compensator uses the **same deterministic quadrature nodes** as the
        unmarked path. An earlier revision drew fresh Monte Carlo samples here,
        which meant ``logprob`` and ``event_logprob_matrix`` integrated over
        different nodes and disagreed by a random amount on every call — the exact
        failure mode ``_event_terms`` exists to prevent. Quadrature also makes the
        marked NLL reproducible, which frozen contracts require.
        """
        if self.f_marks is None:
            raise RuntimeError(
                "THPTemporalProcess.marked_logprob() requires n_marks > 0; "
                "this instance was built unmarked."
            )
        B = event_times.shape[0]
        device = event_times.device
        seq_lens = input_mask.sum(dim=1).long().clamp(min=1)

        hidden, delta_t = self._encode(event_times, seq_lens)  # (B, T+1, H), (B, T)

        # Per-mark projections: bef_all[b, i, k] = f_k(h_i)
        bef_all = self.f_marks(hidden)      # (B, T+1, M)
        bef_ev = bef_all[:, :-1, :]         # (B, T, M) — h_0..h_{T-1}

        # λ_k(t_i) = softplus(f_k(h_{i-1}) + β_k · Δt_i)
        lambda_k = F.softplus(
            bef_ev + self.beta_marks * delta_t.unsqueeze(-1)
        )  # (B, T, M)

        # log λ(t_i) = log Σ_k λ_k(t_i), and log p(k|t_i) = log λ_k − log λ.
        # Going through logsumexp makes the mark distribution normalise exactly
        # (the epsilon is inside both terms, so they cancel to machine precision).
        log_lambda_k = torch.log(lambda_k + 1e-20)                   # (B, T, M)
        log_lambda = torch.logsumexp(log_lambda_k, dim=-1)           # (B, T)
        mark_log_probs = log_lambda_k - log_lambda.unsqueeze(-1)     # (B, T, M)

        # Compensator over observed intervals, pooled over marks
        frac = self._quadrature_fractions(tuple(delta_t.shape), device)  # (S, B, T)
        node_dt = frac * delta_t.unsqueeze(0)
        lambda_s = F.softplus(
            bef_ev.unsqueeze(0) + self.beta_marks * node_dt.unsqueeze(-1)
        ).sum(dim=-1)                                                # (S, B, T)
        comp_per = (lambda_s * delta_t.unsqueeze(0)).mean(dim=0)     # (B, T)

        event_ll = (log_lambda - comp_per) * input_mask              # (B, T)

        # Survival compensator from last event to t1
        arange_B = torch.arange(B, device=device)
        bef_last = bef_all[arange_B, seq_lens, :]                    # (B, M)
        t_last = event_times[arange_B, (seq_lens - 1).clamp(min=0)]
        dt_surv = (t1 - t_last).clamp(min=0.0)                       # (B,)
        frac_surv = self._quadrature_fractions((B,), device)         # (S, B)
        node_surv = frac_surv * dt_surv.unsqueeze(0)
        lambda_surv = F.softplus(
            bef_last.unsqueeze(0) + self.beta_marks * node_surv.unsqueeze(-1)
        ).sum(dim=-1)                                                # (S, B)
        survival = (lambda_surv * dt_surv.unsqueeze(0)).mean(dim=0)  # (B,)

        return event_ll, survival, mark_log_probs

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

        ht_ext = torch.cat([torch.zeros(1, device=device), ht], dim=0)
        t_last = ht_ext[k_m]                        # (M,)
        dt = (t_query - t_last).clamp(min=0.0)

        if self.n_marks > 0:
            # Ground intensity of the marked process: λ(t) = Σ_k λ_k(t)
            f_marks_sel = self.f_marks(h_select)    # (M, K)
            return F.softplus(
                f_marks_sel + self.beta_marks * dt.unsqueeze(-1)
            ).sum(dim=-1)                            # (M,)

        f_select = self.f(h_select).squeeze(-1)     # (M,)
        return F.softplus(f_select + self.beta.squeeze() * dt)
