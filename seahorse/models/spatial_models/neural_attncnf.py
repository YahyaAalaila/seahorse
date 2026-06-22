"""Faithful attentive CNF spatial decoder for the shared Neural STPP family."""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import Decoder
from ..model_registry import register_spatial
from .neural_jumpcnf import (
    HAS_TORCHDIFFEQ,
    _build_fc_odefunc,
    _euler_cnf_solve,
    _gaussian_loglik,
    _normalize_hidden_dims,
    _odeint_adj,
    _odeint_std,
)


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


def _rms_norm(tensor: Tensor) -> Tensor:
    return tensor.pow(2).mean().sqrt()


def max_rms_norm(shapes):
    def _norm(tensor):
        total = 0
        out = []
        for shape in shapes:
            numel = 1
            for d in shape:
                numel *= int(d)
            next_total = total + numel
            out.append(_rms_norm(tensor[total:next_total]))
            total = next_total
        if total != tensor.numel():
            raise ValueError("Shapes do not total to the full size of the tensor.")
        return max(out)

    return _norm


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
        nonself_connections: bool = False,
        energy_regularization: float = 0.0,
        jacnorm_regularization: float = 0.0,
        use_adjoint: bool = True,
    ):
        super().__init__()
        self.func = func
        self.dim = dim
        self.tol = tol
        self.method = method
        self.nonself_connections = bool(nonself_connections)
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
        e = torch.randn_like(x)[:, : self.dim]
        energy = torch.zeros(1, device=x.device, dtype=x.dtype)
        jacnorm = torch.zeros(1, device=x.device, dtype=x.dtype)
        initial_state = (t0, t1, e, x, logpx, energy, jacnorm)

        if intermediate_states > 1:
            tt = torch.linspace(
                self.start_time,
                self.end_time,
                intermediate_states,
                device=t0.device,
                dtype=t0.dtype,
            )
        else:
            tt = torch.tensor(
                [self.start_time, self.end_time],
                device=t0.device,
                dtype=t0.dtype,
            )

        if HAS_TORCHDIFFEQ:
            odeint_fn = _odeint_adj if self.use_adjoint else _odeint_std
            solution = odeint_fn(
                self,
                initial_state,
                tt,
                rtol=tol,
                atol=tol,
                method=method,
            )
        else:  # pragma: no cover
            solution = _euler_cnf_solve(
                self,
                initial_state,
                tt,
                n_steps=max(50, intermediate_states * 10 or 50),
            )

        if intermediate_states > 1:
            y = solution[3]
            _, _, _, _, logpy, energy, jacnorm = tuple(s[-1] for s in solution)
        else:
            _, _, _, y, logpy, energy, jacnorm = tuple(s[-1] for s in solution)

        regularization = (
            self.energy_regularization * (energy - energy.detach())
            + self.jacnorm_regularization * (jacnorm - jacnorm.detach())
        )
        return y, logpy, regularization.reshape(
            () if regularization.numel() == 1 else regularization.shape
        )

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

            if self.nonself_connections:
                dx_div = self.func(t, x, rm_nonself_grads=True)
                dx_div = dx_div * ratio.reshape(-1, *([1] * (x.ndim - 1)))
            else:
                dx_div = dx

            if not self.training:
                div = _divergence_bf(dx_div[:, : self.dim], x, self.training)
            else:
                vjp = torch.autograd.grad(
                    dx_div[:, : self.dim],
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


class SelfonlyGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_logits):
        return attn_logits

    @staticmethod
    def backward(ctx, grads):
        grads = torch.diagonal(grads, dim1=0, dim2=1)
        grads = torch.diag_embed(grads).permute(2, 3, 0, 1)
        return grads


def _update_attn_weights(attn_weights: Tensor, attn_multiplier: Optional[Tensor]) -> Tensor:
    if attn_multiplier is not None:
        attn_weights = attn_weights * attn_multiplier[..., None]
        attn_weights = attn_weights / attn_weights.sum(1, keepdim=True)
    return attn_weights


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        rm_nonself_grads: bool = False,
        attn_multiplier: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        t_steps, batch_size, _ = x.shape
        q, k, v = map(
            lambda a: a.reshape(t_steps, batch_size, self.num_heads, self.head_dim),
            torch.split(self.in_proj(x), self.embed_dim, dim=-1),
        )
        attn_logits = torch.einsum("tbhd,sbhd->tsbh", q, k) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask[..., None, None]
        attn_weights = F.softmax(attn_logits, dim=1)
        attn_weights = _update_attn_weights(attn_weights, attn_multiplier)
        attn = torch.einsum("tsbh,sbhd->tbhd", attn_weights, v).reshape(
            t_steps, batch_size, -1
        )

        if rm_nonself_grads:
            attn_logits_keyonly = torch.einsum(
                "tbhd,sbhd->tsbh", q.detach(), k
            ) / math.sqrt(self.head_dim)
            attn_logits_queryonly = torch.einsum(
                "tbhd,sbhd->tsbh", q, k.detach()
            ) / math.sqrt(self.head_dim)

            attn_logits_keyonly = SelfonlyGradients.apply(attn_logits_keyonly)
            attn_logits = attn_logits_queryonly + (
                attn_logits_keyonly - attn_logits_keyonly.detach()
            )
            if attn_mask is not None:
                attn_logits = attn_logits + attn_mask[..., None, None]
            attn_weights = F.softmax(attn_logits, dim=1)
            attn_weights = _update_attn_weights(attn_weights, attn_multiplier)

            selfonly_mask = ~(
                torch.triu(torch.ones(t_steps, t_steps), diagonal=1)
                + torch.tril(torch.ones(t_steps, t_steps), diagonal=-1)
            ).bool().to(attn_weights.device)
            selfonly_attn_weights = attn_weights * selfonly_mask[..., None, None]
            attn_vpath = torch.einsum(
                "tsbh,sbhd->tbhd", selfonly_attn_weights.detach(), v
            ).reshape(t_steps, batch_size, -1)
            attn_spath = torch.einsum(
                "tsbh,sbhd->tbhd", attn_weights, v.detach()
            ).reshape(t_steps, batch_size, -1)
            modified_attn = attn_spath + (attn_vpath - attn_vpath.detach())
            attn = attn.detach() + (modified_attn - modified_attn.detach())

        attn = self.out_proj(attn)
        return attn, attn_weights.detach()


class L2MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.q_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.v_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_weight.view(self.embed_dim, self.embed_dim))
        nn.init.xavier_uniform_(self.v_weight.view(self.embed_dim, self.embed_dim))

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        rm_nonself_grads: bool = False,
        attn_multiplier: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        t_steps, batch_size, _ = x.shape
        q = k = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        squared_dist = (
            torch.einsum("tbhd,tbhd->tbh", q, q).unsqueeze(1)
            + torch.einsum("sbhd,sbhd->sbh", k, k).unsqueeze(0)
            - 2 * torch.einsum("tbhd,sbhd->tsbh", q, k)
        )
        attn_logits = -squared_dist / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask[..., None, None]
        attn_weights = F.softmax(attn_logits, dim=1)
        attn_weights = _update_attn_weights(attn_weights, attn_multiplier)
        a = torch.einsum("mhd,nhd->hmn", self.q_weight, self.q_weight) / math.sqrt(
            self.head_dim
        )
        xa = torch.einsum("tbm,hmn->tbhn", x, a)
        pxa = torch.einsum("tsbh,sbhm->tbhm", attn_weights, xa)

        if rm_nonself_grads:
            q_detach = q.detach()
            k_detach = k.detach()
            attn_logits_keyonly = -(
                torch.einsum("tbhd,tbhd->tbh", q_detach, q_detach).unsqueeze(1)
                + torch.einsum("sbhd,sbhd->sbh", k, k).unsqueeze(0)
                - 2 * torch.einsum("tbhd,sbhd->tsbh", q_detach, k)
            ) / math.sqrt(self.head_dim)
            attn_logits_queryonly = -(
                torch.einsum("tbhd,tbhd->tbh", q, q).unsqueeze(1)
                + torch.einsum("sbhd,sbhd->sbh", k_detach, k_detach).unsqueeze(0)
                - 2 * torch.einsum("tbhd,sbhd->tsbh", q, k_detach)
            ) / math.sqrt(self.head_dim)
            attn_logits_keyonly = SelfonlyGradients.apply(attn_logits_keyonly)
            attn_logits = attn_logits_queryonly + (
                attn_logits_keyonly - attn_logits_keyonly.detach()
            )
            if attn_mask is not None:
                attn_logits = attn_logits + attn_mask[..., None, None]
            attn_weights = F.softmax(attn_logits, dim=1)
            attn_weights = _update_attn_weights(attn_weights, attn_multiplier)

            selfonly_mask = ~(
                torch.triu(torch.ones(t_steps, t_steps), diagonal=1)
                + torch.tril(torch.ones(t_steps, t_steps), diagonal=-1)
            ).bool().to(attn_weights.device)
            selfonly_attn_weights = attn_weights * selfonly_mask[..., None, None]
            pxa_vpath = torch.einsum(
                "tsbh,sbhm->tbhm", selfonly_attn_weights.detach(), xa
            )
            pxa_spath = torch.einsum(
                "tsbh,sbhm->tbhm", attn_weights, xa.detach()
            )
            modified_pxa = pxa_spath + (pxa_vpath - pxa_vpath.detach())
            pxa = pxa.detach() + (modified_pxa - modified_pxa.detach())

        pxav = torch.einsum("tbhm,mhd->tbhd", pxa, self.v_weight).reshape(
            t_steps, batch_size, self.embed_dim
        )
        return self.out_proj(pxav), attn_weights.detach()


class EventTimeEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        div_term = torch.exp(
            torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, event_times: Tensor) -> Tensor:
        n_batch, steps = event_times.shape
        pe = torch.zeros(n_batch, steps, self.dim, device=event_times.device, dtype=event_times.dtype)
        pe[:, :, 0::2] = torch.sin(event_times[..., None] * self.div_term)
        pe[:, :, 1::2] = torch.cos(event_times[..., None] * self.div_term)
        return pe


class TanhGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.weight) * x


class ActNorm(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer("initialized", torch.tensor(0))

    def forward(self, x: Tensor, logpx: Optional[Tensor] = None):
        if not bool(self.initialized.item()):
            with torch.no_grad():
                x_ = x.reshape(-1, x.shape[-1])
                batch_mean = torch.mean(x_, dim=0)
                batch_var = torch.var(x_, dim=0, unbiased=False)
                batch_var = torch.max(
                    batch_var,
                    torch.tensor(0.2, device=batch_var.device, dtype=batch_var.dtype),
                )
                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)

        bias = self.bias.expand_as(x)
        weight = self.weight.expand_as(x)
        y = (x + bias) * torch.exp(weight)
        if logpx is None:
            return y
        return y, logpx - self._logdetgrad(x)

    def inverse(self, y: Tensor, logpy: Optional[Tensor] = None):
        if not bool(self.initialized.item()):
            raise RuntimeError("ActNorm.inverse requires initialized parameters.")
        bias = self.bias.expand_as(y)
        weight = self.weight.expand_as(y)
        x = y * torch.exp(-weight) - bias
        if logpy is None:
            return x
        return x, logpy + self._logdetgrad(x)

    def _logdetgrad(self, x: Tensor) -> Tensor:
        shape = (1,) * (x.ndim - 1) + (self.num_features,)
        return self.weight.view(*shape).expand_as(x).contiguous().view(x.size(0), -1).sum(
            1, keepdim=True
        )


class SelfAttentiveODEFunc(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dims: Sequence[int],
        aux_dim: int,
        actfn: str,
        time_offset: float,
        *,
        nblocks: int = 2,
        num_heads: int = 4,
        l2_attn: bool = False,
        layer_type: str = "concat",
    ):
        super().__init__()
        if len(hidden_dims) < 2:
            raise ValueError("SelfAttentiveCNF requires at least two hidden_dims values.")
        self.dim = dim
        self.aux_dim = aux_dim
        self.time_offset = time_offset
        self.num_heads = num_heads

        mid_idx = int(math.ceil(len(hidden_dims) / 2))
        self.embed_dim = hidden_dims[mid_idx]
        self.embedding = _build_fc_odefunc(
            dim + aux_dim,
            hidden_dims[:mid_idx],
            out_dim=self.embed_dim,
            layer_type=layer_type,
            actfn=actfn,
            zero_init=False,
        )

        mha = L2MultiheadAttention if l2_attn else MultiheadAttention
        self.self_attns = nn.ModuleList(
            [mha(self.embed_dim, num_heads=num_heads) for _ in range(nblocks)]
        )
        self.attn_actnorms = nn.ModuleList([ActNorm(self.embed_dim) for _ in range(nblocks)])
        self.fcs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim * 4),
                    nn.Softplus(),
                    nn.Linear(self.embed_dim * 4, self.embed_dim),
                )
                for _ in range(nblocks)
            ]
        )
        self.fc_actnorms = nn.ModuleList([ActNorm(self.embed_dim) for _ in range(nblocks)])
        self.attn_gates = nn.ModuleList([TanhGate() for _ in range(nblocks)])
        self.fc_gates = nn.ModuleList([TanhGate() for _ in range(nblocks)])
        self.output_proj = _build_fc_odefunc(
            self.embed_dim,
            hidden_dims[mid_idx:],
            out_dim=self.dim,
            layer_type=layer_type,
            actfn=actfn,
            zero_init=True,
        )
        self.shape = None

    def set_shape(self, shape) -> None:
        self.shape = shape

    @staticmethod
    def _create_self_attn_mask(steps: int) -> Tensor:
        return torch.triu(torch.ones(steps, steps), diagonal=1) * -1e12

    def forward(
        self,
        t: Tensor,
        state: Tensor,
        *,
        rm_nonself_grads: bool = False,
    ) -> Tensor:
        if self.shape is None:
            raise RuntimeError("SelfAttentiveODEFunc.set_shape must be called before forward().")
        steps, batch_size, _ = self.shape
        x = state[:, : self.dim]
        a = state[:, max(self.dim + 1, state.shape[-1] - self.aux_dim) :]
        x = torch.cat([x, a], dim=-1)
        x = self.embedding(t, x)
        x = x.reshape(steps, batch_size, self.embed_dim)

        attn_mask = self._create_self_attn_mask(steps).to(x)
        for norm0, self_attn, gate0, norm1, fc, gate1 in zip(
            self.attn_actnorms,
            self.self_attns,
            self.attn_gates,
            self.fc_actnorms,
            self.fcs,
            self.fc_gates,
        ):
            h, _ = self_attn(
                norm0(x),
                attn_mask=attn_mask,
                rm_nonself_grads=rm_nonself_grads,
            )
            x = x + gate0(h)
            x = x + gate1(fc(norm1(x)))

        dx = self.output_proj(t, x.reshape(-1, self.embed_dim))
        dh = torch.zeros_like(state[:, self.dim :])
        return torch.cat([dx, dh], dim=1)


@register_spatial("neural_attncnf")
class NeuralAttnCNFSpatial(Decoder):
    """Faithful attentive CNF port for the shared-hidden Neural STPP family."""

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
        l2_attn: bool = False,
        naive_hutch: bool = False,
        lowvar_trace: Optional[bool] = None,
        tol: float = 1e-4,
        otreg_strength: float = 1e-4,
        nblocks: int = 2,
        num_heads: int = 4,
        field_cov_dim: int = 0,
        **kwargs,
    ):
        del field_cov_dim, kwargs
        super().__init__(hidden_dim=hidden_dim, spatial_dim=spatial_dim)
        self.dim = spatial_dim
        self.hidden_dims = _normalize_hidden_dims(hidden_dims)
        if len(self.hidden_dims) < 2:
            raise ValueError("NeuralAttnCNFSpatial requires at least two hidden_dims values.")
        self.aux_hidden_dim = int(spatial_aux_dim or hidden_dim // 2)
        if self.aux_hidden_dim <= 0:
            raise ValueError("NeuralAttnCNFSpatial requires spatial_aux_dim > 0")
        self.l2_attn = bool(l2_attn)
        self.lowvar_trace = bool((not naive_hutch) if lowvar_trace is None else lowvar_trace)
        self.naive_hutch = not self.lowvar_trace
        self.nblocks = int(nblocks)
        self.num_heads = int(num_heads)
        if self.nblocks <= 0:
            raise ValueError("NeuralAttnCNFSpatial requires nblocks > 0")
        if self.num_heads <= 0:
            raise ValueError("NeuralAttnCNFSpatial requires num_heads > 0")
        self._energy_reg: Tensor | float = 0.0

        mid_idx = int(math.ceil(len(self.hidden_dims) / 2))
        self.t_embedding_dim = int(self.hidden_dims[mid_idx])
        self.t_embedding = EventTimeEncoding(self.t_embedding_dim)

        self.odefunc = SelfAttentiveODEFunc(
            spatial_dim,
            self.hidden_dims,
            self.aux_hidden_dim + self.t_embedding_dim,
            actfn,
            self.time_offset,
            nblocks=self.nblocks,
            num_heads=self.num_heads,
            l2_attn=self.l2_attn,
            layer_type=layer_type,
        )
        self.cnf = TimeVariableCNF(
            self.odefunc,
            spatial_dim,
            tol=tol,
            method="dopri5",
            nonself_connections=self.lowvar_trace,
            energy_regularization=otreg_strength,
            jacnorm_regularization=otreg_strength,
            use_adjoint=True,
        )

        base_odefunc = _build_fc_odefunc(
            dim=spatial_dim,
            hidden_dims=self.hidden_dims,
            layer_type=layer_type,
            actfn=actfn,
            zero_init=zero_init,
        )
        self.base_cnf = TimeVariableCNF(
            base_odefunc,
            spatial_dim,
            tol=1e-6,
            method="dopri5",
            energy_regularization=1e-4,
            jacnorm_regularization=1e-4,
            use_adjoint=True,
        )

        self.base_dist_params = nn.Sequential(
            nn.Linear(self.aux_hidden_dim + self.t_embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, spatial_dim * 2),
        )
        self.base_dist_params[-1].weight.data.fill_(0.0)
        self.base_dist_params[-1].bias.data.fill_(0.0)

    def _select_aux_tail(self, z_seq: Tensor) -> Tensor:
        if z_seq.shape[-1] < self.aux_hidden_dim:
            raise ValueError(
                f"NeuralAttnCNFSpatial expected hidden width >= {self.aux_hidden_dim}, "
                f"got {z_seq.shape[-1]}"
            )
        return z_seq[..., -self.aux_hidden_dim :]

    def log_prob(self, z, t, s, t_prev, x_field=None):
        raise NotImplementedError("NeuralAttnCNFSpatial.SEQUENCE_COUPLED=True; use sequence_nll().")

    def nll(self, z, t, s, t_prev, x_field=None):
        raise NotImplementedError("NeuralAttnCNFSpatial.SEQUENCE_COUPLED=True; use sequence_nll().")

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
            raise ValueError("AttnCNF event_times and input_mask must have the same shape.")
        if event_times.shape[:2] != spatial_locations.shape[:2]:
            raise ValueError("AttnCNF event_times and spatial_locations must align.")
        if aux_state is not None and event_times.shape[:2] != aux_state.shape[:2]:
            raise ValueError("AttnCNF aux_state must align with event_times.")

        n_batch, steps, dim = spatial_locations.shape
        if dim != self.spatial_dim:
            raise ValueError(f"Expected spatial_dim={self.spatial_dim}, got {dim}")

        spatial_locations = spatial_locations.clone().requires_grad_(True)
        t_embed = self.t_embedding(event_times) / math.sqrt(self.t_embedding_dim)

        if aux_state is not None:
            inputs = [spatial_locations, aux_state, t_embed]
        else:
            inputs = [spatial_locations, t_embed]

        inputs_t = [inp.transpose(0, 1) for inp in inputs]
        norm_fn = max_rms_norm([a.shape for a in inputs_t])
        x = torch.cat(inputs_t, dim=-1)
        self.odefunc.set_shape(x.shape)

        x = x.reshape(steps * n_batch, -1)
        event_times_flat = event_times.transpose(0, 1).reshape(steps * n_batch)
        t0 = event_times_flat + self.time_offset
        t1 = torch.zeros_like(event_times_flat) + self.time_offset

        z, delta_logp, reg_main = self.cnf.integrate(
            t0,
            t1,
            x,
            torch.zeros_like(event_times_flat),
            norm=norm_fn,
        )
        z = z[:, : self.dim]

        base_t = torch.zeros_like(event_times_flat)
        z, delta_logp, reg_base = self.base_cnf.integrate(t1, base_t, z, delta_logp)

        if aux_state is not None:
            cond_inputs = [self._select_aux_tail(aux_state), t_embed]
        else:
            cond_inputs = [t_embed]
        cond = torch.cat(cond_inputs, dim=-1)
        cond = torch.where(
            input_mask[..., None].expand_as(cond).bool(),
            cond,
            torch.zeros_like(cond),
        )
        cond = cond.transpose(0, 1).reshape(steps * n_batch, -1)

        z_params = self.base_dist_params(cond)
        z_mean, z_logstd = torch.split(z_params, self.dim, dim=-1)
        logpz = _gaussian_loglik(z, z_mean, z_logstd).sum(dim=-1)
        logpx = (logpz - delta_logp).reshape(steps, n_batch).transpose(0, 1)

        reg_total = torch.as_tensor(reg_main, device=z.device, dtype=z.dtype).reshape(())
        reg_total = reg_total + torch.as_tensor(reg_base, device=z.device, dtype=z.dtype).reshape(())
        self._energy_reg = reg_total
        return torch.where(input_mask.bool(), logpx, torch.zeros_like(logpx))

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
        logpx = self.sequence_logprob(
            event_times=t_seq.squeeze(-1),
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
        steps, dim = event_locs.shape

        def logprob_fn(s: Tensor) -> Tensor:
            batch_size = s.shape[0]
            bsz_event_times = event_times.unsqueeze(0).expand(batch_size, steps)
            bsz_event_times = torch.cat(
                [
                    bsz_event_times,
                    torch.full(
                        (batch_size, 1),
                        float(t_query),
                        device=bsz_event_times.device,
                        dtype=bsz_event_times.dtype,
                    ),
                ],
                dim=1,
            )
            bsz_spatial_locations = event_locs.unsqueeze(0).expand(batch_size, steps, dim)
            bsz_spatial_locations = torch.cat(
                [bsz_spatial_locations, s.reshape(batch_size, 1, dim)],
                dim=1,
            )
            bsz_aux_state = z_aug.reshape(1, steps + 1, -1).expand(batch_size, -1, -1)
            return self.sequence_logprob(
                event_times=bsz_event_times,
                spatial_locations=bsz_spatial_locations,
                input_mask=None,
                aux_state=bsz_aux_state,
            )[:, -1]

        return logprob_fn


__all__ = [
    "ActNorm",
    "EventTimeEncoding",
    "L2MultiheadAttention",
    "MultiheadAttention",
    "NeuralAttnCNFSpatial",
    "SelfAttentiveODEFunc",
]
