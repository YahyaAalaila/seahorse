"""Conditional-GMM spatial decoder for the canonical NJSDE preset."""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import Decoder
from ..model_registry import register_spatial


_ACTFNS = {
    "softplus": nn.Softplus,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


def _build_mlp(
    dim: int,
    hidden_dims: Sequence[int] | None,
    out_dim: int,
    *,
    actfn: str = "softplus",
) -> nn.Sequential:
    hidden_dims = list(hidden_dims or [])
    if actfn not in _ACTFNS:
        raise ValueError(f"Unknown ConditionalGMM actfn {actfn!r}")
    if hidden_dims:
        dims = [dim] + hidden_dims
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(_ACTFNS[actfn]())
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        return nn.Sequential(*layers)
    return nn.Sequential(nn.Linear(dim, out_dim))


def _normalize_hidden_dims(hidden_dims: Sequence[int] | int | str | None) -> list[int]:
    if hidden_dims is None:
        return [64, 64, 64]
    if isinstance(hidden_dims, int):
        return [hidden_dims]
    if isinstance(hidden_dims, str):
        raw = hidden_dims.strip().strip("[]()")
        if not raw:
            raise ValueError("ConditionalGMM hidden_dims string cannot be empty")
        if "-" in raw:
            parts = [p.strip() for p in raw.split("-")]
        elif "," in raw:
            parts = [p.strip() for p in raw.split(",")]
        else:
            parts = [raw]
        dims = [int(p) for p in parts if p]
    else:
        dims = [int(h) for h in hidden_dims]
    if not dims:
        return []
    return dims


def _gaussian_loglik(z: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    c = torch.tensor(math.log(2.0 * math.pi), device=z.device, dtype=z.dtype)
    inv_sigma = torch.exp(-log_std)
    delta = (z - mean) * inv_sigma
    return -0.5 * (delta * delta + 2.0 * log_std + c)


def _gmm_loglik(z: Tensor, params: Tensor, n_mixtures: int) -> Tensor:
    params = params.reshape(*z.shape, 3, n_mixtures)
    mix_logits = params[..., 0, :]
    means = params[..., 1, :]
    log_stds = params[..., 2, :]
    mix_logprobs = mix_logits - torch.logsumexp(mix_logits, dim=-1, keepdim=True)
    logprobs = _gaussian_loglik(z[..., None], means, log_stds)
    return torch.logsumexp(mix_logprobs + logprobs, dim=-1)


def _gmm_sample(params: Tensor, n_mixtures: int) -> Tensor:
    params = params.reshape(-1, 3, n_mixtures)
    mix_logits = params[:, 0, :]
    means = params[:, 1, :]
    log_stds = params[:, 2, :]
    mix_probs = torch.softmax(mix_logits, dim=-1)
    component = torch.multinomial(mix_probs, 1).reshape(-1)
    chosen_mean = means.gather(1, component.unsqueeze(-1)).squeeze(-1)
    chosen_log_std = log_stds.gather(1, component.unsqueeze(-1)).squeeze(-1)
    return torch.randn_like(chosen_mean) * torch.exp(chosen_log_std) + chosen_mean


@register_spatial("conditional_gmm")
class ConditionalGMMSpatial(Decoder):
    """NJSDE conditional Gaussian mixture spatial decoder.

    The decoder receives the spatial auxiliary slice from the shared neural
    temporal backbone and emits a per-event Gaussian mixture distribution.
    """

    SEQUENCE_COUPLED = True
    USES_NEURAL_AUX_STATE = True

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        *,
        spatial_aux_dim: Optional[int] = None,
        hidden_dims: Sequence[int] | None = None,
        n_mixtures: int = 5,
        actfn: str = "softplus",
        field_cov_dim: int = 0,
        **kwargs,
    ):
        del field_cov_dim, kwargs
        super().__init__(hidden_dim=hidden_dim, spatial_dim=spatial_dim)
        if spatial_aux_dim is None:
            spatial_aux_dim = hidden_dim
        self.aux_hidden_dim = int(spatial_aux_dim)
        if self.aux_hidden_dim <= 0:
            raise ValueError("ConditionalGMMSpatial requires spatial_aux_dim > 0")
        self.n_mixtures = int(n_mixtures)
        if self.n_mixtures <= 0:
            raise ValueError("ConditionalGMMSpatial requires n_mixtures > 0")
        self.hidden_dims = _normalize_hidden_dims(hidden_dims)
        self.actfn = actfn
        out_dim = spatial_dim * self.n_mixtures * 3
        self.gmm_params = _build_mlp(
            self.aux_hidden_dim,
            self.hidden_dims,
            out_dim,
            actfn=actfn,
        )

    def _select_aux(self, z: Tensor) -> Tensor:
        if z.shape[-1] < self.aux_hidden_dim:
            raise ValueError(
                f"ConditionalGMMSpatial expected aux state width >= {self.aux_hidden_dim}, "
                f"got {z.shape[-1]}"
            )
        return z[..., -self.aux_hidden_dim :]

    def _params_from_aux(self, aux_state: Tensor) -> Tensor:
        flat_aux = aux_state.reshape(-1, self.aux_hidden_dim)
        params = self.gmm_params(flat_aux)
        return params.reshape(*aux_state.shape[:-1], self.spatial_dim * 3 * self.n_mixtures)

    def log_prob(self, z, t, s, t_prev, x_field=None):
        del t, t_prev, x_field
        aux = self._select_aux(z)
        params = self._params_from_aux(aux)
        return _gmm_loglik(s, params, self.n_mixtures).sum(dim=-1)

    def nll(self, z, t, s, t_prev, x_field=None):
        return -self.log_prob(z, t, s, t_prev, x_field=x_field)

    def sequence_logprob(
        self,
        *,
        z_seq: Tensor,
        s_seq: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        aux_state = self._select_aux(z_seq)
        params = self._params_from_aux(aux_state)
        logpx = _gmm_loglik(s_seq, params, self.n_mixtures).sum(dim=-1)
        if mask is None:
            return logpx
        return torch.where(mask.bool(), logpx, torch.zeros_like(logpx))

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
        del t_seq, t_prev_seq, lengths, kwargs
        return -self.sequence_logprob(z_seq=z_seq, s_seq=s_seq, mask=mask)

    def sample_spatial(
        self,
        nsamples: int,
        event_times: Tensor,
        spatial_locations: Tensor,
        input_mask: Optional[Tensor] = None,
        aux_state: Optional[Tensor] = None,
    ) -> Tensor:
        del event_times, spatial_locations
        if aux_state is None:
            raise ValueError("ConditionalGMMSpatial.sample_spatial requires aux_state.")
        if input_mask is None:
            input_mask = torch.ones(aux_state.shape[:2], device=aux_state.device, dtype=aux_state.dtype)

        aux = self._select_aux(aux_state)
        params = self._params_from_aux(aux)
        bsz, steps = aux.shape[:2]
        params = params.reshape(bsz * steps, self.spatial_dim, 3, self.n_mixtures)
        params = params.unsqueeze(0).expand(nsamples, -1, -1, -1, -1)
        samples = _gmm_sample(
            params.reshape(-1, 3, self.n_mixtures),
            self.n_mixtures,
        ).reshape(nsamples, bsz, steps, self.spatial_dim)
        return samples * input_mask.unsqueeze(0).unsqueeze(-1)

    def conditional_logprob_fn(
        self,
        t_query: float,
        event_times: Tensor,
        event_locs: Tensor,
        z_aug: Tensor,
    ) -> Callable[[Tensor], Tensor]:
        del t_query, event_times, event_locs
        query_aux = self._select_aux(z_aug[-1:, :])

        def logprob_fn(s: Tensor) -> Tensor:
            aux = query_aux.expand(s.shape[0], -1)
            params = self._params_from_aux(aux)
            return _gmm_loglik(s, params, self.n_mixtures).sum(dim=-1)

        return logprob_fn


__all__ = ["ConditionalGMMSpatial"]
