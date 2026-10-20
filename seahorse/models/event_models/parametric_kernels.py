"""Parametric Hawkes and Poisson event models with exact likelihoods.

Both models share one likelihood implementation: a homogeneous background rate
``mu`` with a Gaussian spatial density, plus (for the Hawkes model) an
exponential temporal trigger ``eta * q * exp(-q * age)`` whose spatial
displacement law ``p(displacement)`` is supplied by the subclass.  The
displacement law integrates to one, so the next-event survival term is
analytic and the per-event NLL decomposes exactly into eventwise terms.

Presets built on this module:

* ``static_hawkes``: Gaussian displacement independent of the event age.
* ``gaussian_poisson``: no triggering (``eta = 0``); a homogeneous temporal
  Poisson process with a normalised Gaussian spatial background.

These are the classical baselines of the benchmark tables; the shared base is
not a preset of its own.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..abstractions import EventCapabilities, EventModel, StateContext
from ..model_registry import register_event


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1.0e-8)
    return math.log(math.expm1(value))


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


class ParametricHawkesEventModel(EventModel):
    """Exponential-decay Hawkes process with a Gaussian background.

    Subclasses provide :meth:`transition_log_density`, the log density of the
    offspring displacement given the event age.
    """

    def __init__(
        self,
        *,
        spatial_dim: int,
        eta_cap: float = 0.95,
        min_history: int = 1,
        regularization_weight: float = 1.0e-5,
        mu_init: float = 2.0,
        eta_init: float = 0.45,
        killing_rate_init: float = 1.0,
        background_std_init: float = 1.0,
        **_: Any,
    ) -> None:
        super().__init__()
        if spatial_dim < 1:
            raise ValueError("spatial_dim must be positive.")
        if not 0.0 < eta_cap < 1.0:
            raise ValueError("eta_cap must lie strictly between zero and one.")

        self.spatial_dim = int(spatial_dim)
        self.eta_cap = float(eta_cap)
        self.min_history = max(int(min_history), 1)
        self.regularization_weight = max(float(regularization_weight), 0.0)

        self.raw_mu = nn.Parameter(torch.tensor(_inverse_softplus(mu_init)))
        self.raw_eta = nn.Parameter(torch.tensor(_logit(eta_init / self.eta_cap)))
        self.raw_killing_rate = nn.Parameter(torch.tensor(_inverse_softplus(killing_rate_init)))
        self.background_mean = nn.Parameter(torch.zeros(self.spatial_dim))
        self.raw_background_std = nn.Parameter(
            torch.full((self.spatial_dim,), _inverse_softplus(background_std_init))
        )

    @property
    def mu(self) -> Tensor:
        return F.softplus(self.raw_mu) + 1.0e-6

    @property
    def eta(self) -> Tensor:
        return self.eta_cap * torch.sigmoid(self.raw_eta)

    @property
    def killing_rate(self) -> Tensor:
        return F.softplus(self.raw_killing_rate) + 1.0e-4

    @property
    def background_variance(self) -> Tensor:
        return (F.softplus(self.raw_background_std) + 1.0e-4).square()

    @staticmethod
    def _diagonal_gaussian_log_density(
        value: Tensor,
        mean: Tensor,
        variance: Tensor,
    ) -> Tensor:
        variance = variance.clamp_min(1.0e-10)
        return -0.5 * (
            math.log(2.0 * math.pi)
            + torch.log(variance)
            + (value - mean).square() / variance
        ).sum(dim=-1)

    def transition_log_density(
        self,
        displacements: Tensor,
        ages: Tensor,
    ) -> Tensor:
        """Return ``log p_age(displacement)`` for ``[..., spatial_dim]`` inputs."""
        raise NotImplementedError

    def _regularizer(self) -> Tensor:
        return self.raw_mu * 0.0

    def _compute(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        device,
    ) -> dict[str, Tensor]:
        times = state.payload.get("times_raw", times).to(device)
        locations = state.payload.get("locations_raw", locations).to(device)
        lengths = lengths.to(device)
        batch_size, max_events = times.shape
        event_indices = torch.arange(max_events, device=device)
        valid_events = event_indices.unsqueeze(0) < lengths.unsqueeze(1)
        target_mask = valid_events & (event_indices.unsqueeze(0) >= self.min_history)

        # Pair axes are [batch, target event, parent event].
        ages = times.unsqueeze(2) - times.unsqueeze(1)
        history_order = event_indices.view(1, 1, -1) < event_indices.view(1, -1, 1)
        pair_mask = history_order & valid_events.unsqueeze(1) & valid_events.unsqueeze(2)
        safe_ages = ages.clamp_min(1.0e-7)
        displacements = locations.unsqueeze(2) - locations.unsqueeze(1)
        transition_log_density = self.transition_log_density(displacements, safe_ages)

        log_eta = torch.where(
            self.eta > 0.0,
            torch.log(self.eta),
            self.eta.new_full((), -torch.inf),
        )
        log_temporal_trigger = (
            log_eta
            + torch.log(self.killing_rate.clamp_min(1.0e-12))
            - self.killing_rate * safe_ages
        ).masked_fill(~pair_mask, -1.0e30)
        history_log_joint = log_temporal_trigger + transition_log_density

        background_log_density = self._diagonal_gaussian_log_density(
            locations,
            self.background_mean,
            self.background_variance,
        )
        background_log_joint = torch.log(self.mu) + background_log_density
        log_joint_intensity = torch.logsumexp(
            torch.cat([background_log_joint.unsqueeze(-1), history_log_joint], dim=-1),
            dim=-1,
        )
        log_total_rate = torch.logsumexp(
            torch.cat(
                [
                    torch.log(self.mu).expand(batch_size, max_events, 1),
                    log_temporal_trigger,
                ],
                dim=-1,
            ),
            dim=-1,
        )

        previous_times = torch.cat([times.new_zeros((batch_size, 1)), times[:, :-1]], dim=1)
        background_integral = self.mu * (times - previous_times).clamp_min(0.0)
        previous_ages = (previous_times.unsqueeze(2) - times.unsqueeze(1)).clamp_min(0.0)
        offspring_integral_terms = self.eta * (
            torch.exp(-self.killing_rate * previous_ages)
            - torch.exp(-self.killing_rate * safe_ages)
        )
        offspring_integral = (offspring_integral_terms * pair_mask).sum(dim=-1)
        survival_integral = background_integral + offspring_integral

        temporal_nll_matrix = -log_total_rate + survival_integral
        spatial_nll_matrix = -(log_joint_intensity - log_total_rate)
        nll_matrix = temporal_nll_matrix + spatial_nll_matrix
        selected = target_mask.to(nll_matrix.dtype)
        total_events = selected.sum()
        denominator = total_events.clamp_min(1.0)
        pure_nll = (nll_matrix * selected).sum() / denominator
        temporal_nll = (temporal_nll_matrix * selected).sum() / denominator
        spatial_nll = (spatial_nll_matrix * selected).sum() / denominator
        per_sequence_counts = selected.sum(dim=-1)
        nll_per_event = (nll_matrix * selected).sum(dim=-1) / per_sequence_counts.clamp_min(1.0)
        regularizer = self._regularizer()

        return {
            "loss": pure_nll + regularizer,
            "nll": pure_nll,
            "temporal_nll": float(temporal_nll.detach().item()),
            "spatial_nll": float(spatial_nll.detach().item()),
            "nll_per_event": nll_per_event,
            "total_events": total_events,
            "nll_matrix": nll_matrix,
            "temporal_nll_matrix": temporal_nll_matrix,
            "spatial_nll_matrix": spatial_nll_matrix,
            "mask": target_mask.to(nll_matrix.dtype),
            "next_event_mask": target_mask.to(nll_matrix.dtype),
            "target_event_indices": event_indices.unsqueeze(0).expand(batch_size, -1),
            "state_regularization_terms": {"parameter_regularizer": regularizer},
            "extra_metrics": {
                "raw_space_nll": pure_nll.detach(),
                "raw_space_temporal_nll": temporal_nll.detach(),
                "raw_space_spatial_nll": spatial_nll.detach(),
                "eta": self.eta.detach(),
                "killing_rate": self.killing_rate.detach(),
            },
        }

    def training_loss(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms=None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> dict[str, Tensor]:
        del state_regularization_terms, x_field_at_events, marks
        if device is None:
            device = times.device
        return self._compute(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
            device=device,
        )

    def intensity(
        self,
        *,
        state: StateContext,
        query_times: Tensor,
        query_locations: Tensor,
        query_lengths=None,
        x_field_at_events=None,
        marks=None,
        device=None,
    ) -> Tensor:
        del query_lengths, x_field_at_events, marks
        if device is None:
            device = query_times.device
        history_times = state.payload["times_raw"].to(device)
        history_locations = state.payload["locations_raw"].to(device)
        lengths = state.payload["lengths"].to(device)
        if history_times.shape[0] != 1:
            raise ValueError("Intensity queries currently require one history sequence.")

        query_times = query_times.reshape(-1).to(device)
        query_locations = query_locations.reshape(-1, self.spatial_dim).to(device)
        n_history = int(lengths[0].item())
        parent_times = history_times[0, :n_history]
        parent_locations = history_locations[0, :n_history]
        ages = query_times.unsqueeze(1) - parent_times.unsqueeze(0)
        valid = ages > 0.0
        safe_ages = ages.clamp_min(1.0e-7)
        displacements = query_locations.unsqueeze(1) - parent_locations.unsqueeze(0)
        transition_log_density = self.transition_log_density(displacements, safe_ages)
        log_eta = torch.where(
            self.eta > 0.0,
            torch.log(self.eta),
            self.eta.new_full((), -torch.inf),
        )
        log_history = (
            log_eta
            + torch.log(self.killing_rate)
            - self.killing_rate * safe_ages
            + transition_log_density
        ).masked_fill(~valid, -1.0e30)
        log_background = torch.log(self.mu) + self._diagonal_gaussian_log_density(
            query_locations,
            self.background_mean,
            self.background_variance,
        )
        return torch.exp(
            torch.logsumexp(
                torch.cat([log_background.unsqueeze(-1), log_history], dim=-1),
                dim=-1,
            )
        )

    def query_surface(
        self,
        *,
        state: StateContext,
        grid_times: Tensor,
        grid_locs: Tensor,
        **kwargs,
    ) -> Tensor:
        del kwargs
        return self.intensity(
            state=state,
            query_times=grid_times,
            query_locations=grid_locs,
            device=grid_times.device,
        )


@register_event("static_gaussian_hawkes")
class StaticGaussianHawkesEventModel(ParametricHawkesEventModel):
    """Exact Hawkes baseline with an age-independent Gaussian displacement law."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        static_std_init: float = 0.40,
        **kwargs: Any,
    ) -> None:
        super().__init__(spatial_dim=spatial_dim, **kwargs)
        self.static_mean = nn.Parameter(torch.zeros(self.spatial_dim))
        self.raw_static_std = nn.Parameter(
            torch.full((self.spatial_dim,), _inverse_softplus(static_std_init))
        )

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="nll",
            metric_key="nll",
            objective_description="static-displacement Gaussian Hawkes exact NLL",
            objective_includes_regularization=self.regularization_weight > 0.0,
            nll_kind="exact",
            nll_path="closed_form",
            nll_description=(
                "raw-space next-event NLL with analytic survival and an "
                "age-independent Gaussian offspring displacement"
            ),
            supports_raw_reporting=False,
            has_intensity=True,
            exposes_eventwise_terms=True,
        )

    @property
    def static_variance(self) -> Tensor:
        return (F.softplus(self.raw_static_std) + 1.0e-4).square()

    def transition_log_density(
        self,
        displacements: Tensor,
        ages: Tensor,
    ) -> Tensor:
        del ages
        return self._diagonal_gaussian_log_density(
            displacements,
            self.static_mean.to(displacements.dtype),
            self.static_variance.to(displacements.dtype),
        )

    def _regularizer(self) -> Tensor:
        if self.regularization_weight <= 0.0:
            return self.raw_mu * 0.0
        return self.regularization_weight * (
            self.static_mean.square().sum() + self.raw_static_std.square().sum()
        )


@register_event("gaussian_poisson")
class GaussianPoissonEventModel(ParametricHawkesEventModel):
    """Exact Gaussian-background Poisson null on the same raw-space support."""

    def __init__(self, *, spatial_dim: int, **kwargs: Any) -> None:
        super().__init__(spatial_dim=spatial_dim, **kwargs)
        self.raw_eta.requires_grad_(False)
        self.raw_killing_rate.requires_grad_(False)

    @property
    def eta(self) -> Tensor:
        # Returning an exact zero removes both the triggering intensity and its
        # compensator while retaining the common likelihood implementation.
        return self.raw_eta.detach() * 0.0

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="nll",
            metric_key="nll",
            objective_description="Gaussian-background Poisson exact NLL",
            objective_includes_regularization=self.regularization_weight > 0.0,
            nll_kind="exact",
            nll_path="closed_form",
            nll_description=(
                "raw-space next-event NLL for a homogeneous temporal Poisson "
                "process with a normalized Gaussian spatial background"
            ),
            supports_raw_reporting=False,
            has_intensity=True,
            exposes_eventwise_terms=True,
        )

    def transition_log_density(
        self,
        displacements: Tensor,
        ages: Tensor,
    ) -> Tensor:
        # No offspring: the trigger term is -inf, so the displacement law never
        # contributes. A zero log density keeps the shared implementation intact.
        del ages
        return torch.zeros(displacements.shape[:-1], device=displacements.device, dtype=displacements.dtype)

    def _regularizer(self) -> Tensor:
        if self.regularization_weight <= 0.0:
            return self.raw_mu * 0.0
        return self.regularization_weight * (
            self.background_mean.square().sum()
            + self.raw_background_std.square().sum()
        )
