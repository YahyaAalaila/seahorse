"""Full-plane next-event sampling for factorized Gaussian-mixture baselines.

This module intentionally does not use the generic finite-rectangle thinning
sampler.  ``GaussianMixtureSpatialModel`` is a normalized density on R^D, so
clipping its draws to an empirical rectangle would change the predictive law.

``sample_factorized`` draws *independent* next-event samples conditional on one
observed history.  The temporal process is sampled from the trained temporal
module and the location is then sampled exactly from the trained shared GMM in
its native model coordinates.  Returned values are converted back to the raw
coordinates expected by the runner's public data path.

Supported temporal modules are deliberately an explicit allow-list.  Adding a
new factorized temporal head requires deriving and testing its next-event law;
falling back to generic thinning would be scientifically ambiguous here.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from seahorse.data.transforms import transform_from_spec
from seahorse.models.spatial_models.gaussian_mixture import GaussianMixtureSpatialModel
from seahorse.models.temporal_models.neural_temporal import (
    RMTPPTemporalProcess,
    THPTemporalProcess,
)
from seahorse.models.temporal_models.parametric_processes import (
    HawkesProcess,
    HomogeneousPoissonProcess,
    SelfCorrectingProcess,
)


_NUM_TOL = 1e-9
_MAX_ADAPTIVE_DEPTH = 32
_MAX_TAIL_REFINEMENTS = 8


def _softplus_scalar(x: float) -> float:
    """Stable scalar softplus, including large learned THP logits."""
    if x > 30.0:
        return x + math.log1p(math.exp(-x))
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _log1pexp(x: float) -> float:
    return _softplus_scalar(x)


def _exp_wait(generator: torch.Generator) -> float:
    # `exponential_` accepts a Generator on CPU and CUDA; clamp avoids log(0)
    # only in the theoretical endpoint, never for ordinary generator output.
    u = float(torch.rand((), generator=generator, device=generator.device).item())
    return -math.log(max(1.0 - u, np.finfo(np.float64).tiny))


def _adaptive_simpson(
    fn: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    atol: float = _NUM_TOL,
    rtol: float = _NUM_TOL,
) -> float:
    """Finite-interval adaptive Simpson integration with an explicit failure.

    The sampler only uses this for the one-dimensional THP hazard.  Raising on
    exhausted refinement is preferable to silently returning an uncontrolled
    approximate draw.
    """
    def simpson(a: float, b: float, fa: float, fm: float, fb: float) -> float:
        return (b - a) * (fa + 4.0 * fm + fb) / 6.0

    fa, fb = fn(lo), fn(hi)
    mid = 0.5 * (lo + hi)
    fm = fn(mid)
    whole = simpson(lo, hi, fa, fm, fb)

    def recurse(a: float, b: float, fa_: float, fm_: float, fb_: float, whole_: float, depth: int) -> float:
        mid_ = 0.5 * (a + b)
        left_mid = 0.5 * (a + mid_)
        right_mid = 0.5 * (mid_ + b)
        flm, frm = fn(left_mid), fn(right_mid)
        left = simpson(a, mid_, fa_, flm, fm_)
        right = simpson(mid_, b, fm_, frm, fb_)
        estimate = left + right
        err = abs(estimate - whole_)
        tolerance = 15.0 * (atol + rtol * abs(estimate))
        if err <= tolerance:
            return estimate + (estimate - whole_) / 15.0
        if depth >= _MAX_ADAPTIVE_DEPTH:
            raise RuntimeError(
                "THP hazard quadrature exceeded the controlled refinement budget."
            )
        return (
            recurse(a, mid_, fa_, flm, fm_, left, depth + 1)
            + recurse(mid_, b, fm_, frm, fb_, right, depth + 1)
        )

    return recurse(lo, hi, fa, fm, fb, whole, 0)


def _thp_hazard(c: float, beta: float, wait: float) -> float:
    """Integrated THP post-history hazard from 0 through ``wait``.

    In q=|beta|*u coordinates the integrand is well-scaled even when beta is
    very small.  Exactly zero beta is a homogeneous process.
    """
    if wait < 0.0:
        raise ValueError("wait must be nonnegative")
    if abs(beta) < 1e-12:
        return _softplus_scalar(c) * wait
    q_end = abs(beta) * wait
    sign = 1.0 if beta > 0.0 else -1.0
    integral_q = _adaptive_simpson(lambda q: _softplus_scalar(c + sign * q), 0.0, q_end)
    return integral_q / abs(beta)


def _thp_total_hazard_bounds(c: float, beta: float, *, tol: float) -> tuple[float, float, float]:
    """Return lower/upper total-hazard bounds and q endpoint for beta < 0.

    For q beyond q_end, softplus(c-q) <= exp(c-q); consequently the returned
    upper bound has a proved exponential tail bound.  We refine rather than
    turn a draw in the unresolved numerical interval into a spurious event.
    """
    assert beta < 0.0
    abs_beta = abs(beta)
    # Choose q so exp(c-q)/|beta| is no more than tol, then integrate in q.
    q_end = max(0.0, c - math.log(max(tol * abs_beta, np.finfo(float).tiny)))
    lower = _adaptive_simpson(lambda q: _softplus_scalar(c - q), 0.0, q_end) / abs_beta
    tail_bound = math.exp(min(c - q_end, 700.0)) / abs_beta
    return lower, lower + tail_bound, q_end


def _bisect_hazard(
    hazard: Callable[[float], float],
    target: float,
    lo: float,
    hi: float,
) -> float:
    if target <= 0.0:
        return lo
    h_hi = hazard(hi)
    if h_hi < target:
        raise RuntimeError("Hazard bracket does not contain target integrated hazard.")
    tolerance = _NUM_TOL * max(1.0, target)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        value = hazard(mid)
        if abs(value - target) <= tolerance:
            return mid
        if value < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _sample_temporal_wait(
    temporal: torch.nn.Module,
    history_times: torch.Tensor,
    generator: torch.Generator,
) -> tuple[float | None, dict[str, Any]]:
    """Sample one waiting time from the concrete trained temporal module."""
    if isinstance(temporal, HomogeneousPoissonProcess):
        rate = float(F.softplus(temporal.log_lambda).detach().cpu().item())
        return _exp_wait(generator) / rate, {"route": "poisson_inverse_hazard", "rate": rate}

    if isinstance(temporal, HawkesProcess):
        mu = float(F.softplus(temporal.log_mu).detach().cpu().item())
        alpha = float(F.softplus(temporal.log_alpha).detach().cpu().item())
        beta = float(F.softplus(temporal.log_beta).detach().cpu().item())
        t_last = float(history_times[-1].detach().cpu().item())
        lag = (t_last - history_times).clamp(min=0.0)
        excitation = alpha * float(torch.exp(-beta * lag).sum().detach().cpu().item())
        target = _exp_wait(generator)

        def hazard(wait: float) -> float:
            return mu * wait + excitation * (-math.expm1(-beta * wait)) / beta

        hi = max(target / mu, 1.0 / beta)
        while hazard(hi) < target:
            hi *= 2.0
        return _bisect_hazard(hazard, target, 0.0, hi), {
            "route": "hawkes_inverse_integrated_hazard",
            "mu": mu,
            "excitation": excitation,
            "beta": beta,
        }

    if isinstance(temporal, SelfCorrectingProcess):
        mu = float(F.softplus(temporal.log_mu).detach().cpu().item())
        beta = float(F.softplus(temporal.log_beta).detach().cpu().item())
        t_last = float(history_times[-1].detach().cpu().item())
        n = int(history_times.numel())
        log_rate_now = mu * t_last - beta * n
        target = _exp_wait(generator)
        # log(1 + mu*target/exp(log_rate_now))/mu, evaluated without overflow.
        wait = _log1pexp(math.log(mu * target) - log_rate_now) / mu
        return wait, {
            "route": "selfcorrecting_inverse_integrated_hazard",
            "mu": mu,
            "beta": beta,
            "history_count": n,
        }

    if isinstance(temporal, RMTPPTemporalProcess):
        n = int(history_times.numel())
        seq = history_times.reshape(1, n)
        lengths = torch.tensor([n], dtype=torch.long, device=history_times.device)
        hidden, _ = temporal._encode(seq, lengths)
        ab = temporal.mlp(hidden[0, n, :])
        a = float(F.softplus(ab[0]).detach().cpu().item())
        b = float(F.softplus(ab[1]).clamp(min=1e-6).detach().cpu().item())
        target = _exp_wait(generator)
        total_hazard = a / b
        if target >= total_hazard:
            return None, {
                "route": "rmtpp_defective_inverse_hazard",
                "a": a,
                "b": b,
                "total_hazard": total_hazard,
                "no_event": True,
            }
        ratio = min(target * b / a, 1.0 - np.finfo(float).eps)
        return -math.log1p(-ratio) / b, {
            "route": "rmtpp_defective_inverse_hazard",
            "a": a,
            "b": b,
            "total_hazard": total_hazard,
            "no_event": False,
        }

    if isinstance(temporal, THPTemporalProcess):
        if int(getattr(temporal, "n_marks", 0)) != 0:
            raise NotImplementedError("Marked THP sampling is outside the unmarked factorized campaign.")
        n = int(history_times.numel())
        seq = history_times.reshape(1, n)
        lengths = torch.tensor([n], dtype=torch.long, device=history_times.device)
        hidden, _ = temporal._encode(seq, lengths)
        c = float(temporal.f(hidden[0, n, :]).squeeze().detach().cpu().item())
        beta = float(temporal.beta.squeeze().detach().cpu().item())
        target = _exp_wait(generator)
        if abs(beta) < 1e-12:
            rate = _softplus_scalar(c)
            return target / rate, {"route": "thp_constant_inverse_hazard", "c": c, "beta": beta}
        if beta > 0.0:
            def hazard(wait: float) -> float:
                return _thp_hazard(c, beta, wait)
            hi = max(1.0 / beta, 1.0)
            while hazard(hi) < target:
                hi *= 2.0
            return _bisect_hazard(hazard, target, 0.0, hi), {
                "route": "thp_adaptive_integral_inverse_hazard",
                "c": c,
                "beta": beta,
            }

        tolerance = _NUM_TOL
        for _ in range(_MAX_TAIL_REFINEMENTS):
            lower, upper, q_end = _thp_total_hazard_bounds(c, beta, tol=tolerance)
            if target > upper:
                return None, {
                    "route": "thp_adaptive_integral_defective_hazard",
                    "c": c,
                    "beta": beta,
                    "total_hazard_lower": lower,
                    "total_hazard_upper": upper,
                    "no_event": True,
                }
            if target < lower:
                hi = q_end / abs(beta)
                return _bisect_hazard(lambda w: _thp_hazard(c, beta, w), target, 0.0, hi), {
                    "route": "thp_adaptive_integral_defective_hazard",
                    "c": c,
                    "beta": beta,
                    "total_hazard_lower": lower,
                    "total_hazard_upper": upper,
                    "no_event": False,
                }
            tolerance *= 1e-3
        raise RuntimeError("THP finite-tail draw remained numerically indeterminate after refinement.")

    raise TypeError(
        "Unsupported temporal model for verified factorized sampling: "
        f"{type(temporal).__name__}."
    )


def _sample_gmm_location(
    spatial: GaussianMixtureSpatialModel,
    *,
    query_time: torch.Tensor,
    history_times: torch.Tensor,
    history_locations: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw exactly from the full-R^D GMM specified by the spatial module."""
    if history_times.numel() == 0:
        sigma = F.softplus(spatial.log_sigma_prior)
        return torch.randn(history_locations.shape[-1], generator=generator, device=history_locations.device) * sigma
    tau = F.softplus(spatial.log_tau)
    sigma = F.softplus(spatial.log_sigma_kernel)
    logits = -tau * (query_time - history_times).clamp(min=0.0)
    weights = torch.softmax(logits, dim=0)
    component = int(torch.multinomial(weights, 1, generator=generator).item())
    noise = torch.randn(history_locations.shape[-1], generator=generator, device=history_locations.device)
    return history_locations[component] + sigma * noise


def _runner_input_history(runner: Any, history: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw public-history values through the runner's fitted data transform."""
    if "times" not in history or "locations" not in history:
        raise ValueError("history requires 'times' and 'locations'.")
    times_raw = np.asarray(history["times"], dtype=np.float64).reshape(-1)
    locs_raw = np.asarray(history["locations"], dtype=np.float64)
    if times_raw.size == 0:
        raise ValueError("factorized next-event sampling requires at least one history event.")
    if locs_raw.ndim != 2 or locs_raw.shape[0] != times_raw.size:
        raise ValueError("history locations must have shape (n_events, spatial_dim).")
    if not (np.isfinite(times_raw).all() and np.isfinite(locs_raw).all()):
        raise ValueError("history contains non-finite time or location values.")
    if np.any(np.diff(times_raw) < 0.0):
        raise ValueError("history times must be nondecreasing.")

    stats = runner.norm_stats
    if bool(stats.get("normalize", False)):
        t_std = max(float(stats.get("time_std", 1.0)), 1e-8)
        loc_std = np.maximum(np.asarray(stats.get("loc_std", np.ones(locs_raw.shape[1])), dtype=np.float64), 1e-8)
        times_input = (times_raw - float(stats.get("time_mean", 0.0))) / t_std
        locs_input = (locs_raw - np.asarray(stats.get("loc_mean", np.zeros(locs_raw.shape[1])), dtype=np.float64)) / loc_std
    else:
        times_input, locs_input = times_raw, locs_raw
    # Trained factorized modules are float32.  This is also the runner's model
    # input dtype; raw values are retained separately and returned as float64.
    return (
        torch.as_tensor(times_input, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(locs_input, dtype=torch.float32, device=device).unsqueeze(0),
    )


def _raw_outputs(
    runner: Any,
    *,
    internal_times: torch.Tensor,
    internal_locations: torch.Tensor,
    transform_spec: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert family-owned then runner-owned affine transforms, in that order."""
    transform = transform_from_spec(transform_spec or {})
    lengths = torch.full((internal_times.shape[0],), 1, dtype=torch.long, device=internal_times.device)
    if transform is not None:
        times_input = transform.inverse_times(internal_times, lengths)
        locs_input = transform.inverse_locations(internal_locations, lengths)
    else:
        times_input, locs_input = internal_times, internal_locations
    stats = runner.norm_stats
    if bool(stats.get("normalize", False)):
        times_raw = times_input * max(float(stats.get("time_std", 1.0)), 1e-8) + float(stats.get("time_mean", 0.0))
        mean = locs_input.new_tensor(np.asarray(stats.get("loc_mean", np.zeros(locs_input.shape[-1])), dtype=np.float32))
        std = locs_input.new_tensor(np.maximum(np.asarray(stats.get("loc_std", np.ones(locs_input.shape[-1])), dtype=np.float32), 1e-8))
        locs_raw = locs_input * std + mean
    else:
        times_raw, locs_raw = times_input, locs_input
    return (
        times_raw.detach().cpu().numpy().astype(np.float64, copy=False),
        locs_raw.detach().cpu().numpy().astype(np.float64, copy=False),
    )


@contextmanager
def _evaluation_mode(model: torch.nn.Module):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def _resolve_sampler_device(
    device: str | torch.device,
    *,
    current_cuda_device: int | None = None,
) -> torch.device:
    """Resolve ``cuda`` to the concrete current CUDA device.

    PyTorch considers ``cuda`` and ``cuda:0`` different ``torch.device``
    values even when they address the same current GPU.  Sampling tensors need
    a concrete index to compare safely with checkpoint parameters.
    """
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        index = torch.cuda.current_device() if current_cuda_device is None else int(current_cuda_device)
        resolved = torch.device("cuda", index)
    return resolved


def _require_model_device(requested: torch.device, model_device: torch.device) -> None:
    if requested != model_device:
        raise ValueError(f"Requested device {requested} does not match model device {model_device}.")


def sample_factorized(
    runner: Any,
    history: dict[str, Any],
    k: int,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """Draw ``k`` independent next events from a trained factorized GMM model.

    ``history`` uses raw public data coordinates, as in the rest of the runner
    predictive API.  The returned arrays are in those same raw units.  A
    successful defective-process draw with no future event has ``no_event=True``,
    ``next_times=NaN`` and a NaN location; it is never coerced to a finite time.
    """
    if k < 0:
        raise ValueError("k must be nonnegative.")
    model = runner.model
    event_model = getattr(model, "event_model", None)
    if event_model is None:
        raise TypeError("runner.model has no event_model.")
    if int(getattr(event_model, "n_marks", 0)) != 0:
        raise NotImplementedError(
            "Marked factorized sampling is outside the unmarked factorized campaign; "
            "this sampler does not silently discard mark draws."
        )
    spatial = getattr(event_model, "spatial_model", None)
    if not isinstance(spatial, GaussianMixtureSpatialModel):
        raise TypeError("sample_factorized only supports GaussianMixtureSpatialModel spatial heads.")
    temporal = getattr(event_model, "temporal_model", None)
    dev = _resolve_sampler_device(device)
    param = next(model.parameters(), None)
    if param is not None:
        _require_model_device(dev, param.device)
    if param is not None and param.dtype != torch.float32:
        raise ValueError("Verified factorized sampler currently requires float32 model parameters.")
    # Keep the RNG on the same concrete CUDA ordinal as model parameters.
    # ``device=dev.type`` would silently select the current default GPU for an
    # explicit ``cuda:N`` request.
    generator = torch.Generator(device=dev)
    generator.manual_seed(int(seed))

    with _evaluation_mode(model):
        input_times, input_locations = _runner_input_history(runner, history, dev)
        length = torch.tensor([input_times.shape[1]], dtype=torch.long, device=dev)
        state = model.state_model.encode_sampling_history(
            times=input_times, locations=input_locations, lengths=length
        )
        internal_times = state.payload["times"][0, : length.item()]
        internal_locations = state.payload["locations"][0, : length.item(), :]
        shifted = (internal_times - internal_times[0]).clamp(min=0.0)
        transform_spec = state.payload.get("input_transform")

        dim = int(internal_locations.shape[1])
        sampled_internal_time = torch.full((k,), float("nan"), device=dev)
        sampled_internal_loc = torch.full((k, dim), float("nan"), device=dev)
        succeeded = np.ones(k, dtype=bool)
        no_event = np.zeros(k, dtype=bool)
        draw_diagnostics: list[dict[str, Any]] = []
        for i in range(k):
            wait, diagnostic = _sample_temporal_wait(temporal, shifted, generator)
            if wait is None:
                no_event[i] = True
                diagnostic = {**diagnostic, "success": True}
                draw_diagnostics.append(diagnostic)
                continue
            q_time = shifted[-1] + float(wait)
            loc = _sample_gmm_location(
                spatial,
                query_time=q_time,
                history_times=shifted,
                history_locations=internal_locations,
                generator=generator,
            )
            sampled_internal_time[i] = q_time + internal_times[0]
            sampled_internal_loc[i] = loc
            diagnostic = {**diagnostic, "success": True}
            draw_diagnostics.append(diagnostic)

        raw_times, raw_locations = _raw_outputs(
            runner,
            internal_times=sampled_internal_time,
            internal_locations=sampled_internal_loc,
            transform_spec=transform_spec if isinstance(transform_spec, dict) else None,
        )
    return {
        "next_times": raw_times,
        "next_locations": raw_locations,
        "success": succeeded,
        "no_event": no_event,
        "diagnostics": {
            "sampler": "factorized_exact_temporal_full_plane_gmm_v1",
            "spatial_support": "R^D (untruncated Gaussian mixture)",
            "coordinate_output": "raw_runner_input_units",
            "draws": draw_diagnostics,
        },
    }


__all__ = ["sample_factorized"]
