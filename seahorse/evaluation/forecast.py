"""Verified next-event and fixed-horizon forecasting routes.

There is deliberately no generic rectangle-thinning fallback. A model lacking a
validated conditional sampler reports ``unsupported`` instead of producing a
plausible but distribution-changing forecast.
"""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

_FACTORISED = frozenset({"poisson_gmm", "hawkes_gmm", "selfcorrecting_gmm", "rmtpp_gmm", "thp_gmm"})
_NATIVE = frozenset({"smash", "diffusion_stpp", "glide"})
_HISTORY_METADATA = (
    "t0", "t1", "window_start", "calendar_window_start",
    "calendar_offset_from_window_start", "time_unit", "time_axis",
    "source_timezone",
)


def _history_arrays(history: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(history["times"], dtype=np.float64)
    locations = np.asarray(history["locations"], dtype=np.float64)
    if times.ndim != 1 or locations.shape != (len(times), 2) or not len(times):
        raise ValueError("history needs nonempty times[N] and locations[N,2].")
    if not np.isfinite(times).all() or not np.isfinite(locations).all() or np.any(np.diff(times) <= 0):
        raise ValueError("history must be finite and strictly chronological.")
    return times, locations


def _route(runner: Any) -> str | None:
    if getattr(runner, "runner_name", "nnstpp") == "statstpp": return "statistical"
    preset = str(runner.config.model.preset)
    if preset in _FACTORISED: return "factorized"
    if preset in _NATIVE: return "native"
    return None


def _unsupported(preset: str) -> dict[str, Any]:
    return {"status": "unsupported", "preset": preset,
            "reason": "no validated native next-event sampler; generic rectangle thinning is disabled",
            "times": np.empty(0), "locations": np.empty((0, 2)),
            "valid": np.empty(0, dtype=bool), "no_event": np.empty(0, dtype=bool)}


def _preserved_history(history: Mapping[str, Any], times: np.ndarray, locations: np.ndarray) -> dict[str, Any]:
    result = {"times": times, "locations": locations}
    for key in _HISTORY_METADATA:
        if key in history: result[key] = history[key]
    return result


def _neural_history_frame(history: Mapping[str, Any], times: np.ndarray, locations: np.ndarray) -> tuple[dict[str, Any], float]:
    """Put one retained neural H-event context in its trained local frame.

    Neural campaign contexts are rebased to their first retained event.  In a
    recursive fixed-horizon rollout, dropping old generated events otherwise
    changes that convention.  The returned offset belongs only to the model
    coordinate frame and is added back to finite sampled times by the caller.
    Statistical histories deliberately do not use this helper: their raw time
    axis and calendar fields carry likelihood-relevant information.
    """
    offset = float(times[0])
    framed = {"times": times - offset, "locations": locations}
    return framed, offset


def forecast_next(runner: Any, history: Mapping[str, Any], *, n_samples: int = 1, seed: int = 0, device: str = "cpu") -> dict[str, Any]:
    """Draw independent conditional next events through a declared sampler.

    Only the factorized route can currently return a valid no-event outcome,
    via its explicit ``no_event`` mask. NaN/Inf from every other route is a
    numerical failure, never a censored event.
    """
    if n_samples < 1: raise ValueError("n_samples must be positive.")
    times, locations = _history_arrays(history)
    preset, route, last = str(runner.config.model.preset), _route(runner), float(times[-1])
    if route is None: return _unsupported(preset)
    diagnostics: dict[str, Any] = {}
    no_event = np.zeros(n_samples, dtype=bool)
    if route == "statistical":
        stat_history = _preserved_history(history, times, locations)
        sampled_t, sampled_s = runner.model.sample_next_event(stat_history, n_samples=n_samples, rng=np.random.default_rng(seed))
        backend = getattr(runner.model, "sampling_method", "statstpp_native_conditional")
    elif route == "factorized":
        from seahorse.experiments.factorized_sampling import sample_factorized
        output = sample_factorized(runner, _preserved_history(history, times, locations), n_samples, device=device, seed=seed)
        sampled_t, sampled_s = output["next_times"], output["next_locations"]
        no_event, diagnostics = np.asarray(output["no_event"], dtype=bool), dict(output.get("diagnostics", {}))
        backend = "factorized_exact_temporal_full_plane_gmm_v1"
    else:
        # Existing decoder paths own each preset's affine token inversion. Fork
        # global Torch RNG so a forecast seed is reproducible and does not alter
        # the caller's training/evaluation RNG stream.
        import torch
        from seahorse.evaluation.predictive.rollout import sample_next_events_diffusion_batch, sample_next_events_smash_batch
        torch_device = torch.device(device)
        cuda_index = None
        if torch_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("A CUDA forecast was requested but CUDA is unavailable.")
            cuda_index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
        devices = [] if cuda_index is None else [cuda_index]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if cuda_index is not None:
                # ``fork_rng`` restores precisely these device states.
                torch.cuda.manual_seed(int(seed))
            if preset == "smash":
                sampled_t, sampled_s = sample_next_events_smash_batch(runner, _preserved_history(history, times, locations), n_samples, device)
            else:  # diffusion_stpp and glide share the documented decoder path
                sampled_t, sampled_s = sample_next_events_diffusion_batch(runner, _preserved_history(history, times, locations), n_samples, device)
        backend = f"{preset}_native_next_event"
    sampled_t, sampled_s = np.asarray(sampled_t, dtype=np.float64), np.asarray(sampled_s, dtype=np.float64)
    if sampled_t.shape != (n_samples,) or sampled_s.shape != (n_samples, 2) or no_event.shape != (n_samples,):
        raise RuntimeError("Sampler returned an invalid next-event shape.")
    valid = ~no_event & np.isfinite(sampled_t) & np.isfinite(sampled_s).all(axis=1) & (sampled_t > last)
    reasons = np.full(n_samples, "", dtype=object)
    reasons[no_event] = "no_event"
    reasons[~no_event & ~valid] = "invalid_native_draw"
    return {"status": "ok", "preset": preset, "backend": backend, "times": sampled_t,
            "locations": sampled_s, "valid": valid, "no_event": no_event,
            "failure_reason": reasons, "diagnostics": diagnostics}


def forecast_horizon(runner: Any, history: Mapping[str, Any], *, t_end: float, n_samples: int, seed: int = 0, device: str = "cpu", history_limit: int = 20, event_cap: int = 10_000) -> dict[str, Any]:
    """Autoregressively forecast to a fixed endpoint with generated history only."""
    times, locations = _history_arrays(history)
    origin = float(history.get("t1", times[-1]))
    if not np.isfinite(origin) or origin < times[-1]:
        raise ValueError("history t1 must be finite and cannot precede its last event.")
    if not np.isfinite(t_end) or t_end <= origin: raise ValueError("t_end must follow the observed history endpoint.")
    if n_samples < 1 or history_limit < 1 or event_cap < 1: raise ValueError("positive n_samples/history_limit/event_cap required.")
    preset = str(runner.config.model.preset)
    route = _route(runner)
    if route is None:
        return {**_unsupported(preset), "t_end": float(t_end), "sample_times": [], "sample_locations": [], "sample_status": []}
    draws_t, draws_s, statuses = [], [], []
    # The Poisson sampler has an exact fixed-horizon law and can explicitly
    # represent zero total intensity as an empty future.  This shortcut is
    # restricted to Poisson: Hawkes uses the recursive path below so H20 is
    # an actual generated-history limit rather than a hidden full history.
    is_poisson = route == "statistical" and preset in {"homogeneous_poisson", "spatial_poisson", "seasonal_poisson"}
    for draw in range(n_samples):
        if is_poisson:
            poisson_history = _preserved_history(history, times, locations)
            try:
                future = runner.model.sample_future(
                    poisson_history, horizon=float(t_end - float(poisson_history.get("t1", times[-1]))),
                    rng=np.random.default_rng(seed + draw * 1_000_003), max_events=event_cap,
                )
                future_t = np.asarray(future["times"], dtype=np.float64)
                future_s = np.asarray(future["locations"], dtype=np.float64).reshape(-1, 2)
                if (future_t.ndim != 1 or future_s.shape != (len(future_t), 2)
                        or not np.isfinite(future_t).all() or not np.isfinite(future_s).all()
                        or np.any(np.diff(future_t) <= 0)
                        or np.any(future_t < float(poisson_history.get("t1", times[-1])))
                        or np.any(future_t >= t_end)):
                    draws_t.append(np.empty(0)); draws_s.append(np.empty((0, 2))); statuses.append("invalid_draw")
                else:
                    draws_t.append(future_t); draws_s.append(future_s); statuses.append("no_event" if not len(future_t) else "complete")
            except RuntimeError as exc:
                # The statistical implementation deliberately raises, rather
                # than truncates, when its cap is reached.
                draws_t.append(np.empty(0)); draws_s.append(np.empty((0, 2)));
                statuses.append("event_cap_exceeded" if "max_events" in str(exc) else "invalid_draw")
            continue
        h_t, h_s = times.copy(), locations.copy(); out_t, out_s = [], []; status = "complete"
        for step in range(event_cap):
            h = _preserved_history(history, h_t[-history_limit:], h_s[-history_limit:])
            # The initial history's t1 may encode a quiet observed interval.
            # Once we append a generated event, its endpoint is the only
            # valid forecast origin for the next conditional draw.
            if out_t:
                h["t1"] = float(h_t[-1])
            if route in {"factorized", "native"}:
                model_history, frame_offset = _neural_history_frame(h, h["times"], h["locations"])
            else:
                model_history, frame_offset = h, 0.0
            one = forecast_next(runner, model_history, n_samples=1,
                                seed=seed + draw * 1_000_003 + step, device=device)
            if one["status"] != "ok" or not bool(one["valid"][0]):
                status = "no_event" if one["status"] == "ok" and bool(one["no_event"][0]) else "invalid_draw"; break
            next_t = float(one["times"][0]) + frame_offset
            next_s = np.asarray(one["locations"][0], dtype=np.float64)
            # A candidate after t_end is right-censored: it establishes no
            # generated event in the fixed horizon and is not an overflow draw.
            if next_t >= t_end: break
            out_t.append(next_t); out_s.append(next_s); h_t=np.append(h_t,next_t); h_s=np.vstack((h_s,next_s))
        else: status = "event_cap_exceeded"
        draws_t.append(np.asarray(out_t, dtype=np.float64)); draws_s.append(np.asarray(out_s, dtype=np.float64).reshape(-1,2)); statuses.append(status)
    return {"status":"ok", "preset":preset, "t_end":float(t_end), "sample_times":draws_t, "sample_locations":draws_s,
            "sample_status":np.asarray(statuses,dtype=object), "history_limit":history_limit, "event_cap":event_cap,
            "provenance":"autoregressive_generated_history_only",
            "neural_history_frame":"rebase retained H events to first retained event; restore sampled times to fixed-horizon axis" if route in {"factorized", "native"} else "raw statistical time axis retained"}


def score_horizon_counts(forecast: Mapping[str, Any], truth_times: Any, *, t_start: float, t_end: float, bin_width: float, truth_locations: Any | None = None, spatial_bounds: tuple[float,float,float,float] | None = None, spatial_bin_width: float | None = None, spatial_grid: tuple[int, int] | None = None) -> dict[str,float]:
    """Score fixed time×space count cells; includes a spatial overflow cell.

    Horizon failures cannot be scored as ordinary short forecasts. Passing no
    spatial arguments is retained only for temporal count controls.
    """
    if forecast.get("status") != "ok" or any(s not in {"complete","no_event"} for s in forecast.get("sample_status", [])):
        return {"count_crps":float("nan"),"count_rmse":float("nan"),"count_mae":float("nan")}
    if not (t_end>t_start and bin_width>0): raise ValueError("invalid time bins")
    n_time_bins = max(1, int(np.ceil((t_end-t_start)/bin_width - 1e-12)))
    edges = t_start + np.arange(n_time_bins+1) * bin_width
    edges[-1] = t_end
    truth=np.asarray(truth_times,dtype=float)
    samples_t = [np.asarray(row, dtype=float) for row in forecast["sample_times"]]
    if (truth.ndim != 1 or not np.isfinite(truth).all()
            or np.any(truth < t_start) or np.any(truth >= t_end)
            or any(row.ndim != 1 or not np.isfinite(row).all() or np.any(row < t_start) or np.any(row >= t_end) for row in samples_t)):
        raise ValueError("Count scoring requires finite event times within [t_start, t_end).")
    def temporal(row): return np.histogram(np.asarray(row),bins=edges)[0]
    actual=temporal(truth); predicted=np.stack([temporal(row) for row in forecast["sample_times"]])
    if truth_locations is not None or spatial_bounds is not None or spatial_bin_width is not None:
        if truth_locations is None or spatial_bounds is None: raise ValueError("spatial scoring requires truth_locations and bounds")
        if spatial_grid is None:
            if spatial_bin_width is None or spatial_bin_width <= 0: raise ValueError("spatial scoring needs a positive bin width or spatial_grid")
            nx = int(np.ceil((spatial_bounds[1] - spatial_bounds[0]) / spatial_bin_width))
            ny = int(np.ceil((spatial_bounds[3] - spatial_bounds[2]) / spatial_bin_width))
        else:
            nx, ny = map(int, spatial_grid)
            if nx < 1 or ny < 1: raise ValueError("spatial_grid must contain positive dimensions")
        xmin,xmax,ymin,ymax=spatial_bounds; xedges=np.linspace(xmin,xmax,nx+1); yedges=np.linspace(ymin,ymax,ny+1)
        tl=np.asarray(truth_locations,dtype=float)
        if tl.shape != (len(truth), 2) or not np.isfinite(tl).all(): raise ValueError("truth_locations must be finite shape (N,2)")
        def grid(ts,ls):
            if ls.shape != (len(ts), 2) or not np.isfinite(ls).all(): raise ValueError("sample locations must be finite shape (N,2)")
            # The source preprocessing rectangles include their outer edges;
            # NumPy also assigns the outer right edge to the final spatial bin.
            inside=(ts>=t_start)&(ts<t_end)&(ls[:,0]>=xmin)&(ls[:,0]<=xmax)&(ls[:,1]>=ymin)&(ls[:,1]<=ymax)
            body=np.histogramdd(np.column_stack((ts[inside],ls[inside,0],ls[inside,1])),bins=(edges,xedges,yedges))[0].reshape(n_time_bins, -1)
            overflow = temporal(ts[~inside])
            return np.column_stack((body, overflow)).ravel()
        actual=grid(truth,tl); predicted=np.stack([grid(ts,np.asarray(ls, dtype=float)) for ts,ls in zip(samples_t,forecast["sample_locations"],strict=True)])
    mean=predicted.mean(axis=0); mae=np.mean(abs(mean-actual)); rmse=np.sqrt(np.mean((mean-actual)**2)); pair=abs(predicted[:,None]-predicted[None,:]).mean(axis=(0,1)); crps=(abs(predicted-actual).mean(axis=0)-.5*pair).mean()
    return {"count_crps":float(crps),"count_rmse":float(rmse),"count_mae":float(mae)}
