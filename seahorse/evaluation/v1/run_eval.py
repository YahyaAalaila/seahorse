"""Evaluate one trained run with the v1.0 metric set and write artifacts.

Usage (CLI)::

    python -m seahorse.evaluation.v1.run_eval --run RUN_DIR --test TEST.jsonl \
        --train TRAIN.jsonl --out OUT_DIR [--gt-params L0_params.json] \
        [--sections nll,temporal,samples,intensity,rollout] [--device cuda]

Outputs (``OUT_DIR``):
  summary.json        scalar metrics + provenance + timings + diagnostics
  per_context.npz     per-context arrays (sequence index, target index, ...)
  rollout.npz         rollout scores per start context (if run)
  intensity.npz       per-sequence intensity-recovery statistics (if run)

See docs/v1.0/evaluation/METRICS_SPEC.md for every definition used here.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import scoring as S
from .loading import load_runner_compat

COMMON_MIN_TARGET = 20  # common target set C20: targets with >= 20 history events


def make_adapter(runner, device):
    preset = runner.config.model.preset
    if preset == "auto_stpp":
        from .families.windowed import AutoSTPPAdapter

        return AutoSTPPAdapter(runner, device)
    if preset == "deep_stpp":
        from .families.windowed import DeepSTPPAdapter

        return DeepSTPPAdapter(runner, device)
    if preset == "nsmpp":
        from .families.joint import NSMPPAdapter

        return NSMPPAdapter(runner, device)
    if preset in ("smash", "diffusion_stpp"):
        from .families.native import NativeAdapter

        return NativeAdapter(runner, device)
    if preset.startswith("neural_") or preset in ("njsde",):
        from .families.neural import NeuralAdapter

        return NeuralAdapter(runner, device)
    if preset.endswith("_gmm") and (preset.startswith("rmtpp") or preset.startswith("thp")):
        from .families.factorized import FactorizedAdapter

        return FactorizedAdapter(runner, device)
    raise NotImplementedError(f"no v1 adapter for preset {preset!r}")


def _git_head(path: Path) -> str | None:
    marker = path / "GIT_COMMIT"  # frozen snapshot (git archive) records its commit here
    if marker.exists():
        return marker.read_text().strip()
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _load_jsonl(path):
    """Load sequences; times get the same float32 strictness repair as the NLL path.

    ``_strict_float32_times`` (used by ``evaluation/likelihood.py``) moves an
    event that ties its predecessor after the float32 cast to the next float32
    value. Applying it here keeps every v1 quantity on the same event times as
    the packaged NLL.
    """
    from seahorse.data.dataset import _strict_float32_times
    from seahorse.utils import load_jsonl

    out = []
    for s in load_jsonl(str(path)):
        t = _strict_float32_times(np.asarray(s["times"], float), context=str(path)).astype(np.float64)
        out.append({"times": t, "locations": np.asarray(s["locations"], float)})
    return out


def _stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "n": 0}
    return {"mean": float(x.mean()), "median": float(np.median(x)), "n": int(x.size)}


class _NLLOnly:
    """Fallback adapter for presets without a family adapter.

    Only the ``nll`` section runs; the per-target NLL is the packaged (fixed)
    eventwise / prefix-difference value, and every target is treated as defined.
    """

    has_intensity = False
    spatial_sampler = "none"
    min_history = 1

    def __init__(self, runner, reason):
        self.runner = runner
        self.reason = reason
        self.nll_kind = runner.model.event_model.capabilities.nll_kind
        self.preset = runner.config.model.preset

    def describe(self):
        return {"family": "nll_only", "preset": self.preset, "nll_kind": self.nll_kind,
                "has_intensity": False, "reason": self.reason, "min_history": 1,
                "spatial_sampler": "none", "spatial_domain": "n/a"}


class Evaluator:
    def __init__(self, run_dir, test_path, train_path, out_dir, *, device, seed=0, K=200, K_flow=None,
                 ctx_stride=1, ctx_stride_flow=None, c=None, tau_grid_n=1024, tau_factor=4.0,
                 gt_params=None, roll_starts_stride=40, roll_first=50, roll_M=50, roll_H=10,
                 intensity_slices=50, intensity_grid=32, max_seqs=None, pkg_check_seqs=2):
        self.t_start = time.time()
        self.pkg_check_seqs = int(pkg_check_seqs)
        self.run_dir = Path(run_dir)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.runner, self.load_info = load_runner_compat(self.run_dir, train_path=train_path)
        self.runner.model.to(self.device)
        self.runner.model.eval()
        self.preset = self.runner.config.model.preset
        try:
            self.ad = make_adapter(self.runner, self.device)
        except NotImplementedError as exc:  # NLL-only evaluation via the packaged (fixed) path
            self.ad = _NLLOnly(self.runner, str(exc))
        self.test = _load_jsonl(test_path)[:max_seqs]
        self.train = _load_jsonl(train_path) if train_path else None
        self.test_path, self.train_path = str(test_path), str(train_path)
        if c is None:
            if self.train is None:
                raise ValueError("need --train to set the CRPS threshold c (max train IET)")
            c = float(max(np.diff(s["times"]).max() for s in self.train if len(s["times"]) > 1))
            c = float(np.ceil(c * 10.0) / 10.0)  # round up to 0.1
        self.c = float(c)
        self.tau_max = float(tau_factor * self.c)
        self.tau_grid = S.tau_grid_quadratic(self.tau_max, int(tau_grid_n))
        is_flow = getattr(self.ad, "spatial_sampler", "").startswith("exact_inverse_flow")
        self.K = int(K_flow if (is_flow and K_flow) else K)
        self.ctx_stride = int(ctx_stride_flow if (is_flow and ctx_stride_flow) else ctx_stride)
        self.gt = None
        if gt_params:
            from .gt_hawkesnest import TravelingWaveGT

            self.gt = TravelingWaveGT.from_file(gt_params)
        self.roll = dict(stride=roll_starts_stride, first=roll_first, M=roll_M, H=roll_H)
        self.int_cfg = dict(slices=intensity_slices, grid=intensity_grid)
        self.summary: dict[str, Any] = {
            "provenance": {
                "run_dir": str(self.run_dir),
                "preset": self.preset,
                "test_path": self.test_path,
                "train_path": self.train_path,
                "code_commit": _git_head(Path(__file__).resolve().parents[3]),
                "load": self.load_info,
                "device": str(self.device),
                "host": platform.node(),
                "torch": torch.__version__,
                "seed": self.seed,
            },
            "adapter": self.ad.describe(),
            "config": {
                "c": self.c, "tau_max": self.tau_max, "tau_grid_n": int(tau_grid_n), "K": self.K,
                "ctx_stride": self.ctx_stride, "common_min_target": COMMON_MIN_TARGET,
                "rollout": self.roll, "intensity": self.int_cfg,
            },
            "timings_sec": {},
            "diagnostics": {},
        }
        self.arrays: dict[str, list] = {}

    # ------------------------------------------------------------------
    def _put(self, key, arr):
        self.arrays.setdefault(key, []).append(np.asarray(arr))

    def _rng(self, *parts):
        from seahorse.evaluation.runtime import derive_seed

        return np.random.default_rng(derive_seed(self.seed, self.preset, *parts))

    def _contexts(self, n):
        k = np.arange(0, n - 1)
        return k

    # ------------------------------------------------------------------
    def section_nll(self):
        """Per-target NLL via the v1 adapters, cross-checked against the packaged path.

        The packaged (fixed) path ``compute_next_event_test_nll`` is run on the
        first ``pkg_check_seqs`` test sequences only (all sequences for presets
        without an adapter). The legacy-protocol mean (all targets 1..n-1, with
        undefined targets counted as 0, as the paper-era benchmark did) is then
        derived from the v1 per-target values; on the checked sequences it
        equals the packaged value to the reported tolerance.
        """
        t0 = time.time()
        from seahorse.evaluation.likelihood import compute_next_event_test_nll

        caps = self.runner.model.event_model.capabilities
        nll_only = isinstance(self.ad, _NLLOnly)
        n_chk = len(self.test) if nll_only else min(self.pkg_check_seqs, len(self.test))
        pkg = None
        if caps.nll_kind == "exact" and n_chk > 0:
            pkg = compute_next_event_test_nll(
                self.runner, [dict(times=s["times"], locations=s["locations"]) for s in self.test[:n_chk]],
                device=self.device)
            per = np.asarray(pkg["per_context_nll"], float)
        consist = []
        offs = 0
        for si, seq in enumerate(self.test):
            n = len(seq["times"])
            k = self._contexts(n)
            tgt = k + 1
            self._put("seq", np.full(len(k), si))
            self._put("target", tgt)
            pk = per[offs : offs + len(k)] if (pkg is not None and si < n_chk) else np.full(len(k), np.nan)
            if nll_only:
                self._put("defined", np.ones(len(k), bool))
                if pkg is not None:
                    self._put("nll_pkg", pk)
                    self._put("nll_v1", pk)
            elif self.ad.has_intensity:
                prep = self.ad.prepare(seq["times"], seq["locations"])
                et = self.ad.event_terms(prep)
                self._put("defined", et.defined[tgt])
                self._put("nll_v1", et.nll[tgt])
                self._put("d_lambda", et.d_lambda[tgt])
                self._put("log_lambda", et.log_lambda[tgt])
                self._put("log_ps", et.log_ps[tgt])
                for key, val in et.extra.items():
                    self._put(key, np.asarray(val)[tgt])
                if pkg is not None:
                    self._put("nll_pkg", pk)
                    if si < n_chk:
                        d = np.abs(pk - et.nll[tgt])[et.defined[tgt]]
                        consist.append(d)
            else:
                self._put("defined", np.ones(len(k), bool))
            if pkg is not None and si < n_chk:
                offs += len(k)
        self._cat()
        A = self.arrays
        res = {"nll_kind_preset": self.ad.nll_kind, "capabilities_nll_kind": caps.nll_kind,
               "pkg_check_seqs": int(n_chk)}
        if "nll_v1" in A:
            dfn = A["defined"].astype(bool)
            common = dfn & (A["target"] >= COMMON_MIN_TARGET)
            legacy = np.where(dfn, A["nll_v1"], 0.0)  # paper-era protocol: undefined targets scored 0
            res.update({
                "nll_defined_mean": float(np.mean(A["nll_v1"][dfn])),
                "nll_defined_n": int(dfn.sum()),
                "nll_common_mean": float(np.mean(A["nll_v1"][common])),
                "nll_common_n": int(common.sum()),
                "n_undefined_targets": int((~dfn).sum()),
                "legacy_protocol_mean": float(np.nanmean(legacy)),
                "legacy_protocol_n": int(len(legacy)),
            })
        if pkg is not None:
            chk = np.isfinite(A["nll_pkg"])
            res.update({
                "legacy_protocol_method": pkg["method"],
                "pkg_mean_on_checked": float(np.mean(A["nll_pkg"][chk])),
                "v1_legacy_mean_on_checked": float(np.mean(np.where(A["defined"], A["nll_v1"], 0.0)[chk])),
                "max_abs_diff_pkg_vs_v1_defined": float(np.max(np.concatenate(consist))) if consist else None,
                "mean_abs_diff_pkg_vs_v1_defined": float(np.mean(np.concatenate(consist))) if consist else None,
            })
            undefined = chk & ~A["defined"].astype(bool)
            res["legacy_undefined_scored_as"] = float(np.mean(A["nll_pkg"][undefined])) if undefined.any() else None
        self.summary["nll"] = res
        self.summary["timings_sec"]["nll"] = time.time() - t0

    def _cat(self):
        for k, v in list(self.arrays.items()):
            if isinstance(v, list):
                self.arrays[k] = np.concatenate(v) if len(v) else np.zeros(0)

    # ------------------------------------------------------------------
    def section_temporal(self):
        if not self.ad.has_intensity:
            return
        t0 = time.time()
        crps, pit_grid, s_c, s_max, lam_y = [], [], [], [], []
        for si, seq in enumerate(self.test):
            n = len(seq["times"])
            prep = self.ad.prepare(seq["times"], seq["locations"])
            k_all = self._contexts(n)
            ok = k_all + 1 >= self.ad.min_history
            k = k_all[ok]
            out_crps = np.full(len(k_all), np.nan)
            out_pg = np.full(len(k_all), np.nan)
            out_sc = np.full(len(k_all), np.nan)
            out_sm = np.full(len(k_all), np.nan)
            out_ly = np.full(len(k_all), np.nan)
            if len(k):
                L = self.ad.cum_hazard(prep, np.zeros_like(k), k, self.tau_grid)
                F = -np.expm1(-L)
                y = seq["times"][k + 1] - seq["times"][k]
                out_crps[ok] = S.crps_from_cdf_grid(F, self.tau_grid, y, self.c)
                ly = np.array([np.interp(y[i], self.tau_grid, L[i]) for i in range(len(k))])
                out_ly[ok] = ly
                out_pg[ok] = -np.expm1(-ly)
                out_sc[ok] = np.exp(-np.array([np.interp(self.c, self.tau_grid, L[i]) for i in range(len(k))]))
                out_sm[ok] = np.exp(-L[:, -1])
            crps.append(out_crps); pit_grid.append(out_pg); s_c.append(out_sc); s_max.append(out_sm); lam_y.append(out_ly)
        A = self.arrays
        A["crps_exact"] = np.concatenate(crps)
        A["pit_grid"] = np.concatenate(pit_grid)
        A["surv_at_c"] = np.concatenate(s_c)
        A["surv_at_taumax"] = np.concatenate(s_max)
        A["lambda_grid_at_y"] = np.concatenate(lam_y)
        A["pit_exact"] = S.pit_from_compensator(A["d_lambda"])
        dfn = A["defined"].astype(bool)
        common = dfn & (A["target"] >= COMMON_MIN_TARGET)
        res = {}
        for name, m in (("common", common), ("defined", dfn)):
            u = A["pit_exact"][m]
            res[name] = {
                "crps_exact_mean": float(np.nanmean(A["crps_exact"][m])),
                "pit_ks": S.ks_uniform(u),
                "pit_tails": S.pit_tail_fractions(u),
                "pit_hist": S.pit_histogram(u),
                "surv_at_c_mean": float(np.nanmean(A["surv_at_c"][m])),
            }
        rel = np.abs(A["lambda_grid_at_y"] - A["d_lambda"]) / np.maximum(A["d_lambda"], 1e-12)
        res["check_grid_vs_nll_compensator"] = {
            "median_rel_err": float(np.nanmedian(rel[dfn])),
            "p99_rel_err": float(np.nanquantile(rel[dfn], 0.99)),
            "max_abs_err": float(np.nanmax(np.abs(A["lambda_grid_at_y"] - A["d_lambda"])[dfn])),
        }
        if "d_lambda_nll_path" in A:
            res["thp_nll_path_vs_gauss_legendre"] = {
                "median_abs_diff": float(np.nanmedian(np.abs(A["d_lambda_nll_path"] - A["d_lambda"])[dfn])),
                "pit_ks_nll_path": S.ks_uniform(S.pit_from_compensator(A["d_lambda_nll_path"])[common]),
            }
        self.summary["temporal"] = res
        self.summary["timings_sec"]["temporal"] = time.time() - t0

    # ------------------------------------------------------------------
    def section_samples(self):
        t0 = time.time()
        A = self.arrays
        K = self.K
        n_ctx = A["target"].shape[0]
        sel = np.zeros(n_ctx, bool)
        sel[:: self.ctx_stride] = True
        sel &= A["defined"].astype(bool) | (not self.ad.has_intensity)
        keys = ["t_crps_fair", "t_pit_rand", "s_es_marg", "s_pit_x", "s_pit_y", "s_hpd_cond", "s_es_cond",
                "frac_invalid_time", "kde_nll", "legacy_frac_pit"]
        out = {k_: np.full(n_ctx, np.nan) for k_ in keys}
        offs = 0
        rt = self._rng("samples")
        for si, seq in enumerate(self.test):
            n = len(seq["times"])
            kk = self._contexts(n)
            idx = np.nonzero(sel[offs : offs + len(kk)])[0]
            if len(idx):
                k = kk[idx]
                b = np.zeros_like(k)
                gi = offs + idx
                prep = self.ad.prepare(seq["times"], seq["locations"])
                y = seq["times"][k + 1] - seq["times"][k]
                s_true = seq["locations"][k + 1]
                if self.ad.has_intensity:
                    L = self.ad.cum_hazard(prep, b, k, self.tau_grid)
                    tau, cens = S.sample_from_cumhazard_grid(L, self.tau_grid, K, rt)
                    valid_t = np.ones_like(tau, bool)
                    # spatial at sampled (finite) times
                    rr, cc = np.nonzero(~cens)
                    locs = np.full((len(k), K, 2), np.nan)
                    if len(rr):
                        tq = seq["times"][k[rr]] + tau[rr, cc]
                        kw = {"grid_n": 64} if getattr(self.ad, "spatial_sampler", "") == "fine_grid" else {}
                        xs, _ = self.ad.sample_space(prep, b[rr], k[rr], tq, rt, **kw)
                        locs[rr, cc] = xs
                    # conditional spatial at the true time (HPD rank PIT)
                    xs_c, lp_c = self.ad.sample_space_k(prep, b, k, seq["times"][k + 1], K, rt)
                    if hasattr(self.ad, "sampling_logpdf"):
                        lp_true = self.ad.sampling_logpdf(prep, b, k, seq["times"][k + 1], s_true[:, None, :])[:, 0]
                    else:
                        lp_true = A["log_ps"][gi]
                    out["s_hpd_cond"][gi] = S.hpd_rank_pit(lp_c, lp_true, rt)
                    out["s_es_cond"][gi] = S.energy_score_fair(xs_c, s_true)
                else:
                    dt, locs = self.ad.sample_next(prep, b, k, K)
                    valid_t = np.isfinite(dt) & (dt > 0)
                    tau = np.where(valid_t, dt, np.nan)
                    cens = np.zeros_like(valid_t)
                    locs = np.where(valid_t[..., None], locs, np.nan)
                    for i, gix in enumerate(gi):
                        if valid_t[i].sum() >= 8:
                            from seahorse.evaluation.likelihood import _joint_sample_kde_next_event_nll

                            out["kde_nll"][gix] = _joint_sample_kde_next_event_nll(
                                next_times=seq["times"][k[i]] + dt[i][valid_t[i]],
                                next_locs=locs[i][valid_t[i]],
                                true_next_time=float(seq["times"][k[i] + 1]),
                                true_next_loc=s_true[i],
                                history_end_time=float(seq["times"][k[i]]),
                            )
                tau_f = np.where(valid_t, tau, np.nan)
                out["frac_invalid_time"][gi] = 1.0 - valid_t.mean(1)
                out["t_crps_fair"][gi] = S.crps_fair_censored(np.where(valid_t, tau, np.nan), y, self.c, valid=valid_t)
                out["t_pit_rand"][gi] = S.randomized_rank_pit(tau_f, y, rt, valid=valid_t)
                out["legacy_frac_pit"][gi] = S.deterministic_fraction_pit(tau_f, y)
                vs = np.isfinite(locs).all(-1)
                out["s_es_marg"][gi] = S.energy_score_fair(locs, s_true, valid=vs)
                out["s_pit_x"][gi] = S.randomized_rank_pit(locs[..., 0], s_true[:, 0], rt, valid=vs)
                out["s_pit_y"][gi] = S.randomized_rank_pit(locs[..., 1], s_true[:, 1], rt, valid=vs)
            offs += len(kk)
        A.update({f"smp_{k_}": v for k_, v in out.items()})
        A["sampled"] = sel
        common = sel & (A["target"] >= COMMON_MIN_TARGET)
        res = {"K": K, "ctx_stride": self.ctx_stride, "n_contexts_sampled": int(sel.sum())}
        for name, m in (("common", common), ("all_sampled", sel)):
            r = {
                "t_crps_fair_mean": float(np.nanmean(out["t_crps_fair"][m])),
                "t_pit_rand_ks": S.ks_uniform(out["t_pit_rand"][m]),
                "t_pit_rand_tails": S.pit_tail_fractions(out["t_pit_rand"][m]),
                "s_es_marg_mean": float(np.nanmean(out["s_es_marg"][m])),
                "s_pit_x_ks": S.ks_uniform(out["s_pit_x"][m]),
                "s_pit_y_ks": S.ks_uniform(out["s_pit_y"][m]),
                "s_pit_marg_ks_mean": 0.5 * (S.ks_uniform(out["s_pit_x"][m])["ks"] + S.ks_uniform(out["s_pit_y"][m])["ks"]),
                "frac_invalid_time_mean": float(np.nanmean(out["frac_invalid_time"][m])),
            }
            if self.ad.has_intensity:
                r.update({
                    "s_hpd_cond_ks": S.ks_uniform(out["s_hpd_cond"][m]),
                    "s_hpd_cond_hist": S.pit_histogram(out["s_hpd_cond"][m]),
                    "s_es_cond_mean": float(np.nanmean(out["s_es_cond"][m])),
                    "crps_exact_mean_same_contexts": float(np.nanmean(A["crps_exact"][m])),
                    "pit_exact_ks_same_contexts": S.ks_uniform(A["pit_exact"][m]),
                })
            else:
                r["kde_nll_mean"] = float(np.nanmean(out["kde_nll"][m]))
            r["t_pit_rand_hist"] = S.pit_histogram(out["t_pit_rand"][m])
            res[name] = r
        self.summary["samples"] = res
        self.summary["timings_sec"]["samples"] = time.time() - t0

    # ------------------------------------------------------------------
    def section_intensity(self):
        if self.gt is None or not self.ad.has_intensity:
            return
        t0 = time.time()
        g = self.int_cfg["grid"]
        n_slices = self.int_cfg["slices"]
        if getattr(self.ad, "spatial_sampler", "").startswith("exact_inverse_flow_attncnf"):
            # joint attentive ODE per density evaluation: coarser grid (documented in METRICS_SPEC §5)
            g, n_slices = 16, 25
        cc = (np.arange(g) + 0.5) / g
        X, Y = np.meshgrid(self.gt.x_range[0] + cc * (self.gt.x_range[1] - self.gt.x_range[0]),
                           self.gt.y_range[0] + cc * (self.gt.y_range[1] - self.gt.y_range[0]), indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel()], -1)
        cell = (self.gt.x_range[1] - self.gt.x_range[0]) * (self.gt.y_range[1] - self.gt.y_range[0]) / (g * g)
        per_seq = []
        all_m, all_t = [], []
        slice_r = []
        for si, seq in enumerate(self.test):
            t = seq["times"]
            lo = t[max(self.ad.min_history - 1, 0)]
            ts = np.linspace(lo, t[-1], n_slices + 2)[1:-1]
            k = np.searchsorted(t, ts, side="left") - 1  # last event strictly before ts
            ok = k >= self.ad.min_history - 1
            ts, k = ts[ok], k[ok]
            prep = self.ad.prepare(t, seq["locations"])
            lam_m = self.ad.intensity(prep, np.zeros_like(k), k, ts, np.broadcast_to(pts, (len(k),) + pts.shape))
            lam_t = np.stack([self.gt.intensity(tt, pts, t, seq["locations"]) for tt in ts])
            sr = [S.pearson(lam_m[i], lam_t[i]) for i in range(len(ts))]
            slice_r.extend(sr)
            dt_slice = (t[-1] - lo) / (len(ts) + 1)
            per_seq.append({
                "seq": si,
                "slice_pearson_mean": float(np.nanmean(sr)),
                "pooled_pearson": S.pearson(lam_m, lam_t),
                "spearman": S.spearman(lam_m, lam_t),
                "log_rmse": float(np.sqrt(np.nanmean((np.log(np.maximum(lam_m, 1e-12)) - np.log(lam_t)) ** 2))),
                "mass_model": float(lam_m.sum() * cell * dt_slice),
                "mass_true": float(lam_t.sum() * cell * dt_slice),
                "n_events_window": int(((t > lo) & (t <= t[-1])).sum()),
            })
            all_m.append(lam_m.ravel())
            all_t.append(lam_t.ravel())
        am, at = np.concatenate(all_m), np.concatenate(all_t)
        self.summary["intensity"] = {
            "slice_pearson_mean": float(np.nanmean(slice_r)),
            "pooled_pearson": S.pearson(am, at),
            "spearman": S.spearman(am, at),
            "log_rmse": float(np.sqrt(np.nanmean((np.log(np.maximum(am, 1e-12)) - np.log(at)) ** 2))),
            "mass_ratio_model_over_true": float(sum(p["mass_model"] for p in per_seq) / sum(p["mass_true"] for p in per_seq)),
            "mass_ratio_true_over_events": float(sum(p["mass_true"] for p in per_seq) / sum(p["n_events_window"] for p in per_seq)),
            "per_sequence": per_seq,
            "grid": g, "slices": n_slices,
        }
        self.summary["timings_sec"]["intensity"] = time.time() - t0

    # ------------------------------------------------------------------
    def section_rollout(self):
        """Rollouts batched across test sequences for each start index k0.

        For start k0 every test sequence contributes its own history 0..k0 (all
        histories have equal length), so one batch holds n_seq x M trajectories.
        """
        t0 = time.time()
        H, M = self.roll["H"], self.roll["M"]
        rr = self._rng("rollout")
        rows = []
        n_min = min(len(s["times"]) for s in self.test)
        first = max(self.roll["first"], self.ad.min_history)
        starts = np.arange(first - 1, n_min - 1 - H, self.roll["stride"])
        nseq = len(self.test)
        for k0 in starts:
            ht0 = np.stack([s["times"][: k0 + 1] for s in self.test])
            hs0 = np.stack([s["locations"][: k0 + 1] for s in self.test])
            seq_prep = self.ad.prepare(ht0, hs0)
            prep = self.ad.rollout_init(seq_prep, int(k0), M)
            Btot = nseq * M
            ok = np.ones(Btot, bool)
            for h in range(H):
                b = np.arange(Btot)
                k = np.full(Btot, prep.T - 1)
                t_last = prep.times[:, -1]
                if self.ad.has_intensity:
                    L = self.ad.cum_hazard(prep, b, k, self.tau_grid)
                    tau, cens = S.sample_from_cumhazard_grid(L, self.tau_grid, 1, rr)
                    tau, cens = tau[:, 0], cens[:, 0]
                    ok &= ~cens
                    tau = np.where(cens, self.tau_max, tau)
                    kw = {"grid_n": 64} if getattr(self.ad, "spatial_sampler", "") == "fine_grid" else {}
                    xs, _ = self.ad.sample_space(prep, b, k, t_last + tau, rr, **kw)
                else:
                    dt, loc = self.ad.sample_next(prep, b, k, 1)
                    tau, xs = dt[:, 0], loc[:, 0]
                    bad = ~(np.isfinite(tau) & (tau > 0))
                    ok &= ~bad
                    tau = np.where(bad, 1e-6, tau)
                prep = self.ad.extend(prep, t_last + tau, xs)
            for si, seq in enumerate(self.test):
                t, s = seq["times"], seq["locations"]
                rs = slice(si * M, (si + 1) * M)
                gen_off = prep.times[rs, k0 + 1 :] - t[k0]
                gen_loc = prep.locs[rs, k0 + 1 :]
                okr = ok[rs]
                true_off = t[k0 + 1 : k0 + 1 + H] - t[k0]
                true_loc = s[k0 + 1 : k0 + 1 + H]
                row = {"seq": si, "k0": int(k0), "frac_invalid": float(1 - okr.mean())}
                for Hh in (1, 5, 10):
                    if Hh > H:
                        continue
                    row[f"es_time_h{Hh}"] = float(S.energy_score_fair(gen_off[None, :, :Hh], true_off[None, :Hh], valid=okr[None])[0])
                    row[f"es_space_h{Hh}"] = float(S.energy_score_fair(
                        gen_loc[None, :, :Hh].reshape(1, M, 2 * Hh), true_loc[None, :Hh].reshape(1, 2 * Hh), valid=okr[None])[0])
                rows.append(row)
        res = {"H": H, "M": M, "n_starts": len(rows), "start_indices": [int(x) for x in starts]}
        for key in rows[0] if rows else []:
            if key.startswith("es_") or key == "frac_invalid":
                res[key] = float(np.nanmean([r[key] for r in rows]))
        self.summary["rollout"] = res
        self.rollout_rows = rows
        self.summary["timings_sec"]["rollout"] = time.time() - t0

    # ------------------------------------------------------------------
    def save(self):
        A = self.arrays
        np.savez_compressed(self.out / "per_context.npz", **{k: np.asarray(v) for k, v in A.items()})
        if getattr(self, "rollout_rows", None):
            (self.out / "rollout_rows.json").write_text(json.dumps(self.rollout_rows))
        self.summary["timings_sec"]["total"] = time.time() - self.t_start
        (self.out / "summary.json").write_text(json.dumps(self.summary, indent=2, default=float))

    def run(self, sections):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        order = ["nll", "temporal", "samples", "intensity", "rollout"]
        for sec in order:
            if isinstance(self.ad, _NLLOnly) and sec != "nll":
                continue
            if sec in sections:
                getattr(self, f"section_{sec}")()
                self.save()  # checkpoint after every section
        return self.summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--train", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--gt-params", default=None)
    p.add_argument("--sections", default="nll,temporal,samples,intensity,rollout")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--K-flow", type=int, default=None)
    p.add_argument("--ctx-stride", type=int, default=1)
    p.add_argument("--ctx-stride-flow", type=int, default=None)
    p.add_argument("--c", type=float, default=None)
    p.add_argument("--tau-grid-n", type=int, default=1024)
    p.add_argument("--roll-M", type=int, default=50)
    p.add_argument("--roll-H", type=int, default=10)
    p.add_argument("--roll-stride", type=int, default=40)
    p.add_argument("--max-seqs", type=int, default=None)
    a = p.parse_args(argv)
    ev = Evaluator(a.run, a.test, a.train, a.out, device=a.device, seed=a.seed, K=a.K, K_flow=a.K_flow,
                   ctx_stride=a.ctx_stride, ctx_stride_flow=a.ctx_stride_flow, c=a.c, tau_grid_n=a.tau_grid_n,
                   gt_params=a.gt_params, roll_M=a.roll_M, roll_H=a.roll_H, roll_starts_stride=a.roll_stride,
                   max_seqs=a.max_seqs)
    ev.run([x.strip() for x in a.sections.split(",") if x.strip()])
    print(json.dumps({k: ev.summary.get(k) for k in ("nll", "timings_sec")}, indent=1, default=float))


if __name__ == "__main__":
    main()
