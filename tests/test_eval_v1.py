"""Tests for the v1.0 evaluation layer (seahorse.evaluation.v1).

Analytic checks of every scoring primitive, plus per-family adapter checks on
tiny randomly initialised presets:

* the adapter's per-event NLL equals the packaged (fixed) benchmark NLL;
* the cumulative hazard on the waiting-time grid, evaluated at the observed
  wait, equals the NLL-path compensator increment;
* exact spatial draws carry log-densities that match the model density;
* CNF inverse-flow draws are invertible (tracked log-density == forward
  log-density of the drawn point).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from seahorse.evaluation.v1 import scoring as S

try:
    from tests.eval_test_helpers import MINI_OVERRIDES, make_saved_run
except ImportError:  # pragma: no cover
    from eval_test_helpers import MINI_OVERRIDES, make_saved_run

RNG = np.random.default_rng


# ---------------------------------------------------------------------------
# scoring primitives
# ---------------------------------------------------------------------------


def _simulate_exp_hawkes(mu, alpha, beta, T, rng):
    """Ogata thinning for lambda(t) = mu + alpha sum exp(-beta (t - t_j))."""
    t, ev = 0.0, []
    excite = 0.0  # sum alpha exp(-beta (t - t_j)) at current t
    while True:
        lam_bar = mu + excite
        w = rng.exponential(1.0 / lam_bar)
        t += w
        excite *= math.exp(-beta * w)
        if t > T:
            return np.asarray(ev)
        if rng.random() * lam_bar <= mu + excite:
            ev.append(t)
            excite += alpha


def _exp_hawkes_compensator(times, mu, alpha, beta):
    out = []
    prev = 0.0
    for i, t in enumerate(times):
        hist = times[:i]
        c = mu * (t - prev) + (alpha / beta) * np.sum(
            np.exp(-beta * (prev - hist)) - np.exp(-beta * (t - hist))
        )
        out.append(c)
        prev = t
    return np.asarray(out)


class TestScoring(unittest.TestCase):
    def test_pit_poisson_is_uniform(self):
        rng = RNG(0)
        rate = 3.0
        dt = rng.exponential(1.0 / rate, size=5000)
        u = S.pit_from_compensator(rate * dt)
        self.assertGreater(S.ks_uniform(u)["p_value"], 0.01)

    def test_pit_hawkes_is_uniform(self):
        rng = RNG(1)
        mu, alpha, beta = 1.0, 0.8, 2.0
        times = _simulate_exp_hawkes(mu, alpha, beta, 2500.0, rng)
        dl = _exp_hawkes_compensator(times, mu, alpha, beta)
        u = S.pit_from_compensator(dl)
        self.assertGreater(len(u), 3000)
        self.assertGreater(S.ks_uniform(u)["p_value"], 0.01)
        # a misspecified (Poisson) compensator is detected
        rate = len(times) / times[-1]
        u_bad = S.pit_from_compensator(rate * np.diff(np.concatenate([[0.0], times])))
        self.assertLess(S.ks_uniform(u_bad)["p_value"], 1e-6)

    def test_crps_grid_matches_exponential_closed_form(self):
        rate, c = 2.5, 1.5
        tg = S.tau_grid_quadratic(4 * c, 4096)
        F = 1.0 - np.exp(-rate * tg)
        ys = np.array([0.0, 1e-4, 0.03, 0.4, 1.2, 1.5, 3.0])
        got = S.crps_from_cdf_grid(np.repeat(F[None], len(ys), 0), tg, ys, c)
        want = np.array([S.crps_exponential_truncated(rate, y, c) for y in ys])
        np.testing.assert_allclose(got, want, atol=2e-6)
        # untruncated limit
        big = 40.0
        tg2 = S.tau_grid_quadratic(big, 8192)
        F2 = 1.0 - np.exp(-rate * tg2)
        got2 = S.crps_from_cdf_grid(F2[None], tg2, np.array([0.3]), big)[0]
        self.assertAlmostEqual(got2, S.crps_exponential(rate, 0.3), places=5)

    def test_fair_crps_is_unbiased(self):
        rng = RNG(2)
        rate, c, y, K = 1.7, 2.0, 0.6, 8
        draws = rng.exponential(1.0 / rate, size=(40000, K))
        est = S.crps_fair_censored(draws, np.full(40000, y), c).mean()
        self.assertAlmostEqual(est, S.crps_exponential_truncated(rate, y, c), delta=3e-3)

    def test_randomized_pit_uniform_with_ties_and_legacy_is_not(self):
        rng = RNG(3)
        N, K = 20000, 32
        support = np.array([0.0, 1.0, 2.0])
        p = np.array([0.5, 0.3, 0.2])
        samples = rng.choice(support, size=(N, K), p=p)
        truth = rng.choice(support, size=N, p=p)
        u = S.randomized_rank_pit(samples, truth, rng)
        self.assertGreater(S.ks_uniform(u)["p_value"], 0.01)
        legacy = S.deterministic_fraction_pit(samples, truth)
        self.assertLess(S.ks_uniform(legacy)["p_value"], 1e-6)
        # continuous case, small K: deterministic fraction has a KS floor ~1/(K+1)
        s2 = rng.random((N, 4))
        t2 = rng.random(N)
        self.assertGreater(S.ks_uniform(S.deterministic_fraction_pit(s2, t2))["ks"], 0.15)
        self.assertGreater(S.ks_uniform(S.randomized_rank_pit(s2, t2, rng))["p_value"], 0.01)

    def test_energy_score_1d_equals_fair_crps(self):
        rng = RNG(4)
        x = rng.normal(size=(50, 16))
        y = rng.normal(size=50)
        es = S.energy_score_fair(x[..., None], y[:, None])
        cr = S.crps_fair_censored(x - 10.0, y - 10.0, 1e9)  # shift to keep x > 0 irrelevant
        np.testing.assert_allclose(es, cr, atol=1e-10)

    def test_inverse_transform_sampler_exponential(self):
        rng = RNG(5)
        rate = 4.0
        tg = S.tau_grid_quadratic(5.0, 2048)
        L = np.repeat((rate * tg)[None], 3, 0)
        tau, cens = S.sample_from_cumhazard_grid(L, tg, 20000, rng)
        self.assertLess(cens.mean(), 1e-6 + math.exp(-rate * 5.0) * 2)
        from scipy.stats import kstest

        self.assertGreater(kstest(tau[0], "expon", args=(0, 1 / rate)).pvalue, 0.01)

    def test_hpd_rank_pit_calibrated_gaussian(self):
        rng = RNG(6)
        N, K = 5000, 50
        mu = rng.normal(size=(N, 2))
        samp = mu[:, None, :] + rng.normal(size=(N, K, 2))
        obs = mu + rng.normal(size=(N, 2))
        lp = lambda x, m: -0.5 * ((x - m) ** 2).sum(-1)  # noqa: E731
        u = S.hpd_rank_pit(lp(samp, mu[:, None, :]), lp(obs, mu), rng)
        self.assertGreater(S.ks_uniform(u)["p_value"], 0.01)
        # an over-concentrated model (true spread 2x) pushes u towards 1
        obs_wide = mu + 2.0 * rng.normal(size=(N, 2))
        u2 = S.hpd_rank_pit(lp(samp, mu[:, None, :]), lp(obs_wide, mu), rng)
        self.assertGreater(np.mean(u2), 0.7)


# ---------------------------------------------------------------------------
# family adapters on tiny random presets
# ---------------------------------------------------------------------------

EXTRA_MINI = {
    "neural_jumpcnf": [
        "model.hidden_dim=8",
        "model.backbone.tpp_hidden_dims=8-8",
        "model.decoder.spatial.hidden_dims=8-8",
    ],
    "neural_attncnf": [
        "model.hidden_dim=8",
        "model.backbone.tpp_hidden_dims=8-8",
        "model.decoder.spatial.hidden_dims=8-8",
        "model.decoder.spatial.num_heads=2",
    ],
    "rmtpp_gmm": [],
    "thp_gmm": [],
}


def _toy_sequence(n=14, seed=0):
    rng = RNG(seed)
    t = np.cumsum(rng.exponential(0.2, size=n)) + 0.05
    s = rng.random((n, 2))
    return t, s


class _AdapterCase(unittest.TestCase):
    preset = None

    @classmethod
    def setUpClass(cls):
        if cls.preset is None:
            raise unittest.SkipTest("base")
        cls.tmp = tempfile.TemporaryDirectory()
        overrides = MINI_OVERRIDES.get(cls.preset, EXTRA_MINI.get(cls.preset, []))
        run = make_saved_run(Path(cls.tmp.name), preset=cls.preset, override_list=overrides)
        from seahorse.evaluation.v1.loading import load_runner_compat
        from seahorse.evaluation.v1.run_eval import make_adapter

        cls.runner, _ = load_runner_compat(run)
        torch.manual_seed(0)
        with torch.no_grad():  # make the random model non-trivial (zero-init last layers)
            for name, p in cls.runner.model.named_parameters():
                if any(k in name for k in ("spatial_decoder", "hawkes_decoder", "kernel")):
                    p.add_(0.05 * torch.randn_like(p))
        cls.runner.model.eval()
        cls.ad = make_adapter(cls.runner, torch.device("cpu"))
        cls.t, cls.s = _toy_sequence()
        cls.prep = cls.ad.prepare(cls.t, cls.s)
        cls.et = cls.ad.event_terms(cls.prep)

    @classmethod
    def tearDownClass(cls):
        if cls.preset is not None:
            cls.tmp.cleanup()

    def test_nll_matches_packaged(self):
        from seahorse.evaluation.likelihood import compute_next_event_test_nll

        pkg = compute_next_event_test_nll(self.runner, [{"times": self.t, "locations": self.s}], device=torch.device("cpu"))
        per = np.asarray(pkg["per_context_nll"], float)
        tgt = np.arange(1, len(self.t))
        d = self.et.defined[tgt]
        self.assertTrue(d.any())
        np.testing.assert_allclose(self.et.nll[tgt][d], per[d], atol=5e-3, rtol=1e-4)

    def test_cum_hazard_matches_compensator(self):
        tgt = np.arange(1, len(self.t))
        d = self.et.defined[tgt]
        k = (tgt - 1)[d]
        tg = S.tau_grid_quadratic(4.0 * float(np.diff(self.t).max()) + 1.0, 2048)
        L = self.ad.cum_hazard(self.prep, np.zeros_like(k), k, tg)
        y = self.t[k + 1] - self.t[k]
        ly = np.array([np.interp(y[i], tg, L[i]) for i in range(len(k))])
        np.testing.assert_allclose(ly, self.et.d_lambda[k + 1], rtol=self.cum_rtol, atol=1e-4)

    cum_rtol = 2e-3

    def test_spatial_samples_have_consistent_logp(self):
        tgt = np.arange(1, len(self.t))
        k = (tgt - 1)[self.et.defined[tgt]][:4]
        b = np.zeros_like(k)
        tq = self.t[k + 1]
        x, lp = self.ad.sample_space(self.prep, b, k, tq, RNG(0))
        self.assertTrue(np.isfinite(x).all())
        if hasattr(self.ad, "sampling_logpdf"):
            lp2 = self.ad.sampling_logpdf(self.prep, b, k, tq, x[:, None, :])[:, 0]
        else:
            lp2 = self.ad.spatial_logpdf(self.prep, b, k, tq, x[:, None, :])[:, 0]
        np.testing.assert_allclose(lp, lp2, atol=self.logp_atol)
        # model density at the truth equals the NLL-path spatial term
        if not hasattr(self.ad, "sampling_logpdf"):
            lpt = self.ad.spatial_logpdf(self.prep, b, k, tq, self.s[k + 1][:, None, :])[:, 0]
            np.testing.assert_allclose(lpt, self.et.log_ps[k + 1], atol=self.logp_atol)

    logp_atol = 1e-4

    def test_rollout_extend_matches_teacher_forcing(self):
        k0 = 6
        rp = self.ad.rollout_init(self.prep, k0, 2)
        rp = self.ad.extend(rp, np.repeat(self.t[k0 + 1], 2), np.repeat(self.s[k0 + 1][None], 2, 0))
        tg = S.tau_grid_quadratic(2.0, 256)
        L1 = self.ad.cum_hazard(rp, np.array([0, 1]), np.array([k0 + 1, k0 + 1]), tg)
        L0 = self.ad.cum_hazard(self.prep, np.array([0]), np.array([k0 + 1]), tg)
        np.testing.assert_allclose(L1, np.repeat(L0, 2, 0), rtol=2e-3, atol=1e-5)


class TestNJSDEAdapter(_AdapterCase):
    preset = "njsde"


class TestJumpCNFAdapter(_AdapterCase):
    preset = "neural_jumpcnf"
    logp_atol = 2e-3

    def test_roundtrip(self):
        from seahorse.evaluation.v1.families.neural_flows import check_roundtrip

        k = np.array([0, 3, 7, 12])
        res = check_roundtrip(self.ad, self.prep, np.zeros_like(k), k, self.t[k] + 0.07, RNG(1), n=4)
        self.assertLess(res["max_abs_logp_err"], 2e-3)
        self.assertLess(res["max_abs_fast_density_err"], 2e-3)


class TestAttnCNFAdapter(_AdapterCase):
    preset = "neural_attncnf"
    logp_atol = 2e-3

    def test_roundtrip(self):
        from seahorse.evaluation.v1.families.neural_flows import check_roundtrip

        k = np.array([0, 3, 7, 12])
        res = check_roundtrip(self.ad, self.prep, np.zeros_like(k), k, self.t[k] + 0.07, RNG(1), n=4)
        self.assertLess(res["max_abs_logp_err"], 2e-3)


class TestAutoSTPPAdapter(_AdapterCase):
    preset = "auto_stpp"

    def test_density_normalised_on_box(self):
        k = np.array([5, 9])
        n = 200
        c = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(c, c, indexing="ij")
        pts_hat = np.stack([X.ravel(), Y.ravel()], -1)
        pts = self.ad._unscale_locs(pts_hat)
        lp = self.ad.spatial_logpdf(self.prep, np.zeros_like(k), k, self.t[k] + 0.05,
                                    np.broadcast_to(pts, (2,) + pts.shape))
        mass = np.exp(lp).mean(-1) * np.exp(self.ad.log_jac_s)
        np.testing.assert_allclose(mass, 1.0, atol=2e-3)


    def test_separable_grid_matches_direct_kernel(self):
        k = np.array([5, 9])
        n = 16
        tq = self.t[k] + 0.05
        probs = self.ad._grid_probs(self.prep, np.zeros_like(k), k, tq, n)
        c = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(c, c, indexing="ij")
        pts = self.ad._unscale_locs(np.stack([X.ravel(), Y.ravel()], -1))
        lam = self.ad.intensity(self.prep, np.zeros_like(k), k, tq, np.broadcast_to(pts, (2,) + pts.shape))
        np.testing.assert_allclose(probs, lam / lam.sum(-1, keepdims=True), rtol=1e-4, atol=1e-7)


class TestDeepSTPPAdapter(_AdapterCase):
    preset = "deep_stpp"


class TestNSMPPAdapter(_AdapterCase):
    preset = "nsmpp"
    cum_rtol = 2e-2  # NLL path: 2-node midpoint per interval; grid: fine trapezoid


class TestRMTPPAdapter(_AdapterCase):
    preset = "rmtpp_gmm"


class TestTHPAdapter(_AdapterCase):
    preset = "thp_gmm"


class TestRadialInverse(unittest.TestCase):
    def test_radial_roundtrip(self):
        from seahorse.models.spatial_models.neural_jumpcnf import HypernetworkRadialFlow
        from seahorse.evaluation.v1.families.neural_flows import _radial_inverse

        torch.manual_seed(0)
        flow = HypernetworkRadialFlow(2, cond_dim=5, nflows=4)
        with torch.no_grad():
            for p in flow.parameters():
                p.add_(0.2 * torch.randn_like(p))
            x = torch.randn(64, 2)
            cond = torch.randn(64, 5)
            y, lp = flow(x, logpx=torch.zeros(64), cond=cond)
            x_rec, logdet = _radial_inverse(flow, y, cond)
        torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(-logdet, lp, atol=1e-4, rtol=1e-4)


class TestGroundTruth(unittest.TestCase):
    """Needs a local HawkesNest Suite 3 checkout: set SEAHORSE_HAWKESNEST_ROOT."""

    ROOT = Path(os.environ.get("SEAHORSE_HAWKESNEST_ROOT", "")) / "suite3_entanglement"

    def test_reproduces_reference_grid(self):
        if not os.environ.get("SEAHORSE_HAWKESNEST_ROOT") or not (
            self.ROOT / "ground_truth/L1_params.json"
        ).exists():
            raise unittest.SkipTest("HawkesNest data not available (SEAHORSE_HAWKESNEST_ROOT unset)")
        from seahorse.evaluation.v1.gt_hawkesnest import TravelingWaveGT

        gt = TravelingWaveGT.from_file(self.ROOT / "ground_truth/L1_params.json")
        g = np.load(self.ROOT / "ground_truth/L1_intensity_grid_r0.npz")
        seq = json.loads((self.ROOT / "jsonl/L1/test.jsonl").read_text().splitlines()[0])
        X, Y = np.meshgrid(g["x_grid"], g["y_grid"], indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel()], -1)
        for i in (10, 50, 90):
            lam = gt.intensity(float(g["t_grid"][i]), pts, np.asarray(seq["times"]), np.asarray(seq["locations"]))
            np.testing.assert_allclose(lam.reshape(X.shape), g["lambda_"][i], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
