"""Likelihood/intensity regression checks for active presets."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from unified_stpp.evaluation.intensity import calc_lamb, eval_intensity
from unified_stpp.models.configs import ConfigRegistry
from unified_stpp.registry import build_model


_DEVICE = "cpu"


def _tiny_batch(B=2, N=6, d=2, seed=0):
    torch.manual_seed(seed)
    times = torch.sort(torch.rand(B, N), dim=-1).values * 2.0
    locs = torch.randn(B, N, d) * 0.4
    lengths = torch.full((B,), N, dtype=torch.long)
    return times, locs, lengths


def _build(preset, hidden_dim=16, seed=0):
    torch.manual_seed(seed)
    return build_model({}, preset=preset, spatial_dim=2, hidden_dim=hidden_dim)


def _history_arrays(N=5, d=2, seed=1):
    rng = np.random.RandomState(seed)
    times = np.sort(rng.rand(N).astype(np.float32)) * 1.5
    locs = rng.randn(N, d).astype(np.float32) * 0.3
    return times, locs


class TestNLLFinite(unittest.TestCase):
    def _check_preset(self, preset):
        model = _build(preset)
        model.eval()
        times, locs, lengths = _tiny_batch()
        with torch.no_grad():
            out = model(times=times, locations=locs, lengths=lengths)
        self.assertIn("nll", out)
        self.assertTrue(
            torch.isfinite(out["nll"]),
            f"{preset}: NLL is not finite — got {out['nll'].item():.4g}",
        )

    def test_auto_stpp_nll_finite(self):
        self._check_preset("auto_stpp")

    def test_deep_stpp_nll_finite(self):
        self._check_preset("deep_stpp")


class TestEvalIntensity(unittest.TestCase):
    _S_GRID = np.array(
        [[0.0, 0.0], [0.5, 0.4], [-0.3, 0.2], [1.0, -0.5]], dtype=np.float32
    )

    def _call(self, preset, correct=False):
        model = _build(preset)
        model.eval()
        history_times, history_locs = _history_arrays()
        return eval_intensity(
            model=model,
            t_query=2.0,
            s_grid=self._S_GRID,
            history_times=history_times,
            history_locs=history_locs,
            t_bias=0.0,
            t_scale=1.0,
            s_bias=np.zeros(2),
            s_scale=np.ones(2),
            device=torch.device(_DEVICE),
            correct_for_normalization=correct,
        )

    def test_auto_stpp_shape(self):
        vals = self._call("auto_stpp")
        self.assertEqual(vals.shape, (4,))

    def test_deep_stpp_shape(self):
        vals = self._call("deep_stpp")
        self.assertEqual(vals.shape, (4,))

    def test_auto_stpp_nonnegative(self):
        vals = self._call("auto_stpp")
        self.assertTrue(np.all(vals >= 0), f"Negative intensity values: {vals}")

    def test_deep_stpp_nonnegative(self):
        vals = self._call("deep_stpp")
        self.assertTrue(np.all(vals >= 0), f"Negative intensity values: {vals}")

    def test_auto_stpp_finite(self):
        vals = self._call("auto_stpp")
        self.assertTrue(np.all(np.isfinite(vals)), f"Non-finite values: {vals}")

    def test_deep_stpp_finite(self):
        vals = self._call("deep_stpp")
        self.assertTrue(np.all(np.isfinite(vals)), f"Non-finite values: {vals}")


class TestPaperPresetStatus(unittest.TestCase):
    def test_neural_presets_are_benchmark_supported(self):
        for preset in ("njsde", "neural_jumpcnf", "neural_attncnf"):
            with self.subTest(preset=preset):
                self.assertEqual(ConfigRegistry.canonical_status(preset), "canonical")


class TestScaleCorrection(unittest.TestCase):
    _S_GRID = np.array([[0.0, 0.0], [0.3, -0.1]], dtype=np.float32)

    def _check_preset(self, preset):
        model = _build(preset)
        model.eval()
        history_times, history_locs = _history_arrays()

        t_scale = 2.5
        s_scale = np.array([3.0, 4.0], dtype=np.float32)
        expected_factor = t_scale * float(np.prod(s_scale))

        kw = dict(
            model=model,
            t_query=2.0,
            s_grid=self._S_GRID,
            history_times=history_times,
            history_locs=history_locs,
            t_bias=0.0,
            t_scale=t_scale,
            s_bias=np.zeros(2, dtype=np.float32),
            s_scale=s_scale,
            device=torch.device(_DEVICE),
        )

        raw = eval_intensity(**kw, correct_for_normalization=False)
        corrected = eval_intensity(**kw, correct_for_normalization=True)

        np.testing.assert_allclose(
            corrected,
            raw / expected_factor,
            rtol=1e-5,
            err_msg=(
                f"{preset}: scale correction must divide by "
                f"t_scale * prod(s_scale) = {expected_factor}"
            ),
        )

    def test_auto_stpp_scale_correction(self):
        self._check_preset("auto_stpp")

    def test_deep_stpp_scale_correction(self):
        self._check_preset("deep_stpp")


class TestCalcLamb(unittest.TestCase):
    def _check_preset(self, preset):
        model = _build(preset)
        model.eval()
        history_times, history_locs = _history_arrays()

        x_range = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
        y_range = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
        t_range = np.array([1.6, 1.8, 2.0], dtype=np.float32)

        lamb = calc_lamb(
            model=model,
            history_times=history_times,
            history_locs=history_locs,
            t_bias=0.0,
            t_scale=1.0,
            s_bias=np.zeros(2, dtype=np.float32),
            s_scale=np.ones(2, dtype=np.float32),
            x_range=x_range,
            y_range=y_range,
            t_range=t_range,
            device=torch.device(_DEVICE),
            correct_for_normalization=True,
        )

        self.assertEqual(lamb.shape, (len(t_range), len(x_range), len(y_range)))
        self.assertTrue(np.all(np.isfinite(lamb)))

    def test_auto_stpp_calc_lamb_shape(self):
        self._check_preset("auto_stpp")

    def test_deep_stpp_calc_lamb_shape(self):
        self._check_preset("deep_stpp")


if __name__ == "__main__":
    unittest.main()
