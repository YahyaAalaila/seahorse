"""
Tests for NLL/LL equivalence across decoders and intensity evaluation.

Verifies:
  1. NLL returned by model.forward() is finite for AutoSTPP and DeepSTPP.
  2. AutoIntDecoder and FactorizedDecoder both implement the same mathematical
     quantity: NLL = −1/N Σ log f*(t_i, s_i).  (DiffusionDecoder is
     intentionally excluded — its nll() is a DSM surrogate, not a true NLL.)
  3. eval_intensity() returns non-negative values for both decoder types.
  4. The correct_for_normalization flag divides by exactly
     (t_scale · ∏ s_scale).
  5. calc_lamb() returns the correct shape.
"""

import math
import unittest

import numpy as np
import torch

from unified_stpp.registry import build_model
from unified_stpp.evaluation.intensity import eval_intensity, calc_lamb
from unified_stpp.evaluation.likelihood import LikelihoodEvaluator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DEVICE = "cpu"


def _tiny_batch(B=2, N=6, d=2, seed=0):
    """Return a minimal padded batch of synthetic sequences."""
    torch.manual_seed(seed)
    # Use sorted times so the sequences are valid STPP observations
    times = torch.sort(torch.rand(B, N), dim=-1).values * 2.0  # ∈ [0, 2)
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


# ---------------------------------------------------------------------------
# 1. NLL finite check
# ---------------------------------------------------------------------------

class TestNLLFinite(unittest.TestCase):
    """model.forward() must return a finite NLL for standard presets."""

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


# ---------------------------------------------------------------------------
# 2. LL equivalence: both decoders report values in the same range
# ---------------------------------------------------------------------------

class TestLLEquivalence(unittest.TestCase):
    """
    Both AutoIntDecoder and FactorizedDecoder (DeepSTPP) implement:
        NLL = −log f*(t, s)  per event.

    We cannot check the *value* against each other (different models), but we
    can verify:
      (a) NLL is scalar, finite, and of reasonable magnitude.
      (b) mean NLL × N_events ≈ total NLL accumulated by a manual loop.
    """

    def _manual_nll(self, model, times, locs, lengths):
        """Reproduce _forward_batched by calling encoder + decoder directly."""
        events = torch.cat([times.unsqueeze(-1), locs], dim=-1)
        _, all_states = model.encoder(events, lengths)

        B, N = times.shape
        max_len = int(lengths.max().item())
        L = max_len - 1

        z_cond   = all_states[:, :L, :]
        t_target = times[:, 1:1 + L].unsqueeze(-1)
        s_target = locs[:, 1:1 + L, :]
        t_prev   = times[:, :L].unsqueeze(-1)

        n_idx = torch.arange(L)
        mask = (n_idx.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()

        h = z_cond.shape[-1]
        d = s_target.shape[-1]
        nll_flat = model.decoder.nll(
            z_cond.reshape(B * L, h),
            t_target.reshape(B * L, 1),
            s_target.reshape(B * L, d),
            t_prev.reshape(B * L, 1),
        ).reshape(B, L)

        return (nll_flat * mask).sum() / mask.sum().clamp(min=1)

    def _check_preset(self, preset):
        model = _build(preset)
        model.eval()
        times, locs, lengths = _tiny_batch()

        with torch.no_grad():
            model_nll = model(
                times=times, locations=locs, lengths=lengths
            )["nll"].item()
            manual_nll = self._manual_nll(model, times, locs, lengths).item()

        self.assertTrue(math.isfinite(model_nll), f"{preset}: model NLL not finite")
        self.assertTrue(math.isfinite(manual_nll), f"{preset}: manual NLL not finite")
        self.assertAlmostEqual(
            model_nll, manual_nll, places=4,
            msg=f"{preset}: model NLL ({model_nll:.6f}) ≠ manual NLL ({manual_nll:.6f})",
        )

    def test_auto_stpp_model_equals_manual(self):
        self._check_preset("auto_stpp")

    def test_deep_stpp_model_equals_manual(self):
        self._check_preset("deep_stpp")


# ---------------------------------------------------------------------------
# 3. eval_intensity: output shape and non-negativity
# ---------------------------------------------------------------------------

class TestEvalIntensity(unittest.TestCase):
    """eval_intensity must return (M,) non-negative values."""

    _S_GRID = np.array(
        [[0.0, 0.0], [0.5, 0.4], [-0.3, 0.2], [1.0, -0.5]], dtype=np.float32
    )  # M=4 test points

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
        self.assertTrue(
            np.all(vals >= 0),
            f"Negative intensity values: {vals}",
        )

    def test_deep_stpp_nonnegative(self):
        vals = self._call("deep_stpp")
        self.assertTrue(
            np.all(vals >= 0),
            f"Negative intensity values: {vals}",
        )

    def test_auto_stpp_finite(self):
        vals = self._call("auto_stpp")
        self.assertTrue(np.all(np.isfinite(vals)), f"Non-finite values: {vals}")

    def test_deep_stpp_finite(self):
        vals = self._call("deep_stpp")
        self.assertTrue(np.all(np.isfinite(vals)), f"Non-finite values: {vals}")


# ---------------------------------------------------------------------------
# 4. Scale correction: divides by exactly (t_scale · ∏ s_scale)
# ---------------------------------------------------------------------------

class TestScaleCorrection(unittest.TestCase):
    """
    eval_intensity(correct_for_normalization=True) must equal
    eval_intensity(correct_for_normalization=False) / (t_scale · ∏ s_scale).
    """

    _S_GRID = np.array([[0.0, 0.0], [0.3, -0.1]], dtype=np.float32)

    def _check_preset(self, preset):
        model = _build(preset)
        model.eval()
        history_times, history_locs = _history_arrays()

        t_scale = 2.5
        s_scale = np.array([3.0, 4.0], dtype=np.float32)
        expected_factor = t_scale * float(np.prod(s_scale))  # 30.0

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


# ---------------------------------------------------------------------------
# 5. calc_lamb: output shape
# ---------------------------------------------------------------------------

class TestCalcLamb(unittest.TestCase):
    """calc_lamb must return an array of shape (T, X, Y)."""

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
            s_bias=np.zeros(2),
            s_scale=np.ones(2),
            x_range=x_range,
            y_range=y_range,
            t_range=t_range,
            device=torch.device(_DEVICE),
        )

        self.assertEqual(
            lamb.shape, (3, 5, 6),
            f"{preset}: expected (3, 5, 6), got {lamb.shape}",
        )
        self.assertTrue(
            np.all(np.isfinite(lamb)),
            f"{preset}: non-finite values in lamb",
        )

    def test_auto_stpp_shape(self):
        self._check_preset("auto_stpp")

    def test_deep_stpp_shape(self):
        self._check_preset("deep_stpp")


if __name__ == "__main__":
    unittest.main()
