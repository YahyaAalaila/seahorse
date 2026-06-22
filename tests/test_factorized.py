"""Tests for the factorized baseline family (poisson_gmm, hawkes_gmm, selfcorrecting_gmm)."""

import unittest

import torch

from seahorse.registry import build_model


# Shared test fixtures
_TIMES = torch.tensor([[0.0, 0.2, 0.5, 0.9]], dtype=torch.float32)
_LOCATIONS = torch.tensor(
    [[[0.0, 0.0], [0.3, -0.1], [0.1, 0.2], [-0.2, 0.1]]], dtype=torch.float32
)
_LENGTHS = torch.tensor([4], dtype=torch.long)
_MASK = torch.tensor([[1.0, 1.0, 1.0, 1.0]])


class TestFactorizedRegistrySmoke(unittest.TestCase):
    """Test 1: config/registry smoke — build_model returns UnifiedSTPP for each preset."""

    def test_presets_build(self):
        from seahorse.models.unified_model import UnifiedSTPP

        for preset in ("poisson_gmm", "hawkes_gmm", "selfcorrecting_gmm"):
            with self.subTest(preset=preset):
                torch.manual_seed(0)
                model = build_model(
                    config={},
                    preset=preset,
                    spatial_dim=2,
                    hidden_dim=16,
                    event_cov_dim=0,
                    field_cov_dim=0,
                )
                self.assertIsInstance(model, UnifiedSTPP)


class TestTemporalComponentSmoke(unittest.TestCase):
    """Test 2: each temporal process class — shape (1,) and finite logprob."""

    def _check(self, cls):
        model = cls()
        t0 = torch.tensor([0.0])
        t1 = torch.tensor([0.9])
        out = model.logprob(_TIMES, _LOCATIONS, _MASK, t0, t1)
        self.assertEqual(out.shape, (1,))
        self.assertTrue(torch.isfinite(out).all(), f"{cls.__name__} logprob not finite: {out}")

    def test_poisson(self):
        from seahorse.models.temporal_models.parametric_processes import (
            HomogeneousPoissonProcess,
        )
        self._check(HomogeneousPoissonProcess)

    def test_hawkes(self):
        from seahorse.models.temporal_models.parametric_processes import HawkesProcess
        self._check(HawkesProcess)

    def test_self_correcting(self):
        from seahorse.models.temporal_models.parametric_processes import SelfCorrectingProcess
        self._check(SelfCorrectingProcess)


class TestSpatialComponentSmoke(unittest.TestCase):
    """Test 3: GaussianMixtureSpatialModel — shape (1, 4) and finite logprob."""

    def test_gmm_spatial(self):
        from seahorse.models.spatial_models.gaussian_mixture import GaussianMixtureSpatialModel

        model = GaussianMixtureSpatialModel()
        out = model.logprob(_TIMES, _LOCATIONS, _MASK)
        self.assertEqual(out.shape, (1, 4))
        self.assertTrue(torch.isfinite(out).all(), f"GMM spatial logprob not finite: {out}")
        # Padding positions should be 0.0 (mask = 1 everywhere here, so check finite is enough)

    def test_gmm_spatial_with_padding(self):
        from seahorse.models.spatial_models.gaussian_mixture import GaussianMixtureSpatialModel

        model = GaussianMixtureSpatialModel()
        times = torch.tensor([[0.0, 0.2, 0.0, 0.0]])     # last 2 are padding
        locs  = torch.tensor([[[0.1, 0.1], [0.2, 0.2], [0.0, 0.0], [0.0, 0.0]]])
        mask  = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        out = model.logprob(times, locs, mask)
        self.assertEqual(out.shape, (1, 4))
        # Padding positions must be 0.0
        self.assertEqual(out[0, 2].item(), 0.0)
        self.assertEqual(out[0, 3].item(), 0.0)
        # Valid positions must be finite
        self.assertTrue(torch.isfinite(out[0, :2]).all())


class TestFactorizedStateModelSmoke(unittest.TestCase):
    """Test 4: FactorizedStateModel — encode_history payload has correct keys."""

    def test_encode_history(self):
        from seahorse.models.state_models.factorized import FactorizedStateModel

        model = FactorizedStateModel()
        ctx = model.encode_history(times=_TIMES, locations=_LOCATIONS, lengths=_LENGTHS)
        payload = ctx.payload
        self.assertIn("times", payload)
        self.assertIn("locations", payload)
        self.assertIn("lengths", payload)
        self.assertTrue(torch.equal(payload["times"], _TIMES))
        self.assertTrue(torch.equal(payload["lengths"], _LENGTHS))


class TestFactorizedEventModelSmoke(unittest.TestCase):
    """Test 5: FactorizedEventModel.training_loss — nll finite, mask correct shape."""

    def test_training_loss(self):
        from seahorse.models.temporal_models.parametric_processes import (
            HomogeneousPoissonProcess,
        )
        from seahorse.models.spatial_models.gaussian_mixture import GaussianMixtureSpatialModel
        from seahorse.models.event_models.factorized import FactorizedEventModel
        from seahorse.models.abstractions import StateContext

        event_model = FactorizedEventModel(
            temporal_model=HomogeneousPoissonProcess(),
            spatial_model=GaussianMixtureSpatialModel(),
            t0=0.0,
            t1=None,  # last-event convention
        )
        state = StateContext(payload={})
        out = event_model.training_loss(
            times=_TIMES,
            locations=_LOCATIONS,
            lengths=_LENGTHS,
            state=state,
        )
        self.assertIn("nll", out)
        self.assertIn("mask", out)
        self.assertTrue(torch.isfinite(out["nll"]), f"nll not finite: {out['nll']}")
        self.assertEqual(out["mask"].shape, (1, 4))

    def test_explicit_t1(self):
        """Test that explicit t1 runs without error and gives finite nll."""
        from seahorse.models.temporal_models.parametric_processes import HawkesProcess
        from seahorse.models.spatial_models.gaussian_mixture import GaussianMixtureSpatialModel
        from seahorse.models.event_models.factorized import FactorizedEventModel
        from seahorse.models.abstractions import StateContext

        event_model = FactorizedEventModel(
            temporal_model=HawkesProcess(),
            spatial_model=GaussianMixtureSpatialModel(),
            t0=0.0,
            t1=1.0,  # fixed observation window end
        )
        state = StateContext(payload={})
        out = event_model.training_loss(
            times=_TIMES,
            locations=_LOCATIONS,
            lengths=_LENGTHS,
            state=state,
        )
        self.assertTrue(torch.isfinite(out["nll"]), f"nll not finite with t1=1.0: {out['nll']}")

    def test_raw_reporting_metrics_follow_transform_correction(self):
        model = build_model(
            config={
                "input_transform": {
                    "type": "zscore",
                    "normalize_time": True,
                    "normalize_space": True,
                    "time_mean": 0.0,
                    "time_std": 2.0,
                    "loc_mean": [0.0, 0.0],
                    "loc_std": [3.0, 4.0],
                }
            },
            preset="poisson_gmm",
            spatial_dim=2,
            hidden_dim=16,
            event_cov_dim=0,
            field_cov_dim=0,
        )
        model.eval()
        with torch.no_grad():
            out = model(times=_TIMES, locations=_LOCATIONS, lengths=_LENGTHS)
        extra = out["extra_metrics"]
        self.assertIn("raw_space_nll", extra)
        self.assertIn("raw_space_temporal_nll", extra)
        self.assertIn("raw_space_spatial_nll", extra)
        expected_correction = float(torch.log(torch.tensor(2.0 * 3.0 * 4.0)).item())
        self.assertAlmostEqual(
            extra["raw_space_nll"],
            float(out["nll"].item()) + expected_correction,
            places=5,
        )


class TestFactorizedEndToEnd(unittest.TestCase):
    """Test 6: end-to-end forward via UnifiedSTPP.forward — nll finite for all presets."""

    def test_all_presets(self):
        for preset in ("poisson_gmm", "hawkes_gmm", "selfcorrecting_gmm"):
            with self.subTest(preset=preset):
                torch.manual_seed(0)
                model = build_model(
                    config={},
                    preset=preset,
                    spatial_dim=2,
                    hidden_dim=16,
                    event_cov_dim=0,
                    field_cov_dim=0,
                )
                model.eval()
                with torch.no_grad():
                    out = model(times=_TIMES, locations=_LOCATIONS, lengths=_LENGTHS)
                self.assertIn("nll", out)
                self.assertTrue(
                    torch.isfinite(out["nll"]),
                    f"{preset}: nll not finite: {out['nll']}",
                )

    def test_batched(self):
        """Test with B=3 sequences of varying lengths."""
        torch.manual_seed(1)
        model = build_model(config={}, preset="hawkes_gmm", spatial_dim=2, hidden_dim=16)
        model.eval()

        times = torch.tensor([
            [0.0, 0.3, 0.7, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.1, 0.4, 0.8],
        ])
        locs = torch.zeros(3, 4, 2)
        lengths = torch.tensor([3, 2, 4], dtype=torch.long)

        with torch.no_grad():
            out = model(times=times, locations=locs, lengths=lengths)
        self.assertTrue(torch.isfinite(out["nll"]))
        self.assertEqual(out["mask"].shape, (3, 4))
        # Verify mask is correct
        expected_n_events = lengths.sum().item()
        self.assertEqual(int(out["total_events"].item()), expected_n_events)


class TestIndependentCNFSmoke(unittest.TestCase):
    """Test 7: IndependentCNF spatial model — shape, finiteness, padding."""

    def _build(self, squash_time: bool):
        from seahorse.models.spatial_models.independent_cnf import IndependentCNF
        return IndependentCNF(
            dim=2,
            hidden_dims=(16, 16),
            layer_type="concat",
            actfn="softplus",
            tol=1e-3,
            squash_time=squash_time,
        )

    def test_squash_time_true(self):
        torch.manual_seed(0)
        model = self._build(squash_time=True)
        model.eval()
        with torch.no_grad():
            out = model.logprob(_TIMES, _LOCATIONS, _MASK)
        self.assertEqual(out.shape, (1, 4))
        self.assertTrue(torch.isfinite(out).all(), f"squash_time=True not finite: {out}")

    def test_squash_time_false(self):
        torch.manual_seed(0)
        model = self._build(squash_time=False)
        model.eval()
        with torch.no_grad():
            out = model.logprob(_TIMES, _LOCATIONS, _MASK)
        self.assertEqual(out.shape, (1, 4))
        self.assertTrue(torch.isfinite(out).all(), f"squash_time=False not finite: {out}")

    def test_padding(self):
        from seahorse.models.spatial_models.independent_cnf import IndependentCNF
        torch.manual_seed(0)
        model = IndependentCNF(dim=2, hidden_dims=(16, 16), tol=1e-3)
        model.eval()
        times = torch.tensor([[0.0, 0.2, 0.0, 0.0]])
        locs  = torch.tensor([[[0.1, 0.1], [0.2, 0.2], [0.0, 0.0], [0.0, 0.0]]])
        mask  = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        with torch.no_grad():
            out = model.logprob(times, locs, mask)
        self.assertEqual(out.shape, (1, 4))
        self.assertEqual(out[0, 2].item(), 0.0)
        self.assertEqual(out[0, 3].item(), 0.0)
        self.assertTrue(torch.isfinite(out[0, :2]).all())


class TestFactorizedCNFEndToEnd(unittest.TestCase):
    """Test 8: end-to-end forward via UnifiedSTPP for CNF presets."""

    def test_all_cnf_presets(self):
        for preset in ("poisson_cnf", "hawkes_cnf", "selfcorrecting_cnf"):
            with self.subTest(preset=preset):
                torch.manual_seed(0)
                model = build_model(
                    config={},
                    preset=preset,
                    spatial_dim=2,
                    hidden_dim=16,
                    event_cov_dim=0,
                    field_cov_dim=0,
                )
                model.eval()
                with torch.no_grad():
                    out = model(times=_TIMES, locations=_LOCATIONS, lengths=_LENGTHS)
                self.assertIn("nll", out)
                self.assertTrue(
                    torch.isfinite(out["nll"]),
                    f"{preset}: nll not finite: {out['nll']}",
                )


class TestFactorizedTVCNFEndToEnd(unittest.TestCase):
    """Test 9: end-to-end forward via UnifiedSTPP for tvcnf presets."""

    def test_all_tvcnf_presets(self):
        for preset in ("poisson_tvcnf", "hawkes_tvcnf", "selfcorrecting_tvcnf"):
            with self.subTest(preset=preset):
                torch.manual_seed(0)
                model = build_model(
                    config={},
                    preset=preset,
                    spatial_dim=2,
                    hidden_dim=16,
                    event_cov_dim=0,
                    field_cov_dim=0,
                )
                model.eval()
                with torch.no_grad():
                    out = model(times=_TIMES, locations=_LOCATIONS, lengths=_LENGTHS)
                self.assertIn("nll", out)
                self.assertTrue(
                    torch.isfinite(out["nll"]),
                    f"{preset}: nll not finite: {out['nll']}",
                )

    def test_squash_time_false_default(self):
        """Ensure tvcnf presets instantiate with squash_time=False."""
        from seahorse.models.configs import (
            PoissonTVCNFConfig, HawkesTVCNFConfig, SelfCorrectingTVCNFConfig,
        )
        for cls in (PoissonTVCNFConfig, HawkesTVCNFConfig, SelfCorrectingTVCNFConfig):
            cfg = cls.from_dict({}, hidden_dim=16, spatial_dim=2)
            self.assertFalse(cfg.squash_time, f"{cls.__name__}.squash_time should be False")


if __name__ == "__main__":
    unittest.main()
