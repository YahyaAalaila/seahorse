from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from unified_stpp.config.schema import STPPConfig
from unified_stpp.models.configs import ConfigRegistry
from unified_stpp.models.configs.nsmpp_deepbasis import NSMPPDeepBasisConfig
from unified_stpp.models.event_models.nsmpp_deepbasis_event import NSMPPDeepBasisEventModel
from unified_stpp.models.event_models.nsmpp_deepbasis_kernel import DeepBasisKernel
from unified_stpp.runner import STPPRunner


def build_model(config=None, preset=None, **dims):
    return ConfigRegistry.build(name=preset, overrides=dict(config or {}), **dims)


def _build_model(**config_overrides):
    cfg = {
        "support_t0": 0.0,
        "support_t1": 1.0,
        "support_space_min": (0.0, 0.0),
        "support_space_max": (1.0, 1.0),
        "decoder": {
            "type": "nsmpp_deepbasis",
            "mu": 1.0,
            "n_basis": 3,
            "basis_dim": 4,
            "nn_width": 6,
            "int_res": 4,
            "numerical_int": True,
            "init_gain": 1.0,
            "init_bias": 0.0,
            "init_std": 0.5,
            "intensity_eps": 1e-5,
            "compensator_chunk_size": 8,
        },
    }
    for key, value in config_overrides.items():
        if key == "decoder":
            cfg["decoder"].update(value)
        else:
            cfg[key] = value
    return build_model(config=cfg, preset="nsmpp", spatial_dim=2)


def _first_metric(run_dir: str | Path, key: str) -> float:
    metrics_path = Path(run_dir) / "metrics.csv"
    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get(key)
            if value not in {None, ""}:
                return float(value)
    raise AssertionError(f"Metric {key!r} not found in {metrics_path}.")


class TestNSMPPDeepBasisKernel(unittest.TestCase):
    def test_manual_low_rank_sum_matches_forward(self):
        torch.manual_seed(0)
        kernel = DeepBasisKernel(
            n_basis=2,
            data_dim=3,
            basis_dim=4,
            init_gain=1.0,
            init_bias=0.0,
            init_std=0.25,
            nn_width=5,
        )
        x = torch.randn(7, 3)
        y = torch.randn(7, 3)

        out = kernel(x, y)
        manual = torch.zeros(7)
        weights = F.softplus(kernel.raw_weights)
        for idx in range(kernel.n_basis):
            manual = manual + weights[idx] * (
                kernel.x_basis[idx](x) * kernel.y_basis[idx](y)
            ).sum(dim=-1)
        self.assertTrue(torch.allclose(out, manual, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.all(weights > 0))


class TestNSMPPDeepBasisConfig(unittest.TestCase):
    def test_bundled_preset_yaml_loads(self):
        cfg = STPPConfig.from_source(preset="nsmpp", config=None)
        self.assertEqual(cfg.model.preset, "nsmpp")
        self.assertEqual(cfg.data.protocol, "raw")
        self.assertTrue(cfg.data.normalize)
        self.assertEqual(cfg.data.batch_size, 25)
        self.assertEqual(cfg.training.optimizer, "adadelta")
        self.assertEqual(cfg.training.batch_size, 25)
        dec = cfg.model.build_overrides["decoder"]
        self.assertAlmostEqual(dec["mu"], 1e-2, places=8)
        self.assertEqual(dec["n_basis"], 2)
        self.assertEqual(dec["basis_dim"], 10)
        self.assertEqual(dec["nn_width"], 5)
        self.assertAlmostEqual(dec["init_gain"], 5e-1, places=8)
        self.assertAlmostEqual(dec["init_std"], 1e-1, places=8)
        self.assertAlmostEqual(dec["init_weight_mean"], -5.0, places=8)
        self.assertEqual(dec["int_res"], 40)
        self.assertAlmostEqual(cfg.training.lr, 1.0, places=8)
        self.assertAlmostEqual(cfg.training.grad_clip, 5.0, places=8)

    def test_deprecated_alias_loads_canonical_preset(self):
        cfg = STPPConfig.from_source(preset="nsmpp_deepbasis_provisional", config=None)
        self.assertEqual(cfg.model.preset, "nsmpp")

    def test_preset_is_benchmark_supported_and_builds(self):
        self.assertEqual(
            ConfigRegistry.canonical_status("nsmpp"),
            "canonical",
        )
        model = _build_model()
        self.assertIsInstance(model.event_model, NSMPPDeepBasisEventModel)

    def test_data_init_overrides_use_train_split_bounds(self):
        sequences = [
            {
                "times": [0.25, 0.75],
                "locations": [[-1.0, 2.0], [3.0, 5.0]],
            },
            {
                "times": [0.5, 1.5],
                "locations": [[0.0, -2.0], [4.0, 6.0]],
            },
        ]
        dm = SimpleNamespace(
            _bundle=SimpleNamespace(
                train_dataset=SimpleNamespace(sequences=sequences),
            )
        )
        overrides = NSMPPDeepBasisConfig.data_init_overrides(dm)
        self.assertAlmostEqual(overrides["support_t0"], 0.25, places=6)
        self.assertAlmostEqual(overrides["support_t1"], 1.5, places=6)
        self.assertEqual(overrides["support_space_min"], (-1.0, -2.0))
        self.assertEqual(overrides["support_space_max"], (4.0, 6.0))


class TestNSMPPDeepBasisEventModel(unittest.TestCase):
    def test_background_rate_matches_configured_mu(self):
        model = _build_model(decoder={"mu": 0.2})
        query = torch.tensor([[[0.7, 0.3, 0.4]]], dtype=torch.float32)
        history = torch.zeros((1, 0, 3), dtype=torch.float32)
        history_mask = torch.zeros((1, 1, 0), dtype=torch.bool)

        got = model.event_model._conditional_intensity(
            query_events=query,
            history_events=history,
            history_mask=history_mask,
        )

        expected = float(model.event_model.mu.detach().item()) + model.event_model.intensity_eps
        self.assertAlmostEqual(float(got.item()), expected, places=6)

    def test_intensity_ignores_padding_and_uses_only_valid_history(self):
        torch.manual_seed(1)
        model = _build_model()
        times = torch.tensor([[0.2, 0.6, 0.0, 0.0]], dtype=torch.float32)
        locs = torch.tensor(
            [[[0.1, 0.2], [0.4, 0.5], [0.0, 0.0], [0.0, 0.0]]],
            dtype=torch.float32,
        )
        lengths = torch.tensor([2], dtype=torch.long)
        state = model.state_model.encode_history(times=times, locations=locs, lengths=lengths)

        query_time = torch.tensor([0.7], dtype=torch.float32)
        query_loc = torch.tensor([[0.3, 0.3]], dtype=torch.float32)
        got = model.event_model.intensity(
            state=state,
            query_times=query_time,
            query_locations=query_loc,
        )

        history = torch.cat([times[:, :2].unsqueeze(-1), locs[:, :2]], dim=-1)[0]
        query_event = torch.cat([query_time.unsqueeze(-1), query_loc], dim=-1)
        raw = (
            model.event_model.raw_mu
            + model.event_model.kernel(
                query_event.expand(history.shape[0], -1),
                history,
            ).sum()
        )
        expected = F.softplus(raw) + model.event_model.intensity_eps
        self.assertTrue(torch.allclose(got.squeeze(0), expected, atol=1e-6, rtol=1e-6))

    def test_event_intensity_uses_only_prior_events(self):
        torch.manual_seed(2)
        model = _build_model()
        times = torch.tensor([[0.1, 0.3, 0.8]], dtype=torch.float32)
        locs = torch.tensor(
            [[[0.2, 0.1], [0.5, 0.4], [0.7, 0.9]]],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3], dtype=torch.long)
        state = model.state_model.encode_history(times=times, locations=locs, lengths=lengths)
        event_vectors = state.payload["event_vectors"]
        valid_mask = state.payload["event_mask"]

        intensities = model.event_model._event_intensities(event_vectors, valid_mask)
        third_event = event_vectors[:, 2:3]
        history = event_vectors[:, :2]
        history_mask = torch.ones(1, 1, 2, dtype=torch.bool)
        expected = model.event_model._conditional_intensity(
            query_events=third_event,
            history_events=history,
            history_mask=history_mask,
        )
        self.assertTrue(torch.allclose(intensities[:, 2], expected[:, 0], atol=1e-6, rtol=1e-6))

    def test_compensator_matches_bruteforce_reference(self):
        torch.manual_seed(3)
        model = _build_model(
            decoder={
                "int_res": 3,
                "compensator_chunk_size": 2,
            }
        )
        times = torch.tensor([[0.2, 0.7]], dtype=torch.float32)
        locs = torch.tensor([[[0.1, 0.2], [0.6, 0.4]]], dtype=torch.float32)
        lengths = torch.tensor([2], dtype=torch.long)
        state = model.state_model.encode_history(times=times, locations=locs, lengths=lengths)
        event_vectors = state.payload["event_vectors"]
        valid_mask = state.payload["event_mask"]

        got = model.event_model._numerical_compensator(event_vectors, valid_mask)

        event_model = model.event_model
        history = event_vectors[0, :2]
        total = 0.0
        for t_val in event_model.support_time_grid:
            for s_val in event_model.support_spatial_grid:
                query = torch.cat([t_val.view(1), s_val], dim=0).view(1, -1)
                hist_mask = history[:, 0] <= t_val
                hist = history[hist_mask]
                if hist.numel() == 0:
                    raw = event_model.raw_mu.detach().to(dtype=query.dtype)
                else:
                    raw = event_model.raw_mu.detach().to(dtype=query.dtype) + event_model.kernel(
                        query.expand(hist.shape[0], -1),
                        hist,
                    ).sum()
                lam = F.softplus(raw) + event_model.intensity_eps
                total += float(lam.item())
        expected = total * float(event_model.unit_vol.item())
        self.assertAlmostEqual(float(got.item()), expected, places=5)

    def test_background_only_compensator_matches_support_volume(self):
        model = _build_model(
            support_t0=0.0,
            support_t1=2.0,
            support_space_min=(0.0, 0.0),
            support_space_max=(3.0, 5.0),
            decoder={
                "mu": 0.2,
                "int_res": 4,
            },
        )
        empty_state = model.state_model.encode_history(
            times=torch.zeros((1, 0), dtype=torch.float32),
            locations=torch.zeros((1, 0, 2), dtype=torch.float32),
            lengths=torch.tensor([0], dtype=torch.long),
        )

        got = model.event_model._numerical_compensator(
            empty_state.payload["event_vectors"],
            empty_state.payload["event_mask"],
        )

        support_volume = (
            (2.0 - 0.0)
            * (3.0 - 0.0)
            * (5.0 - 0.0)
        )
        expected = (
            float(model.event_model.mu.detach().item()) + model.event_model.intensity_eps
        ) * support_volume
        self.assertTrue(torch.isfinite(got).all())
        self.assertAlmostEqual(float(got.item()), expected, places=5)

    def test_intensity_query_matches_query_surface(self):
        torch.manual_seed(4)
        model = _build_model()
        times = torch.tensor([[0.15, 0.5]], dtype=torch.float32)
        locs = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)
        lengths = torch.tensor([2], dtype=torch.long)
        state = model.state_model.encode_history(times=times, locations=locs, lengths=lengths)
        q_t = torch.tensor([0.8, 0.9], dtype=torch.float32)
        q_s = torch.tensor([[0.4, 0.3], [0.2, 0.6]], dtype=torch.float32)
        intensity = model.event_model.intensity(
            state=state,
            query_times=q_t,
            query_locations=q_s,
        )
        surface = model.event_model.query_surface(
            state=state,
            grid_times=q_t,
            grid_locs=q_s,
        )
        self.assertTrue(torch.allclose(intensity, surface, atol=1e-6, rtol=1e-6))


class TestNSMPPDeepBasisSmoke(unittest.TestCase):
    def test_manual_optimizer_step_is_finite(self):
        torch.manual_seed(5)
        model = _build_model(decoder={"int_res": 3})
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-2)
        times = torch.tensor(
            [[0.1, 0.3, 0.7], [0.2, 0.4, 0.8]],
            dtype=torch.float32,
        )
        locs = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.5], [0.6, 0.7]],
                [[0.2, 0.1], [0.4, 0.4], [0.7, 0.6]],
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3, 3], dtype=torch.long)

        optimizer.zero_grad()
        out = model(times=times, locations=locs, lengths=lengths)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertAlmostEqual(float(out["loss"].item()), float(out["nll"].item()), places=6)
        out["loss"].backward()
        optimizer.step()

    def test_eval_nll_reports_per_event_while_training_uses_sequence_mean(self):
        torch.manual_seed(6)
        model = _build_model(decoder={"int_res": 3})
        times = torch.tensor(
            [[0.1, 0.3, 0.7], [0.2, 0.4, 0.8]],
            dtype=torch.float32,
        )
        locs = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.5], [0.6, 0.7]],
                [[0.2, 0.1], [0.4, 0.4], [0.7, 0.6]],
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3, 3], dtype=torch.long)

        train_out = model(times=times, locations=locs, lengths=lengths)
        eval_out = model.eval_forward(times=times, locations=locs, lengths=lengths)

        self.assertTrue(torch.isfinite(train_out["loss"]))
        self.assertTrue(torch.isfinite(eval_out["nll"]))
        self.assertNotAlmostEqual(float(train_out["nll"].item()), float(eval_out["nll"].item()), places=6)
        self.assertAlmostEqual(
            float(train_out["extra_metrics"]["per_event_nll"]),
            float(eval_out["nll"].item()),
            places=6,
        )

    def test_tiny_runner_fit_smoke(self):
        train = [
            {"times": [0.1, 0.3, 0.8], "locations": [[0.1, 0.2], [0.3, 0.4], [0.7, 0.8]]},
            {"times": [0.2, 0.6, 0.9], "locations": [[0.2, 0.1], [0.4, 0.3], [0.8, 0.6]]},
        ]
        val = [
            {"times": [0.15, 0.55, 0.85], "locations": [[0.15, 0.25], [0.35, 0.45], [0.75, 0.65]]},
        ]
        test = [
            {"times": [0.12, 0.52, 0.82], "locations": [[0.12, 0.18], [0.42, 0.32], [0.72, 0.62]]},
        ]
        with tempfile.TemporaryDirectory() as td:
            runner = STPPRunner.from_config_source(
                preset="nsmpp",
                config=None,
                cli_values={
                    "data": {
                        "batch_size": 2,
                        "num_workers": 0,
                    },
                    "training": {
                        "n_epochs": 1,
                        "batch_size": 2,
                        "device": "cpu",
                    },
                    "logging": {
                        "out_dir": td,
                    },
                    "model": {
                        "decoder": {
                            "int_res": 3,
                            "compensator_chunk_size": 4,
                        }
                    },
                },
            )
            result = runner.fit(train, val, test, dataset_id="tiny")
        self.assertTrue(torch.isfinite(torch.tensor(result.val_objective)))
        self.assertTrue(torch.isfinite(torch.tensor(result.test_nll)))

    def test_tiny_runner_fit_is_deterministic_for_same_seed(self):
        train = [
            {"times": [0.1, 0.3, 0.8], "locations": [[0.1, 0.2], [0.3, 0.4], [0.7, 0.8]]},
            {"times": [0.2, 0.6, 0.9], "locations": [[0.2, 0.1], [0.4, 0.3], [0.8, 0.6]]},
        ]
        val = [
            {"times": [0.15, 0.55, 0.85], "locations": [[0.15, 0.25], [0.35, 0.45], [0.75, 0.65]]},
        ]
        test = [
            {"times": [0.12, 0.52, 0.82], "locations": [[0.12, 0.18], [0.42, 0.32], [0.72, 0.62]]},
        ]

        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as td:
                runner = STPPRunner.from_config_source(
                    preset="nsmpp",
                    config=None,
                    cli_values={
                        "data": {
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 123,
                        },
                        "training": {
                            "n_epochs": 1,
                            "batch_size": 2,
                            "device": "cpu",
                        },
                        "logging": {
                            "out_dir": td,
                        },
                        "model": {
                            "decoder": {
                                "mu": 0.1,
                                "int_res": 3,
                                "compensator_chunk_size": 4,
                            }
                        },
                    },
                )
                result = runner.fit(train, val, test, dataset_id="tiny")
                results.append(
                    (
                        result.val_objective,
                        result.test_nll,
                        _first_metric(result.run_dir, "val/objective"),
                    )
                )

        first, second = results
        self.assertAlmostEqual(first[0], second[0], places=6)
        self.assertAlmostEqual(first[1], second[1], places=6)
        self.assertAlmostEqual(first[2], second[2], places=6)


if __name__ == "__main__":
    unittest.main()
