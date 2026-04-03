"""Notebook-faithful intensity cube + Plotly renderer regression tests."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from unified_stpp.evaluation.intensity import calc_lamb_from_runner
from unified_stpp.registry import build_model
from unified_stpp.viz import plot_lambst_interactive


_SEQ = {
    "times": np.array([0.1, 0.4, 0.7, 1.1, 1.6], dtype=np.float32),
    "locations": np.array(
        [
            [0.10, 0.20],
            [0.30, 0.10],
            [0.40, 0.35],
            [0.55, 0.45],
            [0.70, 0.80],
        ],
        dtype=np.float32,
    ),
}


def _mock_runner(*, preset: str, config: dict):
    torch.manual_seed(0)
    model = build_model(
        config=config,
        preset=preset,
        spatial_dim=2,
        hidden_dim=8,
        event_cov_dim=0,
        field_cov_dim=0,
    )
    model.eval()
    return types.SimpleNamespace(
        model=model,
        config=types.SimpleNamespace(model=types.SimpleNamespace(preset=preset)),
        norm_stats={
            "time_mean": 0.0,
            "time_std": 1.0,
            "loc_mean": [0.0, 0.0],
            "loc_std": [1.0, 1.0],
        },
    )


class TestNotebookFaithfulCalcLamb(unittest.TestCase):
    def test_deep_stpp_cube_shape_and_defaults(self):
        runner = _mock_runner(
            preset="deep_stpp",
            config={
                "encoder": {"num_heads": 1, "num_layers": 1},
                "decoder": {
                    "seq_len": 3,
                    "lookahead": 1,
                    "num_points": 4,
                    "n_layers": 1,
                    "constrain_b": False,
                },
                "vae": False,
                "paper_dt_min": 0.0,
                "paper_dt_range": 1.0,
                "paper_loc_min": [0.0, 0.0],
                "paper_loc_range": [1.0, 1.0],
            },
        )
        result = calc_lamb_from_runner(
            runner=runner,
            sequences=[_SEQ],
            seq_idx=0,
            split="test",
            x_nstep=4,
            y_nstep=5,
            t_nstep=4,
            round_time=False,
        )

        self.assertEqual(result.lambs.shape, (4, 4, 5))
        self.assertTrue(np.all(np.isfinite(result.lambs)))
        np.testing.assert_allclose(result.x_range[[0, -1]], [0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(result.y_range[[0, -1]], [0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(result.t_range[[0, -1]], [_SEQ["times"][3], _SEQ["times"][-1]], atol=1e-6)
        np.testing.assert_allclose(result.history_times, _SEQ["times"])
        np.testing.assert_allclose(result.history_locs, _SEQ["locations"])

    def test_auto_stpp_faithful_cube_shape_and_defaults(self):
        runner = _mock_runner(
            preset="auto_stpp_faithful",
            config={
                "decoder": {
                    "lookback": 3,
                    "lookahead": 1,
                    "n_prodnet": 2,
                    "hidden_size": 8,
                    "num_layers": 1,
                    "activation": "tanh",
                    "trunc": False,
                },
                "paper_dt_min": 0.0,
                "paper_dt_range": 1.0,
                "paper_loc_min": [0.0, 0.0],
                "paper_loc_range": [1.0, 1.0],
            },
        )
        result = calc_lamb_from_runner(
            runner=runner,
            sequences=[_SEQ],
            seq_idx=0,
            split="test",
            x_nstep=4,
            y_nstep=5,
            t_nstep=4,
            round_time=False,
        )

        self.assertEqual(result.lambs.shape, (4, 4, 5))
        self.assertTrue(np.all(np.isfinite(result.lambs)))
        self.assertTrue(np.all(result.lambs >= 0.0))
        np.testing.assert_allclose(result.x_range[[0, -1]], [0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(result.y_range[[0, -1]], [0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(result.history_times, _SEQ["times"])
        np.testing.assert_allclose(result.history_locs, _SEQ["locations"])


class TestPlotLambstInteractive(unittest.TestCase):
    def test_single_cube_builds_frames_and_html(self):
        lambs = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        try:
            fig = plot_lambst_interactive(
                lambs,
                np.linspace(0.0, 1.0, 4, dtype=np.float32),
                np.linspace(0.0, 1.0, 5, dtype=np.float32),
                np.linspace(0.5, 1.5, 3, dtype=np.float32),
                show=False,
            )
        except ImportError as exc:
            self.skipTest(str(exc))

        self.assertEqual(len(fig.frames), 3)
        self.assertEqual(len(fig.data), 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "surface.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            html = html_path.read_text()
        self.assertIn("plotly", html.lower())

    def test_multi_cube_builds_side_by_side_frames(self):
        lambs = np.stack(
            [
                np.ones((3, 4, 5), dtype=np.float32),
                np.full((3, 4, 5), 2.0, dtype=np.float32),
            ],
            axis=0,
        )
        try:
            fig = plot_lambst_interactive(
                lambs,
                np.linspace(0.0, 1.0, 4, dtype=np.float32),
                np.linspace(0.0, 1.0, 5, dtype=np.float32),
                np.linspace(0.5, 1.5, 3, dtype=np.float32),
                show=False,
                subplot_titles=["A", "B"],
            )
        except ImportError as exc:
            self.skipTest(str(exc))

        self.assertEqual(len(fig.frames), 3)
        self.assertEqual(len(fig.data), 2)


if __name__ == "__main__":
    unittest.main()
