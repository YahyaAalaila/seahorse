import unittest

from temp_intensity_viz import (
    DEFAULT_T_NSTEP,
    DEFAULT_X_NSTEP,
    DEFAULT_Y_NSTEP,
    _build_future_query_grid,
    _history_overlay_z_level,
    _resolve_neural_stpp_viz_profile,
)


class TestNeuralSTPPVizProfiles(unittest.TestCase):
    def test_cond_gmm_keeps_dense_defaults(self):
        profile = _resolve_neural_stpp_viz_profile(
            preset="njsde",
            x_nstep=DEFAULT_X_NSTEP,
            y_nstep=DEFAULT_Y_NSTEP,
            t_nstep=DEFAULT_T_NSTEP,
            spatial_chunk_size=None,
        )
        self.assertEqual(profile["x_nstep"], DEFAULT_X_NSTEP)
        self.assertEqual(profile["y_nstep"], DEFAULT_Y_NSTEP)
        self.assertEqual(profile["t_nstep"], DEFAULT_T_NSTEP)
        self.assertEqual(profile["spatial_chunk_size"], 4096)
        self.assertFalse(profile["auto_coarsened_grid"])
        self.assertTrue(profile["warnings"])

    def test_jumpcnf_auto_coarsens_dense_defaults(self):
        profile = _resolve_neural_stpp_viz_profile(
            preset="neural_jumpcnf",
            x_nstep=DEFAULT_X_NSTEP,
            y_nstep=DEFAULT_Y_NSTEP,
            t_nstep=DEFAULT_T_NSTEP,
            spatial_chunk_size=None,
        )
        self.assertEqual(profile["x_nstep"], 49)
        self.assertEqual(profile["y_nstep"], 49)
        self.assertEqual(profile["t_nstep"], 21)
        self.assertEqual(profile["spatial_chunk_size"], 1024)
        self.assertTrue(profile["auto_coarsened_grid"])
        self.assertTrue(profile["warnings"])

    def test_attncnf_preserves_explicit_grid_and_chunk_override(self):
        profile = _resolve_neural_stpp_viz_profile(
            preset="neural_attncnf",
            x_nstep=25,
            y_nstep=19,
            t_nstep=7,
            spatial_chunk_size=128,
        )
        self.assertEqual(profile["x_nstep"], 25)
        self.assertEqual(profile["y_nstep"], 19)
        self.assertEqual(profile["t_nstep"], 7)
        self.assertEqual(profile["spatial_chunk_size"], 128)
        self.assertFalse(profile["auto_coarsened_grid"])
        self.assertTrue(profile["warnings"])

    def test_future_query_grid_excludes_last_history_time(self):
        grid = _build_future_query_grid(last_t=10.0, horizon=2.0, n_steps=4)
        self.assertEqual(grid.shape, (4,))
        self.assertGreater(float(grid[0]), 10.0)
        self.assertAlmostEqual(float(grid[-1]), 12.0, places=6)

    def test_history_overlay_stays_on_zero_surface(self):
        frame = [[0.0, 0.0], [0.0, 0.0]]
        self.assertEqual(_history_overlay_z_level(frame), 0.0)

    def test_history_overlay_uses_actual_surface_range(self):
        frame = [[2.0, 4.0], [6.0, 10.0]]
        self.assertAlmostEqual(_history_overlay_z_level(frame), 2.0 + 0.88 * (10.0 - 2.0), places=6)


if __name__ == "__main__":
    unittest.main()
