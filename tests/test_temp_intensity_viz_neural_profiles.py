import unittest

from temp_intensity_viz import (
    DEFAULT_T_NSTEP,
    DEFAULT_X_NSTEP,
    DEFAULT_Y_NSTEP,
    _resolve_neural_stpp_viz_profile,
)


class TestNeuralSTPPVizProfiles(unittest.TestCase):
    def test_cond_gmm_keeps_dense_defaults(self):
        profile = _resolve_neural_stpp_viz_profile(
            preset="neural_stpp_shared_cond_gmm",
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
            preset="neural_stpp_shared_jumpcnf",
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
            preset="neural_stpp_shared_attncnf",
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


if __name__ == "__main__":
    unittest.main()
