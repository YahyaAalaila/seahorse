"""Regression tests for the raw-first data/transform path."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from unified_stpp.config.schema import DataConfig
from unified_stpp.data.dataset import STPPDataset, collate_fn
from unified_stpp.data.registry import DataRegistry
from unified_stpp.data.transforms import ZScoreTransformArtifact
from unified_stpp.models.configs.factorized import FactorizedConfig
from unified_stpp.models.configs.neural_stpp import NeuralSTPPConfig


def _toy_sequences():
    return [
        {
            "times": np.array([1.0, 3.0, 6.0], dtype=np.float32),
            "locations": np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        },
        {
            "times": np.array([2.0, 5.0, 9.0], dtype=np.float32),
            "locations": np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]], dtype=np.float32),
        },
    ]


class TestRawDatasetContract(unittest.TestCase):
    def test_raw_dataset_preserves_coordinates(self):
        seqs = _toy_sequences()
        ds = STPPDataset(seqs, normalize_time=False, normalize_space=False, min_length=1)
        item = ds[0]

        self.assertEqual(ds.coordinate_space, "raw")
        self.assertEqual(item["coordinate_space"], "raw")
        torch.testing.assert_close(item["times"], torch.tensor(seqs[0]["times"]))
        torch.testing.assert_close(item["locations"], torch.tensor(seqs[0]["locations"]))

        batch = collate_fn([ds[0], ds[1]])
        self.assertEqual(batch["coordinate_space"], "raw")
        torch.testing.assert_close(batch["times"][0, :3], torch.tensor(seqs[0]["times"]))
        torch.testing.assert_close(batch["locations"][1, :3], torch.tensor(seqs[1]["locations"]))

    def test_registry_raw_builder_returns_raw_datasets(self):
        seqs = _toy_sequences()
        cfg = DataConfig(protocol="raw", normalize=False, batch_size=2)
        train_ds, val_ds, test_ds = DataRegistry.build_datasets(cfg.protocol, cfg, seqs, seqs, seqs)

        self.assertEqual(train_ds.coordinate_space, "raw")
        self.assertEqual(val_ds.coordinate_space, "raw")
        self.assertEqual(test_ds.coordinate_space, "raw")
        self.assertFalse(train_ds.normalize_time)
        self.assertFalse(train_ds.normalize_space)


class TestRawTransformFitting(unittest.TestCase):
    def test_neural_transform_fits_spatial_zscore_from_raw_sequences(self):
        ds = STPPDataset(_toy_sequences(), normalize_time=False, normalize_space=False, min_length=1)
        dm = SimpleNamespace(train_dataset=ds)

        artifact = NeuralSTPPConfig.fit_transform_artifact(dm)

        self.assertIsInstance(artifact, ZScoreTransformArtifact)
        self.assertFalse(artifact.normalize_time)
        self.assertTrue(artifact.normalize_space)
        self.assertNotEqual(tuple(artifact.loc_mean), (0.0, 0.0))

    def test_factorized_transform_fits_time_and_space_zscore_from_raw_sequences(self):
        ds = STPPDataset(_toy_sequences(), normalize_time=False, normalize_space=False, min_length=1)
        dm = SimpleNamespace(train_dataset=ds)

        artifact = FactorizedConfig.fit_transform_artifact(dm)

        self.assertIsInstance(artifact, ZScoreTransformArtifact)
        self.assertTrue(artifact.normalize_time)
        self.assertTrue(artifact.normalize_space)
        self.assertNotEqual(float(artifact.time_mean), 0.0)
        self.assertNotEqual(tuple(artifact.loc_mean), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
