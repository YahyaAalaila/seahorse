"""
PyTorch Lightning DataModule for STPP datasets — Lightning glue only.

STPPDataModule wraps a DataBundle (produced by STPPConfig.build_data_bundle())
and serves the three DataLoaders that Lightning expects.  All data preparation
logic — protocol dispatch, normalization, collate/sampler selection — lives in
the data registry (seahorse/data/registry.py) and is resolved before this
class is constructed.
"""
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from seahorse.data.registry import DataBundle


class STPPDataModule(pl.LightningDataModule):
    """Lightning DataModule that serves DataLoaders from an already-built DataBundle.

    Parameters
    ----------
    bundle      : resolved data components (datasets, collate fn, batch sampler)
    batch_size  : used when ``bundle.train_batch_sampler`` is None
    num_workers : passed to all DataLoaders
    seed        : seeds the shuffle generator for the train DataLoader
    """

    def __init__(
        self,
        bundle: DataBundle,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self._bundle = bundle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_generator = torch.Generator()
        self._train_generator.manual_seed(seed)

    # ------------------------------------------------------------------
    # Convenience accessors (used by runner for norm-stat extraction)
    # ------------------------------------------------------------------

    @property
    def train_dataset(self):
        return self._bundle.train_dataset

    def get_norm_stats(self, normalize: bool) -> dict:
        """Return normalization statistics from the training dataset.

        Parameters
        ----------
        normalize : whether normalization was applied (from ``DataConfig.normalize``).
                    Stored in the returned dict so callers can round-trip without
                    keeping a separate reference to the config.
        """
        ds = self._bundle.train_dataset
        return {
            "normalize": normalize,
            "coordinate_space": str(getattr(ds, "coordinate_space", "raw")),
            "time_mean": float(getattr(ds, "time_mean", 0.0)),
            "time_std":  float(getattr(ds, "time_std",  1.0)),
            "loc_mean":  list(np.asarray(getattr(ds, "loc_mean", [0.0, 0.0])).tolist()),
            "loc_std":   list(np.asarray(getattr(ds, "loc_std",  [1.0, 1.0])).tolist()),
        }

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def setup(self, stage=None):
        pass  # bundle is fully built at construction time

    def train_dataloader(self):
        if self._bundle.train_batch_sampler is not None:
            return DataLoader(
                self._bundle.train_dataset,
                batch_sampler=self._bundle.train_batch_sampler,
                collate_fn=self._bundle.collate_fn,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )
        return DataLoader(
            self._bundle.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._bundle.collate_fn,
            num_workers=self.num_workers,
            generator=self._train_generator,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self._bundle.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._bundle.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        if self._bundle.test_dataset is None:
            return None
        return DataLoader(
            self._bundle.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._bundle.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def get_original_sequence(self, split: str = "val", idx: int = 0) -> dict:
        """Return sequence times and locations in original (un-normalized) space.

        Args:
            split : "train" | "val" | "test"
            idx   : sequence index within the split

        Returns:
            {"times": np.ndarray (L,), "locations": np.ndarray (L, d)}
        """
        dataset = {
            "train": self._bundle.train_dataset,
            "val":   self._bundle.val_dataset,
            "test":  self._bundle.test_dataset,
        }[split]
        if dataset is None:
            raise ValueError(f"Split {split!r} is not available (dataset is None).")
        seq = dataset.sequences[idx]
        return {
            "times":     np.asarray(seq["times"],     dtype=np.float64),
            "locations": np.asarray(seq["locations"], dtype=np.float64),
        }
