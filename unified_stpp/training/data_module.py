"""
PyTorch Lightning DataModule for STPP datasets.
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from unified_stpp.data import STPPDataset, collate_fn


class STPPDataModule(pl.LightningDataModule):
    def __init__(self, train_seqs, val_seqs, test_seqs=None,
                 batch_size=32, num_workers=0, normalize=True):
        super().__init__()
        self.train_seqs = train_seqs
        self.val_seqs = val_seqs
        self.test_seqs = test_seqs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        # Populated by setup()
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage=None):
        # Guard so Lightning's repeated setup() calls don't recreate datasets.
        if self._train_dataset is not None:
            return

        self._train_dataset = STPPDataset(
            self.train_seqs,
            normalize_time=self.normalize,
            normalize_space=self.normalize,
        )
        # Use training stats for val/test normalization
        self._val_dataset = STPPDataset(
            self.val_seqs,
            normalize_time=self.normalize,
            normalize_space=self.normalize,
            cov_mean=self._train_dataset.cov_mean,
            cov_std=self._train_dataset.cov_std,
        )
        # Override time/location normalization stats with training stats
        self._val_dataset.time_mean = self._train_dataset.time_mean
        self._val_dataset.time_std = self._train_dataset.time_std
        self._val_dataset.loc_mean = self._train_dataset.loc_mean
        self._val_dataset.loc_std = self._train_dataset.loc_std

        if self.test_seqs is not None:
            self._test_dataset = STPPDataset(
                self.test_seqs,
                normalize_time=self.normalize,
                normalize_space=self.normalize,
                cov_mean=self._train_dataset.cov_mean,
                cov_std=self._train_dataset.cov_std,
            )
            self._test_dataset.time_mean = self._train_dataset.time_mean
            self._test_dataset.time_std = self._train_dataset.time_std
            self._test_dataset.loc_mean = self._train_dataset.loc_mean
            self._test_dataset.loc_std = self._train_dataset.loc_std

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self._test_dataset is None:
            return None
        return DataLoader(
            self._test_dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_fn,
            num_workers=self.num_workers,
        )
