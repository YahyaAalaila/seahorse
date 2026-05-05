from torch.utils.data import TensorDataset
import torch

from unified_stpp.data.registry import DataBundle
from unified_stpp.training.data_module import STPPDataModule


def test_batch_sampler_train_loader_allows_zero_workers():
    dataset = TensorDataset(torch.arange(4))
    bundle = DataBundle(
        train_dataset=dataset,
        val_dataset=dataset,
        test_dataset=dataset,
        collate_fn=lambda batch: batch,
        train_batch_sampler=[[0, 1], [2, 3]],
    )

    loader = STPPDataModule(bundle, num_workers=0).train_dataloader()

    assert loader.num_workers == 0
    assert loader.persistent_workers is False

