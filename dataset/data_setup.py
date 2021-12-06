from torchvision import datasets
from torch.utils.data import DataLoader

import torch


def setup_dataset(path: str, data_transforms: torch.nn.Module, opts) -> DataLoader:
    return DataLoader(
        datasets.ImageFolder(path, data_transforms),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=True,
        shuffle=True)
