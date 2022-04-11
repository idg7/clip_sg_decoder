from torchvision import datasets
from torch.utils.data import DataLoader
from dataset.img2txt_dataset import Img2TxtDataset
from dataset.img_txt_label_dataset import ImgTxtLabelDataset

import torch


def setup_dataset(path: str, data_transforms: torch.nn.Module, opts) -> DataLoader:
    return DataLoader(
        datasets.ImageFolder(path, data_transforms),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=True,
        shuffle=True)


def setup_img2txt_dataset(path: str, cls_txt_csv: str, data_transforms: torch.nn.Module, opts) -> DataLoader:
    return DataLoader(
        Img2TxtDataset(path, data_transforms, cls_txt_csv),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=True,
        shuffle=True)


def setup_img_txt_cls_dataset(path: str, cls_txt_csv: str, data_transforms: torch.nn.Module, opts) -> DataLoader:
    return DataLoader(
        ImgTxtLabelDataset(path, data_transforms, cls_txt_csv),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=True,
        shuffle=True)
