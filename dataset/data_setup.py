from builtins import NotImplementedError
from torchvision import datasets
from torch.utils.data import DataLoader
from dataset.img2txt_dataset import Img2TxtDataset
from dataset.img_txt_label_dataset import ImgTxtLabelDataset
from dataset.celeba_mapping_dataset import CelebAMappingDataset
from .celeba_dataset import CelebADataset
from dataset.inversion_dataset import InversionDataset
from dataset.celeba_named_dataset import CelebANamedDataset
from .celeba_2imgs_mapping_dataset import CelebA2ImgsMappingDataset

import torch


def setup_simple_ds(path: str, 
    transforms: torch.nn.Module, opts) -> DataLoader:
    if any([path.startswith(x) for x in ["/home/ssd_storage/datasets/CelebAMask-HQ/", "/home/ssd_storage/datasets/celebA_full"]]):
        return DataLoader(
            CelebADataset(path, transforms),
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            pin_memory=True,
            shuffle=True)
        
    return DataLoader(
        datasets.ImageFolder(path, transforms),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=True,
        shuffle=True)

def setup_2imgs_dataset(
    path: str, 
    a_transforms: torch.nn.Module, 
    b_transforms: torch.nn.Module, opts) -> DataLoader:
    if any([path.startswith(x) for x in ["/home/ssd_storage/datasets/CelebAMask-HQ/", "/home/ssd_storage/datasets/celebA_full"]]):
        return DataLoader(
            CelebAMappingDataset(path, a_transforms, b_transforms),
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            pin_memory=True,
            shuffle=True)
        
    return DataLoader(
        InversionDataset(path, a_transforms, b_transforms),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=True,
        shuffle=True)
    
def setup_2models_dataset(
    path: str, encoder_imgs_paths: str,
    a_transforms: torch.nn.Module, 
    b_transforms: torch.nn.Module, opts) -> DataLoader:

    if any([path.startswith(x) for x in ["/home/ssd_storage/datasets/CelebAMask-HQ/", "/home/ssd_storage/datasets/celebA_full"]]):
        return DataLoader(
            CelebA2ImgsMappingDataset(path, encoder_imgs_paths, a_transforms, b_transforms),
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            pin_memory=True,
            shuffle=True)
        
    raise NotImplementedError()


def setup_dataset(path: str, data_transforms: torch.nn.Module, opts) -> DataLoader:
    if path in ['']:
        CelebADataset(path, data_transforms)
    return DataLoader(
        datasets.ImageFolder(path, data_transforms),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=True,
        shuffle=True)


def setup_img2txt_dataset(path: str, cls_txt_csv: str, data_transforms: torch.nn.Module, opts) -> DataLoader:
    if any([path.startswith(x) for x in ["/home/ssd_storage/datasets/CelebAMask-HQ/", "/home/ssd_storage/datasets/celebA_full"]]):
        return DataLoader(
        CelebANamedDataset(path, cls_txt_csv, 'identity_name', 'hq_img', data_transforms),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=True,
        shuffle=True)
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
