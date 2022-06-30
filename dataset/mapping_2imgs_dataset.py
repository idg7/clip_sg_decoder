import torch.utils.data
from torch import Tensor
import glob
import os
from PIL import Image
from typing import Tuple, Optional


class CelebA2ImgsMappingDataset(torch.utils.data.Dataset):
    def __init__(self, a_dir_path, b_dir_path, a_transforms, b_transforms):
        self.a_dir_path = a_dir_path
        self.b_dir_path = b_dir_path
        self.a_transforms = a_transforms
        self.b_transforms = b_transforms
        self.a_images = glob.glob(os.path.join(self.a_dir_path, '*'))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        a_img = Image.open(self.b_images[idx])
        b_img = os.path.join(self.b_dir_path, os.path.basename(self.a_images[idx]))
        return self.a_transforms(a_img), self.b_transforms(b_img)

    def __len__(self) -> int:
        return len(self.b_images)
