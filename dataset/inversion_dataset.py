import torch.utils.data
from torch import Tensor
import glob
import os
from PIL import Image
from typing import Tuple, Optional


class InversionDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, a_transforms, b_transforms):
        self.dir_path = dir_path
        self.a_transforms = a_transforms
        self.b_transforms = b_transforms
        self.classes = glob.glob(os.path.join(dir_path, '*'))
        self.images = []
        for cl in self.classes:
            self.images = self.images + glob.glob(os.path.join(cl, '*'))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]: 
        img = Image.open(self.images[idx])
        return self.a_transforms(img), self.b_transforms(img)

    def __len__(self) -> int:
        return len(self.images)
