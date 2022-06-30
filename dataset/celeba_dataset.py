import torch.utils.data
from torch import Tensor, nn
import glob
import os
from PIL import Image
from typing import Tuple, Optional


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, dir_path: str, transforms: nn.Module):
        self.dir_path = dir_path
        self.transforms = transforms
        self.images = glob.glob(os.path.join(dir_path, '*'))

    def __getitem__(self, idx: int) -> Tensor: 
        img = Image.open(self.images[idx])
        return self.transforms(img)

    def __len__(self) -> int:
        return len(self.images)
