import torch.utils.data
from torch import Tensor, nn
import glob
import os
from PIL import Image
import pandas as pd
from typing import Tuple, Optional


class CelebANamedDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path: str, naming_fp: str, name_col: str, img_col: str, transforms: nn.Module):
        self.dir_path = dir_path
        self.transforms = transforms
        self.naming = pd.read_csv(naming_fp, index_col=img_col)
        print(self.naming)
        self.name_col = name_col
        self.images = glob.glob(os.path.join(dir_path, '*'))

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        img = Image.open(self.images[idx])
        img_name = os.path.basename(self.images[idx])
        name = self.naming.loc[img_name, self.name_col]
        return self.transforms(img), name

    def __len__(self) -> int:
        return len(self.images)
