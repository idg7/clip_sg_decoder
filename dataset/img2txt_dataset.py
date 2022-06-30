import torch.utils.data
import glob
import os
from PIL import Image
import pandas as pd


class Img2TxtDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path: str, transforms: torch.nn.Module, labels_path: str):
        self.dir_path = dir_path
        self.transforms = transforms
        self.classes = glob.glob(os.path.join(dir_path, '*'))
        self.labels_df = pd.read_csv(labels_path, index_col='class')
        self.labels = []
        self.images = []
        for cl_path in self.classes:
            cl_id = os.path.basename(cl_path)
            cl_label = self.labels_df.loc[cl_id, 'name']
            cl_images = glob.glob(os.path.join(cl_path, '*'))
            self.labels = self.labels + [cl_label for _ in cl_images]
            self.images = self.images + cl_images

    def __getitem__(self, idx):
        return self.transforms(Image.open(self.images[idx])), self.labels[idx]

    def __len__(self):
        return len(self.images)
