import clip
import torch.utils.data
import glob
import os
import torch

import plotly.express as px
import pandas as pd

from tqdm import tqdm
from numpy import random
from torch.utils.data import DataLoader
from skimage import io
from PIL import Image


class ComparisonDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, image_transforms, class_naming: pd.DataFrame):
        self.dir_path = dir_path
        self.image_transforms = image_transforms
        self.classes = glob.glob(os.path.join(dir_path, '*'))
        self.class_naming = class_naming
        self.images = []
        for cl in self.classes:
            class_img = random.choice(glob.glob(os.path.join(cl, '*')), replace=False, size=1)
            self.images.append(class_img[0])

    def __getitem__(self, idx):
        class_id = os.path.dirname(os.path.relpath(self.images[idx], self.dir_path))
        return self.image_transforms(Image.open(self.images[idx])), self.class_naming.loc[class_id, 'Name'], class_id

    def __len__(self):
        return len(self.images)


def check_familiarity(model: torch.nn.Module, dataset, sim_thresh = 0.3):
    model.requires_grad_(False)
    familiar = {'name': [], 'class': [], 'score': []}
    # i = 0
    with torch.set_grad_enabled(False):
        for imgs, names, class_id in tqdm(dataset):
            imgs = imgs.cuda(non_blocking=True)
            text = clip.tokenize(names).cuda(non_blocking=True)

            image_features = model.encode_image(imgs)
            text_features = model.encode_text(text)

            image_features = image_features / image_features.norm(dim=1, p=2)[:, None]
            text_features = text_features / text_features.norm(dim=1, p=2)[:, None]
            sim = torch.mm(image_features, text_features.transpose(0, 1))

            familiar_indices = (sim > sim_thresh).nonzero(as_tuple=True)
            print(familiar_indices)
            max = sim.max(dim=1)
            max_class = max.indices.cpu().detach().numpy()
            max_vals = max.values.cpu().detach().numpy()
            for i in range(sim.shape[0]):
                print(f'{names[i]}: {sim[i][i].item()}')
            for i in range(max_class.shape[0]):
                # if familiar_indices[0][i] == familiar_indices[1][i]:
                if max_class[i].item() == i:
                    name = names[max_class[i].item()]
                    id = class_id[max_class[i].item()]
                    val = max_vals[i]
                    if name not in familiar['name']:
                        familiar['name'].append(name)
                        familiar['class'].append(id)
                        familiar['score'].append(val)
    return pd.DataFrame(familiar)


if __name__ == '__main__':
    model, preprocess = clip.load('RN50x16')
    # naming = pd.read_csv('/home/ssd_storage/datasets/vggface2/identity_meta_no_commas.csv', index_col='Class_ID', sep=' ')
    # dataset = DataLoader(ComparisonDataset('/home/ssd_storage/datasets/vggface2_mtcnn/', preprocess, naming),
    #                      batch_size=512, num_workers=4, pin_memory=True, shuffle=True)

    naming = pd.read_csv('/home/ssd_storage/datasets/Cognitive_exp/adva_familiar/map.csv', index_col='Class_Id',
                         sep=',')
    dataset = DataLoader(ComparisonDataset('/home/ssd_storage/datasets/Cognitive_exp/adva_familiar/new_familiar', preprocess, naming),
                         batch_size=34, num_workers=4, pin_memory=True, shuffle=True)
    check_familiarity(model, dataset).to_csv(
        '/home/ssd_storage/datasets/Cognitive_exp/adva_familiar/acc@1.csv')
    # check_familiarity(model, dataset).to_csv('/home/ssd_storage/experiments/clip_decoder/clip_familiar_vggface2_max_class.csv')

# print(sim.mean())
# sim_cpu = sim.cpu().detach().numpy()
# fig = px.imshow(sim_cpu,
#                 labels=dict(x="Images", y="Text", color="Similarity"),
#                 x=names,
#                 y=names)
# fig.update_xaxes(side="top")
# fig.show()
# sim_thresh = input("enter thresh:")