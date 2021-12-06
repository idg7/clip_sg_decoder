import os
import clip
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.comparison_dataset import ComparisonDataset


class CosineDist(torch.nn.CosineSimilarity):
    def forward(self, x, y):
        return 1 - super(CosineDist, self).forward(x, y)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='ViT-B/32')
    parser.add_argument("--dataset_path", type=str, default='/home/ssd_storage/datasets/objects_faces_rdm')
    parser.add_argument('--rdm_dest', type=str, default='/home/ssd_storage/experiments/clip_decoder')

    return parser.parse_args()


def get_rdm(model: clip.model.CLIP, dataset: DataLoader):
    model.eval()
    dist = CosineDist()
    columns = {}
    rows = []

    with torch.set_grad_enabled(False):
        for img1, label1 in tqdm(dataset):
            label1 = label1[0]
            rows.append(label1)
            img1 = img1.cuda()
            for img2, label2 in dataset:
                label2 = label2[0]
                img2 = img2.cuda()
                emb1 = model.encode_image(img1)
                emb2 = model.encode_image(img2)
                if label2 not in columns:
                    columns[label2] = []
                d = dist(emb1, emb2)
                # print(d[0].item(), label1, label2)
                columns[label2].append(d[0].item())
    return pd.DataFrame(index=rows, columns=columns)


def efficient_rdm(model: clip.model.CLIP, dataset: DataLoader):
    model.eval()
    rows = []
    embeddings = None
    with torch.set_grad_enabled(False):
        for imgs, labels1 in tqdm(dataset):
            rows = rows + [label for label in labels1]
            imgs = imgs.cuda()

            emb = model.encode_image(imgs)
            if embeddings is None:
                embeddings = emb
            else:
                embeddings = torch.cat((embeddings, emb))

    embeddings.requires_grad = False
    embeddings = embeddings / embeddings.norm(dim=1, p=2)[:, None]
    relu = torch.nn.ReLU()
    res = relu(1 - torch.mm(embeddings, embeddings.transpose(0, 1)))
    res = res.cpu().detach().numpy()
    print(res.shape)
    return pd.DataFrame(res,
                        columns=[rows[i] for i in range(res.shape[0])],
                        index=[(rows[i]) for i in range(res.shape[0])])


if __name__ == '__main__':
    args = get_args()
    model, preprocess = clip.load(args.model, device='cuda')
    dataset = DataLoader(
        ComparisonDataset(args.dataset_path, preprocess),
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=False)
    efficient_rdm(model, dataset).to_csv(os.path.join(args.rdm_dest, 'faces_objects_rdm.csv'))

