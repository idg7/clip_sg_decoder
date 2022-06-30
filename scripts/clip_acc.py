from typing import List

import clip
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset.comparison_dataset import ComparisonDataset
from dataset.verification_dataset import VerificationDataset
import argparse
from tqdm import tqdm


def acc(df: pd.DataFrame, val_col: str, label_col: str):
    best_acc = 0
    dists = torch.from_numpy(df[val_col].to_numpy(dtype=float)).cuda()
    labels = torch.from_numpy(df[label_col].to_numpy(dtype=float)).cuda()
    for val in dists:
        curr_acc = torch.mean(((dists < val) == labels).float())
        best_acc = max(curr_acc, best_acc)
    return best_acc


def rdm_dists(model: clip.model.CLIP, dataset: DataLoader):
    labels = []
    embeddings = None

    for imgs, paths in dataset:
        labels = labels + paths
        imgs = imgs.cuda(non_blocking=True)

        emb = model.encode_image(imgs)
        emb = emb.float()
        if embeddings is None:
            embeddings = emb
        else:
            embeddings = torch.cat((embeddings, emb))
    embeddings = embeddings / embeddings.norm(dim=1, p=2)[:, None]
    relu = torch.nn.ReLU()
    res = relu(1 - torch.mm(embeddings, embeddings.transpose(0, 1)))
    return pd.DataFrame(res, columns=labels, index=labels)


def cosine_dist(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    emb1 = emb1 / emb1.norm(dim=1, p=2)[:, None]
    emb2 = emb2 / emb2.norm(dim=1, p=2)[:, None]
    relu = torch.nn.ReLU()
    return float(relu(1 - emb1 @ emb2.T))

def l2_dist(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    return float((emb1 - emb2).norm(p=2))

def measure_dists(model: clip.model.CLIP, dataset: DataLoader):
    labels = []
    dists = []

    for i, (img1, img2, _, _, label) in enumerate(tqdm(dataset)):
        img1 = img1.cuda(non_blocking=True)
        img2 = img2.cuda(non_blocking=True)

        emb1 = model.encode_image(img1)
        emb2 = model.encode_image(img2)
        emb1 = emb1.float()
        emb2 = emb2.float()
        # curr_dist = l2_dist(emb1, emb2)
        curr_dist = cosine_dist(emb1, emb2)
        dists.append(curr_dist)
        labels.append(label)
    return pd.DataFrame({'cos_dists': dists, 'isSame': labels})


def rdm_to_dist_list(rdm: pd.DataFrame, label_col: str):
    dist_list = rdm.stack().reset_index()
    dist_list = dist_list[dist_list['level_0'] != dist_list['level_1']]

    def issame(row):
        return row['level_0'].split('/')[0] == row['level_1'].split('/')[0]

    dist_list[label_col] = dist_list.apply(issame, axis=1)
    dist_list = dist_list.rename(columns={0: 'cos_dists'})

    return dist_list


def get_lfw_model_acc(dataset_path: str, pairs: List[str], labels: List[int], label_col: str = 'isSame'):
    for arch in clip.available_models():
        model, preprocess = clip.load(arch, device='cuda')
        dataset = DataLoader(
            VerificationDataset(dataset_path, pairs, labels, preprocess),
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False)
        dist_list = measure_dists(model, dataset)
        model_acc = acc(dist_list, 'cos_dists', label_col)
        print(f"{arch}: {model_acc * 100}%")


def get_model_acc(dataset_path: str):
    for arch in clip.available_models():
        model, preprocess = clip.load(arch, device='cuda')
        dataset = DataLoader(
            ComparisonDataset(dataset_path, preprocess),
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False)
        rdm = measure_dists(model, dataset)
        dist_list = rdm_to_dist_list(rdm, 'isSame')
        model_acc = acc(dist_list, 'cos_dists', 'isSame')
        print(f"{arch}: {model_acc * 100}%")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default='/home/administrator/datasets/dist_mat_dataset_mtcnn/female')

    return parser.parse_args()


def rdm_acc():
    female = '/home/administrator/datasets/dist_mat_dataset_mtcnn/female'
    male = '/home/administrator/datasets/dist_mat_dataset_mtcnn/male'
    with torch.no_grad():
        print('Female:')
        get_model_acc(female)
        print('Male:')
        get_model_acc(male)


def lfw_acc():
    pairs_file_path = '/home/administrator/PycharmProjects/libi/facial_feature_impact_comparison/lfw_test_pairs.txt'
    lfw_path = '/home/administrator/datasets/lfw-align-128'
    pairs_list = pd.read_csv(pairs_file_path, sep=' ',header=None)
    im1 = pairs_list[0].to_list()
    im2 = pairs_list[1].to_list()
    labels = pairs_list[2].to_list()
    pairs = []
    for i in range(len(im1)):
        pairs.append((im1[i], im2[i]))
    get_lfw_model_acc(lfw_path, pairs, labels)


if __name__ == '__main__':
    with torch.no_grad():
        lfw_acc()
