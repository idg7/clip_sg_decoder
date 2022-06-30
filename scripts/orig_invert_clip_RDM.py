import os
from typing import List

import clip
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.comparison_dataset import ComparisonDataset
from models import get_psp
from torchvision import transforms, datasets


def im_txt_rdm(model: clip.model.CLIP, dataset: DataLoader, texts: List[str]):
    model.eval()
    rows = []
    frequency = 5
    embeddings = None
    clip_resize = transforms.Resize((224, 224))
    with torch.set_grad_enabled(False):
        for i, (imgs, labels1) in enumerate(tqdm(dataset)):
            if (i % frequency) == 0:
                rows = rows + [label for label in labels1]
                imgs = imgs.cuda(non_blocking=True)

                emb = model.encode_image(clip_resize(imgs))
                emb = emb.float()
                if embeddings is None:
                    embeddings = emb
                else:
                    embeddings = torch.cat((embeddings, emb))

        tokens = clip.tokenize(texts).cuda()
        text_feats = model.encode_text(tokens)
        text_feats = text_feats.float()

    text_feats = text_feats / text_feats.norm(dim=1, p=2)[:, None]
    embeddings = embeddings / embeddings.norm(dim=1, p=2)[:, None]
    relu = torch.nn.ReLU()
    res = relu(1 - torch.mm(text_feats, embeddings.transpose(0, 1)))
    res = res.cpu().detach().numpy()
    print(res.shape)
    return pd.DataFrame(res,
                        columns=[rows[i] for i in range(res.shape[1])],
                        index=texts)


def efficient_rdm(model: clip.model.CLIP, psp: torch.nn.Module, dataset: DataLoader, invert_dataset: DataLoader):
    model.eval()
    rows = []
    embeddings = None
    psp_resize = transforms.Resize((256, 256))
    clip_resize = transforms.Resize((224, 224))
    with torch.set_grad_enabled(False):
        for i, (imgs, labels1) in enumerate(tqdm(dataset)):
            if (i % 28) == 0:
                rows = rows + [label for label in labels1]
                imgs = imgs.cuda(non_blocking=True)

                emb = model.encode_image(clip_resize(imgs))
                if embeddings is None:
                    embeddings = emb
                else:
                    embeddings = torch.cat((embeddings, emb))

        for i, (imgs, labels1) in enumerate(tqdm(invert_dataset)):
            if (i % 28) == 0:
                rows = rows + [f'invert_{label}' for label in labels1]
                imgs = imgs.cuda(non_blocking=True)
                inverts = psp(psp_resize(imgs))

                emb = model.encode_image(clip_resize(inverts))
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
    clip_model, sg_preprocess = clip.load('ViT-B/32')
    sg_preprocess.transforms[-1] = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    _, preprocess = clip.load('ViT-B/32')

    # path = '/home/ssd_storage/datasets/ffhq'
    path = '/home/ssd_storage/datasets/celebA_barack_obama'
    # path = '/home/ssd_storage/datasets/celebA_crops'
    clip_model.cuda()
    psp = get_psp()
    psp.cuda()
    dataset = DataLoader(
        ComparisonDataset(path, preprocess),
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=False)
    # invert_dataset = DataLoader(
    #     ComparisonDataset(path, sg_preprocess),
    #     batch_size=1,
    #     num_workers=4,
    #     pin_memory=True,
    #     shuffle=False)
    # efficient_rdm(clip_model, psp, dataset, invert_dataset).to_csv(
    #     os.path.join('/home/ssd_storage/experiments/clip_decoder', 'ffhq_faces_inverts_rdm_normed.csv'))

    print(im_txt_rdm(clip_model, dataset, ['Obama', 'Barack Obama', 'A photo of Barack Obama', 'A photo of Obama']))
