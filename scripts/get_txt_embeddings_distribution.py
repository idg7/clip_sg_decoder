import clip
import os
import torch
from torch import Tensor
from torch.utils import data
import pandas as pd
import sys
sys.path.append('/home/administrator/PycharmProjects/clip_sg_decoder/')
from consts import CLIP_EMBEDDING_DIMS
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import consts
from typing import Tuple
from dataset.data_setup import setup_simple_ds


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('dataset_file', type=str, help='Path to the dataset we wish to normalize')
    parser.add_argument('names_col', type=str, help='Name of column containing names')
    parser.add_argument('dataset_name', type=str, help='How to call the output file')
    parser.add_argument('--arch', default='ViT-B/32', type=str, help='[ViT-B/16 / ViT-B/32 / RN50 / RN101/ RN50x4 / RN50x16]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for normalization')
    
    return parser.parse_args()


def get_clip_norm(model: torch.nn.Module, names: pd.Series, emb_dim: int) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        dataset_size = len(names)
        mean = torch.zeros(emb_dim).cuda()
        for name in tqdm(names):
            name = clip.tokenize(name).cuda(non_blocking=True)
            
            embeddings = model.encode_text(name)
            sum_embeddings = embeddings.sum(dim=0)
            mean += sum_embeddings / dataset_size

        print(mean)

        cov = torch.zeros((emb_dim, emb_dim)).cuda()
        for name in tqdm(names):
            name = clip.tokenize(name).cuda(non_blocking=True)
            
            embeddings = model.encode_text(name)
            centered = embeddings[0] - mean
            centered = centered[:, None]
            cov += (centered @ centered.T) / (dataset_size - 1)
    return mean, cov

if __name__ == '__main__':
    args = get_args()

    model, preprocess = clip.load(args.arch)
    emb_dim = CLIP_EMBEDDING_DIMS[args.arch]
    names = pd.read_csv(args.dataset_file)[args.names_col].unique()

    with torch.set_grad_enabled(False):
        model.cuda()
        mean, cov = get_clip_norm(model, names, emb_dim)
        os.makedirs('/home/ssd_storage/experiments/clip_decoder/norms/', exist_ok=True)
        torch.save({'mean': mean.cpu(), 'cov': cov.cpu(), 'std': torch.sqrt(torch.diagonal(cov)).cpu()}, f'/home/ssd_storage/experiments/clip_decoder/norms/CLIP_{args.arch.replace("/", "_")}_{args.dataset_name}_distribution.pt')
