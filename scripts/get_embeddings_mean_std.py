import clip
import os
import torch
from torch import Tensor
from torch.utils import data
from torchvision import datasets, transforms
import sys
sys.path.append('/home/administrator/PycharmProjects/clip_sg_decoder/')
from consts import CLIP_EMBEDDING_DIMS
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from models.e4e.model_utils import setup_model
from models import get_psp
import consts
from typing import Tuple, List
from dataset.data_setup import setup_simple_ds


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Path to the dataset we wish to normalize')
    parser.add_argument('dataset_name', type=str, help='How to call the output file')
    parser.add_argument('--model', default='CLIP', type=str, help='CLIP or SG')
    parser.add_argument('--arch', default='ViT-B/32', type=str, help='[ViT-B/16 / ViT-B/32 / RN50 / RN101/ RN50x4 / RN50x16] or [E4E / PSP]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for normalization')
    
    return parser.parse_args()


def get_clip_norm(model: torch.nn.Module, loader: data.DataLoader, emb_dim: int) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        dataset_size = len(loader.dataset)
        mean = torch.zeros(emb_dim).cuda()
        for images in tqdm(loader):
            if (type(images) == Tuple) or (type(images) == list):
                images = images[0]
            images = images.cuda(non_blocking=True)
            embeddings = model.encode_image(images)
            sum_embeddings = embeddings.sum(dim=0)
            mean += sum_embeddings / dataset_size

        print(mean)

        cov = torch.zeros((emb_dim, emb_dim)).cuda()
        for images in tqdm(loader):
            if (type(images) == Tuple) or (type(images) == list):
                images = images[0]
            images = images.cuda(non_blocking=True)
            embeddings = model.encode_image(images)
            centered = embeddings[0] - mean
            centered = centered[:, None]
            cov += (centered @ centered.T) / (dataset_size - 1)
    return mean, cov


def get_Wplus_norm(model: torch.nn.Module, loader: data.DataLoader, emb_dim: int) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        dataset_size = len(loader.dataset)
        mean = [torch.zeros(emb_dim).cuda() for _ in range(18)]
        for images in tqdm(loader):
            if type(images) == Tuple:
                images = images[0]
            images = images.cuda(non_blocking=True)
            y, y_latents = model(images, return_latents=True)
            for i in range(18):
                sum_latents = y_latents.sum(dim=0)
                mean[i] += sum_latents[i] / dataset_size
        print(mean)

        cov = [torch.zeros((emb_dim, emb_dim)).cuda() for _ in range(18)]
        for images in tqdm(loader):
            if type(images) == Tuple:
                images = images[0]
            images = images.cuda(non_blocking=True)
            y, y_latents = model(images, return_latents=True)
            for i in range(18):
                centered = y_latents[0, i] - mean[i]
                centered = centered[:, None]
                cov[i] += (centered @ centered.T) / (dataset_size - 1)
        print(cov)
    return mean, cov


if __name__ == '__main__':
    args = get_args()

    if args.model == 'CLIP':
        model, preprocess = clip.load(args.arch)
        emb_dim = CLIP_EMBEDDING_DIMS[args.arch]
    else:
        emb_dim=512
        if args.arch == 'E4E':
            print("Using E4E autoencoder")
            model, opts = setup_model(consts.model_paths['e4e_checkpoint_path'], consts.model_paths['stylegan_weights'])
        elif args.arch == 'PSP':
            print("Using PSP autoencoder")
            model = get_psp()
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    with torch.set_grad_enabled(False):
        model.cuda()
        print(args.dataset_path)
        loader = setup_simple_ds(args.dataset_path, preprocess, args)
        if args.model == 'CLIP':
            mean, cov = get_clip_norm(model, loader, emb_dim)
            os.makedirs('/home/ssd_storage/experiments/clip_decoder/norms/', exist_ok=True)
            torch.save({'mean': mean.cpu(), 'cov': cov.cpu(), 'std': torch.sqrt(torch.diagonal(cov)).cpu()}, f'/home/ssd_storage/experiments/clip_decoder/norms/CLIP_{args.arch.replace("/", "_")}_{args.dataset_name}_distribution.pt')
        else:
            mean, cov = get_Wplus_norm(model, loader, emb_dim)
            for i in range(len(mean)):
                os.makedirs('/home/ssd_storage/experiments/clip_decoder/norms/', exist_ok=True)
                torch.save({'mean': mean[i].cpu(), 'cov': cov[i].cpu(), 'std': torch.sqrt(torch.diagonal(cov[i])).cpu()}, f'/home/ssd_storage/experiments/clip_decoder/norms/W+_{args.arch}_{args.dataset_name}_distribution_{i}.pt')
