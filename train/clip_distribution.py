from torch import nn, Tensor
from torch.utils import data
from torchvision import datasets, transforms


def calc_mean(model: nn.Module, dataset: data.DataLoader):
    mean = 0.0
    for imgs, _ in dataset:
        imgs.cuda(non_blocking=True)
        batch_samples = imgs.size(0)
        feats = model.encode_image(imgs)
        mean += feats.sum(dim=0) / len(dataset.dataset)
    return mean


def calc_covariance(model: nn.Module, dataset: data.DataLoader, mean: Tensor):
    cov = 0.0
    for imgs, _ in dataset:
        batch_size = imgs.size(0)
        imgs.cuda(non_blocking=True)
        feats = model.encode_image(imgs)
        deviation = feats - mean
        for i in range(batch_size):
            sample_feat = deviation[i, None]
            sample_cov = sample_feat.T @ sample_feat
            cov += sample_cov / len(dataset.dataset)
    return cov