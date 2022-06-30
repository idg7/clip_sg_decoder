from typing import Optional

from torch import nn, distributions, Tensor
import torch
import numpy as np
import mlflow


class RealNVP(nn.Module):
    """
    Adapted from https://github.com/senya-ashukha/real-nvp-pytorch
    """
    def __init__(self, nets, nett, masks, prior):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.mask = nn.Parameter(masks, requires_grad=False)
        self.t = nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = nn.ModuleList([nets() for _ in range(len(masks))])
        self.mean_log_det_J = 0
        self.mean_log_prob = 0
        self.limit = 10
        self.count = 0

    def g(self, z: Tensor, h: Optional[Tensor] = None):
        x = z

        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            if h != None:
                xh = torch.cat([x_, h], dim=1)
            else:
                xh = x_
            s = self.s[i](xh) * (1 - self.mask[i])
            t = self.t[i](xh) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x: Tensor, h: Optional[Tensor] = None):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            if h != None:
                zh = torch.cat([z_, h], dim=1)
            else:
                zh = z_
            s = self.s[i](zh) * (1 - self.mask[i])
            t = self.t[i](zh) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x: Tensor, h: Optional[Tensor] = None):
        z, logp = self.f(x, h)
        self.mean_log_det_J += float(logp.mean())
        self.count += 1

        log_prob = self.prior.log_prob(z)
        self.mean_log_prob += float(log_prob.mean())

        if (self.count % self.limit) == 0:
            mlflow.log_metric('mean_log_det_J', self.mean_log_det_J / self.limit, self.count)
            mlflow.log_metric('mean_log_prob', self.mean_log_prob / self.limit, self.count)
            self.mean_log_det_J = 0
            self.mean_log_prob = 0
        return log_prob + logp

    def sample(self, batch_size: int, h: Optional[Tensor] = None):
        z = self.prior.sample([batch_size])
        logp = self.prior.log_prob(z)
        if h != None:
            x = self.g(z, h)
        else:
            x = self.g(z)
        return x


def get_realnvp(x_dim: int = 512, condition_dim: int = 512, depth: int = 5, hidden_dim: Optional[int] = None):
    joined_dim = x_dim + condition_dim
    half_dim = int(x_dim / 2)
    if hidden_dim == None:
        hidden_dim = joined_dim

    nets = lambda: nn.Sequential(nn.Linear(joined_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                                 nn.Linear(hidden_dim, x_dim), nn.Tanh())
    nett = lambda: nn.Sequential(nn.Linear(joined_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                                 nn.Linear(hidden_dim, x_dim))
    masks = torch.from_numpy(np.array([[1] * half_dim + [0] * half_dim] * depth).astype(np.float32)).cuda()
    embedding_distribution = torch.load('/home/ssd_storage/experiments/clip_decoder/celebA_subset_distribution2.pt')
    prior = distributions.MultivariateNormal(embedding_distribution['mean'].cuda(), embedding_distribution['cov'].cuda())
    flow = RealNVP(nets, nett, masks, prior)
    return flow
