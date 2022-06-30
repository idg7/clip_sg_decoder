from torch import nn, Tensor, arccos
import torch

torch.PI = torch.acos(torch.zeros(1)).item() * 2


def cartesian_to_spherical(x: Tensor):
    """
    based on https://en.wikipedia.org/wiki/N-sphere section spherical coordinates
    :param x: points in Rn
    :return: the points in spherical coordinates (Rn sphere)
    """
    r = x.norm(p=2, dim=1)
    phi = []
    for dim in range(0, x.shape[1] - 2):
        x_i = x[:, dim][:, None]
        x_i_to_n_norm = x[:, dim:].norm(p=2, dim=1)[:, None]
        phi_i = arccos(x_i/x_i_to_n_norm)
        phi_i[(x_i_to_n_norm == 0)] = 0
        phi_i[(x_i_to_n_norm == 0) & (x_i_to_n_norm < 0)] = torch.PI
        phi.append(phi_i)
    dim = x.shape[1] - 1
    condition = (x[:, dim] > 0)
    bias = (2 * torch.PI * condition)[:, None]
    factor = (x[:, dim] * (1 - condition.float()))[:, None]

    phi_n = (bias + factor * arccos(x[:, dim - 1:]))[:, 0][:, None]
    phi.append(phi_n)
    return torch.cat(phi, dim=1)
