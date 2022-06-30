import torch
import torch.nn as nn
import torch.distributions as D
from .realNVPv2 import RealNVP


class NormedRealNVP(RealNVP):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True, x_mu=None, x_sig=None, y_mu=None, y_sig=None):
        super(NormedRealNVP, self).__init__(n_blocks, input_size, hidden_size, n_hidden, cond_label_size, batch_norm)

        # Set default values
        if x_mu is None:
            x_mu = torch.zeros(input_size)
        if x_sig is None:
            x_sig = torch.ones(input_size)
        if y_mu is None:
            y_mu = torch.zeros(cond_label_size)
        if y_sig is None:
            y_sig = torch.ones(cond_label_size)
        
        # Normalizing values for x
        self.register_buffer('x_mu', x_mu)
        self.register_buffer('x_sig', x_sig)
        # Normalizing values for conditional vector y
        self.register_buffer('y_mu', y_mu)
        self.register_buffer('y_sig', y_sig)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        x = (x - self.x_mu) / self.x_sig
        y = (y - self.y_mu) / self.y_sig
        return self.net(x, y)

    def inverse(self, u, y=None):
        y = (y - self.y_mu) / self.y_sig
        x, sum_log_abs_det_jacobians = self.net.inverse(u, y)
        x = (x * self.x_sig) + self.x_mu
        return x, sum_log_abs_det_jacobians