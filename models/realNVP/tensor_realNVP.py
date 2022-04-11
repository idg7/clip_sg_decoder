from typing import List
from .realNVPv2 import RealNVP


class TensorRealNVP(RealNVP):
    def __init__(self, x_shape: List[int],
                 n_blocks: int,
                 input_size: int,
                 hidden_dim: int,
                 n_hidden: int,
                 cond_label_size: int,
                 batch_norm: bool):
        super(TensorRealNVP, self).__init__(n_blocks, input_size, hidden_dim, n_hidden, cond_label_size, batch_norm)
        self.x_shape = x_shape

    def forward(self, x, y=None):
        batch = x.shape[0]
        x = x.reshape((batch, -1))
        return self.net(x, y)

    def inverse(self, u, y=None):
        batch = u.shape[0]
        x, logp = self.net.inverse(u, y)
        x = x.reshape([batch] + self.x_shape)
        return x, logp
