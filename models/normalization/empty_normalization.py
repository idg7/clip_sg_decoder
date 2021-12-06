from torch import nn, Tensor


class EmptyNormalization(nn.Module):
    def forward(self, x: Tensor):
        return x
