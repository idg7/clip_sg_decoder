from torch import nn, Tensor
from typing import List


class EmbeddingModel(nn.Module):
    def __init__(self, names: List[str], dim: int = 512):
        self.names = names
        self.naming_map = {name: i for i, name in enumerate(self.names)}
        self.emb = nn.Embedding(len(self.names), dim)

    def forward(self, names: List[str]) -> Tensor:
        idx = [self.naming_map[name] for name in names]
        return self.emb(idx)

    def encode_text(self, names: List[str]) -> Tensor:
        return self.forward(names)