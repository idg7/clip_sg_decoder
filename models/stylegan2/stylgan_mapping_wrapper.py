from typing import List, Union, Optional

from torch import nn, Tensor, zeros

from models.stylegan2.model import EqualLinear


class GeneratorMappingWrapper(nn.Module):
    def __init__(self,
                 n: int,
                 generator: nn.Module,
                 encoding_normalization: nn.Module,
                 lr_mul: float = 0.01,
                 dim: Union[int, List[int]] = 512,
                 use_pretrained_mapping: bool = False,
                 w_center: Optional[Tensor] = None):
        super(GeneratorMappingWrapper, self).__init__()
        layers = []
        if type(dim) == list:
            for i in range(1, len(dim)):
                layers.append(
                    EqualLinear(
                        dim[i-1], dim[i], lr_mul=lr_mul, activation='fused_lrelu'
                    )
                )
        else:
            for i in range(n):
                layers.append(
                    EqualLinear(
                        dim, dim, lr_mul=lr_mul, activation='fused_lrelu'
                    )
                )
        self.pre_mapping = nn.Sequential(*layers)
        self.generator = generator
        self.input_is_latent = not use_pretrained_mapping

        if w_center is None:
            latent_dim = dim # assuming dim is an int
            if type(dim) == list:
                latent_dim = dim[-1]
            w_center = zeros(latent_dim)
        self.w_center = w_center

    def forward(self, styles, return_latents=True):
        return self.generator([self.w_center + self.pre_mapping(s.float()) for s in styles],
                              input_is_latent=self.input_is_latent,
                              return_latents=return_latents)
