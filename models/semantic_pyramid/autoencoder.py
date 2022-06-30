from typing import List, Union, Tuple

from torch import nn, Tensor, IntTensor, randn, zeros
from models.semantic_pyramid.model import VGGFeature, Generator
import consts


class Autoencoder(nn.Module):
    def __init__(self, encoder: VGGFeature, decoder: Generator, dim_z: int):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dim_z = dim_z

    def forward(self,
                x: Union[Tensor, List[Tensor]],
                class_id: IntTensor,
                masks: List[Tensor],
                input_code: bool = False,
                return_latents: bool = False) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """

        :param x: Either an image, or if input_code=True -> a list of features (tensors)
        :param z: A noise tensor from normal distribution
        :param class_id: A tensor containing an integer signifying the class
        :param masks: A list of binary masks for the features
        :param input_code: Boolean stating if the input is a latent code or an image
        :param return_latents: Whether to return the latents
        :return: The created image, and optionally the latent code
        """

        if input_code:
            features = x
            batch = x[0].shape[0]

        else:
            batch = x.shape[0]
            features, fcs = self.encoder(x)
            features = features + fcs[1:]
        # for f in features:
        #     print(f.shape)

        if consts.PREDICT_WITH_RANDOM_Z:
            z = randn(batch, self.dim_z).cuda(non_blocking=True)
        else:
            z = zeros((batch, self.dim_z)).cuda(non_blocking=True)

        # for i in range(len(features)):
        #     print(features[i].shape, masks[i].shape)

        y = self.decoder(z, class_id, features, masks)

        masked_feats = []
        for i in range(len(features)):
            masked_feats.append(features[i] * masks[i])

        if return_latents:
            return y, masked_feats
        else:
            return y
