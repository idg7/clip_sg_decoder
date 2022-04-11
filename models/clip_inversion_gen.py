from typing import List

from torch import nn, Tensor
from models.realNVP import RealNVP, predict
import torch


class CLIPInversionGenerator(nn.Module):
    def __init__(self,
                 autoencoder: nn.Module,
                 mappers: List[RealNVP],
                 autoencoder_facepool: nn.Module):
        super(CLIPInversionGenerator, self).__init__()
        self.autoencoder = autoencoder
        self.mappers = torch.nn.ModuleList(mappers)
        self.autoencoder_facepool = autoencoder_facepool

    def predict(self, x: Tensor):
        x_embedding = x
        log_probs = 0.0
        loss_dict = {}
        w = []
        for i, mapper in enumerate(self.mappers):
            y_hat_latents = predict(mapper, x)
            loss = -mapper.log_prob(y_hat_latents, x_embedding).mean()
            loss = loss.mean()
            loss_dict[f'mapper train loss {i}'] = float(loss)
            log_probs += loss

            w.append(y_hat_latents)

        w = torch.stack(w, dim=1)
        y_hat, _ = self.autoencoder(w, input_code=True, return_latents=True)
        log_probs = -log_probs
        return y_hat, log_probs, loss_dict

    def forward(self, x: Tensor):
        if not self.training:
            return self.predict(x)
        else:
            raise NotImplementedError('Training the clip image inversion is not implemented as part of this model')






