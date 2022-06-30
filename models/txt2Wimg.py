from typing import List, Tuple, Dict

import clip
from torch import nn, Tensor, stack

from models import CLIPInversionGenerator
from models.realNVP import RealNVP, predict


class Txt2WImg(nn.Module):
    def __init__(self,
                 clip_model: clip.model.CLIP,
                 autoencoder: nn.Module,
                 mappers: List[RealNVP],
                 autoencoder_facepool: nn.Module):
        super(Txt2WImg, self).__init__()
        self.clip_model = clip_model
        self.autoencoder = autoencoder
        self.mappers = nn.ModuleList(mappers)
        self.autoencoder_facepool = autoencoder_facepool

    def predict(self, labels: List[str]) -> Tuple[Tensor, float, Dict[str, float]]:
        txt_embeddings = self.clip_model.encode_text(labels)

        log_probs = 0.0
        loss_dict = {}
        w = []
        for i, mapper in enumerate(self.mappers):
            y_hat_latents = predict(mapper, txt_embeddings)
            loss = -mapper.log_prob(y_hat_latents, txt_embeddings).mean()
            loss = loss.mean()
            loss_dict[f'mapper train loss {i}'] = float(loss)
            log_probs += loss

            w.append(y_hat_latents)

        w = stack(w, dim=1)
        y_hat, _ = self.autoencoder(w, input_code=True, return_latents=True)
        log_probs = -log_probs
        return y_hat, log_probs, loss_dict

    def forward(self, x: List[str]) -> Tuple[Tensor, float, Dict[str, float]]:
        if not self.training:
            return self.predict(x)
        else:
            raise NotImplementedError('Training the clip txt2img is not implemented as part of this model')

