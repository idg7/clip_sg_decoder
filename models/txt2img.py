from typing import List

import clip
from torch import nn, Tensor

from models import CLIPInversionGenerator
from models.realNVP import RealNVP, predict


class Txt2Img(nn.Module):
    def __init__(self,
                 txt2img: RealNVP,
                 txt_encoder: nn.Module,
                 clip_inversion: CLIPInversionGenerator,
                 test_with_random_z: bool):
        super(Txt2Img, self).__init__()
        self.txt2img = txt2img
        self.txt_encoder = txt_encoder
        self.clip_inversion = clip_inversion
        self.test_with_random_z = test_with_random_z

    def predict2(self, labels: List[str]):
        # print(labels)
        txt_embeddings = self.txt_encoder.encode_text(labels)
        img_hat_embedding = predict(self.txt2img, txt_embeddings)
        # print(img_hat_embedding.shape)
        # normalized_embeddings = img_hat_embedding / img_hat_embedding.norm(p=2, dim=1)[:, None]
        # print(normalized_embeddings @ normalized_embeddings.transpose(0, 1))
        y_hat, _, _ = self.clip_inversion(img_hat_embedding)
        return y_hat

    def predict(self, txt: List[str]):
        log_probs = 0.0
        loss_dict = {}
        txt_embeddings = self.txt_encoder.encode_text(labels)
        img_clip_embedding = predict(self.txt2img, txt_embeddings)

        txt2img_loss = -self.txt2img.log_prob(img_clip_embedding, txt_embeddings).mean()
        txt2img_loss = txt2img_loss.mean()
        loss_dict[f'txt2img loss'] = float(txt2img_loss)

        y_hat, inversion_log_probs, inversion_loss_dict = self.clip_inversion(img_clip_embedding)
        loss_dict = {**loss_dict, **inversion_loss_dict}
        log_probs = log_probs + txt2img_loss
        return y_hat, log_probs, loss_dict

    def forward(self, x: List[str]):
        if not self.training:
            return self.predict2(x)
        else:
            raise NotImplementedError('Training the clip image inversion is not implemented as part of this model')

