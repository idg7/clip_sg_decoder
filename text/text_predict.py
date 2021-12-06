import clip
import torch

from typing import List
from torch import nn
from models.normalization.angular_z_norm import cartesian_to_spherical


def text_to_image(encoder: nn.Module, text: str, psp: nn.Module, mappers: List[nn.Module], image_logger, gradient_manager, batch_size: int = 2):
    with torch.no_grad():
        gradient_manager.requires_grad(encoder, False)
        encoder.eval()
        gradient_manager.requires_grad(psp, False)
        psp.eval()
        for mapper in mappers:
            gradient_manager.requires_grad(mapper, False)
            mapper.eval()
        text_batch = [text] * batch_size
        tokenized = clip.tokenize(text_batch)
        tokenized = tokenized.cuda()
        print(tokenized.shape)
        text_features = encoder.encode_text(tokenized)
        text_features = text_features.float()
        text_features = cartesian_to_spherical(text_features)
        print(text_features.shape)
        w = []
        for i in range(len(mappers)):
            u = mappers[i].base_dist.sample([int(text_features.shape[0])]).cuda()
            u = torch.zeros((text_features.shape[0], 512)).cuda()
            y_hat_latents, _ = mappers[i].inverse(u, text_features)
            w.append(y_hat_latents)
        w = torch.stack(w, dim=1)
        y_hat, _ = psp(w, input_code=True, return_latents=True)
        image_logger.parse_and_log_images_with_source(y_hat, y_hat, y_hat, title=text, step=100)

