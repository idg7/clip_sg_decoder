import clip
import torch

from typing import List
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader


def im2clip2im(encoder: nn.Module, y: Tensor, mappers: List[nn.Module], autoencoder: nn.Module):
    resize = transforms.Resize((256,256))
    y_embeddings = encoder.encode_image(y)
    y_embeddings = y_embeddings.float()
    w = []
    for i in range(len(mappers)):

        u = mappers[i].base_dist.sample([int(y_embeddings.shape[0])]).cuda()
        # u = torch.zeros((y_embeddings.shape[0], self.args.w_latent_dim)).cuda()
        y_hat_latents, _ = mappers[i].inverse(u, y_embeddings)
        w.append(y_hat_latents)
    w = torch.stack(w, dim=1)
    y_hat, _ = autoencoder(w, input_code=True, return_latents=True)
    return resize(y_hat)


def text_invert_compare(encoder: nn.Module, texts: List[str], psp: nn.Module, mappers: List[nn.Module], gradient_manager, dataset: DataLoader, batch_size: int = 1):
    cossim = nn.CosineSimilarity()
    resize = transforms.Resize((224,224))
    with torch.no_grad():
        gradient_manager.requires_grad(encoder, False)
        encoder.eval()
        gradient_manager.requires_grad(psp, False)
        psp.eval()
        for mapper in mappers:
            gradient_manager.requires_grad(mapper, False)
            mapper.eval()
        tokenized = clip.tokenize(texts)
        tokenized = tokenized.cuda()
        text_features = encoder.encode_text(tokenized)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=1, p=2)[:, None]
        for img, _ in dataset:
            img = img.cuda()
            real_feats = encoder.encode_image(resize(img))
            real_feats = real_feats.float()
            real_feats = real_feats / real_feats.norm(dim=1, p=2)[:, None]
            real_cossim = 1 - (real_feats @ text_features.T)
            print(f'real = {real_cossim}')
            invert = im2clip2im(encoder, img, mappers, psp)
            invert_features = encoder.encode_image(resize(invert))
            invert_features = invert_features.float()
            invert_features = invert_features / invert_features.norm(dim=1, p=2)[:, None]
            invert_cossim = 1 - (invert_features @ text_features.T)
            print(f'invert = {invert_cossim}')