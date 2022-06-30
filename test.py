from models.stylegan2.model import Generator
from torchvision.utils import save_image
from PIL import Image
import torch
import clip
import random


def __load_latent_avg(ckpt, repeat=None):
    if 'latent_avg' in ckpt:
        latent_avg = ckpt['latent_avg']#.cuda()
        if repeat is not None:
            latent_avg = latent_avg.repeat(repeat, 1)
    else:
        latent_avg = None
    return latent_avg


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


if __name__ == '__main__':
    decoder = Generator(1024, 512, 8)
    latent = 512
    mixing = 0.9
    batch = 2
    device = 'cuda'
    decoder.to(device)#.cuda()
    stylegan_weights = './models/stylegan2/stylegan2-ffhq-config-f.pt'
    ckpt = torch.load(stylegan_weights)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open("/home/ssd_storage/datasets/celebA/AKON/002617.jpg")).unsqueeze(0).to(device)
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    image_features = model.encode_image(image)

    # decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
    decoder.load_state_dict(ckpt['g_ema'], strict=False)

    # noise = mixing_noise(batch, latent, mixing, device)
    # images, result_latent = decoder(noise)

    sample_z = torch.randn(1, latent, device=device)
    print(sample_z.size())
    print(image_features.size())
    print(decoder([sample_z]).size())
    print(decoder([image_features]).size())
    # with torch.no_grad():
    #     decoder.eval()
    #     sample, _ = decoder([sample_z])
    #     save_image(
    #         sample,
    #         f"./sample/0.png",
    #         nrow=int(1 ** 0.5),
    #         normalize=True,
    #         range=(-1, 1),
    #     )
