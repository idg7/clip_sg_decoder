import torch


class CLIPSphericalEncoder(torch.nn.Module):
    def __init__(self, clip: torch.nn.Module, out_dim: int = 256):
        super(CLIPSphericalEncoder, self).__init__()
        self.clip = clip

    def forward(self, x):
        self.clip.encode_image(self.face_pool(x))
