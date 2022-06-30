import clip
import torch


class CLIPWrapper(torch.nn.Module):
    def __init__(self, input_size: int, clip: torch.nn.Module):
        super(CLIPWrapper, self).__init__()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((input_size, input_size))
        self.clip = clip

    def forward(self, x):
        self.clip.encode_image(self.face_pool(x))
