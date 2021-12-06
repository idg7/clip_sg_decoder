import clip
import torch
from train.gradient import requires_grad
import consts
from torchvision import transforms


class CosineDist(torch.nn.CosineSimilarity):
    def forward(self, x, y):
        return 1 - super(CosineDist, self).forward(x, y)


class SemanticLoss(torch.nn.Module):
    def __init__(self, image_encoder_arch: str, distance: torch.nn.Module = CosineDist()):
        super(SemanticLoss, self).__init__()
        self.clip, preprocess = clip.load(image_encoder_arch)
        self.clip.cuda()
        requires_grad(self.clip, False)
        self.clip.eval()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((consts.CLIP_IMAGE_RESOLUTION[image_encoder_arch], consts.CLIP_IMAGE_RESOLUTION[image_encoder_arch]))
        self.face_pool.cuda()
        self.distance = distance
        self.distance.cuda()

    def forward(self, x, y):
        x_semantics = self.clip.encode_image(self.face_pool(x))
        y_semantics = self.clip.encode_image(self.face_pool(y))
        distances = self.distance(x_semantics, y_semantics)
        batch_size = distances.size()[0]
        return torch.sum(distances) / batch_size
