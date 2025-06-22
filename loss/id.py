import torch
from torchvision import models, transforms


class VggIDLoss(torch.nn.Module):
    def __init__(self, model_store, weights_path: str, num_classes: int):
        super(VggIDLoss, self).__init__()
        self.model = models.vgg16(num_classes=num_classes)
        self.model.features = torch.nn.DataParallel(self.model.features)
        self.model.cuda()
        self.cossim = torch.nn.CosineSimilarity()
        self.transform = transforms.Compose([
            transforms.Resize([224, 224])

        ])

    def preprocess(self, x):
        return self.transform(x)

    def __dist(self, x, y):
        return 1 - self.cossim(x,y)

    def forward(self, x, y):
        x_emb = self.model(x)
        y_emb = self.model(y)

        return self.__dist(x,y)