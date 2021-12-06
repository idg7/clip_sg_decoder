import torch
from torch.utils import data
from torchvision import datasets, transforms

if __name__ == '__main__':
    dataset = datasets.ImageFolder('/home/ssd_storage/datasets/celebA_crops', transform=transforms.Compose([transforms.Resize(256),
                                 # transforms.CenterCrop(224),
                                 transforms.ToTensor()]))

    loader = data.DataLoader(dataset,
                             batch_size=10,
                             num_workers=0,
                             shuffle=False)

    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*224*224))

    print(mean)
    print(std)
