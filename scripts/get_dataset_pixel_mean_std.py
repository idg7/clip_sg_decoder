import torch
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm

if __name__ == '__main__':
    im_size = 256
    dataset = datasets.ImageFolder("/home/ssd_storage/datasets/processed/phase_perc_size/individual_birds_single_species_{'train': 0.8, 'val': 0.2}",
                                   transform=transforms.Compose([transforms.Resize([im_size, im_size]),
                                                                 transforms.ToTensor()]))

    loader = data.DataLoader(dataset,
                             batch_size=10,
                             num_workers=4,
                             shuffle=False, drop_last=False)

    mean = 0.0
    for images, _ in tqdm(loader):
        images = images.cuda(non_blocking=True)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in tqdm(loader):
        images = images.cuda(non_blocking=True)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*im_size*im_size))

    print(mean)
    print(std)
