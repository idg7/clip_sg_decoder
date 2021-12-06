import clip
import torch
from torch.utils import data
from torchvision import datasets, transforms
from consts import CLIP_EMBEDDING_DIMS
from tqdm import tqdm

if __name__ == '__main__':
    with torch.set_grad_enabled(False):
        clip_arch = 'ViT-B/32'

        model, preprocess = clip.load(clip_arch)

        dataset = datasets.ImageFolder('/home/ssd_storage/datasets/celebA_crops', transform=preprocess)

        loader = data.DataLoader(dataset,
                                 batch_size=1,
                                 num_workers=4,
                                 shuffle=False)
        dataset_size = len(dataset)
        mean = torch.zeros(CLIP_EMBEDDING_DIMS[clip_arch]).cuda()
        for images, _ in tqdm(loader):
            images = images.cuda(non_blocking=True)
            embeddings = model.encode_image(images)
            sum_embeddings = embeddings.sum(dim=0)
            mean += sum_embeddings / dataset_size

        print(mean)

        cov = torch.zeros((CLIP_EMBEDDING_DIMS[clip_arch], CLIP_EMBEDDING_DIMS[clip_arch])).cuda()
        for images, _ in tqdm(loader):
            images = images.cuda(non_blocking=True)
            embeddings = model.encode_image(images)
            centered = embeddings[0] - mean
            centered = centered[:, None]
            cov += (centered @ centered.T) / (dataset_size - 1)

        print(cov)
        torch.save({'mean': mean.cpu(), 'cov': cov.cpu()}, '/home/ssd_storage/experiments/clip_decoder/celebA_subset_distribution2.pt')
