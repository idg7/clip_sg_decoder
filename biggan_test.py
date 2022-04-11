import torch
from PIL import Image
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, one_hot_from_int)

if __name__ == '__main__':
    with torch.no_grad():
        model = BigGAN.from_pretrained('biggan-deep-256')
        truncation = 1.0
        idx = [120] * 40
        class_vector = one_hot_from_int(idx, batch_size=len(idx))
        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=len(idx))
        noise_vector = torch.from_numpy(noise_vector)
        class_vector = torch.from_numpy(class_vector)

        class_vector = torch.rand((len(idx), 1000))
        noise_vector = noise_vector.cuda()
        class_vector = class_vector.cuda()
        softmax = torch.nn.Softmax(dim=1).cuda()
        class_vector = softmax(class_vector)
        model = model.cuda()
        output = model(noise_vector, class_vector, truncation)

        # If you have a sixtel compatible terminal you can display the images in the terminal
        # (see https://github.com/saitoha/libsixel for details)

        # Save results as png images
        save_as_images(output.cpu())
