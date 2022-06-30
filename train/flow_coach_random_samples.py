from typing import Dict, Optional
import random
import torchvision.transforms
from torch import Tensor

from torch.utils.data import DataLoader
from tqdm import tqdm
from image_saver import ImageLogger
from local_model_store import LocalModelStore
from train import GradientManager

import torch
import mlflow
import torch


def tile_w(t: Tensor, dim=18):
    batch_size = t.size(0)
    latent_dim = t.size(1)
    return torch.cat([
        t[i].repeat(18).view(-1, dim, latent_dim) for i in batch_size
    ])


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    # if prob > 0 and random.random() < prob:
    #     return make_noise(batch, latent_dim, 2, device)
    #
    # else:
    return [make_noise(batch, latent_dim, 1, device)]


class RandomFlowCoach(object):
    def __init__(self,
                 opts,
                 encoder: torch.nn.Module,
                 mapping: torch.nn.Module,
                 decoder: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_reduce: torch.optim.lr_scheduler.StepLR,
                 test_dataset: DataLoader,
                 image_logger: ImageLogger,
                 model_store: LocalModelStore,
                 grad_manager: GradientManager,
                 image_size: int,
                 model_save_freq: int,
                 embedding_norm_path: str = None,
                 metric_log_freq: int = 100,
                 image_log_freq: int = 100,
                 max_batch_per_epoch: int = 0,
                 test_frequency: int = 0):
        self.args = opts
        self.encoder = encoder
        self.mapping = mapping
        self.decoder = decoder
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.lr_reduce = lr_reduce
        self.metric_log_freq = metric_log_freq
        self.image_log_freq = image_log_freq
        self.max_batch_per_epoch = max_batch_per_epoch
        self.image_logger = image_logger
        self.global_step = 0
        self.test_frequency = test_frequency
        self.model_save_freq = model_save_freq
        self.model_store = model_store
        self.grad_manager = grad_manager
        self.face_pool = torchvision.transforms.Resize((256, 256))
        self.resize = torchvision.transforms.Resize((image_size, image_size))
        self.mean = torch.nn.parameter.Parameter(torch.zeros(512), False).cuda()
        self.std = torch.nn.parameter.Parameter(torch.ones(512), False).cuda()
        if embedding_norm_path != None:
            embedding_distribution = torch.load(embedding_norm_path)
            self.mean = torch.nn.parameter.Parameter(embedding_distribution['mean'], False).cuda()
            self.std = torch.nn.parameter.Parameter(torch.sqrt(embedding_distribution['cov'].cuda()).diag(), False)

        self.embedding_norm = lambda t: ((t - self.mean) / self.std)

    def train(self, start_epoch, num_epochs):
        test_batches = self.__get_batches(self.test_dataset)
        for epoch in range(start_epoch, num_epochs):
            print('Train:')
            epoch_loss = self.epoch(self.args.num_batches_per_epoch, True)
            if (epoch + 1) % self.model_save_freq == 0:
                self.model_store.save_model(self.mapping, self.optimizer, epoch, epoch_loss, label='flow_model_mapping',
                                            is_best=False)
            if (epoch + 1) % self.test_frequency == 0:
                print('Test:')
                epoch_loss = self.epoch(test_batches, train=False)
        print('Test:')
        epoch_loss = self.epoch(test_batches, train=False)

    def __get_batches(self, dataset: DataLoader) -> int:
        num_batches = len(dataset)
        if 0 < self.max_batch_per_epoch < num_batches:
            num_batches = self.max_batch_per_epoch
        mlflow.log_param('num_batches_per_epoch', num_batches)
        return num_batches

    def __log_images(self, y: Optional[Tensor] = None, y_hat: Optional[Tensor] = None):
        with torch.no_grad():
            if (y == None) and (y_hat == None):
                data_loader_iter = iter(self.test_dataset)
                (images, labels) = next(data_loader_iter)
                y = images.cuda(non_blocking=True)
                y_embeddings = self.encoder.encode_image(self.resize(y))
                y_embeddings = y_embeddings.float()
                y_embeddings = self.embedding_norm(y_embeddings)
                if self.args.test_with_random_z:
                    u = self.mapping.base_dist.sample([int(y_embeddings.shape[0])]).cuda()
                else:
                    u = torch.zeros((y_embeddings.shape[0], self.args.w_latent_dim)).cuda()
                y_hat_latents, _ = self.mapping.inverse(u, y_embeddings)
                y_hat, _ = self.decoder([y_hat_latents], input_is_latent=True)
            title = 'test'
            self.image_logger.parse_and_log_images(y, y_hat, title=title, step=self.global_step)

    def mapping_test_iter(self, y: Optional[Tensor]):
        self.grad_manager.requires_grad(self.decoder, False)
        self.decoder.train(False)
        self.grad_manager.requires_grad(self.encoder, False)
        self.encoder.train(False)
        self.grad_manager.requires_grad(self.mapping, False)
        self.mapping.train(False)

        y_embeddings = self.encoder.encode_image(self.resize(y))
        y_embeddings = y_embeddings.float()
        y_embeddings = self.embedding_norm(y_embeddings)
        if self.args.test_with_random_z:
            u = self.mapping.base_dist.sample([int(y_embeddings.shape[0])]).cuda()
        else:
            u = torch.zeros((y_embeddings.shape[0], self.args.w_latent_dim)).cuda()
        y_hat_latents, _ = self.mapping.inverse(u, y_embeddings)
        log_probs = self.mapping.log_prob(y_hat_latents, y_embeddings)
        y_hat, y_hat_latents = self.decoder([y_hat_latents], return_latents=True, input_is_latent=True)

        loss = - log_probs

        if self.global_step % self.image_log_freq == 0:
            self.__log_images(y, y_hat)

        return loss.mean()

    def mapping_random_test_iter(self):
        self.grad_manager.requires_grad(self.decoder, False)
        self.decoder.train(False)
        self.grad_manager.requires_grad(self.encoder, False)
        self.encoder.train(False)
        self.grad_manager.requires_grad(self.mapping, False)
        self.mapping.train(False)

        z = mixing_noise(self.args.batch_size, self.args.w_latent_dim, self.args.mixing, 'cuda')  # sampling z
        y, y_latents = self.decoder(z, return_latents=True)  # getting latents
        y_hat_embeddings = self.encoder.encode_image(self.resize(y))  # getting clip embedding
        y_hat_embeddings = y_hat_embeddings.float()
        y_hat_embeddings = self.embedding_norm(y_hat_embeddings)

        if self.args.test_with_random_z:
            u = self.mapping.base_dist.sample(
                [int(y_hat_embeddings.shape[0])]).cuda()  # sampling from mapping distribution
        else:
            u = torch.zeros((y_hat_embeddings.shape[0], self.args.w_latent_dim)).cuda()
        y_hat_latents, _ = self.mapping.inverse(u, y_hat_embeddings)  # mapping (z | clip embeddings) -> w

        loss = - self.mapping.log_prob(y_hat_latents, y_hat_embeddings)  #
        loss = loss.mean()

        y_hat, y_hat_latents = self.decoder([y_hat_latents], return_latents=True, input_is_latent=True)

        if self.global_step % self.image_log_freq == 0:
            self.__log_images(y, y_hat)

        return loss.mean()

    def __check_if_W_plus(self, y_hat_latents: Tensor):
        for i in range(y_hat_latents.size(0)):
            for j in range(18):
                same_vec = (y_hat_latents[i, 0, :] == y_hat_latents[i, j, :]).type(torch.float16).mean()
                if float(same_vec) < 1:
                    print("Got W+ instead of W!!!!!")
                    print(same_vec)
                    print(y_hat_latents[i, 0])
                    print(j)
                    print(y_hat_latents[i, j])
                    return

    def mapping_train_iter(self):
        self.grad_manager.requires_grad(self.decoder, False)
        self.decoder.train(False)
        self.grad_manager.requires_grad(self.encoder, False)
        self.encoder.train(False)
        self.grad_manager.requires_grad(self.mapping, True)
        self.mapping.train(True)
        z = mixing_noise(self.args.batch_size, self.args.w_latent_dim, self.args.mixing, 'cuda')
        y_hat, y_hat_latents = self.decoder(z, return_latents=True)
        y_hat_embeddings = self.encoder.encode_image(self.resize(y_hat))
        y_hat_embeddings = y_hat_embeddings.float()
        y_hat_embeddings = self.embedding_norm(y_hat_embeddings)

        loss = - self.mapping.log_prob(y_hat_latents[:, 0, :], y_hat_embeddings)
        loss = loss.mean()

        self.encoder.zero_grad()
        self.mapping.zero_grad()
        self.decoder.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.global_step % self.image_log_freq == 0:
            self.__log_images()

        return loss

    def epoch(self, num_batches: int, train: bool):
        epoch_loss = 0
        loss_report = 0

        pbar = tqdm(range(num_batches))
        if not train:
            data_loader_iter = iter(self.test_dataset)

        with torch.set_grad_enabled(train):
            count = 0
            for i in pbar:
                count += 1
                if not train:
                    if self.args.test_on_dataset:
                        (images, target) = next(data_loader_iter)
                        images = images.cuda(non_blocking=True)
                        loss = self.mapping_test_iter(images)
                    else:
                        loss = self.mapping_random_test_iter()
                else:
                    loss = self.mapping_train_iter()
                self.lr_reduce.step()

                try:
                    float(loss)
                except:
                    print(loss)
                pbar.set_description(f"loss={float(loss)}")
                loss_report += float(loss)

                if (self.global_step % self.metric_log_freq) == 0:
                    mlflow.log_metric('Loss', float(loss_report) / count, self.global_step)
                    loss_report = 0
                    count = 0

                self.global_step += 1
        return epoch_loss
