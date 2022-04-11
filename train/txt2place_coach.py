from typing import Dict, Optional, List
import random
import torchvision.transforms
from torchvision import transforms, utils
from torch import Tensor, nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from image_saver import ImageLogger
from local_model_store import LocalModelStore
from train import GradientManager
from models.realNVP import predict, RealNVP
from models.semantic_pyramid import placeholder_latents, chosen_mask

import mlflow
import torch
import clip



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
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


class Text2PlaceCoach(object):
    def __init__(self,
                 opts,
                 encoder: torch.nn.Module,
                 mappers: Dict[int, RealNVP],
                 autoencoder: torch.nn.Module,
                 optimizers: Dict[int, torch.optim.Optimizer],
                 lr_reduce: Dict[int, torch.optim.lr_scheduler.StepLR],
                 train_dataset: DataLoader,
                 test_dataset: DataLoader,
                 image_logger: ImageLogger,
                 model_store: LocalModelStore,
                 grad_manager: GradientManager,
                 image_size: int,
                 model_save_freq: int,
                 metric_log_freq: int = 100,
                 image_log_freq: int = 100,
                 max_batch_per_epoch: int = 0,
                 test_frequency: int = 0):
        self.args = opts
        self.encoder = encoder
        self.mappers = mappers
        self.autoencoder = autoencoder
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizers = optimizers
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

    def train(self, start_epoch, num_epochs):
        test_batches = self.__get_batches(self.test_dataset)
        for epoch in range(start_epoch, num_epochs):
            print(f'{epoch} Train:')
            epoch_loss = self.epoch(self.args.num_batches_per_epoch, True)
            if (epoch + 1) % self.model_save_freq == 0:
                for i in range(len(self.mappers)):
                    self.model_store.save_model(self.mappers[i], self.optimizers[i], epoch, epoch_loss, label=f'{i}_flow_model_mapping',
                                                is_best=False)
            if (epoch + 1) % self.test_frequency == 0:
                print(f'{epoch} Test:')
                epoch_loss = self.epoch(self.args.num_batches_per_epoch, train=False)
        print(f'{epoch} Test:')
        epoch_loss = self.epoch(self.args.num_batches_per_epoch, train=False)

    def __get_batches(self, dataset: DataLoader) -> int:
        num_batches = len(dataset)
        if 0 < self.max_batch_per_epoch < num_batches:
            num_batches = self.max_batch_per_epoch
        mlflow.log_param('num_batches_per_epoch', num_batches)
        return num_batches

    def __log_images(self,
                     title: str,
                     labels: List[str],
                     x: Tensor,
                     cls: Tensor):
        with torch.no_grad():
            x = x.cuda(non_blocking=True)
            txt_tokens = clip.tokenize(labels).cuda(non_blocking=True)
            txt_embedding = self.encoder.encode_text(txt_tokens)
            print(txt_embedding.shape[0])
            latents = placeholder_latents(txt_embedding.shape[0], 'cuda:0')
            for i in self.mappers:
                mapper = self.mappers[i]
                latents[i] = predict(mapper, txt_embedding)
            masks = chosen_mask(self.mappers.keys(), 'cuda:0')
            y = self.autoencoder(x, cls, masks)
            y_hat = self.autoencoder(latents, cls, masks, input_code=True)

            self.image_logger.parse_and_log_images_with_source(x, y, y_hat, title=title, step=self.global_step,
                                                               names=labels)

    def mapping_test_iter(self, x: Tensor, txt: List[str], cls: Tensor):
        self.grad_manager.requires_grad(self.autoencoder, False)
        self.autoencoder.train(False)
        self.grad_manager.requires_grad(self.encoder, False)
        self.encoder.train(False)
        for key in self.mappers:
            mapper = self.mappers[key]
            self.grad_manager.requires_grad(mapper, False)
            mapper.train(False)

        loss_dict = {}
        x = x.cuda(non_blocking=True)
        tokenized = clip.tokenize(txt).cuda(non_blocking=True)
        txt_embeddings = self.encoder.encode_text(tokenized).float()
        masks = chosen_mask(self.mappers.keys())
        y, y_latents = self.autoencoder(x, cls, masks, return_latents=True)
        log_probs = 0
        loss = 0
        for i in self.mappers:
            y_hat_latents = predict(self.mappers[i], txt_embeddings)
            log_probs += self.mappers[i].log_prob(y_hat_latents, txt_embeddings).mean()
            log_probs = -log_probs
            loss_dict[f'mapper test loss {i}'] = float(log_probs)
            loss += log_probs

        if self.global_step % self.image_log_freq == 0:
            self.__log_images('test', txt, x, cls)

        return loss, loss_dict

    def mapping_train_iter(self, x: Tensor, txt: List[str], cls: Tensor):
        self.grad_manager.requires_grad(self.autoencoder, False)
        self.autoencoder.train(False)
        self.grad_manager.requires_grad(self.encoder, False)
        self.encoder.train(False)
        for key in self.mappers:
            mapper = self.mappers[key]
            self.grad_manager.requires_grad(mapper, False)
            mapper.train(False)

        masks = chosen_mask(self.mappers.keys(), 'cuda:0')
        y, y_latents = self.autoencoder(x, cls, masks, return_latents=True)

        tokenized = clip.tokenize(txt).cuda(non_blocking=True)
        txt_embeddings = self.encoder.encode_text(tokenized).float()

        loss_dict = {}
        total_loss = 0

        for i in self.mappers:
            self.grad_manager.requires_grad(self.mappers[i], True)
            self.mappers[i].train(True)
            # print(i)
            loss = -self.mappers[i].log_prob(y_latents[i], txt_embeddings)
            loss = loss.mean()
            loss_dict[f'mapper train loss {i}'] = float(loss)
            total_loss += float(loss)

            self.encoder.zero_grad()
            self.mappers[i].zero_grad()
            self.autoencoder.zero_grad()
            loss.backward()
            self.optimizers[i].step()

            self.grad_manager.requires_grad(self.mappers[i], False)
            self.mappers[i].train(False)

        if self.global_step % self.image_log_freq == 0:
            self.__log_images('train', txt, x, cls)

        return total_loss, loss_dict

    def sum_loss_dicts(self, iter: Dict[str, float], total: Dict[str, float], count: int):
        for key in iter:
            if key not in total:
                total[key] = 0
            total[key] += iter[key] / count
        return total

    def epoch(self, num_batches: int, train: bool):
        epoch_loss = 0
        loss_report = 0
        cumulative_loss_dict = {}

        pbar = tqdm(range(num_batches))
        if not train:
            data_loader_iter = iter(self.test_dataset)
        else:
            data_loader_iter = iter(self.train_dataset)

        with torch.set_grad_enabled(train):
            count = 0
            for i in pbar:
                count += 1
                (images, labels, cls) = next(data_loader_iter)
                images = images.cuda(non_blocking=True)
                cls = cls.cuda(non_blocking=True)
                if not train:
                    loss, iter_loss_dict = self.mapping_test_iter(images, labels, cls)
                else:
                    loss, iter_loss_dict = self.mapping_train_iter(images, labels, cls)
                for key in self.lr_reduce:
                    self.lr_reduce[key].step()

                cumulative_loss_dict = self.sum_loss_dicts(iter_loss_dict, cumulative_loss_dict, count)
                pbar.set_description(f"loss={float(loss)}")
                loss_report += float(loss)

                if (self.global_step % self.metric_log_freq) == 0:
                    mlflow.log_metric('Loss', float(loss_report) / count, self.global_step)
                    mlflow.log_metrics(cumulative_loss_dict, self.global_step)
                    loss_report = 0
                    count = 0
                    cumulative_loss_dict = {}

                self.global_step += 1
        return epoch_loss
