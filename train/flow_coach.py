from typing import Dict

import torchvision.transforms

from torch.utils.data import DataLoader
from tqdm import tqdm
from image_saver import ImageLogger
from local_model_store import LocalModelStore
from train import GradientManager

import torch
import mlflow


class FlowCoach(object):
    def __init__(self,
                 opts,
                 encoder: torch.nn.Module,
                 mapping: torch.nn.Module,
                 autoencoder: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_reduce: torch.optim.lr_scheduler.StepLR,
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
        self.mapping = mapping
        self.autoencoder = autoencoder
        self.train_dataset = train_dataset
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

    def train(self, num_epochs):
        train_batches = self.__get_batches(self.train_dataset)
        test_batches = self.__get_batches(self.test_dataset)
        for epoch in range(num_epochs):
            print('Train:')
            epoch_loss = self.epoch(self.train_dataset, train_batches, True)
            if (epoch+1) % self.model_save_freq == 0:
                self.model_store.save_model(self.mapping, self.optimizer, epoch, epoch_loss, label='flow_model_mapping', is_best=False)
            if (epoch+1) % self.test_frequency == 0:
                print('Test:')
                epoch_loss = self.epoch(self.test_dataset, test_batches, train=False)

    def __get_batches(self, dataset: DataLoader) -> int:
        num_batches = len(dataset)
        if 0 < self.max_batch_per_epoch < num_batches:
            num_batches = self.max_batch_per_epoch
        mlflow.log_param('num_batches_per_epoch', num_batches)
        return num_batches

    def mapping_iter(self, y: torch.Tensor, train: bool) -> Dict[str, float]:
        self.grad_manager.requires_grad(self.mapping, train)
        self.mapping.train(train)

        embeddings = self.encoder.encode_image(y)
        embeddings = embeddings.float()
        y = self.face_pool(y)
        y_hat, latents = self.autoencoder(y, input_code=False, return_latents=True)
        y_hat = self.resize(y_hat)
        y = self.resize(y)
        loss = -self.mapping.log_prob(latents, embeddings)
        if train:
            self.encoder.zero_grad()
            self.mapping.zero_grad()
            self.decoder.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.global_step % self.image_log_freq == 0:
            latents = self.mapping.sample(y.size(0), embeddings)
            self.autoencoder(latents, input_code=True)
            title = 'train'
            if not train:
                title = 'test'
            self.image_logger.parse_and_log_images(y, y_hat, title=title, step=self.global_step)

        return loss

    def epoch(self, dataset: DataLoader, num_batches: int, train: bool):
        epoch_loss = 0
        loss_report = 0

        pbar = tqdm(range(num_batches))
        data_loader_iter = iter(dataset)

        with torch.set_grad_enabled(train):
            for i in pbar:
                (images, target) = next(data_loader_iter)
                images = images.cuda(non_blocking=True)

                loss = self.mapping_iter(images, train)
                self.lr_reduce.step()

                pbar.set_description(f"loss={float(loss)}")
                loss_report += float(loss)

                if (self.global_step % self.metric_log_freq) == 0:
                    mlflow.log_metric('Loss', loss / self.metric_log_freq, self.global_step)

                self.global_step += 1
        return epoch_loss
