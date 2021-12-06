from typing import Dict

import torchvision.transforms

from loss.gan_loss import d_r1_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from image_saver import ImageLogger
from local_model_store import LocalModelStore
from train import GradientManager
from loss import d_logistic_loss, g_nonsaturating_loss, g_path_regularize
from train.non_leaking import augment

import torch
import mlflow


class Coach(object):
    def __init__(self,
                 opts,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 criterions: Dict[str, torch.nn.Module],
                 loss_factors: Dict[str, float],
                 g_optimizer: torch.optim.Optimizer,
                 d_optimizer: torch.optim.Optimizer,
                 g_lr_reduce: torch.optim.lr_scheduler.StepLR,
                 d_lr_reduce: torch.optim.lr_scheduler.StepLR,
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
                 test_frequency: int = 0,
                 discriminator_lambda: float = 1):
        self.args = opts
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.criterions = criterions
        self.criterion_factors = loss_factors
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_lr_reduce = g_lr_reduce
        self.d_lr_reduce = d_lr_reduce
        self.metric_log_freq = metric_log_freq
        self.image_log_freq = image_log_freq
        self.max_batch_per_epoch = max_batch_per_epoch
        self.image_logger = image_logger
        self.global_step = 0
        self.test_frequency = test_frequency
        self.model_save_freq = model_save_freq
        self.model_store = model_store
        self.grad_manager = grad_manager
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.resize = torchvision.transforms.Resize((image_size, image_size))
        self.discriminator_lambda = discriminator_lambda

    def train(self, num_epochs):
        train_batches = self.__get_batches(self.train_dataset)
        test_batches = self.__get_batches(self.test_dataset)
        for epoch in range(num_epochs):
            print('Train:')
            epoch_loss = self.epoch(self.train_dataset, train_batches, True)
            if (epoch+1) % self.model_save_freq == 0:
                self.model_store.save_model(self.decoder, self.g_optimizer, epoch, epoch_loss, label='generator', is_best=False)
                self.model_store.save_model(self.discriminator, self.d_optimizer, epoch, epoch_loss, label='discriminator', is_best=False)
            if (epoch+1) % self.test_frequency == 0:
                print('Test:')
                epoch_loss = self.epoch(self.test_dataset, test_batches, train=False)

    def __get_batches(self, dataset: DataLoader) -> int:
        num_batches = len(dataset)
        if 0 < self.max_batch_per_epoch < num_batches:
            num_batches = self.max_batch_per_epoch
        mlflow.log_param('num_batches_per_epoch', num_batches)
        return num_batches

    def discriminator_iter(self, y: torch.Tensor, train: bool) -> Dict[str, float]:
        self.grad_manager.requires_grad(self.discriminator, train)
        self.discriminator.train(train)
        self.grad_manager.requires_grad(self.decoder, False)
        self.decoder.train(False)
        self.grad_manager.requires_grad(self.encoder, False)
        self.encoder.train(False)

        embeddings = self.encoder.encode_image(y)

        embeddings = embeddings.float()
        y_hat, latents = self.decoder([embeddings])
        y_hat = self.resize(y_hat)
        y = self.resize(y)

        fake_pred = self.discriminator(y_hat)
        real_pred = self.discriminator(y)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        label = 'Train' if train else 'Test'
        loss_dict = {f"{label} d": float(d_loss), f"{label} real_score": float(real_pred.mean()), f"{label} fake_score": float(fake_pred.mean())}

        if train:
            self.discriminator.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

        return d_loss, loss_dict

    def discriminator_regularize(self, real_img: torch.Tensor):
        self.grad_manager.requires_grad(self.discriminator, True)
        self.discriminator.train(True)
        self.grad_manager.requires_grad(self.decoder, False)
        self.decoder.train(False)
        self.grad_manager.requires_grad(self.encoder, False)
        self.encoder.train(False)
        real_img = self.resize(real_img)
        loss_dict = {}
        real_img.requires_grad = True

        ada_aug_p = self.args.augment_p if self.args.augment_p > 0 else 0.0

        if self.args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)

        else:
            real_img_aug = real_img

        real_pred = self.discriminator(real_img_aug)
        r1_loss = d_r1_loss(real_pred, real_img)

        self.discriminator.zero_grad()
        (self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]).backward()

        self.d_optimizer.step()

        loss_dict["r1"] = float(r1_loss)
        return loss_dict

    def decoder_iter(self, y: torch.Tensor, train: bool) -> Dict[str, float]:
        self.grad_manager.requires_grad(self.discriminator, False)
        self.discriminator.train(False)
        self.grad_manager.requires_grad(self.decoder, train)
        self.decoder.train(train)
        self.grad_manager.requires_grad(self.encoder, train and self.args.train_encoder)
        self.encoder.train(train and self.args.train_encoder)

        label = 'Train' if train else 'Test'
        loss_logs = {f'{label} discriminator loss': 0}
        loss = torch.zeros(1).cuda()
        # print(loss)

        embeddings = self.encoder.encode_image(y)
        embeddings = embeddings.float()
        y_hat, latents = self.decoder([embeddings])
        y_hat = self.resize(y_hat)
        y = self.resize(y)

        if self.discriminator_lambda > 0:
            fake_pred = self.discriminator(y_hat)
            g_loss = g_nonsaturating_loss(fake_pred)
            loss += g_loss * self.discriminator_lambda
            loss_logs[f'{label} discriminator loss'] += float(g_loss)

        for criterion_label in self.criterions:
            if self.criterion_factors[criterion_label] > 0:
                curr_criterion_loss = self.criterions[criterion_label](y, y_hat)
                loss_logs[f'{label} {criterion_label}'] = float(curr_criterion_loss)
                loss += curr_criterion_loss * self.criterion_factors[criterion_label]
                loss_logs[f'{label} sum loss'] = float(loss)

        if train:
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            loss.backward()
            self.g_optimizer.step()

        if self.global_step % self.image_log_freq == 0:
            title = 'train'
            if not train:
                title = 'test'
            self.image_logger.parse_and_log_images(y, y_hat, title=title, step=self.global_step)

        return loss, loss_logs

    def decoder_regularize(self, y: torch.Tensor):
        self.grad_manager.requires_grad(self.discriminator, False)
        self.discriminator.train(False)
        self.grad_manager.requires_grad(self.decoder, True)
        self.decoder.train(True)
        self.grad_manager.requires_grad(self.encoder, True and self.args.train_encoder)
        self.encoder.train(True and self.args.train_encoder)
        loss_dict = {}
        mean_path_length = 0

        # path_batch_size = max(1, self.args.batch // self.args.path_batch_shrink)
        embeddings = self.encoder.encode_image(y)
        embeddings = embeddings.float()
        fake_img, latents = self.decoder([embeddings], return_latents=True)

        path_loss, mean_path_length, path_lengths = g_path_regularize(
            fake_img, latents, mean_path_length
        )
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        weighted_path_loss = self.args.path_regularize * self.args.g_reg_every * path_loss

        # if self.args.path_batch_shrink:
        #     weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

        weighted_path_loss.backward()

        self.g_optimizer.step()
        loss_dict["Train path"] = float(path_loss)
        loss_dict["Train path_length"] = float(path_lengths.mean())

        return loss_dict

    def add_loss_dict(self, loss_dict: Dict[str, float], iter_loss_dict: Dict[str, float], denominator: int):
        for key in iter_loss_dict:
            if key not in loss_dict:
                loss_dict[key] = 0
            loss_dict[key] += iter_loss_dict[key] / denominator
        return loss_dict

    def epoch(self, dataset: DataLoader, num_batches: int, train: bool):
        d_loss_dict = {}
        g_loss_dict = {}

        epoch_loss = 0

        pbar = tqdm(range(num_batches))
        data_loader_iter = iter(dataset)

        with torch.set_grad_enabled(train):
            for i in pbar:
                (images, target) = next(data_loader_iter)
                images = images.cuda(non_blocking=True)

                if self.discriminator_lambda > 0:
                    d_loss, d_loss_iter_dict = self.discriminator_iter(images, train)
                    self.d_lr_reduce.step()
                else:
                    d_loss = torch.zeros(1)
                    d_loss_iter_dict = {}
                g_loss, g_loss_iter_dict = self.decoder_iter(images, train)
                self.g_lr_reduce.step()
                if train and (self.args.g_reg_every != 0) and ((self.global_step % self.args.g_reg_every) == 0):
                    g_reg_dict = self.decoder_regularize(images)
                    g_loss_iter_dict = {**g_loss_iter_dict, **g_reg_dict}
                if train and (self.args.d_reg_every != 0) and ((self.global_step % self.args.d_reg_every) == 0):
                    d_reg_dict = self.discriminator_regularize(images)
                    d_loss_iter_dict = {**d_loss_iter_dict, **d_reg_dict}

                d_loss_dict = self.add_loss_dict(d_loss_dict, d_loss_iter_dict, self.metric_log_freq)
                g_loss_dict = self.add_loss_dict(g_loss_dict, g_loss_iter_dict, self.metric_log_freq)

                pbar.set_description(f"discriminator loss: {str(float(d_loss))}, generator loss: {str(float(g_loss))}")

                if (self.global_step % self.metric_log_freq) == 0:
                    mlflow.log_metrics(d_loss_dict, self.global_step)
                    mlflow.log_metrics(g_loss_dict, self.global_step)
                    d_loss_dict = {}
                    g_loss_dict = {}

                self.global_step += 1
        return epoch_loss
