from typing import Dict, Optional, List
import random
import torchvision.transforms
from torch import Tensor

from torch.utils.data import DataLoader
from tqdm import tqdm
from image_saver import ImageLogger
from local_model_store import LocalModelStore
from train import GradientManager
from models.normalization.angular_z_norm import cartesian_to_spherical

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
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


class WPlusFlowCoach(object):
    def __init__(self,
                 opts,
                 encoder: torch.nn.Module,
                 mappers: List[torch.nn.Module],
                 autoencoder: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_reduce: List[torch.optim.lr_scheduler.StepLR],
                 train_dataset: DataLoader,
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
        self.face_pool = torchvision.transforms.Resize((256, 256))
        self.resize = torchvision.transforms.Resize((image_size, image_size))
        if opts.use_vgg:
            self.mean = torch.nn.parameter.Parameter(torch.zeros(4096), False).cuda()
            self.std = torch.nn.parameter.Parameter(torch.ones(4096), False).cuda()
        else:
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
        # mlflow.log_param('num_batches_per_epoch', num_batches)
        return num_batches

    def im_2clip_2im(self, y: Tensor):
        y_embeddings = self.encoder.encode_image(self.resize(y))
        y_embeddings = y_embeddings.float()
        y_embeddings = self.embedding_norm(y_embeddings)
        if self.args.spherical_coordinates:
            y_embeddings = cartesian_to_spherical(y_embeddings)
        w = []
        for i in range(len(self.mappers)):
            if self.args.test_with_random_z:
                u = self.mappers[i].base_dist.sample([int(y_embeddings.shape[0])]).cuda()
            else:
                u = torch.zeros((y_embeddings.shape[0], self.args.w_latent_dim)).cuda()
            y_hat_latents, _ = self.mappers[i].inverse(u, y_embeddings)
            w.append(y_hat_latents)
        w = torch.stack(w, dim=1)
        y_hat, _ = self.autoencoder(w, input_code=True, return_latents=True)
        return y_hat

    def __log_images(self, title: str, x: Optional[Tensor] = None, y: Optional[Tensor] = None, x2: Optional[Tensor] = None):
        with torch.no_grad():
            if y is None:
                data_loader_iter = iter(self.test_dataset)
                (images, labels) = next(data_loader_iter)
                x = images.cuda(non_blocking=True)
            # if self.args.clip_on_orig:
            #     y = x
            # else:
                y, y_latents = self.autoencoder(self.face_pool(x), return_latents=True)
            x2 = self.im_2clip_2im(y)
            self.image_logger.parse_and_log_images_with_source(x, y, x2, title=title, step=self.global_step)

    def mapping_test_iter(self, x1: Optional[Tensor], x2: Optional[Tensor]):
        self.grad_manager.requires_grad(self.autoencoder, False)
        self.autoencoder.train(False)
        self.grad_manager.requires_grad(self.encoder, False)
        self.encoder.train(False)
        for i in range(len(self.mappers)):
            self.grad_manager.requires_grad(self.mappers[i], False)
            self.mappers[i].train(False)

        loss_dict = {}

        if self.args.clip_on_orig:
            x_embeddings = self.encoder.encode_image(self.resize(x1))
        else:
            y, y_latents = self.autoencoder(self.face_pool(x2), return_latents=True)
            x_embeddings = self.encoder.encode_image(self.resize(y))
        x_embeddings = x_embeddings.float()
        x_embeddings = self.embedding_norm(x_embeddings)
        if self.args.spherical_coordinates:
            x_embeddings = cartesian_to_spherical(x_embeddings)
        log_probs = 0
        loss = 0
        w = []
        for i in range(len(self.mappers)):
            if self.args.test_with_random_z:
                z = self.mappers[i].base_dist.sample([int(x_embeddings.shape[0])]).cuda()
            else:
                z = torch.zeros((x_embeddings.shape[0], self.args.w_latent_dim)).cuda()

            y_hat_latents, _ = self.mappers[i].inverse(z, x_embeddings)
            log_probs += self.mappers[i].log_prob(y_hat_latents, x_embeddings).mean()
            log_probs = -log_probs
            loss_dict[f'mapper test loss {i}'] = float(log_probs)
            loss += log_probs
            w.append(y_hat_latents)

        w = torch.stack(w, dim=1)
        y_hat, y_hat_latents = self.autoencoder(w, input_code=True, return_latents=True)

        if self.global_step % self.image_log_freq == 0:
            self.__log_images('test', x2, self.autoencoder(x2), x1)#, x, y_hat)

        return loss, loss_dict

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

    def mapping_train_iter(self, x1: Tensor, x2: Tensor):
        self.grad_manager.requires_grad(self.autoencoder, False)
        self.autoencoder.train(False)
        self.grad_manager.requires_grad(self.encoder, False)
        self.encoder.train(False)
        for i in range(len(self.mappers)):
            self.grad_manager.requires_grad(self.mappers[i], False)
            self.mappers[i].train(False)

        y, y_latents = self.autoencoder(self.face_pool(x2), return_latents=True)

        if self.args.clip_on_orig:
            y_embeddings = self.encoder.encode_image(self.resize(x1))
        else:
            y_embeddings = self.encoder.encode_image(self.resize(y))
        y_embeddings = y_embeddings.float()
        y_embeddings = self.embedding_norm(y_embeddings)
        if self.args.spherical_coordinates:
            y_embeddings = cartesian_to_spherical(y_embeddings)

        loss_dict = {}
        total_loss = 0

        for i in range(len(self.mappers)):
            self.grad_manager.requires_grad(self.mappers[i], True)
            self.mappers[i].train(True)
            loss = - self.mappers[i].log_prob(y_latents[:, i, :], y_embeddings)
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
            self.__log_images('train', x2, y, x1)

        return total_loss, loss_dict

    def sum_loss_dicts(self, iter: Dict[str, float], total: Dict[str, float], count: int):
        for key in iter:
            if key not in total:
                total[key] = 0
            total[key] += iter[key]
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
                (a_image, b_image) = next(data_loader_iter)
                a_image = a_image.cuda(non_blocking=True)
                b_image = b_image.cuda(non_blocking=True)
                if not train:
                    if self.args.test_on_dataset:
                        loss, iter_loss_dict = self.mapping_test_iter(a_image, b_image)
                    else:
                        loss, iter_loss_dict = self.mapping_random_test_iter()
                else:
                    loss, iter_loss_dict = self.mapping_train_iter(a_image, b_image)
                for lr_reduce in self.lr_reduce:
                    lr_reduce.step()

                cumulative_loss_dict = self.sum_loss_dicts(iter_loss_dict, cumulative_loss_dict, count)
                pbar.set_description(f"loss={float(loss)}")
                loss_report += float(loss)

                if (self.global_step % self.metric_log_freq) == 0:
                    mlflow.log_metric('Loss', float(loss_report) / count, self.global_step)
                    for key in cumulative_loss_dict:
                        cumulative_loss_dict[key] = cumulative_loss_dict[key] / count
                    mlflow.log_metrics(cumulative_loss_dict, self.global_step)
                    loss_report = 0
                    count = 0
                    cumulative_loss_dict = {}

                self.global_step += 1
        return epoch_loss
