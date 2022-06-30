from typing import Dict, List
from models.realNVP import predict, RealNVP
import torchvision.transforms
from torch import Tensor

from torch.utils.data import DataLoader
from tqdm import tqdm
from image_saver import ImageLogger
from local_model_store import LocalModelStore
from train import GradientManager

import mlflow
import torch
import clip


class Txt2ImgFlowCoach(object):
    def __init__(self,
                 opts,
                 txt_encoder: torch.nn.Module,
                 img_encoder: torch.nn.Module,
                 txt2img_mapper: RealNVP,
                 decoder: torch.nn.Module,
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
        self.txt_encoder = txt_encoder
        self.img_encoder = img_encoder
        self.txt2img_mapper = txt2img_mapper
        self.decoder = decoder
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
        # self.face_pool = torchvision.transforms.Resize((256, 256))
        self.resize = torchvision.transforms.Resize((image_size, image_size))

        self.txt_encoder.train(False)
        self.grad_manager.requires_grad(self.txt_encoder, False)
        self.img_encoder.train(False)
        self.grad_manager.requires_grad(self.img_encoder, False)
        self.decoder.train(False)
        self.grad_manager.requires_grad(self.decoder, False)

    def train(self, start_epoch, num_epochs):
        test_batches = self.__get_batches(self.test_dataset)
        for epoch in range(start_epoch, num_epochs):
            print(f'{epoch} Train:')
            epoch_loss = self.epoch(self.args.num_batches_per_epoch, True)
            if (epoch + 1) % self.model_save_freq == 0:
                self.model_store.save_model(self.txt2img_mapper, self.optimizer, epoch, epoch_loss, label=f'txt2img_flow_model_mapping',
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

    def __log_images(self,
                     title: str,
                     labels: List[str],
                     x: Tensor):
        with torch.no_grad():
            txt_tokens = clip.tokenize(labels).cuda(non_blocking=True)
            txt_embeddings = self.txt_encoder.encode_text(txt_tokens)
            img_hat_embedding = predict(self.txt2img_mapper, txt_embeddings)

            x = x.cuda(non_blocking=True)
            img_embedding = self.img_encoder.encode_image(self.resize(x))

            y, _, _ = self.decoder(img_embedding)
            y_hat, _, _ = self.decoder(img_hat_embedding)
            self.image_logger.parse_and_log_images_with_source(x, y, y_hat, title=title, step=self.global_step, names=labels)

    def mapping_iter(self, imgs: Tensor, txt: List[str], train: bool):
        self.grad_manager.requires_grad(self.txt2img_mapper, train)
        self.txt2img_mapper.train(train)

        label = 'text'
        if train:
            label = 'train'

        txt_tokens = clip.tokenize(txt)
        txt_tokens = txt_tokens.cuda(non_blocking=True)
        imgs = imgs.cuda(non_blocking=True)

        img_embeddings = self.img_encoder.encode_image(self.resize(imgs))
        txt_embeddings = self.txt_encoder.encode_text(txt_tokens)

        img_embeddings = img_embeddings.float()
        txt_embeddings = txt_embeddings.float()

        loss_dict = {}
        total_loss = 0

        loss = -self.txt2img_mapper.log_prob(img_embeddings, txt_embeddings)
        loss = loss.mean()
        loss_dict[f'{label} txt2image mapper train loss'] = float(loss)
        total_loss += float(loss)

        img_hat_embeddings = predict(self.txt2img_mapper, txt_embeddings)

        normalized_img_embeddings = img_embeddings.detach()
        normalized_img_hat_embeddings = img_hat_embeddings.detach()
        normalized_txt_embeddings = txt_embeddings.detach()
        normalized_img_embeddings = normalized_img_embeddings / normalized_img_embeddings.norm(dim=1, p=2)[:, None]
        normalized_img_hat_embeddings = normalized_img_hat_embeddings / normalized_img_hat_embeddings.norm(dim=1, p=2)[:, None]
        normalized_txt_embeddings = normalized_txt_embeddings / normalized_txt_embeddings.norm(dim=1, p=2)[:, None]

        # orig_cossim = torch.mm(normalized_img_embeddings, normalized_txt_embeddings.transpose(0, 1))
        # prediction_cossim = torch.mm(normalized_img_embeddings, normalized_img_hat_embeddings.transpose(0, 1))
        # loss_dict[f'{label} mean txt-img cossim'] = float(torch.mean(torch.diagonal(orig_cossim)))
        # loss_dict[f'{label} mean predicted cossim'] = float(torch.mean(torch.diagonal(prediction_cossim)))

        if train:
            self.txt2img_mapper.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.global_step % self.image_log_freq == 0:
            self.__log_images('label', txt, imgs)

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
                (images, text) = next(data_loader_iter)
                images = images.cuda(non_blocking=True)
                if not train:
                    with torch.no_grad():
                        loss, iter_loss_dict = self.mapping_iter(images, text, train)
                else:
                    loss, iter_loss_dict = self.mapping_iter(images, text, train)
                self.lr_reduce.step()

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
