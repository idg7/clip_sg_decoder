from torch import optim
from argparse import ArgumentParser
from torchvision import transforms

from models import get_psp
from train import FlowCoach
from train import GradientManager

from dataset import setup_dataset
from image_saver import ImageLogger
from local_model_store import LocalModelStore

import os
import torch
import clip
import mlflow

import consts

mlflow.set_tracking_uri(consts.MLFLOW_TRACKING_URI)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--exp_dir', default='/home/ssd_storage/experiments/clip_decoder', type=str,
                        help='Path to experiment output directory')
    parser.add_argument('--experiment_name', type=str, default='clip_decoder',
                        help='The specific name of the experiment')

    parser.add_argument('--batch_size', default=12, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=0.00001, type=float, help='LR')
    parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of train dataloader workers')

    parser.add_argument('--semantic_architecture', default="ViT-B/32")  # ViT-B/32 \ RN101 \ RN50x16
    # parser.add_argument('--w_norm_lambda', default=0.1, type=float, help='W-norm loss multiplier factor') # 0.01
    # self.parser.add_argument('--lpips_lambda_crop', default=0, type=float, help='LPIPS loss multiplier factor for inner image region')
    # self.parser.add_argument('--l2_lambda_crop', default=0, type=float, help='L2 loss multiplier factor for inner image region')

    parser.add_argument('--product_image_size', default=256, type=int, help='(size x size) of produced images')
    parser.add_argument("--train_encoder", action="store_true", help="Should train CLIP?")
    parser.add_argument("--train_full_generator", action="store_true", help="Should train CLIP?")

    parser.add_argument('--stylegan_weights', default='./models/stylegan2/stylegan2-ffhq-config-f.pt', type=str,
                        help='Path to StyleGAN model weights')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--mapping_depth', default=12, type=int, help='num of layers in mapping function')
    parser.add_argument('--max_steps', default=100, type=int, help='Maximum number of training steps')
    parser.add_argument('--image_interval', default=50, type=int,
                        help='Interval for logging train images during training')
    parser.add_argument('--board_interval', default=10, type=int, help='Interval for logging metrics to tensorboard')
    parser.add_argument('--val_interval', default=10, type=int, help='Validation interval')
    parser.add_argument('--save_interval', default=25, type=int, help='Model checkpoint interval (epochs)')
    parser.add_argument('--train_dataset_path', default='/home/ssd_storage/datasets/celebA_crops', type=str,
                        help='path to the train dir')
    parser.add_argument('--test_dataset_path', default='/home/ssd_storage/datasets/celebA_crops', type=str,
                        help='path to the validation dir')
    parser.add_argument("--lr_reduce_step", type=int, default=25000, help="after how many steps to reduce lr")

    return parser.parse_args()


if __name__ == '__main__':
    opts = get_args()
    if mlflow.get_experiment_by_name(opts.experiment_name) is None:
        mlflow.create_experiment(opts.experiment_name,
                                 artifact_location=os.path.join(consts.MLFLOW_ARTIFACT_STORE, opts.experiment_name))
    mlflow.set_experiment(opts.experiment_name)

    with mlflow.start_run():
        mlflow.log_param('mapping_depth', opts.mapping_depth)
        mlflow.log_param('lr', opts.lr)
        mlflow.log_param('semantic_architecture', opts.semantic_architecture)

        latent = consts.CLIP_EMBEDDING_DIMS[opts.semantic_architecture]
        mixing = 0.9
        batch = opts.batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"

        clip_model, preprocess = clip.load(opts.semantic_architecture, device=device)
        preprocess.transforms[-1] = transforms.Normalize([0.5065, 0.4118, 0.3635], [0.3436, 0.3095, 0.3044])
        clip_model.cuda()

        mapping = get_realnvp(512, latent, opts.mapping_depth)
        mapping.cuda()

        autoencoder = get_psp()
        autoencoder.cuda()

        optim = optim.Adam(
            mapping.parameters(),
            lr=opts.lr,
            betas=(0.9, 0.99),
        )

        train_ds = setup_dataset(opts.train_dataset_path, preprocess, opts)
        test_ds = setup_dataset(opts.test_dataset_path, preprocess, opts)
        trainer = FlowCoach(opts,
                            clip_model,
                            mapping,
                            autoencoder,
                            optim,
                            torch.optim.lr_scheduler.StepLR(optim, opts.lr_reduce_step),
                            train_ds,
                            test_ds,
                            ImageLogger(os.path.join(opts.exp_dir, opts.experiment_name, 'image_sample')),
                            LocalModelStore('stylegan2', opts.experiment_name, opts.exp_dir),
                            GradientManager(opts),
                            opts.product_image_size,
                            opts.save_interval, opts.board_interval, opts.image_interval, 1000, opts.val_interval)

        trainer.train(opts.max_steps)
