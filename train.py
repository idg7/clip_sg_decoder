from torch import optim
from argparse import ArgumentParser
from torchvision import transforms

from models.stylegan2 import GeneratorMappingWrapper
from models.stylegan2.model import Generator, Discriminator
from loss import setup_generator_loss
from train.coach import Coach
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
    parser.add_argument('--exp_dir', default='/home/ssd_storage/experiments/clip_decoder', type=str, help='Path to experiment output directory')
    parser.add_argument('--experiment_name', type=str, default='clip_decoder', help='The specific name of the experiment')

    parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=1e-4, type=float, help='LR')
    parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of train dataloader workers')

    parser.add_argument('--lpips_lambda', default=1., type=float, help='LPIPS loss multiplier factor')
    parser.add_argument('--semantic_lambda', default=0.9, type=float, help='ID loss multiplier factor')
    parser.add_argument('--l2_lambda', default=1, type=float, help='L2 loss multiplier factor')
    parser.add_argument('--id_lambda', default=0.9, type=float, help='ID (resnet50) loss multiplier factor')
    parser.add_argument('--discriminator_lambda', default=0.0, type=float, help='L2 loss multiplier factor')
    parser.add_argument('--semantic_architecture', default="ViT-B/32") # ViT-B/32 \ RN101 \ RN50x16
    # parser.add_argument('--w_norm_lambda', default=0.1, type=float, help='W-norm loss multiplier factor') # 0.01
    # self.parser.add_argument('--lpips_lambda_crop', default=0, type=float, help='LPIPS loss multiplier factor for inner image region')
    # self.parser.add_argument('--l2_lambda_crop', default=0, type=float, help='L2 loss multiplier factor for inner image region')

    parser.add_argument('--product_image_size', default=256, type=int, help='(size x size) of produced images')
    parser.add_argument('--stylegan_weights', default='./models/stylegan2/stylegan2-ffhq-config-f.pt', type=str, help='Path to StyleGAN model weights')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model model checkpoint')
    parser.add_argument('--mapping_depth', default=12, type=int, help='num of layers in mapping function')
    parser.add_argument('--max_steps', default=100, type=int, help='Maximum number of training steps')
    parser.add_argument('--image_interval', default=50, type=int, help='Interval for logging train images during training')
    parser.add_argument('--board_interval', default=10, type=int, help='Interval for logging metrics to tensorboard')
    parser.add_argument('--val_interval', default=10, type=int, help='Validation interval')
    parser.add_argument('--save_interval', default=25, type=int, help='Model checkpoint interval (epochs)')
    parser.add_argument('--train_dataset_path', default='/home/ssd_storage/datasets/celebA_crops', type=str, help='path to the train dir')
    parser.add_argument('--test_dataset_path', default='/home/ssd_storage/datasets/celebA_crops', type=str, help='path to the validation dir')
    parser.add_argument("--augment_p", type=float, default=0, help="probability of applying augmentation. 0 = use adaptive augmentation",)
    parser.add_argument("--d_reg_every", type=int, default=0, help="interval of the applying r1 regularization") # 16
    parser.add_argument("--g_reg_every", type=int, default=0, help="interval of the applying path length regularization") # 4
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=0, help="weight of the path length regularization") #2
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--lr_reduce_step", type=int, default=25000, help="after how many steps to reduce lr"
    )
    parser.add_argument(
        "--train_encoder", action="store_false", help="Should train CLIP?"
    )

    return parser.parse_args()


if __name__ == '__main__':
    opts = get_args()
    if mlflow.get_experiment_by_name(opts.experiment_name) is None:
        mlflow.create_experiment(opts.experiment_name, artifact_location=os.path.join(consts.MLFLOW_ARTIFACT_STORE, opts.experiment_name))
    mlflow.set_experiment(opts.experiment_name)

    with mlflow.start_run():
        mlflow.log_param('semantic_lambda', opts.semantic_lambda)
        mlflow.log_param('lpips_lambda', opts.lpips_lambda)
        mlflow.log_param('l2_lambda', opts.l2_lambda)
        mlflow.log_param('id_lambda', opts.id_lambda)
        mlflow.log_param('discriminator_lambda', opts.discriminator_lambda)
        mlflow.log_param('mapping_depth', opts.mapping_depth)
        mlflow.log_param('d_reg_every', opts.d_reg_every)
        mlflow.log_param('g_reg_every', opts.g_reg_every)
        mlflow.log_param('r1', opts.r1)
        mlflow.log_param('path_regularize', opts.path_regularize)


        mlflow.log_param('lr', opts.lr)
        mlflow.log_param('semantic_architecture', opts.semantic_architecture)

        decoder = Generator(1024, 512, 8)
        discriminator = Discriminator(opts.product_image_size, channel_multiplier=2)
        latent = consts.CLIP_EMBEDDING_DIMS[opts.semantic_architecture]
        mixing = 0.9
        batch = opts.batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        discriminator = Discriminator(opts.product_image_size, channel_multiplier=2)
        discriminator.to(device)
        decoder.to(device)
        stylegan_weights = './models/stylegan2/stylegan2-ffhq-config-f.pt'
        ckpt = torch.load(stylegan_weights)

        clip_model, preprocess = clip.load(opts.semantic_architecture, device=device)
        # print(preprocess)
        preprocess.transforms[-1] = transforms.Normalize([0.5065, 0.4118, 0.3635], [0.3436, 0.3095, 0.3044])
        # print(preprocess)
        clip_model.cuda()

        # decoder.load_state_dict(ckpt['g_ema'], strict=False)
        decoder.load_state_dict(ckpt["g"])
        latent_avg = ckpt['latent_avg'].to(device)
        # discriminator.load_state_dict(ckpt['d'], strict=False)

        decoder = GeneratorMappingWrapper(opts.mapping_depth, decoder,
                                          dim=[
                                              consts.CLIP_EMBEDDING_DIMS[opts.semantic_architecture],
                                              consts.CLIP_EMBEDDING_DIMS[opts.semantic_architecture],
                                              consts.CLIP_EMBEDDING_DIMS[opts.semantic_architecture],
                                              consts.CLIP_EMBEDDING_DIMS[opts.semantic_architecture],
                                              512, 512, 512, 512
                                          ])
        decoder.cuda()

        loss, lambdas = setup_generator_loss(opts)
        g_reg_ratio = 1
        d_reg_ratio = 1
        if opts.g_reg_every > 0:
            g_reg_ratio = opts.g_reg_every / (opts.g_reg_every + 1)
        if opts.d_reg_every > 0:
            d_reg_ratio = opts.d_reg_every / (opts.d_reg_every + 1)

        if not opts.train_encoder:
            g_optim = optim.Adam(
                decoder.parameters(),
                lr=opts.lr * g_reg_ratio,
                betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
            )
            d_optim = optim.Adam(
                discriminator.parameters(),
                lr=opts.lr * d_reg_ratio,
                betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
            )
        else:
            g_optim = optim.SGD(
                list(decoder.parameters()) + list(clip_model.parameters()),
                lr=opts.lr * g_reg_ratio,
                nesterov=True,
                momentum=0.9,
            )
            d_optim = optim.SGD(
                discriminator.parameters(),
                lr=opts.lr * d_reg_ratio,
                nesterov=True,
                momentum=0.9,
            )

        train_ds = setup_dataset(opts.train_dataset_path, preprocess, opts)
        test_ds = setup_dataset(opts.test_dataset_path, preprocess, opts)
        trainer = Coach(opts,
              clip_model,
              decoder,
              discriminator,
              loss,
              lambdas,
              g_optim,
              d_optim,
                        torch.optim.lr_scheduler.StepLR(g_optim, opts.lr_reduce_step),
                        torch.optim.lr_scheduler.StepLR(d_optim, opts.lr_reduce_step),
              train_ds,
              test_ds,
              ImageLogger(os.path.join(opts.exp_dir, opts.experiment_name, 'image_sample')),
              LocalModelStore('stylegan2', opts.experiment_name, opts.exp_dir),
              opts.product_image_size,
              opts.save_interval, opts.board_interval, opts.image_interval, 1000, opts.val_interval, opts.discriminator_lambda)

        trainer.train(opts.max_steps)
