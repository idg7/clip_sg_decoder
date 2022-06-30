from torch import optim
from argparse import ArgumentParser

from torchvision import transforms

from models.realNVP import RealNVP, TensorRealNVP
from models.semantic_pyramid import Autoencoder, VGGFeature, Generator
from train import GradientManager, Text2PlaceCoach

from dataset import setup_img_txt_cls_dataset
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
    parser.add_argument('--experiment_name', type=str, default='clip txt2semantic pyramid encoder',
                        help='The specific name of the experiment')

    parser.add_argument('--num_batches_per_epoch', default=250, type=int, help='num batches per epoch')

    parser.add_argument('--test_batch_size', default=4, type=int, help='Batch size for testing and inference')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of train dataloader workers')

    # parser.add_argument('--w_norm_lambda', default=0.1, type=float, help='W-norm loss multiplier factor') # 0.01
    # self.parser.add_argument('--lpips_lambda_crop', default=0, type=float, help='LPIPS loss multiplier factor for inner image region')
    # self.parser.add_argument('--l2_lambda_crop', default=0, type=float, help='L2 loss multiplier factor for inner image region')

    parser.add_argument('--product_image_size', default=224, type=int, help='(size x size) of produced images')
    parser.add_argument("--train_encoder", action="store_true", help="Should train CLIP?")
    parser.add_argument("--train_full_generator", action="store_true", help="Should train CLIP?")

    parser.add_argument('--stylegan_weights', default='./models/stylegan2/stylegan2-ffhq-config-f.pt', type=str,
                        help='Path to StyleGAN model weights')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')

    parser.add_argument('--image_interval', default=50, type=int,
                        help='Interval for logging train images during training')
    parser.add_argument('--board_interval', default=10, type=int, help='Interval for logging metrics to tensorboard')
    parser.add_argument('--val_interval', default=10, type=int, help='Validation interval')
    parser.add_argument('--save_interval', default=25, type=int, help='Model checkpoint interval (epochs)')

    parser.add_argument('--test_dataset_path', default='/home/ssd_storage/datasets/Places365/places365_standard/val', type=str,
                        help='path to the validation dir')
    parser.add_argument('--test_dataset_labels_path',
                        default='/home/ssd_storage/experiments/clip_decoder/places365_cls_names.csv', type=str,
                        help='path to the test cls2label file')
    parser.add_argument('--train_dataset_path',
                        default="/home/ssd_storage/datasets/Places365/places365_standard/train",
                        type=str, help='path to the train dir')
    parser.add_argument('--train_dataset_labels_path',
                        default="/home/ssd_storage/experiments/clip_decoder/places365_cls_names.csv", type=str,
                        help='path to the train cls2label file')

    parser.add_argument("--lr_reduce_step", type=int, default=25000, help="after how many steps to reduce lr")
    parser.add_argument("--w_latent_dim", type=int, default=512, help="dim of w latent space")
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")

    parser.add_argument('--n_hidden', type=int, default=5, help='Number of hidden layers in each MADE.')
    parser.add_argument('--n_blocks', type=int, default=5,
                        help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help='hidden dim in s,t for conditional normalizing flow')
    parser.add_argument('--batch_size', default=12, type=int, help='Batch size for training')
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float, help='LR')
    # parser.add_argument('--mapping_depth', default=4, type=int, help='num of layers in mapping function')
    parser.add_argument('--embedding_norm_path',
                        default='/home/ssd_storage/experiments/clip_decoder/celebA_subset_distribution2.pt', type=str,
                        help='Path to embeddings norm')
    parser.add_argument('--semantic_architecture', default="ViT-B/32")  # ViT-B/32 \ RN101 \ RN50x16
    parser.add_argument('--max_steps', default=200, type=int, help='Maximum number of training steps')
    parser.add_argument('--start_epoch', default=0, type=int, help='epoch to start form')
    parser.add_argument('--mapping_ckpt', default=None, type=str,
                        help='where to load the model weights from')  # '/home/hdd_storage/mlflow/artifact_store/clip_decoder/05d9c461360146b58f94b47518935edf/artifacts/flow_model_mapping99.pth'
    parser.add_argument('--W_plus', action='store_false', help='Should work in W+')
    parser.add_argument('--test_with_random_z', action='store_true',
                        help='When predicting clip picture - should ues random Z')
    parser.add_argument('--test_on_dataset', action='store_false',
                        help='When predicting clip picture - attempt to recreate from test dataset')
    parser.add_argument('--clip_on_orig', action='store_false', help='epoch to start form')

    parser.add_argument('--n_class', default=365, type=int)
    parser.add_argument('--dim_z', default=128, type=int)
    parser.add_argument("--dim_class", type=int, default=128)
    parser.add_argument('--autoencoder_model', default='e4e', type=str, help='e4e / psp')
    parser.add_argument('--ckpt', default='/home/administrator/PycharmProjects/semantic-pyramid-pytorch-orig/checkpoint/060000.pt', type=str, help='semantic pyramid checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    opts = get_args()
    if mlflow.get_experiment_by_name(opts.experiment_name) is None:
        mlflow.create_experiment(opts.experiment_name,
                                 artifact_location=os.path.join(consts.MLFLOW_ARTIFACT_STORE, opts.experiment_name))
    mlflow.set_experiment(opts.experiment_name)

    with mlflow.start_run():
        # mlflow.log_param('mapping_depth', opts.mapping_depth)
        mlflow.log_param('train_dataset_path', opts.train_dataset_path)
        mlflow.log_param('test_dataset_path', opts.test_dataset_path)
        mlflow.log_param('lr', opts.lr)
        mlflow.log_param('semantic_architecture', opts.semantic_architecture)
        mlflow.log_param('n_hidden', opts.n_hidden)
        mlflow.log_param('n_blocks', opts.n_blocks)
        mlflow.log_param('hidden_dim', opts.hidden_dim)
        mlflow.log_param('batch_size', opts.batch_size)
        mlflow.log_param('no_batch_norm', opts.no_batch_norm)
        mlflow.log_param('semantic_architecture', opts.semantic_architecture)
        mlflow.log_param('semantic_architecture', opts.semantic_architecture)
        mlflow.log_param('embedding_norm_path', opts.embedding_norm_path)
        mlflow.log_param('W_plus', opts.W_plus)
        mlflow.log_param('test_with_random_z', opts.test_with_random_z)
        mlflow.log_param('test_on_dataset', opts.test_on_dataset)
        mlflow.log_param('autoencoder_model', opts.autoencoder_model)

        latent = consts.CLIP_EMBEDDING_DIMS[opts.semantic_architecture]
        mixing = 0.9
        batch = opts.batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"

        clip_model, preprocess = clip.load(opts.semantic_architecture, device=device)
        preprocess.transforms[-1] = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        clip_model.cuda()

        w_latent_dim = opts.w_latent_dim
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        train_ds = setup_img_txt_cls_dataset(opts.train_dataset_path, opts.train_dataset_labels_path, preprocess, opts)
        test_ds = setup_img_txt_cls_dataset(opts.test_dataset_path, opts.test_dataset_labels_path, preprocess, opts)
        model_store = LocalModelStore('semantic_pyramid', opts.experiment_name, opts.exp_dir)

        vgg = VGGFeature("vgg16", [4, 9, 16, 23, 30], use_fc=True).eval().to(device)
        gen = Generator(opts.n_class, opts.dim_z, opts.dim_class).to(device)

        if opts.ckpt is not None:
            print(f'loading generator from {opts.ckpt}')
            ckpt = torch.load(opts.ckpt, map_location=lambda storage, loc: storage)
            gen.load_state_dict(ckpt["g"])

        autoencoder = Autoencoder(vgg, gen, opts.dim_z)
        autoencoder.cuda()
        mappers = {}
        optims = {}

        # mappers[3] = TensorRealNVP([512, 14, 14], 3, 100352, opts.hidden_dim, opts.n_hidden, latent,
        #                            batch_norm=not opts.no_batch_norm)
        # mappers[4] = TensorRealNVP([512, 7, 7], opts.n_blocks, 25088, opts.hidden_dim, opts.n_hidden, latent,
        #                                batch_norm=not opts.no_batch_norm)
        mappers[5] = RealNVP(opts.n_blocks, 4096, opts.hidden_dim, opts.n_hidden, latent,
                              batch_norm=not opts.no_batch_norm)
        mappers[6] = RealNVP(opts.n_blocks, 1000, opts.hidden_dim, opts.n_hidden, latent,
                                       batch_norm=not opts.no_batch_norm)
        if opts.start_epoch > 0:
            for key in mappers:
                model_store.load_model_and_optimizer(mappers[key], epoch=opts.start_epoch - 1,
                                                 label=f'{key}_flow_model_mapping')

        for key in mappers:
            mappers[key].cuda()
            optims[key] = optim.Adam(
                mappers[key].parameters(),
                lr=opts.lr,
                betas=(0.9, 0.999),
            )

        image_logger = ImageLogger(os.path.join(opts.exp_dir, opts.experiment_name, 'image_sample'))

        trainer = Text2PlaceCoach(opts,
                                  clip_model,
                                  mappers,
                                  autoencoder,
                                  optims,
                                  {i: torch.optim.lr_scheduler.StepLR(optims[i], opts.lr_reduce_step) for i in optims},
                                  train_ds,
                                  test_ds,
                                  image_logger,
                                  model_store,
                                  GradientManager(opts),
                                  opts.product_image_size,
                                  opts.save_interval, opts.board_interval, opts.image_interval, 1000,
                                  opts.val_interval)

    trainer.train(opts.start_epoch, opts.max_steps)
