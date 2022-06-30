from torch import optim
from argparse import ArgumentParser

from torchvision import transforms

from models import RealNVP, get_psp, clip_txt_encoder, get_txt_encoder
from models.e4e.model_utils import setup_model
from train import GradientManager, Text2WFlowCoach

from dataset import setup_img2txt_dataset
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
    parser.add_argument('--experiment_name', type=str, default='txt2w',
                        help='The specific name of the experiment')
    parser.add_argument('--run_name', type=str, default='gpt2 to W',
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

    parser.add_argument('--test_dataset_path', default='/home/ssd_storage/datasets/celebA_crops', type=str,
                        help='path to the validation dir')
    parser.add_argument('--test_dataset_labels_path',
                        default='/home/ssd_storage/experiments/clip_decoder/celebA_cls_names.csv', type=str,
                        help='path to the test cls2label file')
    # parser.add_argument('--train_dataset_path',
    #                     default="/home/ssd_storage/datasets/processed/clip_familiar_vggface2_{'train': 0.7, 'val': 0.2, 'test': 0.1}/train",
    #                     type=str, help='path to the train dir')
    # parser.add_argument('--train_dataset_labels_path',
    #                     default="/home/ssd_storage/experiments/clip_decoder/identity_meta.csv", type=str,
    #                     help='path to the train cls2label file')

    parser.add_argument('--train_dataset_path',
                        default="/home/ssd_storage/datasets/CelebAMask-HQ/CLIP_familiar/ViT-B32",
                        type=str, help='path to the train dir')
    parser.add_argument('--train_dataset_labels_path',
                        default="/home/ssd_storage/experiments/clip_decoder/celeba_hq_names.csv", type=str,
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
    parser.add_argument('--autoencoder_model', default='e4e', type=str, help='e4e / psp')

    parser.add_argument('--txt_architecture', default=None)
    parser.add_argument('--embedding_reduction', default='last')

    return parser.parse_args()


if __name__ == '__main__':
    opts = get_args()
    if mlflow.get_experiment_by_name(opts.experiment_name) is None:
        artifact_loc = os.path.join(consts.MLFLOW_ARTIFACT_STORE, opts.experiment_name)
        if opts.run_name is not None:
            artifact_loc = os.path.join(artifact_loc, opts.run_name)
        mlflow.create_experiment(opts.experiment_name, artifact_location=artifact_loc)
    mlflow.set_experiment(opts.experiment_name)

    with mlflow.start_run(run_name=opts.run_name):
        mlflow.log_params(vars(opts))

        latent = consts.TXT_EMBEDDING_DIMS[opts.semantic_architecture]
        mixing = 0.9
        batch = opts.batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if opts.txt_architecture is None:
            txt_model, _ = clip.load(opts.semantic_architecture, device=device)
            txt_model = clip_txt_encoder(txt_model)
        else:
            txt_model = get_txt_encoder(opts.txt_architecture)
            latent = consts.TXT_EMBEDDING_DIMS[opts.txt_architecture]
        txt_model.cuda()

        e4e_preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        w_latent_dim = opts.w_latent_dim

        train_ds = setup_img2txt_dataset(opts.train_dataset_path, opts.train_dataset_labels_path, e4e_preprocess, opts)
        test_ds = setup_img2txt_dataset(opts.test_dataset_path, opts.test_dataset_labels_path, e4e_preprocess, opts)
        model_store = LocalModelStore('prototype_decoder', opts.experiment_name, opts.exp_dir)

        if opts.autoencoder_model == 'e4e':
            print("Using E4E autoencoder")
            autoencoder, args = setup_model(consts.model_paths['e4e_checkpoint_path'],
                                            consts.model_paths['stylegan_weights'])
        elif opts.autoencoder_model == 'psp':
            print("Using PSP autoencoder")
            autoencoder = get_psp()
        autoencoder.cuda()
        mappers = []
        optims = []
        for i in range(18):
            mapping = RealNVP(n_blocks=opts.n_blocks, input_size=w_latent_dim, hidden_size=opts.hidden_dim, n_hidden=opts.n_hidden, cond_label_size=latent,
                              batch_norm=not opts.no_batch_norm)
            print(mapping)
            mapping.cuda()
            if opts.start_epoch > 0:
                model_store.load_model_and_optimizer(mapping, epoch=opts.start_epoch - 1,
                                                     label=f'{i}_flow_model_mapping')
            mappers.append(mapping)
            # if opts.mapping_ckpt is not None:
            #     mapping, _, _, _ = model_store.load_model_and_optimizer_loc(mapping, model_location=opts.mapping_ckpt)
            optims.append(optim.Adam(
                mapping.parameters(),
                lr=opts.lr,
                betas=(0.9, 0.999),
            ))
        image_logger = ImageLogger(os.path.join(opts.exp_dir, opts.experiment_name, 'image_sample'))

        trainer = Text2WFlowCoach(opts,
                                  txt_model,
                                  mappers,
                                  autoencoder,
                                  optims,
                                  [torch.optim.lr_scheduler.StepLR(optim, opts.lr_reduce_step) for optim in optims],
                                  train_ds,
                                  test_ds,
                                  image_logger,
                                  model_store,
                                  GradientManager(opts),
                                  opts.product_image_size,
                                  opts.save_interval, opts.board_interval, opts.image_interval, 1000,
                                  opts.val_interval)

        trainer.train(opts.start_epoch, opts.max_steps)
