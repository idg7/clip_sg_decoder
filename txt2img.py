from torch import optim
from argparse import ArgumentParser

from torchvision import transforms

from models import RealNVP, get_psp, CLIPInversionGenerator, Txt2Img
from models.e4e.model_utils import setup_model
from train import GradientManager, Txt2ImgFlowCoach

from dataset import setup_img2txt_dataset
from image_saver import ImageLogger
from local_model_store import LocalModelStore
from text import text_to_image, text_invert_compare
from tqdm import tqdm
from glob import glob

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
    parser.add_argument('--experiment_name', type=str, default='clip_txt_decoder',
                        help='The specific name of the experiment')

    parser.add_argument('--num_batches_per_epoch', default=250, type=int, help='num batches per epoch')

    parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of train dataloader workers')

    parser.add_argument('--product_image_size', default=224, type=int, help='(size x size) of produced images')
    parser.add_argument('--stylegan_weights', default='./models/stylegan2/stylegan2-ffhq-config-f.pt', type=str,
                        help='Path to StyleGAN model weights')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')

    parser.add_argument('--test_dataset_path', default='/home/ssd_storage/datasets/celebA_crops', type=str,
                        help='path to the validation dir')
    parser.add_argument('--test_dataset_labels_path',
                        default='/home/ssd_storage/experiments/clip_decoder/celebA_cls_names.csv', type=str,
                        help='path to the test cls2label file')

    parser.add_argument("--w_latent_dim", type=int, default=512, help="dim of w latent space")

    parser.add_argument('--clip2w_n_hidden', type=int, default=5, help='Number of hidden layers in each MADE.')
    parser.add_argument('--clip2w_n_blocks', type=int, default=5,
                        help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
    parser.add_argument('--clip2w_hidden_dim', default=512, type=int,
                        help='hidden dim in s,t for conditional normalizing flow')
    parser.add_argument('--clip2w_weights_path', default='/home/ssd_storage/experiments/clip_decoder/clip_decoder/stylegan2/models/{}_flow_model_mapping174.pth', type=str,
                        help='dir with clip2w weights')
    parser.add_argument('--no_batch_norm', action='store_true')

    parser.add_argument('--txt2img_n_hidden', type=int, default=5, help='Number of hidden layers in each MADE.')
    parser.add_argument('--txt2img_n_blocks', type=int, default=5,
                        help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
    parser.add_argument('--txt2img_hidden_dim', default=512, type=int,
                        help='hidden dim in s,t for conditional normalizing flow')
    parser.add_argument('--txt2img_weights_path',
                        # default='/home/ssd_storage/experiments/clip_decoder/clip_txt_decoder/stylegan2/models/txt2img_flow_model_mapping199.pth',
                        default='/home/hdd_storage/mlflow/artifact_store/clip_txt_decoder/5274a6ae36c049a5a706b72ede3d17f1/artifacts/txt2img_flow_model_mapping199.pth',
                        type=str,
                        help='dir with clip2w weights')

    parser.add_argument('--semantic_architecture', default="ViT-B/32")  # ViT-B/32 \ RN101 \ RN50x16
    # parser.add_argument('--clip2w_weight_step', default=175, type=int, help='Maximum number of training steps')

    parser.add_argument('--W_plus', action='store_false', help='Should work in W+')
    parser.add_argument('--test_with_random_z', action='store_false',
                        help='When predicting clip picture - should use random Z')
    parser.add_argument('--test_on_dataset', action='store_false',
                        help='When predicting clip picture - attempt to recreate from test dataset')
    parser.add_argument('--autoencoder_model', default='e4e', type=str, help='e4e / psp')

    return parser.parse_args()


if __name__ == '__main__':
    opts = get_args()
    if mlflow.get_experiment_by_name(opts.experiment_name) is None:
        mlflow.create_experiment(opts.experiment_name,
                                 artifact_location=os.path.join(consts.MLFLOW_ARTIFACT_STORE, opts.experiment_name))
    mlflow.set_experiment(opts.experiment_name)

    with mlflow.start_run():
        mlflow.log_param('test_batch_size', opts.test_batch_size)
        mlflow.log_param('test_dataset_path', opts.test_dataset_path)
        mlflow.log_param('semantic_architecture', opts.semantic_architecture)
        mlflow.log_param('clip2w n_hidden', opts.clip2w_n_hidden)
        mlflow.log_param('clip2w n_blocks', opts.clip2w_n_blocks)
        mlflow.log_param('clip2w hidden_dim', opts.clip2w_hidden_dim)
        mlflow.log_param('txt2img n_hidden', opts.txt2img_n_hidden)
        mlflow.log_param('txt2img n_blocks', opts.txt2img_n_blocks)
        mlflow.log_param('txt2img hidden_dim', opts.txt2img_hidden_dim)
        mlflow.log_param('semantic_architecture', opts.semantic_architecture)
        mlflow.log_param('W_plus', opts.W_plus)
        mlflow.log_param('test_with_random_z', opts.test_with_random_z)
        mlflow.log_param('test_on_dataset', opts.test_on_dataset)
        mlflow.log_param('autoencoder_model', opts.autoencoder_model)

        latent = consts.CLIP_EMBEDDING_DIMS[opts.semantic_architecture]
        batch = opts.test_batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"

        clip_model, preprocess = clip.load(opts.semantic_architecture, device=device)
        preprocess.transforms[-1] = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        clip_model.cuda()

        w_latent_dim = opts.w_latent_dim

        model_store = LocalModelStore('stylegan2', opts.experiment_name, opts.exp_dir)

        if opts.autoencoder_model == 'e4e':
            print("Using E4E autoencoder")
            autoencoder, args = setup_model(consts.model_paths['e4e_checkpoint_path'],
                                            consts.model_paths['stylegan_weights'])
        elif opts.autoencoder_model == 'psp':
            print("Using PSP autoencoder")
            autoencoder = get_psp()
        autoencoder.cuda()
        mappers = []
        for i in range(18):
            mapping = RealNVP(opts.clip2w_n_blocks, w_latent_dim, opts.clip2w_hidden_dim, opts.clip2w_n_hidden, latent,
                              batch_norm=not opts.no_batch_norm)

            mapping.cuda()
            if opts.clip2w_weights_path is not None:
                model_store.load_model_and_optimizer_loc(mapping, model_location=opts.clip2w_weights_path.format(i))
            mappers.append(mapping)

        txt2img_mapper = RealNVP(opts.txt2img_n_blocks, latent, opts.txt2img_hidden_dim, opts.txt2img_n_hidden, latent,
                                 batch_norm=not opts.no_batch_norm)
        if opts.txt2img_weights_path is not None:
            model_store.load_model_and_optimizer_loc(txt2img_mapper, model_location=opts.txt2img_weights_path)

        image_logger = ImageLogger(os.path.join(opts.exp_dir, opts.experiment_name, 'image_sample'))

        decoder = CLIPInversionGenerator(autoencoder, mappers, transforms.Resize((256, 256)))

        txt2img_generator = Txt2Img(txt2img_mapper, clip_model, decoder, True)

        txt2img_generator = txt2img_generator.cuda()

        txt2img_generator.train(False)
        txt2img_generator.requires_grad_(False)

        people = ['Ray Romano', 'Ray_Romano', 'Tom Hardy', 'Tom_Hardy', 'AKON', 'akon', 'Akon']
        hair = ['Blonde', 'Red hair', 'Ginger', 'Black hair', 'Bald', 'Grey hair', 'Beard']
        ethnicity = ['African american', 'Indian', 'Jewish', 'Caucasian', 'Asian']
        gender = ['Man', 'Woman']
        joined = ['African american woman', 'African man', 'Blonde woman', 'Brunette man']
        eyes = ['Blue eyes', 'Brown eyes', 'Green eyes', 'Sunglasses', 'Glasses']
        behavioral = ['Smiling', 'Laughing', 'Frowning', 'crying', 'Happy', 'Sad', 'Angry', 'Mad']
        personality = ['Nice', 'Mean', 'Smart', 'Dumb', 'Good', 'Bad']
        celebA = [os.path.basename(pth) for pth in glob('/home/ssd_storage/datasets/celebA/*')]
        uri = ['Man with red hair and brown eyes', 'Man with brown eyes and red hair', 'Woman with red hair and brown eyes', 'Woman with brown eyes and red hair']
        txt_input = ['Borat', 'Curly black hair with mustache very nice', 'Curly black hair with mustache very nice', 'Curly black hair with mustache very nice']#['big nose', 'short forehead', 'fat', 'big ears', 'big eyes', 'thick eyebrows'] #behavioral + personality + ['Man with red hair and black beard', 'Man with black beard and red hair']#eyes + uri + hair #people + hair + ethnicity + gender + joined + celebA

        with torch.no_grad():
            for i, label in tqdm(enumerate(txt_input)):
                for j, z_status in enumerate([False, True]):
                    consts.PREDICT_WITH_RANDOM_Z = z_status
                    curr_input = [label] * 2
                    generated = txt2img_generator(curr_input)
                    image_logger.parse_and_log_images(generated, generated, step=i, title=f'{label}, random z={z_status}', names=curr_input)
