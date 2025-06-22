from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob
from torchvision import transforms

from models import RealNVP, get_psp, Txt2WImg,  clip_txt_encoder, get_txt_encoder
from models.e4e.model_utils import setup_model
import util

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
    parser.add_argument('--run_name', type=str, default='sgpt predict',
                        help='The specific name of the experiment')

    parser.add_argument('--num_batches_per_epoch', default=250, type=int, help='num batches per epoch')

    parser.add_argument('--test_batch_size', default=4, type=int, help='Batch size for testing and inference')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of train dataloader workers')

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
    parser.add_argument('--train_dataset_path',
                        default="/home/ssd_storage/datasets/processed/clip_familiar_vggface2_{'train': 0.7, 'val': 0.2, 'test': 0.1}/train",
                        type=str, help='path to the train dir')
    parser.add_argument('--train_dataset_labels_path',
                        default="/home/ssd_storage/experiments/clip_decoder/identity_meta.csv", type=str,
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
    parser.add_argument('--start_epoch', default=200, type=int, help='epoch to start form')
    parser.add_argument('--mapping_ckpt', default=None, type=str,
                        help='where to load the model weights from')  # '/home/hdd_storage/mlflow/artifact_store/clip_decoder/05d9c461360146b58f94b47518935edf/artifacts/flow_model_mapping99.pth'
    parser.add_argument('--W_plus', action='store_false', help='Should work in W+')
    parser.add_argument('--test_with_random_z', action='store_true',
                        help='When predicting clip picture - should ues random Z')
    parser.add_argument('--test_on_dataset', action='store_false',
                        help='When predicting clip picture - attempt to recreate from test dataset')
    parser.add_argument('--clip_on_orig', action='store_false', help='epoch to start form')
    parser.add_argument('--autoencoder_model', default='e4e', type=str, help='e4e / psp')
    
    parser.add_argument('--txt_architecture', type=str, default=None)
    parser.add_argument('--embedding_reduction', type=str, default='last')
    parser.add_argument('--txt2w_weights_path', type=str, default='/home/ssd_storage/experiments/clip_decoder/mlflow/artifact_store/txt2w/gpt2 to W/e735f1bb1fc84b4198d43c56abcbdd73/artifacts/{}_flow_model_mapping199.pth')
    

    return parser.parse_args()


if __name__ == '__main__':
    opts = get_args()
    if mlflow.get_experiment_by_name(opts.experiment_name) is None:
        mlflow.create_experiment(opts.experiment_name,
                                 artifact_location=os.path.join(consts.MLFLOW_ARTIFACT_STORE, opts.experiment_name))
    mlflow.set_experiment(opts.experiment_name)

    with mlflow.start_run():
        mlflow.log_params(vars(opts))

        latent = consts.CLIP_EMBEDDING_DIMS[opts.semantic_architecture]
        mixing = 0.9
        batch = opts.batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"

        clip_model, preprocess = clip.load(opts.semantic_architecture, device=device)
        preprocess.transforms[-1] = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        if opts.txt_architecture is None:
            txt_model = clip_txt_encoder(clip_model)
        else:
            txt_model = get_txt_encoder(opts.txt_architecture, opts.embedding_reduction)
            latent = consts.TXT_EMBEDDING_DIMS[opts.txt_architecture]
        txt_model.cuda()

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
            mapping = RealNVP(opts.n_blocks, w_latent_dim, opts.hidden_dim, opts.n_hidden, latent,
                              batch_norm=not opts.no_batch_norm)

            mapping.cuda()
            if opts.txt2w_weights_path is not None:
                model_store.load_model_and_optimizer_loc(mapping, model_location=opts.txt2w_weights_path.format(i))
            mappers.append(mapping)
        image_logger = ImageLogger(os.path.join(opts.exp_dir, opts.experiment_name, 'image_sample'))

        txt2img_generator = Txt2WImg(txt_model, autoencoder, mappers, transforms.Resize((256, 256)))

        txt2img_generator = txt2img_generator.cuda()

        txt2img_generator.train(False)
        txt2img_generator.requires_grad_(False)
        img_log_dir = os.path.join(opts.exp_dir, opts.experiment_name, 'image_sample')

        people = ['Ray Romano', 'Ray_Romano', 'Tom Hardy', 'Tom_Hardy', 'AKON', 'akon', 'Akon']
        hair = ['Blonde', 'Red hair', 'Ginger', 'Black hair', 'Bald', 'Grey hair', 'Beard']
        ethnicity = ['African american', 'Indian', 'Jewish', 'Caucasian', 'Asian']
        gender = ['Man', 'Woman']
        joined = ['African american woman', 'African man', 'Blonde woman', 'Brunette man']
        eyes = ['Blue eyes', 'Brown eyes', 'Green eyes', 'Sunglasses', 'Glasses']
        behavioral = ['Smiling', 'Laughing', 'Frowning', 'crying', 'Happy', 'Sad', 'Angry', 'Mad']
        personality = ['Nice', 'Mean', 'Smart', 'Dumb', 'Good', 'Bad']
        # celebA = [os.path.basename(pth) for pth in glob('/home/ssd_storage/datasets/celebA/*')]
        uri = ['Man with red hair and brown eyes', 'Man with brown eyes and red hair', 'Woman with red hair and brown eyes', 'Woman with brown eyes and red hair']
        txt_input = ['big nose', 'short forehead', 'fat', 'big ears', 'big eyes', 'thick eyebrows'] + behavioral + personality + ['Man with red hair and black beard', 'Man with black beard and red hair'] + eyes + hair + people + hair + uri + ethnicity + gender + joined + ['Donald_Trump', 'Barack Obama', 'angela_rayner', 'angelina_jolie', 'anthony_hopkins', 'bill_clinton', 'boris_johnson', 'david_cameron', 'donald_trump', 'esther_mcvey', 'george_W_bush', 'hillary_clinton', 'Hugh_Grant','jennifer_aniston','Judi_Dench','Kate_Winslet','keira_knightley','liam_neeson','Martin_Freeman','michael_caine','nicolas_cage','nicola_sturgeon','priti_patel','robert_de_niro','sandra_bullock','theresa_may','tom_hanks']

        with torch.no_grad():
            consts.PREDICT_WITH_RANDOM_Z = False
            for i, label in tqdm(enumerate(txt_input)):
                for j, z_status in enumerate([False, True]):
                    consts.PREDICT_WITH_RANDOM_Z = z_status
                    curr_input = [label] * 2
                    generated, _, _ = txt2img_generator(curr_input)
                    
                    img = util.tensor2im(generated[0])
                    path = os.path.join(img_log_dir, f'{label}_random_z={z_status}.jpg')
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    img.save(path)
                    mlflow.log_artifact(path)


        # people = ['Ray Romano', 'Ray_Romano', 'Tom Hardy', 'Tom_Hardy', 'AKON', 'akon', 'Akon']
        # hair = ['Blonde', 'Red hair', 'Ginger', 'Black hair', 'Bald', 'Grey hair', 'Beard']
        # ethnicity = ['African american', 'Indian', 'Jewish', 'Caucasian', 'Asian']
        # gender = ['Man', 'Woman']
        # joined = ['African american woman', 'African man', 'Blonde woman', 'Brunette man']
        # eyes = ['Blue eyes', 'Brown eyes', 'Green eyes', 'Sunglasses', 'Glasses']
        # behavioral = ['Smiling', 'Laughing', 'Frowning', 'crying', 'Happy', 'Sad', 'Angry', 'Mad']
        # personality = ['Nice', 'Mean', 'Smart', 'Dumb', 'Good', 'Bad']
        # celebA = [os.path.basename(pth) for pth in glob('/home/ssd_storage/datasets/celebA/*')]
        # uri = ['Man with red hair and brown eyes', 'Man with brown eyes and red hair',
        #        'Woman with red hair and brown eyes', 'Woman with brown eyes and red hair']
        # txt_input = behavioral + personality + ['Man with red hair and black beard',
        #                                         'Man with black beard and red hair'] + eyes + uri + hair + people + hair + ethnicity + gender + joined #+ celebA

        # with torch.no_grad():
        #     for i, label in tqdm(enumerate(txt_input)):
        #         for j, z_status in enumerate([False, True]):
        #             consts.PREDICT_WITH_RANDOM_Z = z_status
        #             curr_input = [label] * 2
        #             generated, _, _ = txt2img_generator(curr_input)
        #             image_logger.parse_and_log_images(generated, generated, step=i,
        #                                               title=f'{label}, random z={z_status}', names=curr_input)


