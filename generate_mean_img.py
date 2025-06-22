from argparse import ArgumentParser
import util
from torchvision import transforms

from models import RealNVP, NormedRealNVP, get_psp, ModelInitializer, ImageEncoderWrapper, predict
from models.e4e.model_utils import setup_model

from dataset import CelebADataset
from torch.utils.data import DataLoader
from local_model_store import LocalModelStore
from tqdm import tqdm

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
    parser.add_argument('--experiment_name', type=str, default='memory_decoder',
                        help='The specific name of the experiment')
    parser.add_argument('--run_name', type=str, default='clip familiar CelebA-HQ mtcnn clip2W+ mean images',
                        help='The name of the specific run')

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

    # parser.add_argument('--test_dataset_path', default='/home/ssd_storage/datasets/celebA_crops', type=str,
    #                     help='path to the validation dir')
    # parser.add_argument('--test_dataset_labels_path',
    #                     default='/home/ssd_storage/experiments/clip_decoder/celebA_cls_names.csv', type=str,
    #                     help='path to the test cls2label file')
    # VGGFace2
    # parser.add_argument('--train_dataset_path',
    #                     default="/home/ssd_storage/datasets/processed/clip_familiar_vggface2_{'train': 0.7, 'val': 0.2, 'test': 0.1}/train",
    #                     type=str, help='path to the train dir')
    # parser.add_argument('--train_dataset_labels_path',
    #                     default="/home/ssd_storage/experiments/clip_decoder/identity_meta.csv", type=str,
    #                     help='path to the train cls2label file')

    # CelebA HQ
    parser.add_argument('--train_dataset_path',
                        default="/home/ssd_storage/datasets/CelebAMask-HQ/CLIP_familiar/ViT-B32_mtcnn/",
                        type=str, help='path to the train dir')
    parser.add_argument('--train_dataset_labels_path',
                        default="/home/ssd_storage/experiments/clip_decoder/celeba_hq_names.csv", type=str,
                        help='path to the train cls2label file')

    parser.add_argument('--test_dataset_path',
                        default="/home/ssd_storage/datasets/CelebAMask-HQ/CLIP_familiar/ViT-B32_mtcnn/",
                        type=str, help='path to the train dir')
    parser.add_argument('--test_dataset_labels_path',
                        default="/home/ssd_storage/experiments/clip_decoder/celeba_hq_names.csv", type=str,
                        help='path to the train cls2label file')

    parser.add_argument("--lr_reduce_step", type=int, default=25000, help="after how many steps to reduce lr")
    parser.add_argument("--w_latent_dim", type=int, default=512, help="dim of w latent space")
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")

    parser.add_argument('--clip2w_n_hidden', type=int, default=5, help='Number of hidden layers in each MADE.')
    parser.add_argument('--clip2w_n_blocks', type=int, default=5,
                        help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
    parser.add_argument('--clip2w_hidden_dim', default=512, type=int,
                        help='hidden dim in s,t for conditional normalizing flow')
    # parser.add_argument('--clip2w_weights_path', default='/home/ssd_storage/experiments/clip_decoder/clip_decoder/stylegan2/models/{}_flow_model_mapping174.pth', type=str,
    #                     help='dir with clip2w weights')
    # parser.add_argument('--clip2w_weights_path', default='/home/ssd_storage/experiments/clip_decoder/mlflow/artifact_store/memory_decoder/3e4a65b5470d421db0b9c4eaf82c5737/artifacts/{}_flow_model_mapping199.pth', 
    #                     type=str, help='dir with clip2w weights')
    parser.add_argument('--clip2w_weights_path', default='/home/ssd_storage/experiments/clip_decoder/mlflow/artifact_store/memory_decoder/6e45416f50da4dba9d0bb3a4c375e465/artifacts/{}_flow_model_mapping174.pth', 
                        type=str, help='dir with clip2w weights')
    

    parser.add_argument('--txt2img_n_hidden', type=int, default=5, help='Number of hidden layers in each MADE.')
    parser.add_argument('--txt2img_n_blocks', type=int, default=5,
                        help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
    parser.add_argument('--txt2img_hidden_dim', default=512, type=int,
                        help='hidden dim in s,t for conditional normalizing flow')

    parser.add_argument('--batch_size', default=12, type=int, help='Batch size for training')
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float, help='LR')
    # parser.add_argument('--mapping_depth', default=4, type=int, help='num of layers in mapping function')
    parser.add_argument('--embedding_norm_path',
                        default='/home/ssd_storage/experiments/clip_decoder/celebA_subset_distribution2.pt', type=str,
                        help='Path to embeddings norm')
    parser.add_argument('--semantic_architecture', default="ViT-B/32")  # ViT-B/32 \ RN101 \ RN50x16
    # parser.add_argument('--clip2w_weight_step', default=175, type=int, help='Maximum number of training steps')
    parser.add_argument('--max_steps', default=200, type=int, help='Maximum number of training steps')
    parser.add_argument('--start_epoch', default=0, type=int, help='epoch to start form')
    parser.add_argument('--mapping_ckpt', default=None, type=str,
                        help='where to load the model weights from')  # '/home/hdd_storage/mlflow/artifact_store/clip_decoder/05d9c461360146b58f94b47518935edf/artifacts/flow_model_mapping99.pth'
    parser.add_argument('--W_plus', action='store_false', help='Should work in W+')
    parser.add_argument('--test_with_random_z', action='store_true',
                        help='When predicting clip picture - should use random Z')
    parser.add_argument('--test_on_dataset', action='store_false',
                        help='When predicting clip picture - attempt to recreate from test dataset')
    parser.add_argument('--clip_on_orig', action='store_false', help='epoch to start form')
    parser.add_argument('--spherical_coordinates', action='store_true', help='Should use spherical coordinates')
    parser.add_argument('--autoencoder_model', default='e4e', type=str, help='e4e / psp')
    
    # mapped vectors distribution parameters
    parser.add_argument('--CLIP2W_normed', action='store_true', help='Is the CLIP2W+ mapper normalized?')
    parser.add_argument('--CLIP_norm', type=str, default=None, help='Distribution parameters for the CLIP vectors')
    parser.add_argument('--CLIP_txt_norm', type=str, default=None, help='Distribution parameters for the CLIP txt vectors')

    parser.add_argument('--use_vgg', action='store_true', help='Whether or not to use a model other than CLIP')
    parser.add_argument('--model_weights', type=str, default='/home/administrator/experiments/familiarity/pretraining/vgg16/models/119.pth', help='Path to the used model weights')
    parser.add_argument('--num_cls', type=int, default=8749, help='Number of classes for the encoder')
    parser.add_argument('--train_encoder_imgs_paths', type=str, default='/home/ssd_storage/datasets/CelebAMask-HQ/CLIP_familiar/ViT-B32_mtcnn', help='Where to find the images aligned for the encoder')

    return parser.parse_args()


def mean_emb(all_emb):
    mean_emb = all_emb[0] / len(all_emb)
    for i in range(1, len(all_emb)):
        mean_emb += all_emb[i] / len(all_emb)
    return mean_emb


def predict_mean(img_model, mappers, dl):
    all_w = []
    all_y = []
    for x in dl:
        x = x.cuda()
        y = img_model.encode_image(x)
        all_y.append(y)
        w = []
        for i, mapper in enumerate(mappers):
            latents = predict(mapper, y)
            w.append(latents)
        w = torch.stack(w, dim=1)
        all_w.append(w)
    mean_y = mean_emb(all_y)
    w = []
    for i, mapper in enumerate(mappers):
        latents = predict(mapper, mean_y)
        w.append(latents)
    mean_y_w = torch.stack(w, dim=1)
        
    mean_w = mean_emb(all_w)

    mean_w_img, _ = autoencoder(mean_w, resize=False, input_code=True, return_latents=True)
    mean_y_img, _ = autoencoder(mean_y_w, resize=False, input_code=True, return_latents=True)
    return mean_w_img, mean_y_img

def save_hq_img(tensor_img, name, output_dir):
    img = util.tensor2im(tensor_img[0])
    path = os.path.join(output_dir, f'{name}.jpg')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    mlflow.log_artifact(path)

if __name__ == '__main__':
    opts = get_args()
    if mlflow.get_experiment_by_name(opts.experiment_name) is None:
        mlflow.create_experiment(opts.experiment_name,
                                 artifact_location=os.path.join(consts.MLFLOW_ARTIFACT_STORE, opts.experiment_name))
    mlflow.set_experiment(opts.experiment_name)

    with mlflow.start_run(run_name=opts.run_name):
        mlflow.log_params(vars(opts))
        
        latent = consts.CLIP_EMBEDDING_DIMS[opts.semantic_architecture]
        txt_latent = latent
        mixing = 0.9
        batch = opts.batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_store = LocalModelStore(opts.experiment_name, 'weights', opts.exp_dir)

        if opts.use_vgg:
            latent = 4096
            model_factory = ModelInitializer(['vgg16'])
            img_model = model_factory.get_model('vgg16', False, opts.num_cls)
            model_store.load_model_and_optimizer_loc(img_model, model_location=opts.model_weights)
            img_model = ImageEncoderWrapper(img_model)
            preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            txt_model, _ = clip.load(opts.semantic_architecture, device=device)
        else:
            print('creating clip model...')
            clip_model, preprocess = clip.load(opts.semantic_architecture, device=device)
            img_model = clip_model
            txt_model = clip_model
        img_model.cuda()
        txt_model.cuda()

        img_model.eval()
        txt_model.eval()

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
        autoencoder.eval()
        mappers = []
        
        for i in range(18):
            if opts.CLIP2W_normed:
                mapping = NormedRealNVP(opts.clip2w_n_blocks, w_latent_dim, opts.clip2w_hidden_dim, opts.clip2w_n_hidden, latent, not opts.no_batch_norm)
            else:
                mapping = RealNVP(opts.clip2w_n_blocks, w_latent_dim, opts.clip2w_hidden_dim, opts.clip2w_n_hidden, latent, batch_norm=not opts.no_batch_norm)
            mapping.eval()
            mapping.cuda()
            if opts.clip2w_weights_path is not None:
                model_store.load_model_and_optimizer_loc(mapping, model_location=opts.clip2w_weights_path.format(i))
            mappers.append(mapping)
        
        # output_path = '/home/ssd_storage/experiments/clip_decoder/prototypes'
        output_path = '/home/ssd_storage/experiments/clip_decoder/memories/prototypes_2'
        # output_path = '/home/ssd_storage/experiments/clip_decoder/memories/visual_prototypes'
        input_path = '/home/ssd_storage/datasets/Cognitive_exp/adva_images_and_more/adva_celeb_finetune_mtcnn/images_for_finetuning'
        person_names = ['angela_rayner', 'angelina_jolie', 'anthony_hopkins', 'bill_clinton', 'boris_johnson', 'david_cameron', 'donald_trump', 'esther_mcvey', 'george_W_bush', 'hillary_clinton', 'Hugh_Grant','jennifer_aniston','Judi_Dench','Kate_Winslet','keira_knightley','liam_neeson','Martin_Freeman','michael_caine','nicolas_cage','nicola_sturgeon','priti_patel','robert_de_niro','sandra_bullock','theresa_may','tom_hanks', 'tony_blair']
        person_dir_names = ['angela_rayner', 'angelina_jolie', 'anthony_hopkins', 'bill_clinton', 'boris_johnson', 'david_cameron', 'donald_trump', 'esther_mcvey', 'george_Wbush', 'hillary_clinton', 'Hugh_Grant','jennifer_aniston','Judi_Dench','Kate-Winslet','keira_knightley','liam_neeson','Martin_Freeman','michael_caine','nicolas_cage','nicola_sturgeon','PRITI_PATEL','robert_de_niro','sandra_bullock','theresa_may','tom_hankes', 'tony_blair']
        person_dirs = [os.path.join(input_path, name) for name in person_dir_names]
        
        with torch.no_grad():
            for dir, name in tqdm(zip(person_dirs, person_names)):
                dl = DataLoader(CelebADataset(dir, preprocess),
                    batch_size=1,
                    num_workers=4,
                    pin_memory=True)
                w, y = predict_mean(img_model, mappers, dl)
                # save_hq_img(w, name, os.path.join(output_path, 'mean_w'))
                save_hq_img(y, name, os.path.join(output_path, 'mean_emb'))