from torchvision import transforms, datasets

from models import Generator, get_psp, ModelInitializer, ImageEncoderWrapper
from models.e4e.model_utils import setup_model
from train import RandomFlowCoach, GradientManager, WPlusFlowCoach

from dataset import setup_dataset, setup_2imgs_dataset, setup_2models_dataset
from image_saver import ImageLogger
from local_model_store import LocalModelStore
from text import text_to_image, text_invert_compare
from tqdm import tqdm

import os
import torch
import clip
import mlflow

import consts
from argparse import Namespace


# def get_encoders_txt2img(opts: Namespace):

# def get_encoders_txt2w(opts: Namespace):

def get_encoders_img2w(opts: Namespace):
    model_store = LocalModelStore(opts.experiment_name, 'weights', opts.exp_dir)
    if opts.use_vgg:
        latent = 4096
        model_factory = ModelInitializer(['vgg16'])
        clip_model = model_factory.get_model('vgg16', False, opts.num_cls)
        model_store.load_model_and_optimizer_loc(clip_model, model_location=opts.model_weights)
        clip_model = ImageEncoderWrapper(clip_model)
        clip_preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    else:
        print('creating clip model...')
        clip_model, clip_preprocess = clip.load(opts.semantic_architecture, device=device)
    
    e4e_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    clip_model.cuda()

    # mapping = get_realnvp(512, latent, opts.mapping_depth, opts.hidden_dim)

    w_latent_dim = opts.w_latent_dim

    if opts.autoencoder_model == 'e4e':
        print("Using E4E autoencoder")
        autoencoder, args = setup_model(consts.model_paths['e4e_checkpoint_path'], consts.model_paths['stylegan_weights'])
    elif opts.autoencoder_model == 'psp':
        print("Using PSP autoencoder")
        autoencoder = get_psp()
    autoencoder.cuda()

    return model_store, clip_model