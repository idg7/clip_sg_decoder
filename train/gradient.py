import torch
import clip
from models.stylegan2 import Generator, GeneratorMappingWrapper


def requires_grad_generator_wrapper(model: GeneratorMappingWrapper, flag=True):
    requires_grad_generator(model.generator)

    for p in model.pre_mapping.parameters():
        p.requires_grad = flag


def requires_grad_generator(model: Generator, flag=True):
    if not flag:
        for p in model.parameters():
            p.requires_grad = flag
    if flag:
        for p in model.parameters():
            p.requires_grad = flag
    # if flag:
    #     for p in model.style.parameters():
    #         p.requires_grad = flag


def __requires_grad(model: torch.nn.Module, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def requires_grad(model: torch.nn.Module, flag=True):
    if type(model) == Generator:
        requires_grad_generator(model, flag)
    if type(model) == GeneratorMappingWrapper:
        requires_grad_generator_wrapper(model, flag)
    if type(model) == clip.model.CLIP:
        __requires_grad(model, flag)
        __requires_grad(model.visual, flag)
    else:
        __requires_grad(model, flag)
