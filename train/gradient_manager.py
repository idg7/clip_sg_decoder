import torch
import clip
from models.stylegan2 import Generator, GeneratorMappingWrapper


class GradientManager(object):
    def __init__(self, opts):
        self.train_encoder = opts.train_encoder
        self.train_full_generator = opts.train_full_generator

    def requires_grad_generator_wrapper(self, model: GeneratorMappingWrapper, flag=True):
        self.requires_grad_generator(model.generator)

        for p in model.pre_mapping.parameters():
            p.requires_grad = flag

    def requires_grad_generator(self, model: Generator, flag=True):
        if not flag:
            for p in model.parameters():
                p.requires_grad = flag
        if flag:
            if self.train_full_generator:
                for p in model.parameters():
                    p.requires_grad = flag
            else:
                for p in model.style.parameters():
                    p.requires_grad = flag

    def __requires_grad(self, model: torch.nn.Module, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def requires_grad(self, model: torch.nn.Module, flag=True):
        if type(model) == Generator:
            self.requires_grad_generator(model, flag)
        if type(model) == GeneratorMappingWrapper:
            self.requires_grad_generator_wrapper(model, flag)
        if (type(model) == clip.model.CLIP) and self.train_encoder:
            self.__requires_grad(model, flag)
            self.__requires_grad(model.visual, flag)
        else:
            self.__requires_grad(model, flag)