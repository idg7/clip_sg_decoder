import clip
import torch
import transformers
from .clip_txt import CLIPTxt
from .transformers_txt import TransformersTxt
from .sgpt import SGPT
from typing import Optional


def encoder_to_cude(encoder: torch.nn.Module) -> torch.nn.Module:
    if torch.cuda.is_available():
        encoder.cuda()
    return encoder

def clip_txt_encoder(clip_model: torch.nn.Module) -> CLIPTxt:
    encoder = CLIPTxt(clip_model)
    return encoder_to_cude(encoder)


def gpt2_txt_encoder(arch: str = 'gpt2', embedding_reduction: str = 'last') -> TransformersTxt:
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(arch)
    tokenizer.pad_token = tokenizer.eos_token
    encoder = TransformersTxt(transformers.GPT2Model.from_pretrained(arch), tokenizer, embedding_reduction)
    return encoder_to_cude(encoder)


def roberta_txt_encoder(arch: str = 'roberta-base', embedding_reduction: str = 'first') -> TransformersTxt:
    encoder = TransformersTxt(transformers.RobertaModel.from_pretrained(arch), transformers.RobertaTokenizer.from_pretrained(arch), embedding_reduction)
    return encoder_to_cude(encoder)


def bert_txt_encoder(arch: str = 'bert-base-cased', embedding_reduction: str = 'first') -> TransformersTxt:
    encoder = TransformersTxt(transformers.BertModel.from_pretrained(arch), transformers.BertTokenizer.from_pretrained(arch), embedding_reduction)
    return encoder_to_cude(encoder)


def sgpt(arch: str = "Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit") -> SGPT:
    encoder = SGPT(transformers.AutoModel.from_pretrained(arch), transformers.AutoTokenizer.from_pretrained(arch))
    return encoder_to_cude(encoder)


def get_txt_encoder(arch: str, embedding_reduction: Optional[str] = None) -> TransformersTxt:
    if embedding_reduction is not None:
        if arch.startswith('gpt2'):
            return gpt2_txt_encoder(arch=arch, embedding_reduction=embedding_reduction)
        if arch.startswith('bert'):
            return bert_txt_encoder(arch=arch, embedding_reduction=embedding_reduction)
        if arch.startswith('roberta'):
            return roberta_txt_encoder(arch=arch, embedding_reduction=embedding_reduction)
        if arch == 'sgpt':
            return sgpt()
        else:
            raise NotImplementedError(f'{arch} text encoder is not implemented!')
    else:
        if arch.startswith('gpt2'):
            return gpt2_txt_encoder(arch=arch)
        if arch.startswith('bert'):
            return bert_txt_encoder(arch=arch)
        if arch.startswith('roberta'):
            return roberta_txt_encoder(arch=arch)
        if arch == 'sgpt':
            return sgpt()
        else:
            raise NotImplementedError(f'{arch} text encoder is not implemented!')
