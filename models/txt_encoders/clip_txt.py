import clip
from torch import nn, Tensor, cuda
from typing import List


class CLIPTxt(nn.Module):
    def __init__(self, inner: nn.Module):
        super(CLIPTxt, self).__init__()
        self.inner = inner

    def encode_text(self, txt: List[str]) -> Tensor:
        return self.forward(txt)
    
    def forward(self, txt: List[str]) -> Tensor:
        txt_tokens = clip.tokenize(txt)
        if cuda.is_available():
            txt_tokens = txt_tokens.cuda(non_blocking=True)
        txt_embeddings = self.inner.encode_text(txt_tokens)
        return txt_embeddings
