import transformers
import torch
from torch import nn, Tensor, cuda
from typing import Union, List, Optional


class TransformersTxt(nn.Module):
    def __init__(self, inner: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, embedding_reduce: str = 'mean', padding: Optional[str] = None):
        super(TransformersTxt, self).__init__()
        self.inner = inner
        self.tokenizer = tokenizer
        assert (embedding_reduce in ['mean', 'last', 'first'])
        self.embedding_reduce = embedding_reduce
        if padding is not None:
            self.tokenzier.pad_token = padding#tok.eos_token

    def encode_text(self, txt: List[str]) -> Tensor:
        return self.forward(txt)

    def reduce_embeddings(self, embeddings: Tensor, last_indices: List[int], str_indices: List[int]) -> Tensor:
        if self.embedding_reduce == 'mean':
            # Set every padding's vector to 0
            # means = torch.zeros([embeddings.shape[0], embeddings.shape[2]])
            # if torch.cuda.is_available():
            #     means = means.cuda(non_blocking=True)
            # for batch in range(len(last_indices)):
            #     for i in range(last_indices[batch]):
            #         print(batch)
            #         print(i)
            #         print(embeddings.shape)
            #         means[batch] += embeddings[batch, i, :] / (last_indices[i] + 1)
            means = embeddings.mean(dim=1)
            return means
        elif self.embedding_reduce == 'last':
            return embeddings[str_indices, last_indices, :]
        elif self.embedding_reduce == 'first':
            return embeddings[:,0,:]
        raise NotImplementedError('Non existent embedding choice declared!')
    
    def forward(self, txt: List[str]) -> Tensor:
        last_tok_indices = []
        item_indices = []
        
        for i, item in enumerate(txt):
            item_tokens = self.tokenizer(item, return_tensors="pt")
            item_indices.append(i)
            last_tok_indices.append(item_tokens['input_ids'].shape[1] - 1)

        txt_tokens = self.tokenizer(txt, return_tensors="pt", padding=True)
        if cuda.is_available():
            txt_tokens = txt_tokens.to('cuda:0')

        txt_embeddings = self.inner(**txt_tokens)
        txt_embeddings = self.reduce_embeddings(txt_embeddings.last_hidden_state, last_tok_indices, item_indices)
        return txt_embeddings
