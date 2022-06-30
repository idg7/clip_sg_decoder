import transformers
import torch
from torch import nn, Tensor, cuda
from typing import Union, List, Optional


class SGPT(nn.Module):
    def __init__(self, inner: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, padding: Optional[str] = None):
        super(SGPT, self).__init__()
        self.inner = inner
        self.tokenizer = tokenizer
        if padding is not None:
            self.tokenzier.pad_token = padding#tok.eos_token

    def encode_text(self, txt: List[str]) -> Tensor:
        return self.forward(txt)
        
    def forward(self, txt: List[str]) -> Tensor:
        batch_tokens = self.tokenizer(txt, padding=True, truncation=True, return_tensors="pt")
        if cuda.is_available():
            batch_tokens = batch_tokens.to('cuda:0')
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = self.inner(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings
