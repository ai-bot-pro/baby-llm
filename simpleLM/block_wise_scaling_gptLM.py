r"""
This scaling is known as layer-wise or block-wise scaling from:
[DELIGHT: DEEP AND LIGHT-WEIGHT TRANSFORMER](https://arxiv.org/abs/2008.00623)
- attention layers
- FFN layers
reference: 
- https://github.com/sacmehta/delight/blob/master/fairseq/models/delight_transformer.py
- https://github.com/apple/corenet/blob/main/corenet/modeling/models/language_modeling/general_gpt.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import make_divisible, compute_heads


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout, layer_index, num_qkv_heads=None):
        super().__init__()
        if num_qkv_heads is not None:
            num_heads = num_qkv_heads[layer_index]

        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout, layer_index, ffn_intermediate_sizes=None):
        super().__init__()

        ffn_intermediate_size = 4 * n_embd
        if ffn_intermediate_sizes is not None:
            ffn_intermediate_size = ffn_intermediate_sizes[layer_index]

        self.net = nn.Sequential(
            nn.Linear(n_embd, ffn_intermediate_size),
            nn.ReLU(),
            nn.Linear(ffn_intermediate_size, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout, layer_index, num_qkv_heads, ffn_intermediate_sizes):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            n_head, head_size, n_embd, block_size, dropout, layer_index, num_qkv_heads)
        self.ffwd = FeedFoward(
            n_embd, dropout, layer_index, ffn_intermediate_sizes)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DelightGPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, dropout=0.2,
                 qkv_multipliers=(0.5, 1.0), ffn_multipliers=[0.5, 4.0], ffn_intermediate_divisor=256):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd {n_embd} % n_head {n_head} != 0"

        self.block_size = block_size
        self.n_layer = n_layer

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # init layer/block wise scaling
        self._init_block_wise_scaling_attetion(
            n_layer, n_embd, n_head, qkv_multipliers)
        self._init_block_wise_scaling_ffn(
            n_layer, n_embd, ffn_multipliers, ffn_intermediate_divisor)
        assert self.num_qkv_heads.count() > 0, f"num_qkv_heads is empty"
        assert self.ffn_intermediate_sizes.count() > 0, f"ffn_intermediate_sizes is empty"
        self.blocks = nn.ModuleList(
            *[Block(n_embd, n_head, block_size, dropout, layer_index, self.num_qkv_heads, self.ffn_intermediate_sizes) for layer_index in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # init nn weights
        self.apply(self._init_weights)

    def _init_block_wise_scaling_attention(self, n_layer, n_embd, n_head, qkv_multipliers):
        head_size = n_embd // n_head
        # Each attention layer have different latent dimensions assuming qkv_multipliers[0] != qkv_multipliers[1].
        # This results in variable allocation of parameters in attention layer.
        # This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
        att_qkv_multipliers = [
            round(v, 2)
            for v in np.linspace(
                qkv_multipliers[0],
                qkv_multipliers[1],
                num=n_layer,
                dtype=float,
            )
        ]
        # Make sure that scaled model dimension is divisible by scaled head dimension size.
        query_sizes = [
            int(make_divisible(n_embd * m, divisor=head_size))
            for m in att_qkv_multipliers
        ]

        # compute the number of query, key, and value heads
        # For multi-head attention, the number of heads for query, key, and value are the same.
        self.num_qkv_heads = [
            int(compute_heads(q_size, head_size)) for q_size in query_sizes
        ]
        # self.num_kv_heads = [q_heads for q_heads in self.num_query_heads]

    def _init_block_wise_scaling_ffn(self, n_layer, n_embd, ffn_multipliers,  ffn_intermediate_divisor):
        # Each FFN layer have different latent dimensions assuming ffn_multipliers[0] != ffn_multipliers[1].
        # This results in variable allocation of parameters in FFN layer.
        # This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
        ffn_inter_multipliers = [
            round(v, 2)
            for v in np.linspace(
                ffn_multipliers[0],
                ffn_multipliers[1],
                num=n_layer,
                dtype=float,
            )
        ]
        self.ffn_intermediate_sizes = [
            int(
                make_divisible(n_embd * m, divisor=ffn_intermediate_divisor)
            )
            for m in ffn_inter_multipliers
        ]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        # (B,T,C)
        for layer_index in range(self.n_layer):
            x = self.blocks[layer_index](x)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        output = []
        self.eval()  # Otherwise batch normalization will raise an error.
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            output.append(idx_next[0].tolist()[0])
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        self.train()
        return output
