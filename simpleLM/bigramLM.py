import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1234)


# just a Embedding layer vocab_size -> vocab_size
class BigramLanguageModel(nn.Module):
    """
    nn.Embedding:
    A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.
    """

    def __init__(self, vocab_size, nn_type="embedding"):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # an Embedding module containing vocab_size tensors of vocab_size,
        # like BigramLanguageModel train each token weight P(token_i|token_i-1)
        if nn_type == "embedding":
            self.logits = nn.Embedding(vocab_size, vocab_size)
        else:
            self.logits = nn.Parameter(torch.zeros((vocab_size, vocab_size)))

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.logits(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # no sampling generation
    def generate(self, idx, max_new_tokens):
        output = []
        self.eval()  # Otherwise batch normalization will raise an error.
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
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
