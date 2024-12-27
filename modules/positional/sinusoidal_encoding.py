import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    # https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html
    # from https://nlp.seas.harvard.edu/annotated-transformer/#positional-encoding
    # from attention is all u need positional encoding(sinusoidal version)
    """

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


if __name__ == "__main__":
    from d2l import torch as d2l

    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, : X.shape[1], :]
    d2l.plot(
        torch.arange(num_steps),
        P[0, :, 6:10].T,
        xlabel="Row (position)",
        figsize=(6, 2.5),
        legend=["Col %d" % d for d in torch.arange(6, 10)],
    )
    d2l.plt.show()
