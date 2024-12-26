import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        """Rotary positional embedding
        Reference : https://blog.eleuther.ai/rotary-embeddings/
        Paper: https://arxiv.org/pdf/2104.09864.pdf
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
            precision: precision to use for numerical values
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
        self.sin_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
        self.precision = precision

    def forward(self, x, seq_len: int = 0):
        """
        Args:
            x: Input x with T X B X C
            seq_len: Sequence length of input x
        """
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos().view(emb.size(0), 1, 1, emb.size(1))
            self.sin_cached = emb.sin().view(emb.size(0), 1, 1, emb.size(1))
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


if __name__ == "__main__":
    # from d2l import torch as d2l

    torch.manual_seed(0)
    T = 3
    B = 1
    C = 2
    sample = torch.randn(T, B, C)  # TBC
    rope_pos_emd = RotaryPositionalEmbedding(dim=C)
    cos, sin = rope_pos_emd(sample, T)
    print(cos, sin)

    query = sample.view(T, B, 1, C)
    new_query, new_key = apply_rotary_pos_emb(query, query, cos, sin)
    print(new_query, new_key)

    module_scripted = torch.jit.script(rope_pos_emd)
    apply_rotary_scripted = torch.jit.script(apply_rotary_pos_emb)
    # Test several different lengths
    for T in [3, 5, 10]:
        sample = torch.randn(T, B, C)
        # Run forward pass with the original module
        cos_original, sin_original = rope_pos_emd(sample, T)
        query = sample.view(T, B, 1, C)
        new_query, new_key = apply_rotary_pos_emb(query, query, cos_original, sin_original)

        # Run forward pass with the scripted module
        cos_scripted, sin_scripted = module_scripted(sample, T)
        new_query_scripted, new_key_scripted = apply_rotary_scripted(
            query, query, cos_scripted, sin_scripted
        )
        # Ensure the outputs are the same
        # print(cos_original, cos_scripted) # check cos_original==cos_scripted
        print(torch.allclose(cos_original, cos_scripted))
        print(torch.allclose(sin_original, sin_scripted))
        print(torch.allclose(new_query, new_query_scripted))
        print(torch.allclose(new_key, new_key_scripted))

    # d2l.plt.show()
