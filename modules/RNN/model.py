from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    block_size: int = None  # length of the input sequences of integers
    vocab_size: int = None  # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4


class RNNCell(nn.Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """

    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht


class GRUCell(nn.Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.

    - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259) (GRU)
    - [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) (LSTM)

    GRU:
    input hidden state ->
    the reset gate (long-term memory) -> hidden state -> the update gate (short-term memory)
    -> output hidden state

    LSTM:
    input hidden state ->
                                                            (candidate memory cell)
                                                                /|\
    the forget gate (long-term memory) -> the input gate -> hidden state -> the output gate
    -> output hidden state

    """

    def __init__(self, config):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        # first use the reset gate to wipe some channels of the hidden state to zero
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        # calculate the candidate new hidden state hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))
        # calculate the switch gate that determines if each channel should be updated at all
        z = F.sigmoid(self.xh_to_z(xh))
        # blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        return ht


class NN_LSTM(nn.LSTM):
    # https://github.com/pytorch/pytorch/blob/main/torch/backends/cudnn/rnn.py
    # https://developer.nvidia.com/cudnn dnn kenerl
    # https://docs.nvidia.com/deeplearning/cudnn/latest/api/overview.html (cudnn be) shim layer(libcudnn.so) -> (libcudnn_adv.so)
    # https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-adv-library.html#cudnnrnnalgo-t
    # https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/nn/modules/rnn.cpp#L547

    # cellMode:
    # _VF.lstm_cell
    pass

class NN_GRU(nn.GRU):
    # https://github.com/pytorch/pytorch/blob/main/torch/backends/cudnn/rnn.py
    # https://developer.nvidia.com/cudnn dnn kenerl
    # https://docs.nvidia.com/deeplearning/cudnn/latest/api/overview.html (cudnn be) shim layer(libcudnn.so) -> (libcudnn_adv.so)
    # https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-adv-library.html#cudnnrnnalgo-t
    # https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/nn/modules/rnn.cpp#L698

    # cellMode:
    # _VF.rnn_tanh_cell
    # _VF.rnn_relu_cell
    pass

class NN_RNN(nn.RNN):
    # https://github.com/pytorch/pytorch/blob/main/torch/backends/cudnn/rnn.py
    # https://developer.nvidia.com/cudnn dnn kenerl
    # https://docs.nvidia.com/deeplearning/cudnn/latest/api/overview.html (cudnn be) shim layer(libcudnn.so) -> (libcudnn_adv.so)
    # https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-adv-library.html#cudnnrnnalgo-t
    # https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/nn/modules/rnn.cpp#L400

    # cellMode:
    # _VF.rnn_tanh_cell
    # _VF.rnn_relu_cell
    pass


class RNN(nn.Module):
    """
    Recurrent Neural Net language model: either a vanilla RNN recurrence or a GRU.
    Did not implement an LSTM because its API is a bit more annoying as it has
    both a hidden state and a cell state, but it's very similar to GRU and in
    practice works just as well.
    from: https://github.com/karpathy/makemore/blob/master/makemore.py#L303
    """

    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2))  # the starting hidden state
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embeddings table
        if cell_type == "rnn":
            self.cell = RNNCell(config)
        elif cell_type == "gru":  # GRU(gated recurrent unit) -> LSTM (long short-term memory)
            self.cell = GRUCell(config)
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx)  # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        hprev = self.start.expand((b, -1))  # expand out the batch dimension
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :]  # (b, n_embd)
            ht = self.cell(xt, hprev)  # (b, n_embd2)
            hprev = ht
            hiddens.append(ht)

        # decode the outputs
        hidden = torch.stack(hiddens, 1)  # (b, t, n_embd2)
        logits = self.lm_head(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    for cell_type in ["rnn", "gru"]:
        model = RNN(
            ModelConfig(
                block_size=16,  # length of the input sequences of integers
                vocab_size=27,  # the input integers are in range [0 .. vocab_size -1]
                # parameters below control the sizes of each model slightly differently
                n_layer=4,
                n_embd=64,
                n_embd2=64,
                n_head=4,
            ),
            cell_type=cell_type,
        ).to(DEVICE)

        # print the number of parameters in the model
        model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(model)
        print(model_million_params, "M parameters")

        # forward
        x = torch.zeros(50, 1)
        x = x.type(torch.LongTensor).to(DEVICE)
        print(model(x)[0].shape)


"""
RNN(
  (wte): Embedding(27, 64)
  (cell): RNNCell(
    (xh_to_h): Linear(in_features=128, out_features=64, bias=True)
  )
  (lm_head): Linear(in_features=64, out_features=27, bias=True)
)
0.011803 M parameters
RNN(
  (wte): Embedding(27, 64)
  (cell): GRUCell(
    (xh_to_z): Linear(in_features=128, out_features=64, bias=True)
    (xh_to_r): Linear(in_features=128, out_features=64, bias=True)
    (xh_to_hbar): Linear(in_features=128, out_features=64, bias=True)
  )
  (lm_head): Linear(in_features=64, out_features=27, bias=True)
)
0.028315 M parameters
"""
