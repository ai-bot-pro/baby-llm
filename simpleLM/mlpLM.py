import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import math

# We don't want to use one-hot encoding since it's too sparse, nor binary
# coding that may introduce biases. So we encode each input character
# into char_encoding_len inputs having different patterns of 1 and 0, so that
# each pattern is always composed of 50% ones and 50% zeros.
#
# For example if we have 14 inputs per symbol, we can rapresent:
#
#   14! / (6!*6!) = 3432 total symbols
#
# We call this permutation coding.

# 不使用独热编码(one-hot)，因为它太稀疏，也不想使用可能引入偏差的二进制编码。
# 因此，将每个输入字符编码为具有不同1和0模式的 char_encoding_len 个输入，
# 以便每种模式始终由50% 的1和50% 的0组成。
# 例如，如果我们每个符号有14个输入，我们可以表示：
# 14! / (6!*6!) = 3432 个总符号
# 我们称之为排列编码。
def gen_coding_patterns(char_encoding_len, vocab_size, device):
    # Calculate if there are enough permutations of char_encoding_len
    # length bits 10101010 pattern to actually represent vocab_size
    # symbols. Otherwise this function would run forever...
    # 计算一下，用 char_encoding_len 长度的比特位 10101010 模式
    # 是否有足够的排列来实际表示 vocab_size 个符号。
    # 否则这个函数会运行很长时间……
    permutations = math.factorial(char_encoding_len) / \
        (math.factorial(char_encoding_len//2) *
         math.factorial(char_encoding_len//2))
    if permutations < vocab_size:
        print(f"Insufficient 'char_encoding_len' permutations value {permutations} for vocabulary size {vocab_size}.")
        exit(1)
    print(f"'char_encoding_len' permutations value {permutations} | vocabulary size {vocab_size}.")

    # We want the result of this function to be stable, so let's
    # create a PRNG with a seed which is always the same.
    r = random.Random()
    r.seed(1234)

    pattern = "01"*(char_encoding_len//2)
    patterns = {}
    while len(patterns) != vocab_size:
        pattern = list(pattern)
        r.shuffle(pattern)
        pattern = "".join(pattern)
        patterns[pattern] = True
    string_lists = list(patterns)
    int_lists = [[int(char) for char in string] for string in string_lists]
    tensors = torch.tensor(int_lists, device=device, dtype=torch.float)
    return tensors

# Function to convert tensor of indexes to permutation patterns that
# we give as input to our neural network.
def encode_chars(tensor, char_encoding_len, device, encoded_patterns):
    encoded = torch.zeros(*tensor.shape, char_encoding_len).to(device)
    # Iterating over each element in the input tensor to set the
    # corresponding position in the one-hot tensor
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            encoded[i, j] = encoded_patterns[tensor[i, j]]
    encoded = encoded.to(device)
    return encoded

# Toy language model.
# use MLP nn
# forward: N chars in input -> next char prediction.
class MLPLanguageModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_nodes,
                 char_encoding_len,
                 block_size,
                 encoded_patterns,
                 use_batch_norm=True,
                 use_active_fn="relu",  # sigmoid/relu/silu(swiglu)
                 dropout_rate=0.0):
        super().__init__()
        self.encoded_patterns = encoded_patterns

        self.use_batch_norm = use_batch_norm
        self.block_size = block_size
        self.char_encoding_len = char_encoding_len

        input_size = char_encoding_len * block_size
        self.fc1 = nn.Linear(input_size, hidden_nodes)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_nodes)
        self.do1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes // 2)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_nodes // 2)
        self.do2 = nn.Dropout(dropout_rate)

        hidden_nodes //= 2

        self.fc3 = nn.Linear(hidden_nodes, hidden_nodes)
        if use_batch_norm:
            self.bn3 = nn.BatchNorm1d(hidden_nodes)
        self.do3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(hidden_nodes, hidden_nodes // 2)
        if use_batch_norm:
            self.bn4 = nn.BatchNorm1d(hidden_nodes // 2)
        self.do4 = nn.Dropout(dropout_rate)

        hidden_nodes //= 2

        self.fc5 = nn.Linear(hidden_nodes, vocab_size)

        self.active_fn = F.relu
        # https://en.wikipedia.org/wiki/Activation_function
        match use_active_fn:
            case "sigmoid":
                self.active_fn = F.sigmoid
            case "relu":
                self.active_fn = F.relu
            case "silu":
                self.active_fn = F.silu
            case "swiglu":
                self.active_fn = F.silu

    def forward(self, inp, targets=None):
        x = self.fc1(inp)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.active_fn(x)
        x = self.do1(x)

        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.active_fn(x)
        x = self.do2(x)

        x = self.fc3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = self.active_fn(x)
        x = self.do3(x)

        x = self.fc4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = self.active_fn(x)
        x = self.do4(x)

        x = self.fc5(x)
        logits = x

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets.view(-1))

        return logits, loss

    def generate(self, ctx, max_new_tokens):
        output = []
        self.eval()  # Otherwise batch normalization will raise an error.
        for _ in range(max_new_tokens):
            # crop context to the last block_size tokens
            idx_cond = ctx[:, -self.block_size*self.char_encoding_len:]
            # get the predictions
            logits, loss = self(idx_cond)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            output.append(idx_next[0].tolist()[0])
            # append sampled index to the running sequence
            ctx = torch.cat((ctx, self.encoded_patterns[idx_next][0]), dim=1)
        self.train()
        return output
