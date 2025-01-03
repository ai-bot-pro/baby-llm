import torch
import time
import os
import random
import numpy as np
import pickle

# hyperparameters
# no config this, just a simple process
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
dropout = 0.2

# mlp
char_encoding_len = 12  # Number of inputs for each character. Must be even.
use_batch_norm = True  # use batch normalization?
max_hidden_nodes = 2048  # Wider (first) hidden layer size of the funnel
active_fn = "relu"  # sigmoid / relu / silu(swiglu)

# attention
n_embd = 384
n_head = 6
n_layer = 6

# moe
num_experts = 8  # experts num
top_k = 2  # top-k experts
nn_init = "kaiming_normal"  # kaiming_normal / xavier_normal
capacity_factor = 0.0  # expert capacity factor
aux_loss_coef = 0.01  # load auxiliary loss coefficient

# block-wise scaling
qkv_multiplier_min = 0.5
qkv_multiplier_max = 1.0
ffn_multiplier_min = 0.5
ffn_multiplier_max = 4.0
ffn_intermediate_divisor = 256

# mla_moeLM
model_config_file = "mla_moe_config.json"

# ------------
# bigramLM / mlpLM / gptLM / moeLM / moa_moeLM / block_wise_scaling_gptLM
model_name = "gptLM"
compile = False  # use PyTorch 2.0 to compile the model to be faster
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
dataset = "shakespeare_char"
resume = False  # resume model checkpoint to train
ckpt_file = "loss.pth"  # resume model checkpoint file

_cur_work_dir = os.path.dirname(os.path.realpath(__file__))
exec(open(f"{_cur_work_dir}/args.py").read())
# change global args by custom --key=value
change_global_args()  # type: ignore # noqa: F821

qkv_multipliers = [qkv_multiplier_min, qkv_multiplier_max]
ffn_multipliers = [ffn_multiplier_min, ffn_multiplier_max]
torch.manual_seed(1337)


# with open(dataset, 'r', encoding='utf-8') as f:
#    print(f"read dataset:{dataset}")
#    text = f.read()
# here are all the unique characters that occur in this text
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# create a mapping from characters to integers
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
#
# Train and test splits
# data = torch.tensor(encode(text), dtype=torch.long)
# n = int(0.9*len(data)) # first 90% will be train, rest val
# train_data = data[:n]
# val_data = data[n:]

# prepare: datasets -> tokenizer --encode--> tokenids (train.bin, val.bin)
data_dir = os.path.join(_cur_work_dir, "datasets", dataset)
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
vocab_size = None
encode = None
decode = None
meta = {}
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {vocab_size} (inside {meta_path})")
    stoi, itos = meta["stoi"], meta["itos"]

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ll):
        return "".join([itos[i] for i in ll])

    if "block_size" in meta:
        block_size = meta[block_size]
else:
    print(f"need tokenizer meta data from {meta_path},please check{meta_path} is exists?")
    exit()

# data loading


def get_batch(split):
    if model_name == "mlpLM":
        return get_mlp_batch(split)
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # x = torch.stack([data[i:i+block_size] for i in ix])
    # y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    if device == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_mlp_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = []
    for i in ix:
        # block = data[i:i+block_size]
        block = torch.from_numpy((data[i : i + block_size]).astype(np.int64))
        char_tensors = [encoded_patterns[idx] for idx in block]
        char_tensors = torch.stack(char_tensors).view(-1)
        x.append(char_tensors)
    x = torch.stack(x)
    # y = torch.stack([data[i+block_size:i+block_size+1] for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[i + block_size : i + block_size + 1]).astype(np.int64)) for i in ix]
    )
    x, y = x.to(device), y.to(device)

    # For 1/4 of our batches, set the first N random elements in 'x' to
    # zero, so that the network learn how to start a sequence from
    # an incomplete prompt.
    num_batches_to_modify = batch_size // 4
    for batch_index in range(num_batches_to_modify):
        N = random.randint(1, block_size)
        x[batch_index, :N] = 0

    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = None
match model_name:
    case "bigramLM":
        from bigramLM import BigramLanguageModel

        model = BigramLanguageModel(vocab_size)
    case "mlpLM":
        from mlpLM import MLPLanguageModel, gen_coding_patterns

        # Generate the patters for the inputs encoding
        encoded_patterns = gen_coding_patterns(char_encoding_len, vocab_size, device)
        model = MLPLanguageModel(
            vocab_size,
            max_hidden_nodes,
            char_encoding_len,
            block_size,
            encoded_patterns,
            use_batch_norm=use_batch_norm,
            use_active_fn=active_fn,
            dropout_rate=dropout,
        )
    case "gptLM":
        from gptLM import GPTLanguageModel

        model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout)
    case "block_wise_scaling_gptLM":
        from block_wise_scaling_gptLM import DelightGPTLanguageModel

        model = DelightGPTLanguageModel(
            vocab_size,
            n_embd,
            block_size,
            n_layer,
            n_head,
            dropout,
            qkv_multipliers,
            ffn_multipliers,
            ffn_intermediate_divisor,
        )
    case "moeLM":
        from moeLM import SparseMoELanguageModel

        model = SparseMoELanguageModel(
            vocab_size,
            n_head,
            num_experts,
            top_k,
            n_layer,
            n_embd,
            block_size,
            dropout,
            nn_init=nn_init,
        )
    case "moa_moeLM":
        from moa_moeLM import SparseMoAMoELanguageModel

        model = SparseMoAMoELanguageModel(
            vocab_size,
            n_head,
            num_experts,
            top_k,
            n_layer,
            n_embd,
            block_size,
            dropout,
            nn_init=nn_init,
            capacity_factor=capacity_factor,
            aux_loss_coef=aux_loss_coef,
        )
    case "mla_moeLM":
        from mla_moeLM import MlaSparseMoELanguageModel, ModelArgs
        import json

        model_config = {}
        if os.path.exists(model_config_file):
            with open(model_config_file, "r") as f:
                model_config = json.load(f)
                print(json.dumps(model_config, indent=4, sort_keys=True))
        args = ModelArgs(**model_config)
        args.vocab_size = vocab_size
        args.max_seq_len = block_size
        print(f"model:{model_name} args:{args}")
        model = MlaSparseMoELanguageModel(args, nn_init=nn_init)

if model is None:
    raise ValueError("Unknown model name")

m = model.to(device)
# print the number of parameters in the model
model_million_params = sum(p.numel() for p in m.parameters()) / 1e6
print(m)
print(model_million_params, "M parameters")

# If argument resume is set, load the model checkpoint
# and generate some text with it.
if resume is True:
    torch.manual_seed(int(time.time() * 1000))
    m.load_state_dict(torch.load(ckpt_file))
    if model_name == "mlpLM":
        context = torch.zeros(
            (1, (block_size * char_encoding_len)), dtype=torch.float, device=device
        )
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=1024)).strip())
    # exit(0)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# compile the model
if compile and model_name in ["gptLM", "moeLM"]:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# Log the loss in some target file
if model_name == "mlpLM":
    model_id = f"loss_{model_name}_BA:{batch_size}_BL:{block_size}_PAR:{model_million_params:.2f}_E:{char_encoding_len}_V:{vocab_size}_AF:{active_fn}_BN:{use_batch_norm}_LR:{learning_rate}_DR:{dropout}_{os.path.basename(dataset)}"
else:
    model_id = f"loss_{model_name}_BA:{batch_size}_BL:{block_size}_PAR:{model_million_params:.2f}_V:{vocab_size}_LR:{learning_rate}_DR:{dropout}_{os.path.basename(dataset)}"

model_filename = model_id + ".pth"
log_name = model_id + ".log"

# If a model with this parameters was already trained, don't overwrite
# the weights and loss log.
if os.path.exists(model_filename):
    # sys.exit(f"Pretrained weights found for this model: {model_filename}. If you want to proceed remove the file.")
    model_filename = "resume_" + model_filename
    log_name = "resume_" + log_name
    print(f"Pretrained weights,log resume write to {model_filename} .")

loss_file = open(log_name, "w")
print("Logging to", log_name)


model.train()
minloss = 10  # Track minimum validation loss found so far.
iter_duration = 0  # iter time
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        best_so_far = losses["val"] < minloss
        minloss = min(minloss, losses["val"])
        print(
            f">>> step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, min loss {minloss:.4f}, {iter_duration*1000:.2f} ms per step"
        )
        if iter > 0:
            loss_file.write(f"{iter} {losses['train']:.4f} {losses['val']:.4f}\n")
            loss_file.flush()

        if model_name == "mlpLM":
            context = torch.zeros(
                (1, (block_size * char_encoding_len)), dtype=torch.float, device=device
            )
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)

        if best_so_far:
            # generate from the model
            print(decode(m.generate(context, max_new_tokens=200)).strip())
            torch.save(m.state_dict(), model_filename)
            print("Saving model ", model_filename)

    iter_start_time = time.time()

    # sample a batch of data
    xb, yb = get_batch("train")
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    iter_duration = time.time() - iter_start_time
