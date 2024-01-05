import torch
from bigramLM import BigramLanguageModel
import sys,time,os

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

char_encoding_len = 12 # Number of inputs for each character. Must be even.
use_batch_norm = False # use batch normalization?
max_hidden_nodes = 2048 # Wider (first) hidden layer size of the funnel
active_fn = "relu" # sigmoid / relu / silu(swiglu)
# ------------
model_name="bigramLM" # bigramLM / mlpLM / gptLM

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
filename = 'input.txt' if len(sys.argv) < 2 else sys.argv[1]
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
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
        from mlpLM import MLPLanguageModel
        model = MLPLanguageModel(vocab_size, max_hidden_nodes,
                                 char_encoding_len, block_size,
                                 use_batch_norm=use_batch_norm,
                                 use_active_fn=active_fn,
                                 dropout_rate=dropout)
    case "gptLM":
        from gptLM import GPTLanguageModel
        model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head)
if model is None:
    raise ValueError("Unknown model name")

m = model.to(device)
# print the number of parameters in the model
model_million_params = sum(p.numel() for p in m.parameters())/1e6
print(m)
print(model_million_params, 'M parameters')

# If the second argument is a model name, we just load the model
# and generate some text with it.
if len(sys.argv) == 3:
    torch.manual_seed(int(time.time()*1000))
    m.load_state_dict(torch.load(sys.argv[2]))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    #context = torch.zeros((1,(block_size*char_encoding_len)), dtype=torch.float, device=device)
    #print(decode(m.generate(context, max_new_tokens=1024)).strip())
    exit(0)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Log the loss in some target file
model_id = f"loss_BA:{batch_size}_BL:{block_size}_PAR:{model_million_params:.2f}_E:{char_encoding_len}_V:{vocab_size}_BN:{use_batch_norm}_LR:{learning_rate}_DR:{dropout}_{os.path.basename(filename)}"
model_filename = model_id+".pth"

# If a model with this parameters was already trained, don't overwrite
# the weights and loss log.
if os.path.exists(model_filename):
    sys.exit(f"Pretrained weights found for this model: {model_filename}. If you want to proceed remove the file.")

loss_file = open(model_id,'w')
print("Logging to", model_id)

minloss = 10 # Track minimum validation loss found so far.
iter_duration = 0 # iter time
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        best_so_far = losses['val'] < minloss
        minloss = min(minloss,losses['val'])
        print(f">>> step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, min loss {minloss:.4f}, {iter_duration*1000:.2f} ms per step")
        if iter > 0:
            loss_file.write(f"{iter} {losses['train']:.4f} {losses['val']:.4f}\n")
            loss_file.flush()

        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        #context = torch.zeros((1,(block_size*char_encoding_len)), dtype=torch.float, device=device)
        if best_so_far:
            # generate from the model
            print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
            #print(decode(m.generate(context, max_new_tokens=200)).strip())
            torch.save(m.state_dict(),model_filename)
            print("Saving model ",model_filename)

    iter_start_time = time.time()

    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    iter_duration = time.time() - iter_start_time
