"""
### Full Fine-Tuning (FFT)
Adjusts all parameters of the LLM using task-specific data.
like pre-training, but resume ckpt.pt;
- use like prompt-generate_text supervised datasets with tokenizer(sp bpe)
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

from _datasets.loader import Task
from export import model_export

# -----------------------------------------------------------------------------
dataset_name = "cosmopedia_stories"  # tokenized dataset name
csv_file_path = ""  # csv datasets file path
data_dir = "./datas"  # tokenizer datasets dir
# I/O
out_dir = "out"
eval_interval = 5000
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
_init_from = "resume"

# data
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256
prompt_max_len = 256
text_max_len = 256

# chatglm|custom; use chatglm vocab or custom trained
vocab_source = "chatglm"
vocab_size = 64793  # the chatglm tokenizer has 64793 tokens

# model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
# adamw optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = True  # use PyTorch 2.0 to compile the model to be faster

# wandb logging
wandb_log = False  # disabled by default
wandb_project = "baby_llm_llama2"
wandb_run_name_suffix = ""
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + wandb_run_name_suffix

# huggingface upload training results
hf_upload = False  # disabled by default
repo_id = "weege007/babyllm"
hf_models_dir = "/models"

# estimate loss datasets
# defualt "", use: "train,val" | "train" | "val"
estimate_loss_split_datasets = ""

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]

_cur_work_dir = os.path.dirname(os.path.realpath(__file__))
exec(open(f"{_cur_work_dir}/args.py").read())
# change global args by custom --key=value
# overrides from command line
change_global_args()  # type: ignore # noqa: F821
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

estimate_loss_datasets = (
    [] if len(estimate_loss_split_datasets) == 0 else estimate_loss_split_datasets.split(",")
)

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# validating checks
assert vocab_source in ["chatglm", "custom"]
assert vocab_source == "custom" and vocab_size > 0, "The custom vocab_size > 0 "
assert vocab_source == "chatglm" and vocab_size == 64793, "The vocab from chatGLM has 64793 tokens"

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    # this process will do logging, checkpointing etc.
    master_process = ddp_rank == 0
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(
        f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len"
    )

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
# for later use in torch.autocast
device_type = "cuda" if "cuda" in device else "cpu"
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else autocast(device_type=device_type, dtype=ptdtype)

# task-specific setup
iter_batches = partial(
    Task().sft_getitem_batches,
    dataset_name=dataset_name,
    data_dir=data_dir,
    batch_size=batch_size,
    device=device,
    num_workers=0,
    max_seq_len=max_seq_len,
    prompt_max_len=prompt_max_len,
    text_max_len=text_max_len,
    csv_file_path=csv_file_path,
)

# init these up here, can override if _init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)  # start with model_args from command line
if _init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
elif _init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in [
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "vocab_size",
        "multiple_of",
        "max_seq_len",
    ]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
m = model.to(device)
# print the number of parameters in the model
model_million_params = sum(p.numel() for p in m.parameters()) / 1e6
print(m)
print(f"{model_million_params}M parameters")

# initialize a GradScaler. If enabled=False scaler is a no-op
# torch.cuda.amp.GradScaler 是一个用于自动混合精度训练的 PyTorch 工具，它可以帮助加速模型训练并减少显存使用量。具体来说，GradScaler 可以将梯度缩放到较小的范围，以避免数值下溢或溢出的问题，同时保持足够的精度以避免模型的性能下降。
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer AdamW
# 见论文：Decoupled Weight Decay Regularization
# https://arxiv.org/abs/1711.05101
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if _init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
# torch.compile 通过 JIT 将 PyTorch 代码编译成优化的内核，使 PyTorch 代码运行得更快
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches


@torch.no_grad()
def estimate_loss(estimate_loss_datasets):
    out = {"train": 0.0, "val": 0.0}
    model.eval()
    # 分别对训练集和验证集进行估计
    for split in estimate_loss_datasets:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        # 遍历每批次模型的losses, 对每批次losses算均值loss
        for k in range(eval_iters):
            X, Y, loss_mask = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    r"""
    learning rate decay scheduler (cosine with warmup)
    https://zh.d2l.ai/chapter_optimization/lr-scheduler.html#id8
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def push_to_hf_hub(repo_id, model_path_or_fileobj, hf_models_dir, model_million_params):
    print(f"upload checkpoint to {repo_id}")
    upload_file(
        path_or_fileobj=model_path_or_fileobj,
        path_in_repo=os.path.join(hf_models_dir, f"{model_million_params}M.pt"),
        repo_id=repo_id,
    )


# logging
# https://docs.wandb.ai/quickstart
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
train_batch_iter = iter_batches(split="train")
X, Y, loss_mask = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if len(estimate_loss_datasets) > 0 and iter_num % eval_interval == 0 and master_process:
        # train训练数据用于参数（权重和偏置）的学习，
        # val验证数据用于参数的性能评估
        losses = estimate_loss(estimate_loss_datasets)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "loss/train": losses["train"],
                        "loss/val": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    },
                    step=iter_num,
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")
        # 验证集 loss值变小则保存模型
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                model_export(raw_model, os.path.join(out_dir, "model.bin"))
                if hf_upload:
                    from huggingface_hub import upload_file

                    # popen
                    with Pool() as p:
                        p.apply_async(
                            push_to_hf_hub,
                            (
                                repo_id,
                                os.path.join(out_dir, "ckpt.pt"),
                                hf_models_dir,
                                model_million_params,
                            ),
                        )
                        p.apply_async(
                            push_to_hf_hub,
                            (
                                repo_id,
                                os.path.join(out_dir, "model.bin"),
                                hf_models_dir,
                                model_million_params,
                            ),
                        )
                        p.close()
                        p.join()

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            logits = model(X, Y)
            loss = raw_model.last_loss
            if loss_mask:
                loss_mask = loss_mask.view(-1)
                loss = torch.sum(loss * loss_mask) / loss_mask.sum()

            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, loss_mask = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        # 调用 GradScaler 的 backward() 方法计算梯度并缩放
        # 根据前向传播计算的loss值， 反向传播 更新模型参数权重
        # 例子看下这个：https://zhuanlan.zhihu.com/p/447113449
        scaler.scale(loss).backward()
    # clip the gradient
    # 梯度裁剪
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    # 调用 scaler.step(optimizer) 来更新模型参数
    scaler.step(optimizer)
    # 使用 scaler.update() 更新 GradScaler 对象的内部状态
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
