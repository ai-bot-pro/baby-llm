import multiprocessing
import os
import sys
import time
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.RNN.model import DEVICE, RNN, ModelConfig


class CharDataset(Dataset):
    """
    char dataset (map)
    """

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}  # inverse mapping

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1 : 1 + len(ix)] = ix
        y[: len(ix)] = ix
        y[len(ix) + 1 :] = -1  # index -1 will mask the loss at the inactive locations
        return x, y

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1  # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1  # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = "".join(self.itos[i] for i in ix)
        return word


def load_data(input_file):
    # preprocessing of the input text file
    with open(input_file, "r") as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words]  # get rid of any leading or trailing white space
    words = [w for w in words if w]  # get rid of any empty strings
    chars = sorted(list(set("".join(words))))  # all the possible characters
    max_word_length = max(len(w) for w in words)
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print("".join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(
        1000, int(len(words) * 0.1)
    )  # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(
        f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples"
    )

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset


class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=int(1e10)
        )
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except (
            StopIteration
        ):  # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(DEVICE) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()  # reset model back to training mode
    return mean_loss


# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence

        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def print_samples(model, train_dataset, test_dataset, top_k=-1, num=10):
    """samples from the model and pretty prints the decoded samples"""
    X_init = torch.zeros(num, 1, dtype=torch.long).to(DEVICE)
    top_k = top_k if top_k != -1 else None
    steps = (
        train_dataset.get_output_length() - 1
    )  # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to("cpu")
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist()  # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        # separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print("-" * 80)
    for lst, desc in [(train_samples, "in train"), (test_samples, "in test"), (new_samples, "new")]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print("-" * 80)


if __name__ == "__main__":
    model_ckpt_dir = "./datas/models/RNN"
    os.makedirs(model_ckpt_dir, exist_ok=True)
    # system inits Tensorboard
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    work_dir = "./datas/tensorboard/RNN"
    os.makedirs(work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=work_dir)

    # load dataset
    train_ds, test_ds = load_data("./datas/names.txt")
    vocab_size = train_ds.get_vocab_size()
    block_size = train_ds.get_output_length()
    print(f"dataset determined that: {vocab_size}, {block_size}")

    # sample
    top_k = -1  # for sampling, -1 means no top-k

    # train
    batch_size = 32  # batch size during optimization
    learning_rate = 5e-4
    weight_decay = 0.01
    max_steps = -1  # max number of optimization steps to run for, or -1 for infinite.

    # model config
    config = ModelConfig(
        block_size=block_size,  # length of the input sequences of integers
        vocab_size=vocab_size,  # the input integers are in range [0 .. vocab_size -1]
        # parameters below control the sizes of each model slightly differently
        n_layer=4,
        n_embd=64,
        n_embd2=64,
        n_head=4,
    )

    # model
    cell_type = os.getenv("CELL_TYPE", "rnn")  # rnn/gru
    model = RNN(config=config, cell_type=cell_type).to(DEVICE)
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model, DEVICE)
    print(model_million_params, "M parameters")
    model_name = "GaussianMLPVAEModel"
    model_id = (
        f"loss_{model_name}_BA:{batch_size}_PAR:{model_million_params:.2f}_LR:{learning_rate}_name"
    )
    model_filename = model_id + ".pth"
    model_ckpt_path = os.path.join(model_ckpt_dir, model_filename)

    if os.path.exists(model_ckpt_path):
        torch.manual_seed(int(time.time() * 1000))
        model.load_state_dict(torch.load(model_ckpt_path, weights_only=True))
        # sample from decoder with noise vector
        generated_images = model.sample(batch_size)

    if os.getenv("SAMPLE_ONLY", "") == "true":
        print_samples(model, train_ds, test_ds, top_k, num=50)
        sys.exit()

    # ------------train---------------------------
    # init optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    # init dataloader
    batch_loader = InfiniteDataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=True if DEVICE == "cuda" else False,
        num_workers=multiprocessing.cpu_count(),
    )

    # training loop
    best_loss = None
    step = 0
    while True:
        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(DEVICE) for t in batch]
        X, Y = batch

        # feed into the model
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_ds, batch_size=100, max_batches=10)
            test_loss = evaluate(model, test_ds, batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(work_dir, "model.pt")
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(model, train_ds, test_ds, top_k, num=10)

        step += 1
        # termination conditions
        if max_steps >= 0 and step >= max_steps:
            break
