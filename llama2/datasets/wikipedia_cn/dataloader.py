import glob
import random
import os
from torch import from_numpy
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist
import numpy as np


class PretokDataset(IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, data_dir, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        self.data_dir = data_dir

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(
            f"Created a PretokDataset with rng seed {seed} | worker_id {worker_id}")
        if self.vocab_source == "chatglm":
            # the .bin files are in data directory
            bin_dir = os.path.join(self.data_dir, f"")
        elif self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            bin_dir = os.path.join(self.data_dir, f"tok{self.vocab_size}")
        # see preprocess.py check train/test split percentage
        test_shard_filenames = sorted(
            glob.glob(os.path.join(bin_dir, "*.test.bin")))
        train_shard_filenames = sorted(
            glob.glob(os.path.join(bin_dir, "*.train.bin")))
        if self.split == "train":
            shard_filenames = train_shard_filenames[:]
        else:
            shard_filenames = test_shard_filenames[:]
        assert len(shard_filenames) > 0, f"No bin files found in {bin_dir}"
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y
