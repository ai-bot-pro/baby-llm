import sys
import os

from torch.utils.data import DataLoader

from tinyshakespeare.dataloader import PretokDataset as TinyShakespeareDataset
from tinystories.dataloader import PretokDataset as TinyDataset
from wikipedia_cn.dataloader import PretokDataset as WikipediaCnDataset
from cosmopedia_stories.dataloader import ChatGLMPretokSftDataset as CosmopediaStoriesDataset


class Task:
    r"""
    use pytorch DataLoader to load data, for pre-training, sft, rl(reward)
    pytorch dataloader is a wrapper of iterable/getitem dataset, which is a wrapper of iterable/getitem sampler
    see: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """

    def __init__(self) -> None:
        r"""
        init task to register customer datasets
        """
        # pre-training datasets, just split text files
        # if use only one file, u need set num_workers=0
        # use iter stream to load data
        self.datasetClass = {
            "tinyshakespeare": TinyShakespeareDataset,
            "tinystories": TinyDataset,
            "wikipedia_cn": WikipediaCnDataset,
            # just merge all datasets into one to train
        }

        # sft(supervised fine-tuning) datasets, only one struct file
        # just getitem to get data
        self.sftDatasetClass = {
            "cosmopedia_stories": CosmopediaStoriesDataset,
            # just merge all datasets into one to train
        }

    def iter_batches(self, batch_size, device, num_workers=0, dataset_name="tinystories", **dataset_kwargs):
        assert dataset_name in self.datasetClass, f"{dataset_name} pre-training dataset don't support"
        ds = self.datasetClass[dataset_name](**dataset_kwargs)
        num_workers = 0 if dataset_name == "wikipedia_cn" else num_workers
        dl = DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

    def sft_getitem_batches(self, batch_size, device, num_workers=0, dataset_name="", **dataset_kwargs):
        assert dataset_name in self.sftDatasetClass, f"{dataset_name} sft dataset don't support"
        ds = self.sftDatasetClass[dataset_name](**dataset_kwargs)
        dl = DataLoader(
            ds, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True, num_workers=num_workers
        )
        for step, (x, y, loss_mask) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)
            yield x, y, loss_mask


class VisionDataloaderTask:
    r"""
    use torchvision DataLoader to load data, for pre-training, sft, rl(reward)
    torchvision dataloader is a wrapper of iterable dataset, which is a wrapper of iterable sampler
    see: https://pytorch.org/vision/stable/datasets.html
    """


class AudioDataloaderTask:
    r"""
    use torchaudio DataLoader to load data, for pre-training, sft, rl(reward)
    torchaudio dataloader is a wrapper of iterable dataset, which is a wrapper of iterable sampler
    see: https://pytorch.org/audio/stable/index.html
    """


class VideoDataloaderTask:
    r"""
    use pytorchvideo DataLoader to load data, for pre-training, sft, rl(reward)
    pytorchvideo dataloader is a wrapper of iterable dataset, which is a wrapper of iterable sampler
    see: https://pytorchvideo.readthedocs.io/en/latest/data.html
    """


class HFDataloaderTask:
    r"""
    use huggingface DataLoader to load data, for pre-training, sft, rl(reward)
    see: https://huggingface.co/docs/datasets/index
    """


class CuDFDataloaderTask:
    r"""
    use  DataLoader to load data, for pre-training, sft, rl(reward)
    see: https://docs.rapids.ai/api/cudf/stable/
    """


def task_datasetClass(data_dir,
                      vocab_size, vocab_source="custom",
                      dataset_name="tinystories", batch_size=1,
                      max_seq_len=512, device="cpu"):
    iter_batches = partial(
        Task().iter_batches,
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        vocab_source=vocab_source,
        device=device,
        num_workers=0,
    )
    train_batch_iter = iter_batches(split="train")
    X, Y = next(train_batch_iter)  # fetch the very first batch
    print(X.shape, Y.shape)
    print(X[0])
    print(Y[0])


def task_sftDatasetClass(csv_file_path,
                         prompt_max_len=128, text_max_len=128,
                         dataset_name="cosmopedia_stories", batch_size=1,
                         max_seq_len=512, device="cpu"):
    iter_batches = partial(
        Task().sft_getitem_batches,
        dataset_name=dataset_name,
        batch_size=batch_size,
        device=device,
        num_workers=0,
        max_seq_len=max_seq_len,
        prompt_max_len=prompt_max_len,
        text_max_len=text_max_len,
    )
    train_batch_iter = iter_batches(split="train", csv_file_path=csv_file_path)
    X, Y, loss_mask = next(train_batch_iter)  # fetch the very first batch
    print(X.shape, Y.shape, loss_mask.shape)
    print(X[0])
    print(Y[0])
    print(loss_mask[0])


if __name__ == "__main__":
    import argparse
    from functools import partial

    r"""
    python3 ./llama2/datasets/loader.py task_datasetClass --dataset_name=tinyshakespeare --vocab_size=323 --data_dir=./datas/datasets
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=[
                        "task_datasetClass", "task_sftDatasetClass"])
    parser.add_argument("-dn", "--dataset_name", type=str, default="",
                        help="dataset_name from datasets dir")
    parser.add_argument("-vsrc", "--vocab_source", type=str, default="",
                        help="vocab_source")
    parser.add_argument("-vs", "--vocab_size", type=int, default=4096,
                        help="vocab size")
    parser.add_argument("-d", "--data_dir", type=str, default="",
                        help="vocab data dir")
    parser.add_argument("-cfp", "--csv_file_path", type=str, default="",
                        help="csv_file_path")

    args = parser.parse_args()
    print(args)

    if args.stage == "task_datasetClass":
        task_datasetClass(args.data_dir, args.vocab_size,
                          dataset_name=args.dataset_name, vocab_source=args.vocab_source)
    elif args.stage == "task_sftDatasetClass":
        task_sftDatasetClass(args.csv_file_path)
