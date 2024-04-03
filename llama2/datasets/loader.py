from torch.utils.data import DataLoader

from datasets.tinystories.dataloader import PretokDataset as TinyDataset
from datasets.wikipedia_cn.dataloader import PretokDataset as WikipediaCnDataset
from datasets.cosmopedia_stories.dataloader import ChatGLMPretokSftDataset as CosmopediaStoriesDataset


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
