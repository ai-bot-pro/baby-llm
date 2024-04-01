from datasets.tinystories.dataloader import PretokDataset as TinyDataset
from datasets.wikipedia_cn.dataloader import PretokDataset as WikipediaCnDataset
from torch.utils.data import DataLoader


class Task:
    def __init__(self) -> None:
        r"""
        init task to register customer datasets
        """
        self.datasetClass = {
            "tinystories": TinyDataset,
            "wikipedia_cn": WikipediaCnDataset,
            # just merge all datasets into one to train
        }

    @classmethod
    def iter_batches(self, batch_size, device, num_workers=0, dataset_name="tinystories", **dataset_kwargs):
        ds = self.datasetClass[dataset_name](**dataset_kwargs)
        dl = DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y
