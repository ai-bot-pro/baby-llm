from dataclasses import dataclass
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class VQVAEModelTrainArgs:
    # train
    learning_rate: float = 2e-4
    epochs: int = 50


class VQVAEBaseModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def loss_function(self, x, x_hat, commitment_loss, codebook_loss) -> torch.Tensor:
        pass

    def sample(self, batch_size):
        pass

    def generate(self, x):
        pass
