from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from modules.VAE.VQVAE.base import DEVICE, VQVAEBaseModel


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def retrieve_random_codebook(self, random_indices):
        quantized = F.embedding(random_indices, self.embedding)
        quantized = quantized.transpose(1, 3)

        return quantized

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        # 码本损失
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        # 承诺损失
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)

        # vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        # perplexity 困惑度
        avg_probs = torch.mean(encoding_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (
            quantized_latents.permute(0, 3, 1, 2).contiguous(),
            commitment_loss,
            embedding_loss,
            perplexity,
        )  # [B x D x H x W]


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.resblock(input)


@dataclass
class VQVAEModelArgs:
    in_channels: int
    embedding_dim: int
    num_embeddings: int
    hidden_dims: list = None
    beta: float = 0.25
    img_size: int = 64


class VQVAEModel(VQVAEBaseModel):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super(VQVAEModel, self).__init__()
        self.args = VQVAEModelArgs(**kwargs)

        self.embedding_dim = self.args.embedding_dim
        self.num_embeddings = self.args.num_embeddings
        self.img_size = self.args.img_size
        self.beta = self.args.beta

        modules = []
        hidden_dims = self.args.hidden_dims
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        in_channels = self.args.in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, self.args.embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU(),
            )
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(
            self.args.num_embeddings, self.args.embedding_dim, self.beta
        )

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    self.args.embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1
                ),
                nn.LeakyReLU(),
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1
                    ),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1], out_channels=3, kernel_size=4, stride=2, padding=1
                ),
                nn.Tanh(),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.Tensor) Input torch.Tensor to encoder [N x C x H x W]
        :return: torch.Tensor of latent codes
        """
        result = self.encoder(input)
        return result

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.Tensor) [B x D x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: torch.Tensor):
        z = self.encode(input)
        z_quantized, commitment_loss, codebook_loss, perplexity = self.vq_layer(z)
        x_hat = self.decode(z_quantized)
        return x_hat, commitment_loss, codebook_loss, perplexity

    def loss_function(self, x, x_hat, commitment_loss, codebook_loss) -> dict:
        recon_loss = F.mse_loss(x, x_hat)
        loss = recon_loss + commitment_loss * self.beta + codebook_loss
        return loss

    @torch.no_grad()
    def sample(self, batch_size: int):
        random_indices = (
            torch.floor(torch.rand(batch_size // 2, 8, 8) * self.num_embeddings).long().to(DEVICE)
        )

        codes = self.vq_layer.retrieve_random_codebook(random_indices)
        x_hat = self.decoder(codes.to(DEVICE))
        return x_hat

    @torch.no_grad()
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


if __name__ == "__main__":
    input_dim = 3  # in_channels/out_channels
    # hidden_dim = 512
    embedding_dim = 64
    n_embeddings = 512
    img_size = 64  # w:32,h:32
    beta = 0.25

    model = VQVAEModel(
        **VQVAEModelArgs(
            in_channels=input_dim,
            embedding_dim=embedding_dim,
            num_embeddings=n_embeddings,
            img_size=img_size,
            beta=beta,
        ).__dict__
    ).to(DEVICE)
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(model_million_params, "M parameters")
