from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.VAE.VQVAE.base import DEVICE, VQVAEBaseModel


"""
向量量化部分代码控制流程：

flowchart TD
    A[开始] --> B[初始化参数]
    B --> C[创建并初始化嵌入向量]
    C --> D[初始化 EMA 计数器和权重]
    D --> E[结束初始化]

    subgraph 编码过程
    F[输入张量 x] --> G[展平 x 并计算距离]
    G --> H[找到最近的嵌入向量索引]
    H --> I[返回量化向量及其索引]
    end

    subgraph 随机获取嵌入向量
    J[输入随机索引] --> K[获取对应嵌入向量]
    K --> L[调整维度顺序]
    L --> M[返回量化向量]
    end

    subgraph 前向传播
    N[输入张量 x] --> O[展平 x 并计算距离]
    O --> P[找到最近的嵌入向量索引]
    P --> Q[更新 EMA 计数器和权重]
    Q --> R[计算量化损失、承诺损失和困惑度]
    R --> S[返回结果]
    end
"""


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=(4, 4, 3, 1), stride=2):
        super(Encoder, self).__init__()

        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_size

        self.strided_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, stride, padding=1)
        self.strided_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, stride, padding=1)

        self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_3, padding=1)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_4, padding=0)

        self.proj = nn.Conv2d(
            hidden_dim, output_dim, kernel_size=1
        )  # nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.strided_conv_1(x)
        x = self.strided_conv_2(x)

        x = F.relu(x)
        y = self.residual_conv_1(x)
        y = y + x

        x = F.relu(y)
        y = self.residual_conv_2(x)
        y = y + x

        y = self.proj(y)
        return y


class VQEmbeddingEMA(nn.Module):
    """
    实现了一个使用指数移动平均(EMA)更新的向量量化嵌入层
    """

    def __init__(
        self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5
    ):
        super(VQEmbeddingEMA, self).__init__()
        # 承诺损失的权重系数
        self.commitment_cost = commitment_cost
        # EMA更新的衰减率
        self.decay = decay
        # 用于数值稳定性的小常数
        self.epsilon = epsilon

        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        # 计算输入向量与码本中所有向量的距离
        distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2
        # 选择最近的码本向量的索引
        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))

    def retrieve_random_codebook(self, random_indices):
        quantized = F.embedding(random_indices, self.embedding)
        quantized = quantized.transpose(1, 3)

        return quantized

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2
        # 1. 向量量化
        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        # 2. 码本更新（训练时）
        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(
                encodings, dim=0
            )
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        # 3. 损失计算
        # 码本损失
        codebook_loss = F.mse_loss(x.detach(), quantized)
        # 承诺损失
        e_latent_loss = F.mse_loss(x, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        # perplexity 困惑度
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, commitment_loss, codebook_loss, perplexity


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=(1, 3, 2, 2), stride=2):
        super(Decoder, self).__init__()

        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_sizes

        self.in_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_1, padding=0)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, padding=1)

        self.strided_t_conv_1 = nn.ConvTranspose2d(
            hidden_dim, hidden_dim, kernel_3, stride, padding=0
        )
        self.strided_t_conv_2 = nn.ConvTranspose2d(
            hidden_dim, output_dim, kernel_4, stride, padding=0
        )

    def forward(self, x):
        x = self.in_proj(x)

        y = self.residual_conv_1(x)
        y = y + x
        x = F.relu(y)

        y = self.residual_conv_2(x)
        y = y + x
        y = F.relu(y)

        y = self.strided_t_conv_1(y)
        y = self.strided_t_conv_2(y)

        return y


@dataclass
class VQVAEModelArgs:
    input_dim: int
    hidden_dim: int
    latent_dim: int
    n_embeddings: int
    output_dim: int
    commitment_beta: float = 0.25



class VQVAEModel(VQVAEBaseModel):
    def __init__(
        self,
        **kwargs,
    ):
        self.args = VQVAEModelArgs(**kwargs)
        super(VQVAEModel, self).__init__()
        self.encoder = Encoder(
            input_dim=self.args.input_dim,
            hidden_dim=self.args.hidden_dim,
            output_dim=self.args.latent_dim,
        )
        self.codebook = VQEmbeddingEMA(
            n_embeddings=self.args.n_embeddings,
            embedding_dim=self.args.latent_dim,
        )
        self.decoder = Decoder(
            input_dim=self.args.latent_dim,
            hidden_dim=self.args.hidden_dim,
            output_dim=self.args.output_dim,
        )
        self.mse_loss = nn.MSELoss()
        self.commitment_beta = self.args.commitment_beta
        self.n_embeddings = self.args.n_embeddings

    def forward(self, x):
        z = self.encoder(x)
        z_quantized, commitment_loss, codebook_loss, perplexity = self.codebook(z)
        x_hat = self.decoder(z_quantized)

        return x_hat, commitment_loss, codebook_loss, perplexity

    def loss_function(self, x, x_hat, commitment_loss, codebook_loss) -> torch.Tensor:
        recon_loss = self.mse_loss(x, x_hat)
        loss = recon_loss + commitment_loss * self.commitment_beta + codebook_loss
        return loss

    @torch.no_grad()
    def sample(self, batch_size):
        random_indices = (
            torch.floor(torch.rand(batch_size // 2, 8, 8) * self.n_embeddings).long().to(DEVICE)
        )

        codes = self.codebook.retrieve_random_codebook(random_indices)
        x_hat = self.decoder(codes.to(DEVICE))
        return x_hat

    @torch.no_grad()
    def generate(self, x):
        x_hat, _, _, _ = self.forward(x)
        return x_hat


if __name__ == "__main__":
    input_dim = 3
    hidden_dim = 512
    latent_dim = 16
    n_embeddings = 512
    output_dim = 3

    model = VQVAEModel(
        **VQVAEModelArgs(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_embeddings=n_embeddings,
            output_dim=output_dim,
        ).__dict__
    ).to(DEVICE)
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(model_million_params, "M parameters")
