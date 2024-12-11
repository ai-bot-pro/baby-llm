import torch
import torch.nn as nn


"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    """
        x -> z = encoder(x)
    将输入数据编码成潜在空间中的概率分布, 这个分布后续会被用来：
    1. 采样生成潜在向量z
    2. 计算KL散度作为正则化项
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        # input_dim -> hidden_dim -> hidden_dim -> latent_dim
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        # 使用LeakyReLU作为激活函数（斜率为0.2）
        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        """
        使用对数方差而不是直接方差的原因：
        - 数值稳定性更好
        - 避免方差为负数
        - 便于后续计算KL散度
        """
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        # encoder produces mean and log of variance
        # (i.e., parateters of simple tractable normal distribution "q"
        # 潜在向量的均值
        mean = self.FC_mean(h_)
        # 潜在向量方差的对数值
        log_var = self.FC_var(h_)

        return mean, log_var


class Decoder(nn.Module):
    """
    x_hat = decoder(z)
    """

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        # 第一层：从潜在维度(latent_dim)映射到隐藏维度(hidden_dim)
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        # 第二层：保持隐藏维度不变
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出层：从隐藏维度映射到输出维度(output_dim，通常等于原始数据维度)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        # 使用LeakyReLU作为中间层的激活函数（避免死亡ReLU问题）
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        # 最后一层使用sigmoid激活函数，将输出压缩到[0,1]区间（适用于图像数据）
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class GaussianMLPVAEModel(nn.Module):
    """
    x -> z = encoder(x) -> x_hat = decoder(z)
    VAE本身是一个编码器-解码器结构的自编码器，只不过编码器的输出是一个分布，而解码器的输入是该分布的一个样本。
    另外，训练时，在损失函数中，除了要让重建图像和原图像更接近以外，还要让输出的分布和标准正态分布更加接近。
    常见的描述分布之间相似度的指标叫做KL散度。只要把KL散度的公式套进损失函数里
    """

    def __init__(self, x_dim, hidden_dim, latent_dim, device=DEVICE):
        super(GaussianMLPVAEModel, self).__init__()
        self.Encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.Decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var

    @torch.no_grad()
    def sample(self, batch_size):
        # 随机采样 bathc_size 个噪声数据
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        generated_images = self.Decoder(noise)
        return generated_images

    @torch.no_grad()
    def generate(self, x, batch_size):
        x = x.view(batch_size, self.x_dim).to(self.device)
        x_hat, _, _ = self.forward(x)
        return x_hat

    def loss_function(self, x, x_hat, mean, log_var) -> torch.Tensor:
        """
        计算变分自编码器的损失函数值。

        该函数将重构损失(reconstruction loss) 和KL散度(Kullback-Leibler Divergence, KLD) 结合起来， 以计算总损失。

        参数:
        - x: 真实数据，用于计算重构损失。
        - x_hat: 重构数据，模型生成的数据。
        - mean: 隐变量的均值。
        - log_var: 隐变量的方差的对数。

        返回:
        - total_loss: 重构损失和KL散度的总和。
        """

        # 计算重构损失，使用二元交叉熵作为损失函数
        reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

        # 计算KL散度，用于正则化损失，避免过拟合
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # 返回总损失，即重构损失和KL散度的和
        return reconstruction_loss + KLD


if __name__ == "__main__":
    # Model Hyperparameters
    x_dim = 784
    hidden_dim = 400
    latent_dim = 200

    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
    model = GaussianMLPVAEModel(Encoder=encoder, Decoder=decoder).to(DEVICE)
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(model_million_params, "M parameters")
