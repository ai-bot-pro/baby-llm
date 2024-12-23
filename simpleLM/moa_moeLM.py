r"""
from:
- "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity": https://arxiv.org/pdf/2101.03961.pdf
- "ModuleFormer: Modularity Emerges from Mixture-of-Experts": https://arxiv.org/pdf/2306.04640.pdf

- use THE SPARSELY-GATED MIXTURE-OF-EXPERTS ATTENTION LAYER
- use THE SPARSELY-GATED MIXTURE-OF-EXPERTS(mlp) LAYER
"""


import torch
import torch.nn as nn
from torch.nn import functional as F


class SparseMoEMultiHeadAttention(nn.Module):
    """
    spare moe + multiple heads of self-attention in parallel
    more detail see: Attention Expert from JetMoE: Reaching Llama2 Performance with 0.1M Dollars
    https://arxiv.org/pdf/2404.07413.pdf
    """

    def __init__(
        self,
        num_heads,
        head_size,
        n_embed,
        block_size,
        dropout,
        num_experts=8,
        top_k=2,
        reduce_bias=True,
    ):
        super(SparseMoEMultiHeadAttention, self).__init__()

        # 偏置是可学习的参数，通常用于线性层（如全连接层）和卷积层中: a = Wx + Bias
        # 模型中引入偏置项，有助于模型更好地拟合训练数据和提高模型的表达能力
        # 在训练过程中，模型会通过梯度下降等优化算法自动学习到合适的偏置值，从而使模型的预测更准确。
        self.p_reduce_bias = None
        if reduce_bias:
            self.p_reduce_bias = torch.nn.Parameter(torch.empty(n_embed))
            torch.nn.init.zeros_(self.p_reduce_bias)

        self.n_embed = n_embed
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_experts = num_experts
        self.top_k = min(top_k, self.num_experts)

        assert self.top_k > 0, "topk must > 0"
        assert self.num_heads > 0, "num_heads must > 0"
        assert num_heads % self.top_k == 0, f"need num_heads:{num_heads}%top_k:{self.top_k} == 0"

        # num_heads = topk * num_key_val_heads
        # kv_proj_size = num_key_val_heads * head_size
        # num_heads * head_size = topk * kv_proj_size
        self.num_key_val_heads = int(num_heads / top_k)
        self.kv_proj_size = self.num_key_val_heads * head_size

        self.input_linear = ParallelExperts(num_experts, n_embed, self.kv_proj_size)
        self.output_linear = ParallelExperts(num_experts, self.kv_proj_size, n_embed)

        self.router = NoisyTopkRouter(n_embed, num_experts, self.top_k)

        self.k_proj = torch.nn.Linear(n_embed, self.kv_proj_size, bias=False)
        self.v_proj = torch.nn.Linear(n_embed, self.kv_proj_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # B:bsz, S:seq_len=block_size, C:feat_dim=n_embed
        bsz, seq_len, feat_dim = x.size()

        # H:num_heads, kvH:num_key_val_heads, D:head_size
        query_states = self.map(x)  # B S H*D
        key_states = self.k_proj(x)  # B S kvH*D
        value_states = self.v_proj(x)  # B S kvH*D

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_size).transpose(
            1, 2
        )  # B H S D
        key_states = key_states.view(
            bsz, seq_len, self.num_key_val_heads, self.head_size
        ).transpose(1, 2)  # B kvH S D
        value_states = value_states.view(
            bsz, seq_len, self.num_key_val_heads, self.head_size
        ).transpose(1, 2)  # B kvH S D

        # repeat k/v heads if num_key_val_heads < num_heads, it's true
        key_states = key_states.repeat(1, self.top_k, 1, 1)  # B H S D
        value_states = value_states.repeat(1, self.top_k, 1, 1)  # B H S D

        # (B H S D) @ (B H D S) * D**-0.5 -> (B H S S)
        attn_weights = query_states @ key_states.transpose(2, 3) * self.head_size**-0.5
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_size)

        # check attention weights shape
        if attn_weights.size() != (bsz, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # cuasle sequence masked fill with -inf
        attn_weights = attn_weights.masked_fill(
            self.tril[:seq_len, :seq_len] == 0, float("-inf")
        )  # (B H S S)

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # dropout, if trainning loss have some overfit happen, open it
        attn_weights = self.dropout(attn_weights)
        # attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # (B H S S) @ (B H S D) -> (B H S D)
        attn_output = attn_weights @ value_states
        # attn_output = torch.matmul(attn_weights, value_states)

        # check attention output shape
        if attn_output.size() != (bsz, self.num_heads, seq_len, self.head_size):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, seq_len, self.head_size)}, but is"
                f" {attn_output.size()}"
            )

        # 内存连续的张量意味着张量的元素在内存中是按照其在张量中的顺序连续存储的，没有间隔
        # 调用一些需要连续张量作为输入的函数(reshape)时可能会引发错误。
        # 因此，在执行一些操作之前，需要确保张量是连续的
        attn_output = attn_output.transpose(1, 2).contiguous()  # B S H D
        # num_heads(H) * head_size(D) = topk * kv_proj_size
        attn_output = attn_output.reshape(
            bsz, seq_len, self.top_k, self.kv_proj_size
        )  # B S topk kv_proj_size

        attn_output = self.reduce(attn_output)
        attn_output = attn_output.view(bsz, seq_len, -1)

        attn_output = self.dropout(attn_output)

        return attn_output
        # return attn_output, attn_weights

    def map(self, x):
        # 解析输入张量的形状，获取批次大小（bsz）、序列长度（length）和输入特征维度（emb_size）。
        bsz, length, emb_size = x.size()
        # 将输入张量 x 重新整形为二维张量，形状为 (bsz * length, emb_size)，以便进行批次级别的处理。
        x = x.reshape(-1, emb_size)
        # 调用 compute_gate 方法计算门控损失。
        self.compute_gate(x)

        # 根据 batch_index 提取每个样本所属的专家输入，形状为 (num_experts, expert_size)。
        expert_inputs = x[self.batch_index]
        # 将专家输入传递给 input_linear 层，使用专家大小信息进行线性变换，得到专家输出。
        expert_outputs = self.input_linear(expert_inputs, self.expert_size)

        # 创建一个全零张量 zeros，形状为 (bsz * length * top_k, kv_proj_size)，数据类型和设备与 expert_outputs 相同。
        zeros = torch.zeros(
            (bsz * length * self.top_k, self.kv_proj_size),
            dtype=expert_outputs.dtype,
            device=expert_outputs.device,
        )
        # 使用 index_add 方法将专家输出根据 index_sorted_experts 分散到全零张量 zeros 中，得到混合输出张量 y。
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        # 将混合输出张量 y 重新整形为四维张量，形状为(bsz, length, top_k, kv_proj_size)。
        y = y.view(bsz, length, self.top_k, -1)
        return y

    def reduce(self, x: torch.Tensor):
        # 解析输入张量的形状，获取批次大小（bsz）、序列长度（length）、专家数量（k）和嵌入维度（emb_size=kv_proj_size）。
        bsz, length, k, emb_size = x.size()
        # 将输入张量 x 重新整形为二维张量，形状为 (bsz * length * k, emb_size)。
        x = x.reshape(-1, emb_size)

        # 根据 index_sorted_experts 提取每个样本所属的专家输入，形状为 (num_experts, expert_size)。
        expert_inputs = x[self.index_sorted_experts]
        # 将专家输入传递给 output_linear 层，使用专家大小信息进行线性变换，得到专家输出。
        expert_outputs = self.output_linear(expert_inputs, self.expert_size)

        # 将专家输出乘以对应的门控值。
        expert_outputs = expert_outputs * self.batch_gates[:, None]

        # 创建一个全零张量 zeros，形状为 (bsz * length, n_embed)，数据类型和设备与 expert_outputs 相同。
        zeros = torch.zeros(
            (bsz * length, self.n_embed), dtype=expert_outputs.dtype, device=expert_outputs.device
        )
        # 使用 index_add 方法将乘以门控值的专家输出张量根据 batch_index 分散到全零张量 zeros 中，得到降维后的输出张量 y。
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        # 将降维后的输出张量 y 重新整形为三维张量，形状为 (bsz, length, n_embed)。
        y = y.view(bsz, length, self.n_embed)
        # 如果设置了偏置项，则将偏置项添加到输出张量 y 中。
        if self.p_reduce_bias is not None:
            y = y + self.p_reduce_bias
        return y

    def compute_gate(self, x):
        self.top_k_gates, top_k_indices = self.router(x)

        self.batch_gates, self.batch_index, expert_size, self.index_sorted_experts = compute_gating(
            self.top_k, self.num_experts, self.top_k_gates, top_k_indices
        )
        self.expert_size = expert_size.tolist()


class Expert(nn.Module):
    # Expert module
    """An MLP is a simple linear layer followed by a non-linearity i.e. each Expert"""

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size) -> None:
        """
        Initialize the ParallelExperts module.
        like a Expert pool
        maybe manager diff export pool for feature to load :)

        Args:
            num_experts (int): Number of experts.
            input_size (int): Size of the input.
            output_size (int): Size of the output.
            bias (bool): Whether to include bias terms.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        self.reset_parameters()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self):
        return "num_experts={}, input_size={}, output_size={}".format(
            self.num_experts, self.input_size, self.output_size
        )

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the model.
        """
        nn.init.uniform_(self.weight, -1.0 / self.weight.size(1), 1.0 / self.weight.size(1))

    def forward(self, inputs, expert_size):
        """
        Forward pass of the ParallelExperts module.

        Args:
            inputs (Tensor): Input tensor.
            expert_size: Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i in range(self.num_experts):
            output_list.append(F.linear(input_list[i], self.weight[i]))
        results = torch.cat(output_list, dim=0)
        return results


class NoisyTopkRouter(nn.Module):
    # noisy top-k gating
    r"""
    实现了一个带有噪声的 Top-k 门控路由器，它通过对路由器logits添加噪声来引入随机性，
    并使用 softmax 函数将加权的输出转换为概率分布，从而确定每个样本分配给哪些专家处理。
    """

    def __init__(self, n_embed, num_experts, top_k):
        r"""
        n_embed：表示输入特征的维度大小。
        num_experts：表示专家的数量。
        top_k：表示每个样本将被分配给的前 k 个专家。
        """
        super(NoisyTopkRouter, self).__init__()
        self.aux_loss = 0.0
        self.top_k = top_k
        self.num_experts = num_experts
        # layer for router logits
        # topkroute_linear：一个线性层，用于生成路由器的logits，
        # 其输入维度为 n_embed，输出维度为 num_experts。
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # noise_linear：一个线性层，用于生成噪声的logits，
        # 其输入维度同样为 n_embed，输出维度为 num_experts。
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        # mh_output 是来自多头自注意力模块的输出张量。
        # 通过 topkroute_linear 层将 mh_output 输入，得到原始的路由器logits。
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        # 通过 noise_linear 层将 mh_output 输入，得到噪声logits。
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        # torch.randn_like: 从标准正态/高斯分布（均值为0，标准差为1）中随机采样
        # 添加一个缩放后的单位高斯噪声到路由器logits中。
        # 缩放系数使用 F.softplus 函数应用到噪声logits上，然后与单位高斯噪声相乘。
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        # 加入噪声的logits
        noisy_logits = logits + noise

        # 对加入噪声后的logits执行 top-k 操作，
        # 得到每个样本前 k 个最大值的logits和对应的索引。
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        # 创建一个与 noisy_logits 张量相同形状的全 -inf 的张量 zeros。
        zeros = torch.full_like(noisy_logits, float("-inf"))
        # 使用 scatter 函数将每个样本前 k 个最大值的logits分散到 zeros 张量中，
        # 得到稀疏的logits。
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        # 对稀疏的logits执行 softmax 操作，得到路由器的输出。
        router_output = F.softmax(sparse_logits, dim=-1)

        # 训练时才计算辅助loss值, 为了专家之间的负载平衡
        if self.training:
            self.aux_loss = compute_aux_loss(self.num_experts, router_output, indices, noisy_logits)

        # 返回路由器的输出以及对应的索引。
        return router_output, indices


@torch.jit.script
def compute_gating(
    k: int, num_experts: int, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor
):
    """
    Compute gating values for the mixture of experts based on probabilities and top-k indices.

    Args:
        k (int): Number of experts to select.
        num_experts (int): Total number of experts.
        top_k_gates (torch.Tensor): Gating values for top-k experts (batch_size x k).
        top_k_indices (torch.Tensor): Indices of top-k experts (batch_size x k).

    Returns:
        torch.Tensor: Batch-level gating values.
        torch.Tensor: Batch-level expert indices.
        torch.Tensor: Expert size for each expert.
        torch.Tensor: Sorted indices of top-k experts.
    """
    zeros = torch.zeros(
        [top_k_gates.size(0), num_experts], dtype=top_k_gates.dtype, device=top_k_gates.device
    )
    gates = zeros.scatter(-1, top_k_indices, 1)
    # 计算每个专家被选择的次数，即每列中值为 1 的数量，得到专家大小（expert_size）。
    expert_size = gates.long().sum(0)
    # 将顶部 k 个专家的门控值和索引展平为一维张量，并对专家索引进行排序。
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    # 根据专家索引的排序结果，确定每个样本所属的批次索引（batch_index）。
    batch_index = index_sorted_experts.div(k, rounding_mode="trunc")
    # 提取排序后的专家门控值，得到批次级别的门控值（batch_gates）。
    batch_gates = top_k_gates[index_sorted_experts]

    return batch_gates, batch_index, expert_size, index_sorted_experts


def compute_aux_loss(
    num_experts: int, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor, logits: torch.Tensor
):
    """
    Calculate and return the auxiliary loss based on the accumulated statistics.
    switch transformers: https://arxiv.org/pdf/2101.03961.pdf
    A. Differentiable Load Balancing Loss

    Args:
        num_experts (int): The number of experts.
        top_k_gates (tensor): k个最大值的对应logits, 其每个元素表示对应logit概率值。
        top_k_gates (tensor): k个最大值的对应logits索引, 其每个元素表示logit对应索引值。
        logits (tensor): 其每个元素表示对应logit概率值。

    Returns:
        torch.Tensor: The calculated auxiliary loss.
    """
    # 对logits进行softmax操作，得到每个类别的概率分布
    probs = torch.softmax(logits, dim=-1)
    zeros = torch.zeros_like(probs)
    # Convert zeros to match top_k_gates dtype
    zeros = zeros.to(top_k_gates.dtype)
    gates = zeros.scatter(-1, top_k_indices, top_k_gates)

    # 获取 logits 张量的批次大小，即样本数量
    count = logits.size(0)
    # 计算每个专家被选中的概率之和，即将概率沿着批次维度求和。
    probs = probs.sum(0)
    # 计算每个专家被选中的频率，即计算门控值大于0的次数（即专家被选中的次数），
    # 然后将其沿着批次维度求和。
    freq = (gates > 0).float().sum(0)
    # 计算 logits 张量经过 softmax 处理后的平方和的对数。
    # 这里首先使用 softmax 函数将 logits 转换为概率分布，
    # 然后计算概率分布的每个样本的平方和，并取对数，最后将结果沿着批次维度求和。
    lsesq = (torch.log(torch.exp(logits).sum(dim=-1)) ** 2).sum()

    # 计算专家选择损失，其计算方式为对每个专家的概率和频率进行归一化，然后计算它们的点积，最后将结果乘以专家数量。
    switchloss = (
        num_experts * (F.normalize(probs, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()
    )
    # 计算 z 损失，即 logits 的对数平方和除以样本数量
    zloss = lsesq / count
    # 将专家选择损失和 z 损失加权相加得到最终的辅助损失
    loss = switchloss + 0.1 * zloss

    return loss


class BaseMoE(nn.Module):
    """the basic sparse mixture of experts module"""

    def __init__(self, n_embed, num_experts, top_k):
        super(BaseMoE, self).__init__()
        self.top_k = min(top_k, num_experts)
        self.num_experts = num_experts
        self.n_embed = n_embed

        self.router = NoisyTopkRouter(n_embed, num_experts, self.top_k)

    def forward(self, x):
        pass


class SparseMoE(BaseMoE):
    """the sparse mixture of experts module"""

    def __init__(self, n_embed, num_experts, top_k, dropout):
        super(SparseMoE, self).__init__(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed, dropout=dropout) for _ in range(num_experts)])

    def forward(self, x):
        # gating_output 表示每个样本被分配给每个专家的概率分数，
        # indices 表示每个样本被分配给的专家索引。
        gating_output, indices = self.router(x)

        # torch.zeros_like(x) 创建一个与输入张量 x 具有相同形状的全零张量 final_output，用于存储最终的输出。
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        # flat_x 和 flat_gating_output 将输入和门控输出展平为二维张量，以便进行批处理处理。
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            # 创建一个掩码 expert_mask，用于标识哪些输入应该由当前专家处理。
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                # 如果掩码中有任何真值，表示当前专家需要处理这些输入，则从 flat_x 中提取相应的输入，并将其传递给当前专家进行处理。
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                # 从 flat_gating_output 中提取当前专家的门控输出，并将其乘以专家的输出，以获得加权的输出。
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output
                # We need to scatter_add the weighted outputs to their original positions in the batch
                # 使用 masked_scatter_ 函数将加权的输出散布到 final_output 张量的对应位置。
                final_output.masked_scatter_(expert_mask.unsqueeze(-1), weighted_output)

        return final_output.view_as(x)


class SparseMoEWithCapacity(BaseMoE):
    r"""
    加入Expert Capacity:每个专家(expert)能够处理的最大token数量的阈值
    - 如果某个专家分配到的令牌数超过了它的容量,那么超出部分的令牌将不会被该层处理,可能会导致信息损失
    - 适当增加专家容量可以减少这种溢出情况,但也会增加计算和通信开销。因此需要在专家容量和效率之间权衡。
    - 在Switch Transformer中,作者通过上采样和下采样技术,允许不同层使用不同数量的专家,从而优化计算和内存利用率
    合理设置专家容量因子可以平衡模型性能效果和效率
    code detail from:
    - https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L384
    - https://github.com/AviSoori1x/makeMoE/blob/main/makeMoE_from_Scratch_with_Expert_Capacity.ipynb
    """

    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super(SparseMoEWithCapacity, self).__init__(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        # add capacity_factor
        self.capacity_factor = capacity_factor

    def forward(self, x):
        # Assuming x has shape [batch_size, seq_len, n_embd]
        batch_size, seq_len, _ = x.shape
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Flatten the batch and sequence dimensions to treat each token independently
        # Now shape [batch_size * seq_len, n_embd]
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # 将 tokens_per_batch 除以 self.num_experts，得到每个专家应该处理的平均token数量，
        # 然后乘以 self.capacity_factor 来调整基本容量。最后，将结果转换为整数，得到每个专家的最终容量
        tokens_per_batch = batch_size * seq_len * self.top_k
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)

        # 创建与 flat_x 相同形状的全零张量 updates，用于存储每个tokens的更新值。
        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            r"""
            对每个专家进行循环遍历，并根据其索引和门控输出选择相应的tokens。
            如果选择的token数量超过了专家的容量，则只选择前面的部分。
            然后，将选定的token输入到相应的专家中，得到专家输出，
            并根据门控输出对其进行加权。将加权输出累加到 updates 张量中。
            """
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)

            limited_indices = (
                selected_indices[:expert_capacity]
                if selected_indices.numel() > expert_capacity
                else selected_indices
            )
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)

        # 将 updates 张量重新整形为与 x 相同的形状，并将其加到 final_output 中，得到最终输出。
        final_output += updates.view(batch_size, seq_len, -1)

        return final_output


class Block(nn.Module):
    """
    Mixture of Experts Transformer block: communication followed by computation (SparseMoE-multi-head self attention + SparseMoE)
    that may be repeated several number of times; Copy pasting key architecture variables for clarity
    """

    def __init__(
        self, n_embed, n_head, num_experts, top_k, block_size, dropout, capacity_factor=0.0
    ):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head

        # echo block auxiliary loss for training
        # moe self-attention auxiliary loss + moe auxiliary loss
        self.aux_loss = 0.0

        self.sa = SparseMoEMultiHeadAttention(
            n_head, head_size, n_embed, block_size, dropout, num_experts=num_experts, top_k=top_k
        )

        if capacity_factor >= 1.0:
            self.smoe = SparseMoEWithCapacity(
                n_embed, num_experts, top_k, dropout, capacity_factor=capacity_factor
            )
        else:
            self.smoe = SparseMoE(n_embed, num_experts, top_k, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        if self.training:
            self.aux_loss = self.sa.router.aux_loss + self.smoe.router.aux_loss
        return x


class SparseMoAMoELanguageModel(nn.Module):
    """
    putting  all together to crease a sparse mixture of experts language model
    """

    def __init__(
        self,
        vocab_size,
        n_head,
        num_experts,
        top_k,
        n_layer,
        n_embed,
        block_size,
        dropout,
        nn_init="kaiming_normal",
        capacity_factor=0.0,
        aux_loss_coef=0.01,
    ):
        super().__init__()
        self.aux_loss_coef = aux_loss_coef
        self.block_size = block_size
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embed,
                    n_head=n_head,
                    num_experts=num_experts,
                    top_k=top_k,
                    block_size=block_size,
                    dropout=dropout,
                    capacity_factor=capacity_factor,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Jeremy Howard的Fastai第2部分有一个非常出色的讲座，
        # 从零开始实现了这些初始化方法：https://course.fast.ai/Lessons/lesson17.html
        # [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf) Kaiming He
        # [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) Xavier Glorot
        # 这里默认使用Kaiming He初始化(Kaiming 正态分布)
        def init_weights(m):
            if isinstance(m, (nn.Linear)):
                if nn_init == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight)
                else:
                    nn.init.xavier_normal_(m.weight)

        self.apply(init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        # x = self.blocks(x)  # (B,T,C)
        aux_loss = 0.0
        for block in self.blocks:
            x = block(x)
            if self.training:
                aux_loss += block.aux_loss

        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        if targets is not None and self.training:
            loss += self.aux_loss_coef * aux_loss.to(loss.device)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        output = []
        self.eval()  # Otherwise batch normalization will raise an error.
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            output.append(idx_next[0].tolist()[0])
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        self.train()
        return output
