import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Expert(nn.Module):
    # Expert module
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

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
        self.top_k = top_k
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
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        # 加入噪声的logits
        noisy_logits = logits + noise

        # 对加入噪声后的logits执行 top-k 操作，
        # 得到每个样本前 k 个最大值的logits和对应的索引。
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        # 创建一个与 noisy_logits 张量相同形状的全 -inf 的张量 zeros。
        zeros = torch.full_like(noisy_logits, float('-inf'))
        # 使用 scatter 函数将每个样本前 k 个最大值的logits分散到 zeros 张量中，
        # 得到稀疏的logits。
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        # 对稀疏的logits执行 softmax 操作，得到路由器的输出。
        router_output = F.softmax(sparse_logits, dim=-1)

        # 返回路由器的输出以及对应的索引。
        return router_output, indices


class SparseMoE(nn.Module):
    """ the sparse mixture of experts module """

    def __init__(self, n_embed, num_experts, top_k, dropout):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList(
            [Expert(n_embed, dropout=dropout) for _ in range(num_experts)])
        self.top_k = top_k

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
                final_output.masked_scatter_(
                    expert_mask.unsqueeze(-1), weighted_output)

        return final_output.view_as(x)


class SparseMoEWithCapacity(nn.Module):
    r"""
    加入Expert Capacity:每个专家(expert)能够处理的最大token数量的阈值
    - 如果某个专家分配到的令牌数超过了它的容量,那么超出部分的令牌将不会被该层处理,可能会导致信息损失
    - 适当增加专家容量可以减少这种溢出情况,但也会增加计算和通信开销。因此需要在专家容量和效率之间权衡。
    - 在Switch Transformer中,作者通过上采样和下采样技术,允许不同层使用不同数量的专家,从而优化计算和内存利用率
    合理设置专家容量因子可以平衡模型性能和效率
    code detail from: 
    - https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L384
    - https://github.com/AviSoori1x/makeMoE/blob/main/makeMoE_from_Scratch_with_Expert_Capacity.ipynb
    """

    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super(SparseMoEWithCapacity, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed)
                                     for _ in range(num_experts)])
        self.top_k = top_k
        # add capacity_factor
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts

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
        expert_capacity = int(
            (tokens_per_batch / self.num_experts) * self.capacity_factor)

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

            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel(
            ) > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(
                    1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)

        # 将 updates 张量重新整形为与 x 相同的形状，并将其加到 final_output 中，得到最终输出。
        final_output += updates.view(batch_size, seq_len, -1)

        return final_output


class Block(nn.Module):
    """ 
    Mixture of Experts Transformer block: communication followed by computation (multi-head self attention + SparseMoE) 
    that may be repeated several number of times; Copy pasting key architecture variables for clarity
    """

    def __init__(self, n_embed, n_head, num_experts, top_k, block_size, dropout, capacity_factor=0.0):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(
            n_head, head_size, n_embed, block_size, dropout)
        if capacity_factor >= 1.0:
            self.smoe = SparseMoEWithCapacity(
                n_embed, num_experts, top_k, dropout, capacity_factor=capacity_factor)
        else:
            self.smoe = SparseMoE(n_embed, num_experts, top_k, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x


class SparseMoELanguageModel(nn.Module):
    """
    putting  all together to crease a sparse mixture of experts language model
    """

    def __init__(self, vocab_size, n_head, num_experts, top_k, n_layer, n_embed, block_size, dropout, nn_init="kaiming_normal", capacity_factor=0.0):
        super().__init__()
        self.block_size = block_size
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, num_experts=num_experts,
                                    top_k=top_k,  block_size=block_size, dropout=dropout, capacity_factor=capacity_factor) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        output = []
        self.eval()  # Otherwise batch normalization will raise an error.
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
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
