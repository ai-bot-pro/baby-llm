import math
from typing import Literal, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

# from: deepseekv2
# model:
# - https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py
# model config
# - https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json


@dataclass
class ModelArgs:
    hidden_size: int = 384  # n_embed or d_model
    vocab_size: int = None
    max_seq_len: int = 256  # block_size for tril mask
    n_layer: int = 3  # num_layers

    # RMS normalization
    rms_norm_eps: float = 1e-5

    # attention
    num_heads: int = 4  # n_head
    dropout: float = 0.0
    # attention weight with LoRA rank
    q_lora_rank: int = 76
    qk_rope_head_dim: int = 8
    qk_nope_head_dim: int = 16
    kv_lora_rank: int = 28
    v_head_dim: int = 16

    # positional embedding use fixed embedding (max_seq_len) context length
    # if want scaling long seq, use RoPE (YaRN)

    # mlp/moe
    first_k_dense_replace: int = 0  # 0: all MoE
    moe_layer_freq: int = 1
    moe_intermediate_size: int = 128  # MLP/MoE inter hidden size
    num_experts_per_tok: int = 4
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    routed_scaling_factor: float = 1.0
    scoring_func: Literal["softmax", "sigmoid"] = "softmax"
    aux_loss_alpha: float = 0.001
    seq_aux: bool = True
    # topk selection algorithm
    norm_topk_prob: bool = False
    topk_method: Literal["greedy", "group_limited_greedy"] = "greedy"
    topk_group: int = 1
    n_group: int = 1
    ep_size: int = 1


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MLA(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        # tril buffer for Square Matrix(Tensor) mask, if q_len == kv_len, is ok
        self.register_buffer(
            "tril", torch.tril(torch.ones(model_args.max_seq_len, model_args.max_seq_len))
        )

        self.hidden_size = model_args.hidden_size
        self.num_heads = model_args.num_heads
        # self.head_dim = model_args.hidden_size // model_args.num_heads
        self.attention_dropout = model_args.dropout

        # self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        self.q_lora_rank = model_args.q_lora_rank
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)  # math.sqrt(self.q_head_dim)

        self.kv_lora_rank = model_args.kv_lora_rank
        self.v_head_dim = model_args.v_head_dim

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape  # hidden_states

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(B, T, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]  # T

        # use RoPE to apply different freqs to each head, if not, use input positional embeddings to learn
        # q_pe, k_pe = apply_rope(q_pe, k_pe, freqs_cis)

        query_states = k_pe.new_empty(B, self.num_heads, T, self.q_head_dim)  # BHTC
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(B, self.num_heads, T, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        if attn_weights.size() != (B, self.num_heads, T, kv_seq_len):  # BHTC
            raise ValueError(
                f"Attention weights should be of size {(B, self.num_heads, T, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (B, 1, T, kv_seq_len):  # batch_size,1,q_len,kv_len
                raise ValueError(
                    f"Attention mask should be of size {(B, 1, T, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        else:
            attn_weights = attn_weights.masked_fill(
                self.tril[:T, :T] == 0, float("-inf")
            )  # (B, H, T, T)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (B, self.num_heads, T, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(B, self.num_heads, T, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, T, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        output = nn.functional.dropout(
            attn_output, p=self.attention_dropout, training=self.training
        )
        return output


class Expert(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(model_args.hidden_size, model_args.moe_intermediate_size)
        self.down_proj = nn.Linear(model_args.moe_intermediate_size, model_args.hidden_size)
        self.up_proj = nn.Linear(model_args.hidden_size, model_args.moe_intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.top_k = model_args.num_experts_per_tok
        self.n_routed_experts = model_args.n_routed_experts
        self.routed_scaling_factor = model_args.routed_scaling_factor
        self.scoring_func = model_args.scoring_func
        self.alpha = model_args.aux_loss_alpha
        self.seq_aux = model_args.seq_aux
        self.topk_method = model_args.topk_method
        self.n_group = model_args.n_group
        self.topk_group = model_args.topk_group

        # topk selection algorithm
        self.norm_topk_prob = model_args.norm_topk_prob
        self.gating_dim = model_args.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape  # batch_size, seq_len, hidden_size
        ### compute gating score
        x = x.view(-1, C)
        logits = F.linear(x.type(torch.float32), self.weight.type(torch.float32), None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            group_scores = scores.view(B * T, self.n_group, -1).max(dim=-1).values  # [n, n_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(B * T, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(B * T, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(B, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(B, T, -1)
                ce = torch.zeros(B, self.n_routed_experts, device=x.device)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(B, T * aux_topk, device=x.device),
                ).div_(T * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.num_experts_per_tok = model_args.num_experts_per_tok

        if hasattr(model_args, "ep_size") and model_args.ep_size > 1:
            assert model_args.ep_size == dist.get_world_size()
            self.ep_size = model_args.ep_size
            self.experts_per_rank = model_args.n_routed_experts // model_args.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        Expert(model_args)
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(model_args.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = model_args.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList(
                [Expert(model_args) for i in range(model_args.n_routed_experts)]
            )
        self.gate = MoEGate(model_args)
        if model_args.n_shared_experts is not None:
            model_args.moe_intermediate_size = (
                model_args.moe_intermediate_size * model_args.n_shared_experts
            )
            self.shared_experts = Expert(model_args)

    def forward(self, x: torch.Tensor):
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(x)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(x.dtype).view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(x, topk_idx, topk_weight).view(*orig_shape)
        if self.model_args.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape

        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
            output_splits = (
                tokens_per_expert_group.view(self.ep_size, -1).sum(1).cpu().numpy().tolist()
            )
            gathered_tokens = sorted_tokens.new_empty(
                tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            dist.all_to_all(
                list(gathered_tokens.split(output_splits)),
                list(sorted_tokens.split(input_split_sizes)),
            )
            tokens_per_expert_post_gather = tokens_per_expert_group.view(
                self.ep_size, self.experts_per_rank
            ).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        if self.ep_size > 1:
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            dist.all_to_all(
                list(gathered_tokens.split(input_split_sizes)),
                list(new_x.split(output_splits)),
            )
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )

        return final_out


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MLP(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.hidden_size = model_args.hidden_size
        self.intermediate_size = model_args.moe_intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DecoderLayer(nn.Module):
    def __init__(self, model_args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = model_args.hidden_size

        self.self_attn = MLA(model_args)

        self.mlp = (
            MoE(model_args)
            if (
                model_args.n_routed_experts is not None
                and layer_idx >= model_args.first_k_dense_replace
                and layer_idx % model_args.moe_layer_freq == 0
            )
            else MLP(model_args)
        )
        self.input_layernorm = RMSNorm(model_args.hidden_size, eps=model_args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(model_args.hidden_size, eps=model_args.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Args:
            x (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        # Self Attention
        x = x + self.self_attn(self.input_layernorm(x))
        # Fully Connected
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x


class MlaSparseMoELanguageModel(nn.Module):
    """
    putting all( MLA + spares MoE or dense MLP) together to create generative Causal language model
    """

    def __init__(
        self,
        model_args: ModelArgs,
        nn_init="kaiming_normal",
    ):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(model_args.vocab_size, model_args.hidden_size)
        self.position_embedding_table = nn.Embedding(model_args.max_seq_len, model_args.hidden_size)
        self.blocks = nn.Sequential(
            *[DecoderLayer(model_args, i) for i in range(model_args.n_layer)]
        )
        self.ln_f = RMSNorm(
            model_args.hidden_size, eps=model_args.rms_norm_eps
        )  # final RMS layer norm
        self.lm_head = nn.Linear(model_args.hidden_size, model_args.vocab_size)
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
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

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


if __name__ == "__main__":
    args = ModelArgs(vocab_size=26)
    print(args)
    model = MlaSparseMoELanguageModel(args)
    print(model)
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model_million_params, "M parameters")
