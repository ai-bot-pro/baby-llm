from dataclasses import dataclass
import math
from collections.abc import Callable
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


from transformers.cache_utils import Cache


try:
    from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    raise ImportError("Plese run `pip install -U fla-core`")

"""
kimi-linear model config and modules
- https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/blob/main/config.json
- https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/blob/main/configuration_kimi.py
- https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/blob/main/modeling_kimi.py

# references:
- MLA: [2024.5 DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434)
- DeltaNets: [2024.6 Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)
- Gated-DeltaNets: [2024.12 Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- MLA+Gated-DeltaNets: [2025.10 Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)


baby-llm KISS(👶🏻-🦈) Kimi-Linear model
"""


@dataclass
class LinearAttentionArgs:
    full_attn_layers: Optional[list[int]] = None
    kda_layers: Optional[list[int]] = None
    num_heads: int = 4
    head_dim: int = 96  # hidden_size//num_heads
    short_conv_kernel_size: int = 4  # CNN kernel size*size for KDA


@dataclass
class ModelArgs:
    # Dim(fat) and Layers(deep) scaling
    hidden_size: int = 384  # n_embed or d_model
    vocab_size: int = None
    max_seq_len: int = 256  # block_size for tril mask
    n_layer: int = 4  # num_layers

    # RMS normalization
    rms_norm_eps: float = 1e-5
    # rms_norm_eps: float = 1e-6

    # Self-Attention
    # mla
    num_heads: int = 4  # n_head
    dropout: float = 0.0
    # attention weight with LoRA rank
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 8
    kv_lora_rank: int = 28
    v_head_dim: int = 16
    mla_use_nope: bool = True
    # linear_attn_config for MLA KDA layers
    linear_attn_config: Optional[LinearAttentionArgs] = None

    # positional embedding use fixed embedding (max_seq_len) context length
    # if want scaling long seq, use RoPE (YaRN)

    # FFN (MLP/MoE)
    # - mlp (don't use MoE)
    intermediate_size: int = 640  # mlp hidden size

    # - mlp/moe
    # share experts (mlp)
    # share_experts_intermediate_size: int = moe_intermediate_size * n_shared_experts
    # moe
    moe_renormalize: bool = True
    first_k_dense_replace: int = 0  # 0: all MoE
    moe_layer_freq: int = 1
    moe_intermediate_size: int = 128  # MLP/MoE inter hidden size
    num_experts_per_tok: Optional[int] = 4
    n_routed_experts: Optional[int] = 4
    n_shared_experts: Optional[int] = 1
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

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self):
        return self.n_routed_experts is not None and self.num_experts_per_tok is not None

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config.kda_layers is not None
                and len(self.linear_attn_config.kda_layers) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int):
        return (
            self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config.kda_layers
        )


class DynamicCache:
    """
    Dynamic cache for Kimi model.
    Inspired by Qwen3-Next
    """
    is_compileable = False

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        if config.linear_attn_config is not None:
            self.layer_types = []
            for i in range(config.num_hidden_layers):
                if config.is_kda_layer(i):
                    self.layer_types.append("linear_attention")
                else:
                    self.layer_types.append("full_attention")
        else:
            self.layer_types = ["full_attention"] * config.num_hidden_layers

        self.transformer_layers = [
            i for i in range(config.num_hidden_layers) if self.layer_types[i] == "full_attention"
        ]

        linear_layers = [i for i in range(
            config.num_hidden_layers) if self.layer_types[i] == "linear_attention"]
        self.last_linear_layer = linear_layers[-1] if linear_layers else -1

        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                    0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                    0, beam_idx)

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx][0].device
                beam_idx = beam_idx.to(device)
                q_conv, k_conv, v_conv = self.conv_states[layer_idx]
                self.conv_states[layer_idx] = (
                    q_conv.index_select(0, beam_idx),
                    k_conv.index_select(0, beam_idx),
                    v_conv.index_select(0, beam_idx)
                )
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(
                    0, beam_idx)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        """We have a previous state if the last linear (conv) layer was already updated."""
        if self.last_linear_layer == -1:
            return False
        return self.conv_states[self.last_linear_layer] is not None


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
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class BlockSparseMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.ffn_dim = config.moe_intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)   # gate
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)   # down
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)   # up

        self.act_fn = F.silu

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(
            self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MLP(nn.Module):
    """
    for kimi-linear MLP module
    - non-MoE: MLP as FFN
    - MoE: MLP as shared expert
    """

    def __init__(self, config: ModelArgs, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class MLAAttention(nn.Module):
    """
    Multi-Latent Attention adapted from deepseek-v3
    """

    def __init__(self, config: ModelArgs, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # for GQA, now set equal to num_heads and just one group
        # self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_heads = config.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        try:
            self.q_lora_rank = config.q_lora_rank
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.kv_lora_rank = config.kv_lora_rank
            self.v_head_dim = config.v_head_dim
            self.qk_nope_head_dim = config.qk_nope_head_dim
            self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.use_nope = config.mla_use_nope
            self.scaling = self.q_head_dim ** (-0.5)
        except Exception as e:
            raise ValueError(
                f"Kimi MLA config is not found or not properly formatted: {e}")

        assert self.q_lora_rank is None
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.q_head_dim, bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )
        self.is_causal = True
        assert self.use_nope

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1,
                     self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(
            k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(
            k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        # KISS
        # now compute attention with eager attention (u can also use flash attention here)
        attention_interface: Callable = eager_attention_forward
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        # add dropout for attention output for small models to train
        attn_output = nn.functional.dropout(
            attn_output, p=self.attention_dropout, training=self.training
        )

        return attn_output


class DeltaAttention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config.short_conv_kernel_size
        self.head_dim = config.linear_attn_config.head_dim
        self.num_heads = config.linear_attn_config.num_heads
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        self.layer_idx = layer_idx

        assert self.mode in [
            'chunk', 'fused_recurrent'], f"Not suppoerted mode `{self.mode}`."

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(
            self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        # https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/convolution.py#L793
        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu',
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu'
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation='silu'
        )

        self.A_log = torch.nn.Parameter(torch.log(torch.empty(
            self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(1, 1, -1, 1))

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation='sigmoid')
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params: Optional[DynamicCache] = None,
        **kwargs: dict,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask", None)

            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                    "(0 = padding). 3D masks are not supported here."
                )
        use_cache = cache_params is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get('cu_seqlens', None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None
        if cache_params is not None:
            if cache_params.conv_states[self.layer_idx] is not None:
                conv_state_q, conv_state_k, conv_state_v = cache_params.conv_states[
                    self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
        beta = self.b_proj(hidden_states).float().sigmoid()

        q, k = map(lambda x: rearrange(
            x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # use triton JIT runtime pre compilation kernel to speed up;
        # NOTE: don't use torch.compile here
        if mode == 'chunk':  # for training and short seq inference
            # https://github.com/fla-org/flash-linear-attention/blob/v0.4.0/fla/ops/kda/chunk.py#L248
            # https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py#L179
            # ChunkKDAFunction forward and backward
            # forward:
            # https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_intra.py#L387
            # https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_intra.py#L27 (inter)
            # https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_intra.py#L117 (intra)
            # backward:
            # https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_intra.py#L480
            # https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_intra.py#L193 (intra)
            #
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            # https://github.com/fla-org/flash-linear-attention/blob/v0.4.0/fla/ops/kda/fused_recurrent.py#L11
            # https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py#L192
            # FusedRecurrentFunction forward
            # https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py#L21
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = recurrent_state
            cache_params.conv_states[self.layer_idx] = (
                conv_state_q, conv_state_k, conv_state_v)

        g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
        o = self.o_norm(o, g)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o


class MoEGate(nn.Module):
    """
    MoEGate adapted from Deepseek-V2(mla_moeLM).
    Parameter correspondences:
        num_experts -> n_routed_experts
        num_experts_per_token -> num_experts_per_tok
        num_expert_group -> n_group
        moe_router_activation_func -> scoring_func
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.scoring_func
        self.num_expert_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim))
        )

        self.e_score_correction_bias = nn.Parameter(
            torch.empty((self.num_experts))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.moe_router_activation_func}"
            )

        # select top-k experts (group_limited_greedy)
        assert not self.training
        scores_for_choice = scores.view(bsz * seq_len, -1)
        scores_for_choice += self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(
                bsz * seq_len, self.num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )  # [n, num_expert_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, num_expert_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, num_expert_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                bsz * seq_len, self.num_expert_group, self.num_experts // self.num_expert_group
            )
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        # norm gate to sum 1
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        # must multiply the scaling factor
        topk_weight = topk_weight * self.routed_scaling_factor

        # expert-level computation auxiliary loss
        aux_loss = None
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha

        return topk_idx, topk_weight, aux_loss


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


class SparseMoeBlock(nn.Module):
    """
    Adapted from Deepseek-V2's MOE implementation
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.experts_per_rank = config.n_routed_experts
        self.ep_rank = 0  # for single gpu
        self.experts = nn.ModuleList(
            [
                BlockSparseMLP(config)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MLP(config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states: torch.Tensor):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        else:
            # raise NotImplementedError("Training mode is not supported in SparseMoeBlock")
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

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

        outs = torch.cat(outputs, dim=0) if len(
            outputs) else sorted_tokens.new_empty(0)

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


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        if config.is_kda_layer(layer_idx):
            self.is_linear_attn = True
            self.self_attn = DeltaAttention(config=config, layer_idx=layer_idx)
        elif config.is_mla:
            self.is_linear_attn = False
            self.self_attn = MLAAttention(config=config, layer_idx=layer_idx)
        else:
            raise NotImplementedError

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = SparseMoeBlock(config)
        else:
            self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        # **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.is_linear_attn is False:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                # **kwargs,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                # **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class KDASparseMoELanguageModel(nn.Module):
    """
    putting all( KDA(MLA(sparse) + Gated DeltaNet(linear)) + spares MoE or dense MLP) together to create generative Causal language model
    """

    def __init__(
        self,
        model_args: ModelArgs,
        nn_init="kaiming_normal",
    ):
        super().__init__()
        self.model_args = model_args
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
            idx_cond = idx[:, -self.model_args.max_seq_len:]
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


"""
python3 simpleLM/kda_moeLM.py
"""
if __name__ == "__main__":
    import os
    os.getenv("TOKENIZERS_PARALLELISM", "false")

    args = ModelArgs(vocab_size=26)
    print(args)
    model = KDASparseMoELanguageModel(args)
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model_million_params, "M parameters")
    print(model)


"""
# MLA + MoE LM configuration example output:

ModelArgs(hidden_size=384, vocab_size=26, max_seq_len=256, n_layer=4, rms_norm_eps=1e-05, num_heads=4, dropout=0.0, q_lora_rank=None, qk_nope_head_dim=16, qk_rope_head_dim=8, kv_lora_rank=28, v_head_dim=16, mla_use_nope=True, linear_attn_config=None, intermediate_size=640, moe_renormalize=True, first_k_dense_replace=0, moe_layer_freq=1, moe_intermediate_size=128, num_experts_per_tok=4, n_routed_experts=4, n_shared_experts=1, routed_scaling_factor=1.0, scoring_func='softmax', aux_loss_alpha=0.001, seq_aux=True, norm_topk_prob=False, topk_method='greedy', topk_group=1, n_group=1, ep_size=1)

3.392538 M parameters

KDASparseMoELanguageModel(
  (token_embedding_table): Embedding(26, 384)
  (position_embedding_table): Embedding(256, 384)
  (blocks): Sequential(
    (0): DecoderLayer(
      (self_attn): MLAAttention(
        (q_proj): Linear(in_features=384, out_features=96, bias=False)
        (kv_a_proj_with_mqa): Linear(in_features=384, out_features=36, bias=False)
        (kv_a_layernorm): RMSNorm()
        (kv_b_proj): Linear(in_features=28, out_features=128, bias=False)
        (o_proj): Linear(in_features=64, out_features=384, bias=False)
      )
      (block_sparse_moe): SparseMoeBlock(
        (experts): ModuleList(
          (0-3): 4 x BlockSparseMLP(
            (w1): Linear(in_features=384, out_features=128, bias=False)
            (w2): Linear(in_features=128, out_features=384, bias=False)
            (w3): Linear(in_features=384, out_features=128, bias=False)
          )
        )
        (gate): MoEGate()
        (shared_experts): MLP(
          (gate_proj): Linear(in_features=384, out_features=128, bias=False)
          (up_proj): Linear(in_features=384, out_features=128, bias=False)
          (down_proj): Linear(in_features=128, out_features=384, bias=False)
        )
      )
      (input_layernorm): RMSNorm()
      (post_attention_layernorm): RMSNorm()
    )
    (1): DecoderLayer(
      (self_attn): MLAAttention(
        (q_proj): Linear(in_features=384, out_features=96, bias=False)
        (kv_a_proj_with_mqa): Linear(in_features=384, out_features=36, bias=False)
        (kv_a_layernorm): RMSNorm()
        (kv_b_proj): Linear(in_features=28, out_features=128, bias=False)
        (o_proj): Linear(in_features=64, out_features=384, bias=False)
      )
      (block_sparse_moe): SparseMoeBlock(
        (experts): ModuleList(
          (0-3): 4 x BlockSparseMLP(
            (w1): Linear(in_features=384, out_features=128, bias=False)
            (w2): Linear(in_features=128, out_features=384, bias=False)
            (w3): Linear(in_features=384, out_features=128, bias=False)
          )
        )
        (gate): MoEGate()
        (shared_experts): MLP(
          (gate_proj): Linear(in_features=384, out_features=128, bias=False)
          (up_proj): Linear(in_features=384, out_features=128, bias=False)
          (down_proj): Linear(in_features=128, out_features=384, bias=False)
        )
      )
      (input_layernorm): RMSNorm()
      (post_attention_layernorm): RMSNorm()
    )
    (2): DecoderLayer(
      (self_attn): MLAAttention(
        (q_proj): Linear(in_features=384, out_features=96, bias=False)
        (kv_a_proj_with_mqa): Linear(in_features=384, out_features=36, bias=False)
        (kv_a_layernorm): RMSNorm()
        (kv_b_proj): Linear(in_features=28, out_features=128, bias=False)
        (o_proj): Linear(in_features=64, out_features=384, bias=False)
      )
      (block_sparse_moe): SparseMoeBlock(
        (experts): ModuleList(
          (0-3): 4 x BlockSparseMLP(
            (w1): Linear(in_features=384, out_features=128, bias=False)
            (w2): Linear(in_features=128, out_features=384, bias=False)
            (w3): Linear(in_features=384, out_features=128, bias=False)
          )
        )
        (gate): MoEGate()
        (shared_experts): MLP(
          (gate_proj): Linear(in_features=384, out_features=128, bias=False)
          (up_proj): Linear(in_features=384, out_features=128, bias=False)
          (down_proj): Linear(in_features=128, out_features=384, bias=False)
        )
      )
      (input_layernorm): RMSNorm()
      (post_attention_layernorm): RMSNorm()
    )
    (3): DecoderLayer(
      (self_attn): MLAAttention(
        (q_proj): Linear(in_features=384, out_features=96, bias=False)
        (kv_a_proj_with_mqa): Linear(in_features=384, out_features=36, bias=False)
        (kv_a_layernorm): RMSNorm()
        (kv_b_proj): Linear(in_features=28, out_features=128, bias=False)
        (o_proj): Linear(in_features=64, out_features=384, bias=False)
      )
      (block_sparse_moe): SparseMoeBlock(
        (experts): ModuleList(
          (0-3): 4 x BlockSparseMLP(
            (w1): Linear(in_features=384, out_features=128, bias=False)
            (w2): Linear(in_features=128, out_features=384, bias=False)
            (w3): Linear(in_features=384, out_features=128, bias=False)
          )
        )
        (gate): MoEGate()
        (shared_experts): MLP(
          (gate_proj): Linear(in_features=384, out_features=128, bias=False)
          (up_proj): Linear(in_features=384, out_features=128, bias=False)
          (down_proj): Linear(in_features=128, out_features=384, bias=False)
        )
      )
      (input_layernorm): RMSNorm()
      (post_attention_layernorm): RMSNorm()
    )
  )
  (ln_f): RMSNorm()
  (lm_head): Linear(in_features=384, out_features=26, bias=True)
)

"""
