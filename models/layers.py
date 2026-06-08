from typing import Tuple, Optional, Sequence, Any, NamedTuple, Literal
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange

from models.common import trunc_normal_init_, unwrap_tensor
from models.flash_attention_prefixlm_v2 import flash_attn_varlen_prefixlm
from flash_attn_interface import flash_attn_with_kvcache


Carry = dict[str, Any]
CosSin = Tuple[Tensor, Tensor]
AttnType = Literal["causal", "prefixlm"]


def find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: Tensor, cos_sin: CosSin):
    # x:   [..., seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim] OR [..., seq_len, head_dim]
    # Use FP32 RoPE, as in Transformers OLMo and FlashAttention
    # 
    # https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/olmo/modular_olmo.py#L139-L152
    # https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/rotary.h#L126-L133
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len, base, **kwargs):
        super().__init__()
        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, **kwargs) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32, **kwargs)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self, position_ids: Tensor):
        if position_ids is not None:
            return self.cos_cached[position_ids], self.sin_cached[position_ids]

        return self.cos_cached, self.sin_cached


class LinearInit(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 batch_out_features: Sequence[int] = (),
                 init_std: Optional[float] = None,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        # Truncated LeCun normal init
        if init_std is None:
            init_std = 1.0 / (in_features ** 0.5)

        # Parameters
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((math.prod(batch_out_features) * out_features, in_features), **kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((math.prod(batch_out_features) * out_features, ), **kwargs))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class ScaledEmbeddingInit(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 **kwargs):
        super().__init__()
        self.scale = 1.0 / init_std

        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim), **kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.scale * F.embedding(input, self.embedding_weight)


class Cache(NamedTuple):
    """A static cache layer that stores the key and value states as static tensors. Built for `torch.compile` support."""
    keys: Tensor
    values: Tensor

    @classmethod
    def create(cls, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, **kwargs):
        return cls(keys=torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), **kwargs),
                   values=torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), **kwargs))


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, attn_type, init_std_in=None, init_std_out=None, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.attn_type = attn_type

        self.gqkv_proj = LinearInit(hidden_size, self.head_dim, batch_out_features=(2 * self.num_heads + 2 * self.num_key_value_heads, ),
                                   bias=False, init_std=init_std_in, **kwargs)
        self.o_proj = LinearInit(head_dim * num_heads, hidden_size,
                                 bias=False, init_std=init_std_out, **kwargs)

    def forward(self, hidden_states: Tensor, cos_sin: Optional[CosSin], cache: Optional[Cache] = None, cache_lengths: Optional[Tensor] = None, **seq_info) -> Tensor:
        # hidden_states, gqkv: [..., seq_len, hidden_size]
        gqkv = self.gqkv_proj(hidden_states)

        # Split head (last dimension of projected qkv)
        gqkv = rearrange(gqkv, "... (h hd) -> ... h hd", h=2 * self.num_heads + 2 * self.num_key_value_heads)
        gate, query, key, value = gqkv.split((self.num_heads, self.num_heads, self.num_key_value_heads, self.num_key_value_heads), dim=-2)
        # query, key, value: [..., seq_len, num_heads, head_dim]
        # RoPE
        if cos_sin is not None:
            query = apply_rotary_pos_emb(query, cos_sin)
            key = apply_rotary_pos_emb(key, cos_sin)

        is_causal = self.attn_type == "causal"
        if cache is None:
            # flash attn (training)
            attn_output = flash_attn_varlen_prefixlm(query, key, value, is_causal, **{name: unwrap_tensor(tensor) for name, tensor in seq_info.items()})
        else:
            # Regardless of auto / non-autoregressive, apply attention based on current concatenated with cache.
            attn_output = flash_attn_with_kvcache(q=query, k=key, v=value,
                                                  k_cache=cache.keys, v_cache=cache.values, cache_seqlens=cache_lengths,
                                                  num_splits=1,  # Must set to support torch.compile tracing.
                                                  causal=is_causal)  # causal can always be False for PrefixLM. during AR generation seqlen is 1, so causal masking won't matter.

        # attn_output: [..., seq_len, num_heads, head_dim]
        attn_output = rearrange(torch.sigmoid(gate) * attn_output, "... h hd -> ... (h hd)")  # type: ignore
        return self.o_proj(attn_output)


def load_balancing_loss_func(routing_probs: Tensor, selected_experts: Tensor, num_experts: int) -> Tensor:
    """Qwen-style auxiliary load-balancing loss for top-k MoE routing."""
    tokens_per_expert = torch.bincount(
        selected_experts.reshape(-1),
        minlength=num_experts,
    ).to(torch.float32) / selected_experts.shape[0]
    router_prob_per_expert = routing_probs.mean(dim=0)
    return torch.sum(tokens_per_expert * router_prob_per_expert) * num_experts


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, init_std_in=None, init_std_out=None, **kwargs):
        super().__init__()
        self.gate_up_proj = LinearInit(hidden_size, intermediate_size, batch_out_features=(2, ),
                                       bias=False, init_std=init_std_in, **kwargs)
        self.down_proj    = LinearInit(intermediate_size, hidden_size,
                                       bias=False, init_std=init_std_out, **kwargs)

    def forward(self, x, **_seq_info):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class SparseMoESwiGLU(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_experts: int,
                 top_k: int,
                 norm_topk_prob: bool,
                 init_std_in=None,
                 init_std_out=None,
                 **kwargs):
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got top_k={top_k}, num_experts={num_experts}")
        if init_std_in is None:
            init_std_in = 1.0 / (hidden_size ** 0.5)
        if init_std_out is None:
            init_std_out = 1.0 / (intermediate_size ** 0.5)

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

        self.gate = LinearInit(hidden_size, num_experts, bias=False, init_std=init_std_in, **kwargs)
        self.gate_up_weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((num_experts, 2 * intermediate_size, hidden_size), **kwargs),
                std=init_std_in
            )
        )
        self.down_weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((num_experts, hidden_size, intermediate_size), **kwargs),
                std=init_std_out
            )
        )

    def forward(self,
                x: Tensor,
                moe_context: Optional[dict[str, Any]] = None,
                total_seqlen: Optional[Tensor] = None,
                **_seq_info) -> Tensor:
        original_shape = x.shape
        hidden_states = x.reshape(-1, self.hidden_size)

        router_logits = self.gate(hidden_states)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)
        gate_up_weight = self.gate_up_weight.to(hidden_states.dtype)
        down_weight = self.down_weight.to(hidden_states.dtype)
        for expert_idx in range(self.num_experts):
            token_idx, topk_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue

            expert_input = hidden_states[token_idx]
            gate_up = F.linear(expert_input, gate_up_weight[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, down_weight[expert_idx])
            expert_output = expert_output * routing_weights[token_idx, topk_idx, None]
            final_hidden_states.index_add_(0, token_idx, expert_output.to(hidden_states.dtype))

        if moe_context is not None and torch.is_grad_enabled():
            aux_routing_probs = routing_probs
            aux_selected_experts = selected_experts
            if total_seqlen is not None:
                valid_tokens = int(unwrap_tensor(total_seqlen).item())
                aux_routing_probs = aux_routing_probs[:valid_tokens]
                aux_selected_experts = aux_selected_experts[:valid_tokens]

            aux_loss = load_balancing_loss_func(aux_routing_probs, aux_selected_experts, self.num_experts)
            moe_context.setdefault("aux_losses", []).append(aux_loss)

        return final_hidden_states.reshape(original_shape)
