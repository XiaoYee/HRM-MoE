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


def load_balancing_loss_func(routing_weights: Tensor, selected_experts: Tensor, num_experts: int) -> Tensor:
    """Qwen-style auxiliary load-balancing loss for top-k MoE routing."""
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).to(torch.float32)
    tokens_per_expert = expert_mask.mean(dim=0)
    router_prob_per_expert = routing_weights.mean(dim=0)
    return torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_experts


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


class SparseMoEExpertShard(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 expert_in_one_shard: int,
                 shard_idx: int,
                 init_std_in=None,
                 init_std_out=None,
                 **kwargs):
        super().__init__()
        if expert_in_one_shard <= 0:
            raise ValueError(f"expert_in_one_shard must be positive, got {expert_in_one_shard}")

        if init_std_in is None:
            init_std_in = 1.0 / (hidden_size ** 0.5)
        if init_std_out is None:
            init_std_out = 1.0 / (intermediate_size ** 0.5)

        self.expert_in_one_shard = expert_in_one_shard
        self.shard_idx = shard_idx
        self.expert_offset = shard_idx * expert_in_one_shard

        self.gate_up_weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((expert_in_one_shard, 2 * intermediate_size, hidden_size), **kwargs),
                std=init_std_in
            )
        )
        self.down_weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((expert_in_one_shard, hidden_size, intermediate_size), **kwargs),
                std=init_std_out
            )
        )

    def forward(self,
                hidden_states: Tensor,
                selected_experts: Tensor,
                routing_weights: Tensor,
                final_hidden_states: Tensor) -> Tensor:
        for local_idx in range(self.expert_in_one_shard):
            expert_idx = self.expert_offset + local_idx
            token_idx, topk_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue

            expert_input = hidden_states[token_idx]
            gate_up = F.linear(expert_input, self.gate_up_weight[local_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.down_weight[local_idx])
            expert_output = expert_output * routing_weights[token_idx, topk_idx, None]
            final_hidden_states.index_add_(0, token_idx, expert_output.to(hidden_states.dtype))

        return final_hidden_states


def grouped_mm(input: Tensor, weight: Tensor, offsets: Tensor) -> Tensor:
    if hasattr(F, "grouped_mm"):
        return F.grouped_mm(input, weight, offs=offsets)
    return torch._grouped_mm(input, weight, offs=offsets)


class SparseMoEGroupedExperts(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_experts: int,
                 init_std_in=None,
                 init_std_out=None,
                 **kwargs):
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")

        if init_std_in is None:
            init_std_in = 1.0 / (hidden_size ** 0.5)
        if init_std_out is None:
            init_std_out = 1.0 / (intermediate_size ** 0.5)

        self.num_experts = num_experts
        self.expert_in_one_shard = num_experts

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
                hidden_states: Tensor,
                selected_experts: Tensor,
                routing_weights: Tensor,
                final_hidden_states: Tensor) -> Tensor:
        num_tokens = hidden_states.shape[0]
        if num_tokens == 0:
            return final_hidden_states

        top_k = selected_experts.shape[-1]
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        flat_token_idx = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(top_k)

        sorted_experts, sort_idx = torch.sort(flat_experts, stable=True)
        sorted_token_idx = flat_token_idx.index_select(0, sort_idx)
        sorted_weights = flat_weights.index_select(0, sort_idx)
        sorted_hidden_states = hidden_states.index_select(0, sorted_token_idx)

        tokens_per_expert = torch.bincount(sorted_experts, minlength=self.num_experts)
        max_tokens_per_expert = int(tokens_per_expert.max().item())
        expert_offsets = torch.zeros((self.num_experts + 1,), device=hidden_states.device, dtype=torch.long)
        expert_offsets[1:] = torch.cumsum(tokens_per_expert, dim=0)
        positions_in_expert = torch.arange(sorted_experts.shape[0], device=hidden_states.device) - expert_offsets.index_select(0, sorted_experts)

        grouped_hidden_states = hidden_states.new_zeros((self.num_experts, max_tokens_per_expert, hidden_states.shape[-1]))
        grouped_hidden_states[sorted_experts, positions_in_expert] = sorted_hidden_states

        gate_up = torch.bmm(grouped_hidden_states, self.gate_up_weight.transpose(1, 2))
        gate, up = gate_up.chunk(2, dim=-1)
        grouped_expert_output = torch.bmm(F.silu(gate) * up, self.down_weight.transpose(1, 2))
        expert_output = grouped_expert_output[sorted_experts, positions_in_expert]
        expert_output = expert_output * sorted_weights[:, None]
        final_hidden_states.index_add_(0, sorted_token_idx, expert_output.to(hidden_states.dtype))
        return final_hidden_states


class SparseMoEExpertCollection(nn.Module):
    def __init__(self, experts: Sequence[nn.Module], expert_in_one_shard: int):
        super().__init__()
        self.layers = nn.ModuleList(experts)
        self.expert_in_one_shard = expert_in_one_shard

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, idx: int) -> nn.Module:
        return self.layers[idx]

    def forward(self,
                hidden_states: Tensor,
                selected_experts: Tensor,
                routing_weights: Tensor,
                final_hidden_states: Tensor) -> Tensor:
        if self.expert_in_one_shard == 1:
            for expert_idx, expert_layer in enumerate(self.layers):
                token_idx, topk_idx = torch.where(selected_experts == expert_idx)
                if token_idx.numel() == 0:
                    continue

                expert_input = hidden_states[token_idx]
                expert_output = expert_layer(expert_input)
                expert_output = expert_output * routing_weights[token_idx, topk_idx, None]
                final_hidden_states.index_add_(0, token_idx, expert_output.to(hidden_states.dtype))
        else:
            for expert_shard in self.layers:
                final_hidden_states = expert_shard(hidden_states, selected_experts, routing_weights, final_hidden_states)

        return final_hidden_states


class SparseMoESwiGLU(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_experts: int,
                 top_k: int,
                 norm_topk_prob: bool,
                 implementation: Literal["origin", "shard", "grouped"] = "origin",
                 expert_in_one_shard: int = 1,
                 init_std_in=None,
                 init_std_out=None,
                 **kwargs):
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got top_k={top_k}, num_experts={num_experts}")
        if expert_in_one_shard <= 0 or num_experts % expert_in_one_shard != 0:
            raise ValueError(
                f"expert_in_one_shard must divide num_experts, got "
                f"expert_in_one_shard={expert_in_one_shard}, num_experts={num_experts}"
            )

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.implementation = implementation
        self.expert_in_one_shard = expert_in_one_shard

        self.gate = LinearInit(hidden_size, num_experts, bias=False, init_std=init_std_in, **kwargs)
        if implementation == "grouped":
            self.experts = SparseMoEGroupedExperts(hidden_size=hidden_size,
                                                   intermediate_size=intermediate_size,
                                                   num_experts=num_experts,
                                                   init_std_in=init_std_in,
                                                   init_std_out=init_std_out,
                                                   **kwargs)
            return

        if expert_in_one_shard == 1:
            experts = [
                SwiGLU(hidden_size=hidden_size,
                       intermediate_size=intermediate_size,
                       init_std_in=init_std_in,
                       init_std_out=init_std_out,
                       **kwargs)
                for _ in range(num_experts)
            ]
        else:
            experts = [
                SparseMoEExpertShard(hidden_size=hidden_size,
                                     intermediate_size=intermediate_size,
                                     expert_in_one_shard=expert_in_one_shard,
                                     shard_idx=shard_idx,
                                     init_std_in=init_std_in,
                                     init_std_out=init_std_out,
                                     **kwargs)
                for shard_idx in range(num_experts // expert_in_one_shard)
            ]
        self.experts = SparseMoEExpertCollection(experts, expert_in_one_shard)

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

        final_hidden_states = self.experts(
            hidden_states,
            selected_experts,
            routing_weights,
            torch.zeros_like(hidden_states),
        )

        if moe_context is not None and torch.is_grad_enabled():
            aux_routing_probs = routing_probs
            aux_selected_experts = selected_experts
            if total_seqlen is not None:
                valid_tokens = int(unwrap_tensor(total_seqlen).item())
                aux_routing_probs = aux_routing_probs[:valid_tokens]
                aux_selected_experts = aux_selected_experts[:valid_tokens]

            aux_loss = load_balancing_loss_func(aux_routing_probs, aux_selected_experts, self.num_experts)
            moe_context.setdefault("aux_losses", []).append(aux_loss)
            with torch.no_grad():
                expert_counts = torch.bincount(aux_selected_experts.reshape(-1), minlength=self.num_experts).to(torch.float32)
                moe_context.setdefault("expert_counts", []).append(expert_counts)

        return final_hidden_states.reshape(original_shape)
