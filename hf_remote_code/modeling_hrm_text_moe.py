import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_hrm_text_moe import HrmTextMoEConfig


class HrmTextMoEScaledEmbedding(nn.Module):
    def __init__(self, config: HrmTextMoEConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))
        self.scale = float(config.embedding_scale)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.scale * F.embedding(input_ids, self.weight)


class HrmTextMoERotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: Tensor, cos_sin: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (_rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)


class HrmTextMoEAttention(nn.Module):
    def __init__(self, config: HrmTextMoEConfig):
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        out_features = (2 * self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        self.gqkv_proj = nn.Linear(config.hidden_size, out_features, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        cos_sin: tuple[Tensor, Tensor],
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
        use_cache: bool = True,
        token_type_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        gqkv = self.gqkv_proj(hidden_states)
        gqkv = gqkv.view(batch_size, seq_len, 2 * self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        gate, query, key, value = gqkv.split(
            (self.num_heads, self.num_heads, self.num_key_value_heads, self.num_key_value_heads),
            dim=2,
        )
        query = _apply_rotary(query, cos_sin)
        key = _apply_rotary(key, cos_sin)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        new_past = (key, value) if use_cache else None
        is_causal = past_key_value is None and token_type_ids is None and seq_len > 1
        attn_mask = None
        if past_key_value is None and token_type_ids is not None and seq_len > 1:
            prefix = token_type_ids.to(torch.bool)
            if not bool(prefix.all().item()):
                pos = torch.arange(seq_len, device=hidden_states.device)
                causal = pos[None, :] <= pos[:, None]
                prefix_block = prefix[:, :, None] & prefix[:, None, :]
                allowed = causal[None, :, :] | prefix_block
                attn_mask = allowed[:, None, :, :]

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = torch.sigmoid(gate) * attn_output
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), new_past


class HrmTextMoEGroupedExperts(nn.Module):
    def __init__(self, config: HrmTextMoEConfig):
        super().__init__()
        self.num_experts = int(config.moe_num_experts)
        self.intermediate_size = int(config.moe_intermediate_size)
        self.hidden_size = int(config.hidden_size)
        self.gate_up_weight = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size))
        self.down_weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))

    def forward(self, hidden_states: Tensor, selected_experts: Tensor, routing_weights: Tensor) -> Tensor:
        num_tokens = hidden_states.shape[0]
        if num_tokens == 0:
            return hidden_states

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
        positions = torch.arange(sorted_experts.shape[0], device=hidden_states.device) - expert_offsets.index_select(0, sorted_experts)

        grouped_hidden = hidden_states.new_zeros((self.num_experts, max_tokens_per_expert, self.hidden_size))
        grouped_hidden[sorted_experts, positions] = sorted_hidden_states

        gate_up = torch.bmm(grouped_hidden, self.gate_up_weight.to(hidden_states.dtype).transpose(1, 2))
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        grouped_output = torch.bmm(activated, self.down_weight.to(hidden_states.dtype).transpose(1, 2))

        expert_output = grouped_output[sorted_experts, positions] * sorted_weights[:, None]
        final_hidden = torch.zeros_like(hidden_states)
        final_hidden.index_add_(0, sorted_token_idx, expert_output.to(hidden_states.dtype))
        return final_hidden


class HrmTextMoESparseMLP(nn.Module):
    def __init__(self, config: HrmTextMoEConfig):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.moe_num_experts)
        self.top_k = int(config.moe_top_k)
        self.norm_topk_prob = bool(config.moe_norm_topk_prob)
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = HrmTextMoEGroupedExperts(config)

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        hidden_states = x.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        routing_weights = routing_weights.to(hidden_states.dtype)
        output = self.experts(hidden_states, selected_experts, routing_weights)
        return output.reshape(original_shape)


class HrmTextMoEBlock(nn.Module):
    def __init__(self, config: HrmTextMoEConfig):
        super().__init__()
        self.attn = HrmTextMoEAttention(config)
        self.mlp = HrmTextMoESparseMLP(config)
        self.norm_eps = float(config.rms_norm_eps)

    def _norm(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), eps=self.norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos_sin: tuple[Tensor, Tensor],
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
        use_cache: bool = True,
        token_type_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        attn_out, new_past = self.attn(
            self._norm(hidden_states),
            cos_sin=cos_sin,
            past_key_value=past_key_value,
            use_cache=use_cache,
            token_type_ids=token_type_ids,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self._norm(hidden_states))
        return hidden_states, new_past


class HrmTextMoEStack(nn.Module):
    def __init__(self, config: HrmTextMoEConfig):
        super().__init__()
        self.layers = nn.ModuleList([HrmTextMoEBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm_eps = float(config.rms_norm_eps)
        self.rotary_emb = HrmTextMoERotaryEmbedding(
            dim=int(config.head_dim),
            max_position_embeddings=int(config.max_position_embeddings),
            base=float(config.rope_theta),
        )

    def _norm(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), eps=self.norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        past_key_values: Optional[list[Optional[tuple[Tensor, Tensor]]]] = None,
        use_cache: bool = True,
        token_type_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[list[Optional[tuple[Tensor, Tensor]]]]]:
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        cos_sin = self.rotary_emb(position_ids)
        new_past = [] if use_cache else None
        for layer, past in zip(self.layers, past_key_values):
            hidden_states, layer_past = layer(
                hidden_states,
                cos_sin=cos_sin,
                past_key_value=past,
                use_cache=use_cache,
                token_type_ids=token_type_ids,
            )
            if use_cache:
                new_past.append(layer_past)
        return self._norm(hidden_states), new_past


class HrmTextMoEModel(nn.Module):
    def __init__(self, config: HrmTextMoEConfig):
        super().__init__()
        self.embed_tokens = HrmTextMoEScaledEmbedding(config)
        self.z_L_init = nn.Parameter(torch.empty(config.hidden_size), requires_grad=False)
        self.H_module = HrmTextMoEStack(config)
        self.L_module = HrmTextMoEStack(config)
        self.H_cycles = int(config.H_cycles)
        self.L_cycles = int(config.L_cycles)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: Optional[dict[str, list]] = None,
        use_cache: bool = True,
        token_type_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[dict[str, list]]]:
        hidden_states = self.embed_tokens(input_ids)
        z_h = hidden_states
        z_l = self.z_L_init.to(hidden_states.dtype).view(1, 1, -1).expand_as(hidden_states)

        if past_key_values is None:
            past_key_values = {
                "H": [None] * self.H_cycles,
                "L": [None] * (self.H_cycles * self.L_cycles),
            }
        new_past = {"H": [], "L": []} if use_cache else None

        for i in range(self.H_cycles):
            for k in range(i * self.L_cycles, (i + 1) * self.L_cycles):
                z_l, layer_past = self.L_module(
                    z_l + z_h,
                    position_ids=position_ids,
                    past_key_values=past_key_values["L"][k],
                    use_cache=use_cache,
                    token_type_ids=token_type_ids,
                )
                if use_cache:
                    new_past["L"].append(layer_past)
            z_h, layer_past = self.H_module(
                z_h + z_l,
                position_ids=position_ids,
                past_key_values=past_key_values["H"][i],
                use_cache=use_cache,
                token_type_ids=token_type_ids,
            )
            if use_cache:
                new_past["H"].append(layer_past)

        return z_h, new_past


class HrmTextMoEForCausalLM(PreTrainedModel):
    config_class = HrmTextMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["HrmTextMoEBlock"]

    def __init__(self, config: HrmTextMoEConfig):
        super().__init__(config)
        self.model = HrmTextMoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _init_weights(self, module):
        return

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _past_length(self, past_key_values: Optional[dict[str, list]]) -> int:
        if past_key_values is None:
            return 0
        for stack_values in past_key_values.values():
            for cycle_values in stack_values:
                if cycle_values:
                    key = cycle_values[0][0]
                    return int(key.shape[2])
        return 0

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        def reorder_layer(layer_past):
            if layer_past is None:
                return None
            key, value = layer_past
            return key.index_select(0, beam_idx), value.index_select(0, beam_idx)

        return {
            stack_name: [
                [reorder_layer(layer_past) for layer_past in cycle_past]
                if cycle_past is not None else None
                for cycle_past in stack_past
            ]
            for stack_name, stack_past in past_key_values.items()
        }

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[dict[str, list]] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        del attention_mask, inputs_embeds, kwargs
        return_dict = True if return_dict is None else return_dict
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if input_ids is None:
            raise ValueError("input_ids is required")

        past_len = self._past_length(past_key_values)
        if position_ids is None:
            position_ids = torch.arange(
                past_len,
                past_len + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        hidden_states, new_past = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            token_type_ids=token_type_ids,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, new_past)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past,
        )
