from typing import Literal, Optional
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from pydantic import BaseModel

from models.layers import SwiGLU, SparseMoESwiGLU, AttnType, Attention, Cache, RotaryEmbedding, find_multiple


class InitConfig(BaseModel):
    in_std: float

    attn_out_std: float
    ff_out_std: float


class TransformerConfig(BaseModel):
    # Input config
    max_seq_len: int

    # Transformer config
    n_layers: int

    hidden_size: int
    num_heads: int
    expansion: float

    # Qwen-style sparse MoE FFN. moe_num_experts=0 keeps the dense SwiGLU.
    moe_num_experts: int = 0
    moe_top_k: int = 1
    moe_intermediate_size: Optional[int] = None
    moe_norm_topk_prob: bool = True
    moe_router_aux_loss_coef: float = 0.0

    attn_type: AttnType = "prefixlm"

    init_type: Literal["fixed_normal", "lecun_normal", "megatron"]
    init_std: Optional[float] = None

    norm_type: Literal["pre", "post"]
    norm_eps: float

    pos_emb_type: Literal["rope", "none"]
    rope_theta: Optional[float] = None

    # [Computed properties]
    @property
    def intermediate_size(self):
        # Automatic compute "intermediate_size" from "expansion"
        # NOTE: The formula is to match the number of GLU parameters to a vanilla Transformer with same expansion
        return find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)
    
    @property
    def init_config(self):
        match self.init_type:
            case "fixed_normal":
                in_std = attn_out_std = ff_out_std = self.init_std if self.init_std is not None else 0.02  # defaults to 0.02, as in OLMo 2
            case "lecun_normal":
                in_std = attn_out_std = 1.0 / math.sqrt(self.hidden_size)
                ff_out_std = 1.0 / math.sqrt(self.intermediate_size)
            case "megatron":
                in_std = self.init_std if self.init_std is not None else 1.0 / math.sqrt(self.hidden_size)
                attn_out_std = ff_out_std = in_std / math.sqrt(2.0 * self.n_layers)
            case _:
                raise NotImplementedError()
            
        return InitConfig(in_std=in_std, attn_out_std=attn_out_std, ff_out_std=ff_out_std)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            attn_type=config.attn_type,

            init_std_in=config.init_config.in_std,
            init_std_out=config.init_config.attn_out_std
        )
        if config.moe_num_experts > 0:
            moe_intermediate_size = config.moe_intermediate_size
            if moe_intermediate_size is None:
                moe_intermediate_size = find_multiple(max(1, config.intermediate_size // config.moe_top_k), 256)

            self.mlp = SparseMoESwiGLU(
                hidden_size=config.hidden_size,
                intermediate_size=moe_intermediate_size,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
                norm_topk_prob=config.moe_norm_topk_prob,

                init_std_in=config.init_config.in_std,
                init_std_out=config.init_config.ff_out_std
            )
        else:
            self.mlp = SwiGLU(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,

                init_std_in=config.init_config.in_std,
                init_std_out=config.init_config.ff_out_std
            )
        
        self.forward = getattr(self, f"_forward_{config.norm_type}")  # Avoid branching logic in "forward" for torch.compile compatibility
        self.norm = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

    # [Forward logic]
    def _forward_pre(self, x: Tensor, **seq_info) -> Tensor:  # Pre Norm
        attn_seq_info = seq_info
        if "moe_context" in seq_info:
            attn_seq_info = {k: v for k, v in seq_info.items() if k != "moe_context"}
        x = x + self.attn(self.norm(x), **attn_seq_info)
        return x + self.mlp(self.norm(x), **seq_info)
    
    def _forward_post(self, x: Tensor, **seq_info) -> Tensor:  # Post Norm
        attn_seq_info = seq_info
        if "moe_context" in seq_info:
            attn_seq_info = {k: v for k, v in seq_info.items() if k != "moe_context"}
        x = self.norm(x + self.attn(x, **attn_seq_info))
        return self.norm(x + self.mlp(x, **seq_info))


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.head_hint = {"in":  {"dim": config.hidden_size, "init_std": config.init_config.in_std},
                          "out": {"dim": config.hidden_size, "init_std": config.init_config.in_std}}  # Hint for LMHead init

        # Position embeddings
        if config.pos_emb_type == "rope":
            assert config.rope_theta is not None
            self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, config.max_seq_len, base=config.rope_theta)

        # Layers
        self.layers = nn.ModuleList([TransformerBlock(config) for _layer_idx in range(config.n_layers)])

        # Use final norm only for prenorm
        self.norm_f = lambda x: x
        if config.norm_type == "pre":
            self.norm_f = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

        # Create cache function
        self.create_cache = lambda **kwargs: [Cache.create(**kwargs, num_heads=config.num_heads, head_dim=config.hidden_size // config.num_heads) for _i in range(config.n_layers)]

    def forward(self, x: Tensor, cache: Optional[list[Cache]] = None, **seq_info) -> Tensor:
        seq_info["cos_sin"] = self.rotary_emb(seq_info.pop("position_ids", None)) if hasattr(self, "rotary_emb") else None

        # Forward layers
        for layer_id, layer in enumerate(self.layers):
            x = layer(x, **seq_info, cache=cache[layer_id] if cache is not None else None)

        return self.norm_f(x)
