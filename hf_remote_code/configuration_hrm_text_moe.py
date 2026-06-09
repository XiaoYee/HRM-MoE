from transformers import PretrainedConfig


class HrmTextMoEConfig(PretrainedConfig):
    model_type = "hrm_text_moe"

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=1536,
        intermediate_size=512,
        num_hidden_layers=16,
        num_attention_heads=12,
        num_key_value_heads=12,
        head_dim=128,
        H_cycles=2,
        L_cycles=3,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        initializer_range=0.025515518153991442,
        embedding_scale=39.191835884530846,
        prefix_lm=True,
        moe_num_experts=64,
        moe_top_k=8,
        moe_intermediate_size=512,
        moe_implementation="grouped",
        moe_norm_topk_prob=True,
        moe_router_aux_loss_coef=0.0,
        condition_mapping=None,
        use_cache=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.embedding_scale = embedding_scale
        self.prefix_lm = prefix_lm
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_implementation = moe_implementation
        self.moe_norm_topk_prob = moe_norm_topk_prob
        self.moe_router_aux_loss_coef = moe_router_aux_loss_coef
        self.condition_mapping = condition_mapping or {
            "direct": "<|object_ref_start|>",
            "cot": "<|object_ref_end|>",
            "noisy": "<|quad_start|>",
            "synth": "<|quad_end|>",
        }
        self.use_cache = use_cache
