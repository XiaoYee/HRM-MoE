import torch
from torch import Tensor

from models.transformer import Transformer, TransformerConfig

class TransformerWrapper(Transformer):
    def __init__(self, config_dict: dict) -> None:
        config = TransformerConfig(**config_dict)
        super().__init__(config)
        self.moe_num_experts = config.moe_num_experts
        self.moe_router_aux_loss_coef = config.moe_router_aux_loss_coef if self.moe_num_experts > 0 else 0.0

    def forward(self, carry: None, x: Tensor, **kwargs) -> tuple[None, torch.Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return None, super().forward(x, **kwargs)

    def compute_train_extra_args(self, train_state):
        return {}

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> None:
        return None
