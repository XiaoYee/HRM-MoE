from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import BaseModel

from models.layers import LinearInit, ScaledEmbeddingInit, Carry
from models.common import IGNORE_LABEL_ID, packing_sequence_sum


class LMHeadConfig(BaseModel):
    vocab_size: int


class LMHead(nn.Module):
    def __init__(self, model: nn.Module, config_dict: dict) -> None:
        super().__init__()
        self.model = model
        # Create cache function
        self.create_cache = self.model.create_cache
        # Train extra args function
        self.compute_train_extra_args = self.model.compute_train_extra_args

        config = LMHeadConfig(**config_dict)
        head_hint: dict = self.model.head_hint  # pyright: ignore[reportAssignmentType]

        # LMHead input and output
        self.embed_tokens = ScaledEmbeddingInit(config.vocab_size, head_hint["in"]["dim"], init_std=head_hint["in"]["init_std"])  # pyright: ignore[reportArgumentType]
        self.lm_head = LinearInit(head_hint["out"]["dim"], config.vocab_size, bias=False, init_std=head_hint["out"]["init_std"])  # pyright: ignore[reportArgumentType]

    def forward(self, carry: Carry, batch: dict[str, Tensor], **kwargs) -> Tuple[Carry, Tensor] | Tuple[Carry, Tensor, dict[str, Tuple[Tensor, Tensor]]]:
        # Token embedding
        input_embedding = self.embed_tokens(batch["inputs"])

        # Model forward
        moe_context = None
        if "labels" in batch and getattr(self.model, "moe_num_experts", 0) > 0:
            moe_context = {"aux_losses": [], "expert_counts": []}

        model_kwargs = {k: v for k, v in batch.items() if k not in ("inputs", "labels")}
        if moe_context is not None:
            model_kwargs["moe_context"] = moe_context

        new_carry, logits = self.model(carry,
                                       input_embedding,
                                       **model_kwargs,
                                       **kwargs)
        logits = self.lm_head(logits)

        # Loss & Metrics
        if "labels" in batch:
            # Masks & labels
            labels = batch["labels"]
            masks = labels != IGNORE_LABEL_ID

            # Loss (CE in F32)
            ce_loss = F.cross_entropy(logits.to(torch.float32), labels.to(torch.long), ignore_index=IGNORE_LABEL_ID, reduction="sum")
            # AllReduce loss divisor. Divide by mean of valid tokens across all processes, as gradient will be averaged.
            loss_divisor = masks.sum().to(torch.float32)
            dist.all_reduce(loss_divisor, op=dist.ReduceOp.AVG)
            loss = ce_loss / loss_divisor

            if moe_context is not None and moe_context["aux_losses"]:
                aux_loss = torch.stack(moe_context["aux_losses"]).mean()
                aux_loss_scaled = aux_loss * getattr(self.model, "moe_router_aux_loss_coef", 0.0)
                loss = loss + aux_loss_scaled

            # Accuracy
            with torch.no_grad():
                is_correct = torch.argmax(logits, dim=-1) == labels
                local_valid_counts = masks.sum()
                # Sequence-level statistics
                seq_num_tokens_correct = packing_sequence_sum(is_correct, batch["cu_seqlens"])
                seq_num_valid_tokens = packing_sequence_sum(masks, batch["cu_seqlens"])
                seq_is_valid = seq_num_valid_tokens > 0
                # Metrics
                metrics = {
                    "loss": (ce_loss.detach(), local_valid_counts),
                    "accuracy": (is_correct.sum(), local_valid_counts),
                    "exact_accuracy": (((seq_num_tokens_correct == seq_num_valid_tokens) & seq_is_valid).sum(), seq_is_valid.sum()),
                }

                if moe_context is not None and moe_context["aux_losses"]:
                    one = torch.ones((), dtype=torch.float32, device=loss.device)
                    expert_counts = torch.stack(moe_context["expert_counts"]).sum(dim=0)
                    expert_count_total = expert_counts.sum().clamp_min(1.0)
                    metrics["moe_aux_loss"] = (aux_loss.detach(), one)
                    metrics["moe_aux_loss_scaled"] = (aux_loss_scaled.detach(), one)
                    metrics["moe_total_loss"] = (loss.detach(), one)
                    metrics["moe_max_expert_frac"] = ((expert_counts.max() / expert_count_total).detach(), one)

            return new_carry, loss, metrics

        return new_carry, logits
