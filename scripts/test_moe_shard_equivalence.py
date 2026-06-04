#!/usr/bin/env python
"""Check origin and shard MoE implementations are numerically equivalent."""

import torch
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

flash_prefixlm_stub = types.ModuleType("models.flash_attention_prefixlm_v2")
flash_prefixlm_stub.flash_attn_varlen_prefixlm = None
sys.modules.setdefault("models.flash_attention_prefixlm_v2", flash_prefixlm_stub)

flash_interface_stub = types.ModuleType("flash_attn_interface")
flash_interface_stub.flash_attn_with_kvcache = None
sys.modules.setdefault("flash_attn_interface", flash_interface_stub)

from models.layers import SparseMoESwiGLU


def copy_origin_to_shard(origin: SparseMoESwiGLU, shard: SparseMoESwiGLU) -> None:
    with torch.no_grad():
        shard.gate.weight.copy_(origin.gate.weight)

        for expert_idx, origin_expert in enumerate(origin.experts):
            shard_idx = expert_idx // shard.expert_in_one_shard
            local_idx = expert_idx % shard.expert_in_one_shard
            shard_expert = shard.experts[shard_idx]
            shard_expert.gate_up_weight[local_idx].copy_(origin_expert.gate_up_proj.weight)
            shard_expert.down_weight[local_idx].copy_(origin_expert.down_proj.weight)


def assert_expert_grads_equal(origin: SparseMoESwiGLU, shard: SparseMoESwiGLU) -> None:
    for expert_idx, origin_expert in enumerate(origin.experts):
        shard_idx = expert_idx // shard.expert_in_one_shard
        local_idx = expert_idx % shard.expert_in_one_shard
        shard_expert = shard.experts[shard_idx]

        torch.testing.assert_close(
            origin_expert.gate_up_proj.weight.grad,
            shard_expert.gate_up_weight.grad[local_idx],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            origin_expert.down_proj.weight.grad,
            shard_expert.down_weight.grad[local_idx],
            rtol=1e-5,
            atol=1e-6,
        )


def main() -> None:
    torch.manual_seed(0)
    kwargs = dict(
        hidden_size=16,
        intermediate_size=32,
        num_experts=8,
        top_k=3,
        norm_topk_prob=True,
    )

    origin = SparseMoESwiGLU(**kwargs, expert_in_one_shard=1)
    shard = SparseMoESwiGLU(**kwargs, expert_in_one_shard=4)
    copy_origin_to_shard(origin, shard)

    x_origin = torch.randn(5, 7, 16, requires_grad=True)
    x_shard = x_origin.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_origin)

    origin_context = {"aux_losses": [], "expert_counts": []}
    shard_context = {"aux_losses": [], "expert_counts": []}

    out_origin = origin(x_origin, moe_context=origin_context)
    out_shard = shard(x_shard, moe_context=shard_context)

    torch.testing.assert_close(out_origin, out_shard, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        torch.stack(origin_context["aux_losses"]),
        torch.stack(shard_context["aux_losses"]),
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        torch.stack(origin_context["expert_counts"]),
        torch.stack(shard_context["expert_counts"]),
        rtol=0,
        atol=0,
    )

    loss_origin = (out_origin * grad).sum() + torch.stack(origin_context["aux_losses"]).sum()
    loss_shard = (out_shard * grad).sum() + torch.stack(shard_context["aux_losses"]).sum()
    loss_origin.backward()
    loss_shard.backward()

    torch.testing.assert_close(x_origin.grad, x_shard.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(origin.gate.weight.grad, shard.gate.weight.grad, rtol=1e-5, atol=1e-6)
    assert_expert_grads_equal(origin, shard)

    print("origin/shard MoE equivalence: ok")


if __name__ == "__main__":
    main()
