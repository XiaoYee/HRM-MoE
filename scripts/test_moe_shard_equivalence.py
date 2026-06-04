#!/usr/bin/env python
"""Check origin, shard, grouped, grouped_triton, grouped_cutlass, and grouped_ep MoE implementations."""

import torch
import os
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


def copy_origin_to_moe(origin: SparseMoESwiGLU, target: SparseMoESwiGLU) -> None:
    with torch.no_grad():
        target.gate.weight.copy_(origin.gate.weight)
        target_experts = getattr(target.experts, "local_experts", target.experts)

        for expert_idx, origin_expert in enumerate(origin.experts):
            if hasattr(target_experts, "gate_up_weight"):
                target_experts.gate_up_weight[expert_idx].copy_(origin_expert.gate_up_proj.weight)
                target_experts.down_weight[expert_idx].copy_(origin_expert.down_proj.weight)
            else:
                shard_idx = expert_idx // target_experts.expert_in_one_shard
                local_idx = expert_idx % target_experts.expert_in_one_shard
                shard_expert = target_experts[shard_idx]
                shard_expert.gate_up_weight[local_idx].copy_(origin_expert.gate_up_proj.weight)
                shard_expert.down_weight[local_idx].copy_(origin_expert.down_proj.weight)


def assert_expert_grads_equal(origin: SparseMoESwiGLU, target: SparseMoESwiGLU, rtol: float, atol: float) -> None:
    target_experts = getattr(target.experts, "local_experts", target.experts)
    for expert_idx, origin_expert in enumerate(origin.experts):
        origin_gate_up_grad = origin_expert.gate_up_proj.weight.grad
        origin_down_grad = origin_expert.down_proj.weight.grad
        if origin_gate_up_grad is None:
            origin_gate_up_grad = torch.zeros_like(origin_expert.gate_up_proj.weight)
        if origin_down_grad is None:
            origin_down_grad = torch.zeros_like(origin_expert.down_proj.weight)

        if hasattr(target_experts, "gate_up_weight"):
            target_gate_up_grad = (
                torch.zeros_like(target_experts.gate_up_weight[expert_idx])
                if target_experts.gate_up_weight.grad is None
                else target_experts.gate_up_weight.grad[expert_idx]
            )
            target_down_grad = (
                torch.zeros_like(target_experts.down_weight[expert_idx])
                if target_experts.down_weight.grad is None
                else target_experts.down_weight.grad[expert_idx]
            )
        else:
            shard_idx = expert_idx // target_experts.expert_in_one_shard
            local_idx = expert_idx % target_experts.expert_in_one_shard
            shard_expert = target_experts[shard_idx]
            target_gate_up_grad = (
                torch.zeros_like(shard_expert.gate_up_weight[local_idx])
                if shard_expert.gate_up_weight.grad is None
                else shard_expert.gate_up_weight.grad[local_idx]
            )
            target_down_grad = (
                torch.zeros_like(shard_expert.down_weight[local_idx])
                if shard_expert.down_weight.grad is None
                else shard_expert.down_weight.grad[local_idx]
            )

        torch.testing.assert_close(
            origin_gate_up_grad,
            target_gate_up_grad,
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            origin_down_grad,
            target_down_grad,
            rtol=rtol,
            atol=atol,
        )


def main() -> None:
    torch.manual_seed(0)
    device_name = os.environ.get("MOE_EQUIV_DEVICE", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MOE_EQUIV_DEVICE=cuda requested, but CUDA is not available")
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    close_rtol = float(os.environ.get("MOE_EQUIV_RTOL", "3e-2" if dtype == torch.bfloat16 else "1e-5"))
    close_atol = float(os.environ.get("MOE_EQUIV_ATOL", "2e-2" if dtype == torch.bfloat16 else "1e-6"))
    aux_rtol = float(os.environ.get("MOE_EQUIV_AUX_RTOL", "1e-4" if dtype == torch.bfloat16 else "1e-6"))
    aux_atol = float(os.environ.get("MOE_EQUIV_AUX_ATOL", "1e-5" if dtype == torch.bfloat16 else "1e-7"))

    hidden_size = int(os.environ.get("MOE_EQUIV_HIDDEN", "16" if device.type == "cpu" else "1536"))
    intermediate_size = int(os.environ.get("MOE_EQUIV_INTERMEDIATE", "32" if device.type == "cpu" else "512"))
    num_experts = int(os.environ.get("MOE_EQUIV_EXPERTS", "8" if device.type == "cpu" else "64"))
    top_k = int(os.environ.get("MOE_EQUIV_TOPK", "3" if device.type == "cpu" else "8"))
    batch_size = int(os.environ.get("MOE_EQUIV_BATCH", "5" if device.type == "cpu" else "2"))
    seq_len = int(os.environ.get("MOE_EQUIV_SEQ", "7" if device.type == "cpu" else "16"))
    shard_size = int(os.environ.get("MOE_EQUIV_SHARD_SIZE", "4" if device.type == "cpu" else "8"))

    kwargs = dict(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        norm_topk_prob=True,
    )

    origin = SparseMoESwiGLU(**kwargs, implementation="origin", expert_in_one_shard=1).to(device=device, dtype=dtype)
    shard = SparseMoESwiGLU(**kwargs, implementation="shard", expert_in_one_shard=shard_size).to(device=device, dtype=dtype)
    grouped = SparseMoESwiGLU(**kwargs, implementation="grouped", expert_in_one_shard=1).to(device=device, dtype=dtype)
    grouped_triton = SparseMoESwiGLU(**kwargs, implementation="grouped_triton", expert_in_one_shard=1).to(device=device, dtype=dtype)
    grouped_cutlass = SparseMoESwiGLU(**kwargs, implementation="grouped_cutlass", expert_in_one_shard=1).to(device=device, dtype=dtype)
    grouped_ep = SparseMoESwiGLU(**kwargs, implementation="grouped_ep", expert_in_one_shard=1).to(device=device, dtype=dtype)
    copy_origin_to_moe(origin, shard)
    copy_origin_to_moe(origin, grouped)
    copy_origin_to_moe(origin, grouped_triton)
    copy_origin_to_moe(origin, grouped_cutlass)
    copy_origin_to_moe(origin, grouped_ep)

    x_origin = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
    x_shard = x_origin.detach().clone().requires_grad_(True)
    x_grouped = x_origin.detach().clone().requires_grad_(True)
    x_grouped_triton = x_origin.detach().clone().requires_grad_(True)
    x_grouped_cutlass = x_origin.detach().clone().requires_grad_(True)
    x_grouped_ep = x_origin.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_origin)

    origin_context = {"aux_losses": [], "expert_counts": []}
    shard_context = {"aux_losses": [], "expert_counts": []}
    grouped_context = {"aux_losses": [], "expert_counts": []}
    grouped_triton_context = {"aux_losses": [], "expert_counts": []}
    grouped_cutlass_context = {"aux_losses": [], "expert_counts": []}
    grouped_ep_context = {"aux_losses": [], "expert_counts": []}

    out_origin = origin(x_origin, moe_context=origin_context)
    out_shard = shard(x_shard, moe_context=shard_context)
    out_grouped = grouped(x_grouped, moe_context=grouped_context)
    out_grouped_triton = grouped_triton(x_grouped_triton, moe_context=grouped_triton_context)
    out_grouped_cutlass = grouped_cutlass(x_grouped_cutlass, moe_context=grouped_cutlass_context)
    out_grouped_ep = grouped_ep(x_grouped_ep, moe_context=grouped_ep_context)

    torch.testing.assert_close(out_origin, out_shard, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(out_origin, out_grouped, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(out_origin, out_grouped_triton, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(out_origin, out_grouped_cutlass, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(out_origin, out_grouped_ep, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(
        torch.stack(origin_context["aux_losses"]),
        torch.stack(shard_context["aux_losses"]),
        rtol=aux_rtol,
        atol=aux_atol,
    )
    torch.testing.assert_close(
        torch.stack(origin_context["aux_losses"]),
        torch.stack(grouped_context["aux_losses"]),
        rtol=aux_rtol,
        atol=aux_atol,
    )
    torch.testing.assert_close(
        torch.stack(origin_context["aux_losses"]),
        torch.stack(grouped_triton_context["aux_losses"]),
        rtol=aux_rtol,
        atol=aux_atol,
    )
    torch.testing.assert_close(
        torch.stack(origin_context["aux_losses"]),
        torch.stack(grouped_cutlass_context["aux_losses"]),
        rtol=aux_rtol,
        atol=aux_atol,
    )
    torch.testing.assert_close(
        torch.stack(origin_context["aux_losses"]),
        torch.stack(grouped_ep_context["aux_losses"]),
        rtol=aux_rtol,
        atol=aux_atol,
    )
    torch.testing.assert_close(
        torch.stack(origin_context["expert_counts"]),
        torch.stack(shard_context["expert_counts"]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.stack(origin_context["expert_counts"]),
        torch.stack(grouped_context["expert_counts"]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.stack(origin_context["expert_counts"]),
        torch.stack(grouped_triton_context["expert_counts"]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.stack(origin_context["expert_counts"]),
        torch.stack(grouped_cutlass_context["expert_counts"]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.stack(origin_context["expert_counts"]),
        torch.stack(grouped_ep_context["expert_counts"]),
        rtol=0,
        atol=0,
    )

    loss_origin = (out_origin * grad).sum() + torch.stack(origin_context["aux_losses"]).sum()
    loss_shard = (out_shard * grad).sum() + torch.stack(shard_context["aux_losses"]).sum()
    loss_grouped = (out_grouped * grad).sum() + torch.stack(grouped_context["aux_losses"]).sum()
    loss_grouped_triton = (out_grouped_triton * grad).sum() + torch.stack(grouped_triton_context["aux_losses"]).sum()
    loss_grouped_cutlass = (out_grouped_cutlass * grad).sum() + torch.stack(grouped_cutlass_context["aux_losses"]).sum()
    loss_grouped_ep = (out_grouped_ep * grad).sum() + torch.stack(grouped_ep_context["aux_losses"]).sum()
    loss_origin.backward()
    loss_shard.backward()
    loss_grouped.backward()
    loss_grouped_triton.backward()
    loss_grouped_cutlass.backward()
    loss_grouped_ep.backward()

    torch.testing.assert_close(x_origin.grad, x_shard.grad, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(x_origin.grad, x_grouped.grad, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(x_origin.grad, x_grouped_triton.grad, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(x_origin.grad, x_grouped_cutlass.grad, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(x_origin.grad, x_grouped_ep.grad, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(origin.gate.weight.grad, shard.gate.weight.grad, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(origin.gate.weight.grad, grouped.gate.weight.grad, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(origin.gate.weight.grad, grouped_triton.gate.weight.grad, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(origin.gate.weight.grad, grouped_cutlass.gate.weight.grad, rtol=close_rtol, atol=close_atol)
    torch.testing.assert_close(origin.gate.weight.grad, grouped_ep.gate.weight.grad, rtol=close_rtol, atol=close_atol)
    assert_expert_grads_equal(origin, shard, close_rtol, close_atol)
    assert_expert_grads_equal(origin, grouped, close_rtol, close_atol)
    assert_expert_grads_equal(origin, grouped_triton, close_rtol, close_atol)
    assert_expert_grads_equal(origin, grouped_cutlass, close_rtol, close_atol)
    assert_expert_grads_equal(origin, grouped_ep, close_rtol, close_atol)

    print("origin/shard/grouped/grouped_triton/grouped_cutlass/grouped_ep MoE equivalence: ok")


if __name__ == "__main__":
    main()
