#!/usr/bin/env python
"""Distributed correctness check for grouped_ep MoE."""

import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

flash_prefixlm_stub = types.ModuleType("models.flash_attention_prefixlm_v2")
flash_prefixlm_stub.flash_attn_varlen_prefixlm = None
sys.modules.setdefault("models.flash_attention_prefixlm_v2", flash_prefixlm_stub)

flash_interface_stub = types.ModuleType("flash_attn_interface")
flash_interface_stub.flash_attn_with_kvcache = None
sys.modules.setdefault("flash_attn_interface", flash_interface_stub)

from models.layers import SparseMoESwiGLU


def _zero_if_none(grad: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(ref) if grad is None else grad.clone()


def _assert_close_with_stats(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    rank: int,
) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError:
        with torch.no_grad():
            actual_f = actual.detach().float()
            expected_f = expected.detach().float()
            diff = actual_f - expected_f
            actual_norm = actual_f.norm()
            expected_norm = expected_f.norm()
            diff_norm = diff.norm()
            denom = actual_norm * expected_norm
            cosine = (actual_f.flatten() @ expected_f.flatten()) / denom.clamp_min(1e-30)
            ratio = actual_norm / expected_norm.clamp_min(1e-30)
            print(
                f"[rank{rank}] {name} mismatch: "
                f"actual_norm={actual_norm.item():.6g} expected_norm={expected_norm.item():.6g} "
                f"diff_norm={diff_norm.item():.6g} norm_ratio={ratio.item():.6g} "
                f"cosine={cosine.item():.6g} max_abs={diff.abs().max().item():.6g}",
                flush=True,
            )
        raise


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed EP equivalence")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    hidden_size = int(os.environ.get("MOE_EP_EQUIV_HIDDEN", "128"))
    intermediate_size = int(os.environ.get("MOE_EP_EQUIV_INTERMEDIATE", "64"))
    num_experts = int(os.environ.get("MOE_EP_EQUIV_EXPERTS", str(world_size)))
    top_k = int(os.environ.get("MOE_EP_EQUIV_TOPK", "2"))
    batch_size = int(os.environ.get("MOE_EP_EQUIV_BATCH", "2"))
    seq_len = int(os.environ.get("MOE_EP_EQUIV_SEQ", "8"))
    dtype_name = os.environ.get("MOE_EP_EQUIV_DTYPE", "bfloat16")
    backend = os.environ.get("MOE_EP_EQUIV_BACKEND", "bmm")
    rtol = float(os.environ.get("MOE_EP_EQUIV_RTOL", "3e-2"))
    atol = float(os.environ.get("MOE_EP_EQUIV_ATOL", "2e-2"))

    if num_experts % world_size != 0:
        raise RuntimeError(f"{num_experts=} must divide {world_size=}")
    if dtype_name not in ("bfloat16", "float32"):
        raise RuntimeError(f"Unsupported {dtype_name=}; expected bfloat16 or float32")
    if backend not in ("loop", "bmm", "triton", "cutlass"):
        raise RuntimeError(f"Unsupported {backend=}; expected loop, bmm, triton, or cutlass")
    os.environ["HRM_MOE_EP_BACKEND"] = backend
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32

    kwargs = dict(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        norm_topk_prob=True,
    )

    torch.manual_seed(1234)
    origin = SparseMoESwiGLU(**kwargs, implementation="origin", expert_in_one_shard=1).to(device=device, dtype=dtype)
    ep = SparseMoESwiGLU(**kwargs, implementation="grouped_ep", expert_in_one_shard=1).to(device=device, dtype=dtype)

    with torch.no_grad():
        ep.gate.weight.copy_(origin.gate.weight)
        local_experts = ep.experts.local_experts
        start = ep.experts.local_expert_start
        for local_idx in range(ep.experts.experts_per_rank):
            origin_expert = origin.experts[start + local_idx]
            local_experts.gate_up_weight[local_idx].copy_(origin_expert.gate_up_proj.weight)
            local_experts.down_weight[local_idx].copy_(origin_expert.down_proj.weight)

    torch.manual_seed(9000 + rank)
    x_origin = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
    x_ep = x_origin.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_origin)

    origin_context = {"aux_losses": [], "expert_counts": []}
    ep_context = {"aux_losses": [], "expert_counts": []}
    out_origin = origin(x_origin, moe_context=origin_context)
    out_ep = ep(x_ep, moe_context=ep_context)
    _assert_close_with_stats("output", out_ep, out_origin, rtol=rtol, atol=atol, rank=rank)
    _assert_close_with_stats(
        "aux_loss",
        torch.stack(origin_context["aux_losses"]),
        torch.stack(ep_context["aux_losses"]),
        rtol=1e-4,
        atol=1e-5,
        rank=rank,
    )
    _assert_close_with_stats(
        "expert_counts",
        torch.stack(origin_context["expert_counts"]),
        torch.stack(ep_context["expert_counts"]),
        rtol=0,
        atol=0,
        rank=rank,
    )

    loss_origin = (out_origin * grad).sum() + torch.stack(origin_context["aux_losses"]).sum()
    loss_ep = (out_ep * grad).sum() + torch.stack(ep_context["aux_losses"]).sum()
    loss_origin.backward()
    loss_ep.backward()

    _assert_close_with_stats("input_grad", x_ep.grad, x_origin.grad, rtol=rtol, atol=atol, rank=rank)
    _assert_close_with_stats("router_grad", ep.gate.weight.grad, origin.gate.weight.grad, rtol=rtol, atol=atol, rank=rank)

    local_experts = ep.experts.local_experts
    start = ep.experts.local_expert_start
    global_counts = origin_context["expert_counts"][0].clone()
    dist.all_reduce(global_counts)
    for expert_id in range(num_experts):
        origin_expert = origin.experts[expert_id]
        origin_gate_grad = _zero_if_none(origin_expert.gate_up_proj.weight.grad, origin_expert.gate_up_proj.weight)
        origin_down_grad = _zero_if_none(origin_expert.down_proj.weight.grad, origin_expert.down_proj.weight)
        dist.all_reduce(origin_gate_grad)
        dist.all_reduce(origin_down_grad)

        owner_rank = expert_id // ep.experts.experts_per_rank
        if owner_rank != rank:
            continue

        local_idx = expert_id - start
        _assert_close_with_stats(
            f"expert_{expert_id}_gate_up_grad count={global_counts[expert_id].item():.0f}",
            local_experts.gate_up_weight.grad[local_idx],
            origin_gate_grad,
            rtol=rtol,
            atol=atol,
            rank=rank,
        )
        _assert_close_with_stats(
            f"expert_{expert_id}_down_grad count={global_counts[expert_id].item():.0f}",
            local_experts.down_weight.grad[local_idx],
            origin_down_grad,
            rtol=rtol,
            atol=atol,
            rank=rank,
        )

    if rank == 0:
        print(f"distributed grouped_ep MoE equivalence: ok ({backend=}, {dtype_name=})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
