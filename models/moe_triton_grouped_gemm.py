from __future__ import annotations

import os
from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from models.moe_profile import record_moe_profile_phase


try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _env_nonnegative_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}")
    return parsed


SM_MARGIN = _env_nonnegative_int(
    "HRM_MOE_TRITON_SM_MARGIN",
    _env_nonnegative_int("XTUNER_SM_MARGIN", 0),
)
M_GROUPED_BLOCK_M = _env_positive_int("HRM_MOE_TRITON_BLOCK_M", 128)


def _available_sms(device: torch.device) -> int:
    return max(1, torch.cuda.get_device_properties(device).multi_processor_count - SM_MARGIN)


def _torch_grouped_linear(input: Tensor, weight: Tensor, tokens_per_expert: Tensor) -> Tensor:
    outputs = []
    start = 0
    for expert_idx, tokens in enumerate(tokens_per_expert.detach().cpu().tolist()):
        end = start + int(tokens)
        if end > start:
            outputs.append(F.linear(input[start:end], weight[expert_idx]))
        start = end
    if not outputs:
        return input @ weight[0].T
    return torch.cat(outputs, dim=0)


if _TRITON_AVAILABLE:

    def _maybe_autotune(configs: list):
        if os.environ.get("HRM_MOE_TRITON_AUTOTUNE", "0") == "1":
            return configs
        return configs[:1]


    def _m_gemm_autotune_config():
        return _maybe_autotune([
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 6}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 10}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 14}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 6}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 10}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 14}, num_stages=3, num_warps=8),
        ])


    def _k_gemm_autotune_config():
        return _maybe_autotune([
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 6}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 10}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 14}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 6}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 10}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 14}, num_stages=3, num_warps=8),
        ])


    @triton.jit
    def _grouped_launch(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
        grid_m = tl.cdiv(m, block_m)
        grid_n = tl.cdiv(n, block_n)
        width = group_m * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * group_m, group_m)
        remaining_pid = pid - group_id * width
        pid_m = group_id * group_m + (remaining_pid % group_size)
        pid_n = (pid % width) // group_size
        return pid_m, pid_n


    @triton.autotune(configs=_m_gemm_autotune_config(), key=["N", "K"])
    @triton.jit
    def _m_grouped_gemm_bk_kernel(
        A,
        B,
        C,
        pad_starts,
        group_starts,
        group_ends,
        m_indices_pad,
        M_pad_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        dtype_a: tl.constexpr,
        dtype_b: tl.constexpr,
        dtype_c: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        dtypeA = tl.bfloat16 if dtype_a == 0 else tl.float16
        dtypeB = tl.bfloat16 if dtype_b == 0 else tl.float16
        dtypeC = tl.bfloat16 if dtype_c == 0 else tl.float16
        blocks = tl.num_programs(axis=0)
        start_pid = tl.program_id(axis=0)
        M_pad = tl.load(M_pad_ptr)
        num_pid_m = tl.cdiv(M_pad, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_tiles = num_pid_m * num_pid_n

        for tile_id in tl.range(start_pid, num_tiles, blocks):
            pid_m, pid_n = _grouped_launch(tile_id, M_pad, N, BLOCK_M, BLOCK_N, GROUP_M)
            group = tl.load(m_indices_pad + pid_m).to(tl.int32)
            pad_off = tl.load(pad_starts + group).to(tl.int32)
            group_start = (tl.load(group_starts + group) + (pid_m * BLOCK_M - pad_off)).to(tl.int32)
            group_end = tl.load(group_ends + group).to(tl.int32)

            a_ptr = A.to(tl.pointer_type(dtypeA))
            b_ptr = B.to(tl.pointer_type(dtypeB))
            c_ptr = C.to(tl.pointer_type(dtypeC))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[group_end, K],
                strides=[K, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[(group + 1) * N, K],
                strides=[K, 1],
                block_shape=[BLOCK_N, BLOCK_K],
            )
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[group_end, N],
                strides=[N, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )

            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            offs_k = 0
            offs_bn = (pid_n * BLOCK_N).to(tl.int32)
            for _ in tl.range(0, tl.cdiv(K, BLOCK_K)):
                a = a_desc.load([group_start, offs_k])
                b = b_desc.load([group * N + offs_bn, offs_k])
                accumulator = tl.dot(a, b.T, acc=accumulator, input_precision="tf32x3")
                offs_k += BLOCK_K

            c_desc.store([group_start, offs_bn], accumulator.to(dtypeC))


    @triton.autotune(configs=_m_gemm_autotune_config(), key=["N", "K"])
    @triton.jit
    def _m_grouped_gemm_bn_kernel(
        A,
        B,
        C,
        pad_starts,
        group_starts,
        group_ends,
        m_indices_pad,
        M_pad_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        dtype_a: tl.constexpr,
        dtype_b: tl.constexpr,
        dtype_c: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        dtypeA = tl.bfloat16 if dtype_a == 0 else tl.float16
        dtypeB = tl.bfloat16 if dtype_b == 0 else tl.float16
        dtypeC = tl.bfloat16 if dtype_c == 0 else tl.float16
        blocks = tl.num_programs(axis=0)
        start_pid = tl.program_id(axis=0)
        M_pad = tl.load(M_pad_ptr)
        num_pid_m = tl.cdiv(M_pad, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_tiles = num_pid_m * num_pid_n

        for tile_id in tl.range(start_pid, num_tiles, blocks):
            pid_m, pid_n = _grouped_launch(tile_id, M_pad, N, BLOCK_M, BLOCK_N, GROUP_M)
            group = tl.load(m_indices_pad + pid_m).to(tl.int32)
            pad_off = tl.load(pad_starts + group).to(tl.int32)
            group_start = (tl.load(group_starts + group) + (pid_m * BLOCK_M - pad_off)).to(tl.int32)
            group_end = tl.load(group_ends + group).to(tl.int32)

            a_ptr = A.to(tl.pointer_type(dtypeA))
            b_ptr = B.to(tl.pointer_type(dtypeB))
            c_ptr = C.to(tl.pointer_type(dtypeC))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[group_end, K],
                strides=[K, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[(group + 1) * K, N],
                strides=[N, 1],
                block_shape=[BLOCK_K, BLOCK_N],
            )
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[group_end, N],
                strides=[N, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )

            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            offs_k = 0
            offs_bk = 0
            offs_bn = (pid_n * BLOCK_N).to(tl.int32)
            for _ in tl.range(0, tl.cdiv(K, BLOCK_K)):
                a = a_desc.load([group_start, offs_k])
                b = b_desc.load([group * K + offs_bk, offs_bn])
                accumulator = tl.dot(a, b, acc=accumulator, input_precision="tf32x3")
                offs_k += BLOCK_K
                offs_bk += BLOCK_K

            c_desc.store([group_start, offs_bn], accumulator.to(dtypeC))


    @triton.jit
    def _repeat_interleave_kernel(group_ptr, repeats_ptr, repeat_cum_ptr, output_ptr):
        pid = tl.program_id(axis=0)
        repeat = tl.load(repeats_ptr + pid)
        start = tl.load(repeat_cum_ptr + pid) - repeat
        group = tl.load(group_ptr + pid)
        for r in range(repeat):
            tl.store(output_ptr + start + r, group)


    def _repeat_interleave(group_indices: Tensor, repeats: Tensor, repeat_cum: Tensor, output: Tensor) -> None:
        _repeat_interleave_kernel[(len(repeats),)](group_indices, repeats, repeat_cum, output)


    @triton.autotune(configs=_k_gemm_autotune_config(), key=["M", "N"])
    @triton.jit
    def _k_grouped_gemm_kernel(
        A,
        B,
        C,
        group_starts,
        group_ends,
        num_groups: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K,
        dtype_a: tl.constexpr,
        dtype_b: tl.constexpr,
        dtype_c: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        dtypeA = tl.bfloat16 if dtype_a == 0 else tl.float16
        dtypeB = tl.bfloat16 if dtype_b == 0 else tl.float16
        dtypeC = tl.bfloat16 if dtype_c == 0 else tl.float16
        blocks = tl.num_programs(axis=0)
        start_pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_tiles = num_pid_m * num_pid_n * num_groups

        for tile_id in tl.range(start_pid, num_tiles, blocks):
            group = tile_id // (num_pid_m * num_pid_n)
            group_start = tl.load(group_starts + group).to(tl.int32)
            group_end = tl.load(group_ends + group).to(tl.int32)
            tile_in_group = tile_id % (num_pid_m * num_pid_n)
            pid_m, pid_n = _grouped_launch(tile_in_group, M, N, BLOCK_M, BLOCK_N, GROUP_M)
            tokens = group_end - group_start

            a_ptr = A.to(tl.pointer_type(dtypeA))
            b_ptr = B.to(tl.pointer_type(dtypeB))
            c_ptr = C.to(tl.pointer_type(dtypeC))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[group_end, M],
                strides=[M, 1],
                block_shape=[BLOCK_K, BLOCK_M],
            )
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[group_end, N],
                strides=[N, 1],
                block_shape=[BLOCK_K, BLOCK_N],
            )
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[(group + 1) * M, N],
                strides=[N, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )

            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            offs_k = 0
            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N
            for _ in range(0, tl.cdiv(tokens, BLOCK_K)):
                a = a_desc.load([group_start + offs_k, offs_am])
                b = b_desc.load([group_start + offs_k, offs_bn])
                accumulator = tl.dot(a.T, b, acc=accumulator, input_precision="tf32x3")
                offs_k += BLOCK_K

            c_desc.store([group * M + offs_am, offs_bn], accumulator.to(dtypeC))


    def _dtype_id(dtype: torch.dtype) -> int:
        if dtype == torch.bfloat16:
            return 0
        if dtype == torch.float16:
            return 1
        raise TypeError(f"Triton grouped GEMM only supports bf16/fp16, got {dtype}")


    def _m_grouped_gemm(input: Tensor, weight: Tensor, tokens_per_expert: Tensor, trans_b: bool) -> Tensor:
        if input.shape[0] == 0:
            return input @ (weight[0] if not trans_b else weight[0].T)

        M, K = input.shape
        if trans_b:
            num_groups, N, weight_k = weight.shape
        else:
            num_groups, weight_k, N = weight.shape
        if weight_k != K:
            raise ValueError(f"grouped GEMM K mismatch: input K={K}, weight K={weight_k}")

        input = input.contiguous()
        weight = weight.contiguous()
        tokens_per_expert = tokens_per_expert.to(device=input.device, dtype=torch.int64)
        output = input.new_empty(M, N)

        block_m = M_GROUPED_BLOCK_M
        m_per_group_padding = triton.cdiv(tokens_per_expert, block_m) * block_m
        m_pad = m_per_group_padding.sum()
        repeats = (m_per_group_padding // block_m).to(torch.int32)
        m_indices_pad = torch.empty(M // block_m + num_groups, device=input.device, dtype=torch.int64)
        _repeat_interleave(
            torch.arange(num_groups, device=input.device, dtype=torch.int32),
            repeats,
            repeats.cumsum(0),
            m_indices_pad,
        )
        pad_start = m_per_group_padding.cumsum(0) - m_per_group_padding
        group_end = tokens_per_expert.cumsum(0)
        group_start = group_end - tokens_per_expert

        num_sms = _available_sms(input.device)

        def grid(_meta):
            return (num_sms,)

        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device=input.device, dtype=torch.int8)

        triton.set_allocator(alloc_fn)
        kernel = _m_grouped_gemm_bk_kernel if trans_b else _m_grouped_gemm_bn_kernel
        kernel[grid](
            input,
            weight,
            output,
            pad_start,
            group_start,
            group_end,
            m_indices_pad,
            m_pad,
            M,
            N,
            K,
            _dtype_id(input.dtype),
            _dtype_id(weight.dtype),
            _dtype_id(output.dtype),
            BLOCK_M=block_m,
        )
        return output


    def _k_grouped_gemm(A: Tensor, B: Tensor, tokens_per_expert: Tensor) -> Tensor:
        A = A.contiguous()
        B = B.contiguous()
        K, M = A.shape
        K_b, N = B.shape
        if K != K_b:
            raise ValueError(f"grouped wgrad K mismatch: A K={K}, B K={K_b}")

        tokens_per_expert = tokens_per_expert.to(device=A.device, dtype=torch.int64)
        num_groups = tokens_per_expert.shape[0]
        C = A.new_empty(num_groups, M, N)
        group_end = tokens_per_expert.cumsum(0)
        group_start = group_end - tokens_per_expert
        num_sms = _available_sms(A.device)

        def grid(_meta):
            return (num_sms,)

        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device=A.device, dtype=torch.int8)

        triton.set_allocator(alloc_fn)
        _k_grouped_gemm_kernel[grid](
            A,
            B,
            C,
            group_start,
            group_end,
            num_groups,
            M,
            N,
            K,
            _dtype_id(A.dtype),
            _dtype_id(B.dtype),
            _dtype_id(C.dtype),
        )
        return C


class _TritonGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, tokens_per_expert: Tensor) -> Tensor:
        output = _m_grouped_gemm(input, weight, tokens_per_expert, trans_b=True)
        ctx.save_for_backward(input, weight, tokens_per_expert)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, weight, tokens_per_expert = ctx.saved_tensors
        with record_moe_profile_phase("grad_output_contiguous"):
            grad_output = grad_output.contiguous()
        with record_moe_profile_phase("grad_input_gemm"):
            grad_input = _m_grouped_gemm(grad_output, weight, tokens_per_expert, trans_b=False)
        with record_moe_profile_phase("grad_weight_gemm"):
            grad_weight = _k_grouped_gemm(grad_output, input, tokens_per_expert)
        return grad_input, grad_weight, None


def triton_grouped_linear(input: Tensor, weight: Tensor, tokens_per_expert: Tensor) -> Tensor:
    if input.shape[0] == 0:
        return input @ weight[0].T
    if not input.is_cuda:
        return _torch_grouped_linear(input, weight, tokens_per_expert)
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton grouped GEMM requested on CUDA, but Triton is not available")
    return _TritonGroupedLinear.apply(input, weight, tokens_per_expert)
