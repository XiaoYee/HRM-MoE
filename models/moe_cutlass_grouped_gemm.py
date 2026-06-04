from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


try:
    from grouped_gemm import backend as _grouped_gemm_backend
except Exception:
    _grouped_gemm_backend = None


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


class _CutlassGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, tokens_per_expert: Tensor) -> Tensor:
        input = input.contiguous()
        weight = weight.contiguous()
        batch_sizes = tokens_per_expert.to(device="cpu", dtype=torch.int64)
        output = _grouped_gemm_backend.gmm(input, weight, batch_sizes, trans_a=False, trans_b=True)
        ctx.save_for_backward(input, weight, batch_sizes)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, weight, batch_sizes = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = _grouped_gemm_backend.gmm(grad_output, weight, batch_sizes, trans_a=False, trans_b=False)
        grad_weight = _grouped_gemm_backend.gmm(grad_output, input, batch_sizes, trans_a=True, trans_b=False)
        return grad_input, grad_weight, None


def cutlass_grouped_linear(input: Tensor, weight: Tensor, tokens_per_expert: Tensor) -> Tensor:
    if input.shape[0] == 0:
        return input @ weight[0].T
    if not input.is_cuda:
        return _torch_grouped_linear(input, weight, tokens_per_expert)
    if _grouped_gemm_backend is None:
        raise RuntimeError("CUTLASS grouped GEMM requested on CUDA, but grouped_gemm is not available")
    return _CutlassGroupedLinear.apply(input, weight, tokens_per_expert)
