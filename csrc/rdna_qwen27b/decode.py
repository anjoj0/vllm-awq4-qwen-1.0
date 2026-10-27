"""Launch the existing W4A16 arithmetic with a qualified small-M tile."""
import torch
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm_kernel,
)
from vllm.utils.torch_utils import direct_register_custom_op


def _down_gemm(a: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
               zeros: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = weight.shape[1] * 8
    # Called only by Qwen27BDownProjectionKernel's qualified 3..9-row path.
    assert 3 <= m <= 9 and (k, n) == (17408, 5120)
    result = torch.empty((m, n), device=a.device, dtype=a.dtype)
    triton_w4a16_gemm_kernel[(1, n // 64)](
        a, weight, scales, zeros, result, m, n, k,
        a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
        result.stride(0), result.stride(1),
        group_size=32, HAS_ZP=True, ZP_BIAS=0,
        BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=1,
    )
    return result


def _fake(a, weight, scales, zeros):
    return torch.empty((a.shape[0], weight.shape[1] * 8), device=a.device, dtype=a.dtype)


direct_register_custom_op("qwen27b_down_gemm", _down_gemm, mutates_args=[], fake_impl=_fake)


def down_gemm(a, weight, scales, zeros):
    return torch.ops.vllm.qwen27b_down_gemm(a, weight, scales, zeros)
