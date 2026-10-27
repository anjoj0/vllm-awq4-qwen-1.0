"""Transient W4 -> FP16 dequantization for large, validated prefill shapes.

Retains the native ExLlama shuffle and asymmetric group32 arithmetic. Only
one layer's dense weight exists at a time; the model is not kept in FP16.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _dequant_skinny(B, S, Z, W, K: tl.constexpr, N: tl.constexpr,
                    G: tl.constexpr, BS: tl.constexpr, T: tl.constexpr):
    flat = tl.program_id(0) * T + tl.arange(0, T)
    n, k = flat // K, flat % K
    packed = tl.load(B + n * BS + k // 8, n < N, 0)
    shift = (k % 8 // 2) * 4 + (k % 2) * 16
    q = (packed >> shift) & 15
    zero = (tl.load(Z + (n // 8) * (K // G) + k // G, n < N, 0)
            >> (4 * (n % 8))) & 15
    scale = tl.load(S + n * (K // G) + k // G, n < N, 1)
    weight = (q - zero).to(scale.dtype) * scale
    tl.store(W + flat, weight, n < N)


def dequant_matmul(a, packed, scales, group_size, zeros, out=None):
    k, n = a.shape[1], packed.shape[0]
    weight = torch.empty((n, k), device=a.device, dtype=a.dtype)
    _dequant_skinny[(triton.cdiv(n * k, 1024),)](
        packed, scales, zeros, weight, k, n, group_size, packed.stride(0), 1024)
    return torch.mm(a, weight.t(), out=out)
