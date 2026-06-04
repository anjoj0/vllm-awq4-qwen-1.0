import torch  # MUST come before `from . import _C` so libc10.so / libtorch.so are dlopen'd first

from . import _C  # noqa: F401


def mmq_q4_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    version: int = 0,
    w_zeros: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    AWQ-INT4 MMQ kernel for gfx1151.

    version: 0 = v0 scalar reference (always-correct, slow)
             1 = v1 WMMA + LDS staging
             2 = v2 small-M decode
    w_zeros: optional (N, K/32) int8 per-group zero points (asymmetric quant);
             pass None for symmetric uint4b8 (kernel uses zero=8 baseline).
    """
    if w_zeros is None:
        w_zeros = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.awq_mmq_gfx1151.mmq_q4_gemm(x, w_packed, scales, w_zeros, version)


def mmq_q4_gemm_kmajor(
    x: torch.Tensor,
    w_packed_t: torch.Tensor,
    scales_t: torch.Tensor,
    zeros_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode-oriented K-major AWQ-INT4 GEMM for small-M shapes."""
    if zeros_t is None:
        zeros_t = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor(
        x, w_packed_t, scales_t, zeros_t)


def mmq_q4_gemm_kmajor_wmma(
    x: torch.Tensor,
    w_packed_t: torch.Tensor,
    scales_t: torch.Tensor,
    zeros_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """TritonW4A16-style K-major fp16-WMMA decode GEMM for small-M shapes."""
    if zeros_t is None:
        zeros_t = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma(
        x, w_packed_t, scales_t, zeros_t)



def mmq_q4_gemm_kmajor_wmma_v5(
    x: torch.Tensor,
    w_packed_t: torch.Tensor,
    scales_t: torch.Tensor,
    zeros_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """K-major fp16-WMMA decode GEMM with an M<=16 specialization."""
    if zeros_t is None:
        zeros_t = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v5(
        x, w_packed_t, scales_t, zeros_t)



def mmq_q4_gemm_kmajor_wmma_v6(
    x: torch.Tensor,
    w_packed_t: torch.Tensor,
    scales_t: torch.Tensor,
    zeros_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """K-major fp16-WMMA decode GEMM with an M<=16,N=64 specialization."""
    if zeros_t is None:
        zeros_t = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v6(
        x, w_packed_t, scales_t, zeros_t)



def mmq_q4_gemm_kmajor_wmma_v7(
    x: torch.Tensor,
    w_packed_t: torch.Tensor,
    scales_t: torch.Tensor,
    zeros_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """K-major fp16-WMMA decode GEMM with M<=16,N=64 metadata staging."""
    if zeros_t is None:
        zeros_t = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v7(
        x, w_packed_t, scales_t, zeros_t)


def mmq_q4_gemm_kmajor_wmma_v8(
    x: torch.Tensor,
    w_packed_t: torch.Tensor,
    scales_t: torch.Tensor,
    zeros_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """K-major fp16-WMMA decode GEMM with M<=8,N=128 metadata staging."""
    if zeros_t is None:
        zeros_t = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v8(
        x, w_packed_t, scales_t, zeros_t)


def mmq_q4_gemm_kmajor_wmma_v9(
    x: torch.Tensor,
    w_packed_t: torch.Tensor,
    scales_t: torch.Tensor,
    zeros_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """K-major fp16-WMMA decode GEMM with M<=16,N=64 B LDS swizzle."""
    if zeros_t is None:
        zeros_t = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v9(
        x, w_packed_t, scales_t, zeros_t)
