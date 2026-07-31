// pybind11 / TORCH_LIBRARY bindings for the AWQ-INT4 MMQ HIP custom op.
//
// Exposes torch.ops.awq_mmq_gfx1100.mmq_q4_gemm(x, w_packed, scales, version) -> out.
//
// version: 0 = v0 scalar reference, 1 = v1 WMMA + LDS staging, 2 = v2 small-M decode.
//
// Tensor contracts (verified against vLLM v0.20.0 compressed_tensors_wNa16.py):
//   x         : (M, K)        fp16,  CUDA, contiguous
//   w_packed  : (N, K / 8)    int32, CUDA, contiguous
//                 8 uint4 values per int32, low-nibble first (shifts 0,4,...,28).
//                 weight_type = uint4b8: stored values [0,15] decode to signed [-8,7]
//                 via subtraction by 8 (matches llama.cpp's __vsubss4 recenter).
//   scales    : (N, K / 32)   fp16,  CUDA, contiguous (group_size=32 for cyankiwi AWQ4)
//   out       : (M, N)        fp16,  CUDA, contiguous, allocated here

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void launch_mmq_q4_gemm_gfx1100(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    at::Tensor& out,
    int64_t version);

void launch_mmq_q4_gemm_kmajor_gfx1100(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t,
    at::Tensor& out);

void launch_mmq_q4_gemm_kmajor_wmma_gfx1100(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t,
    at::Tensor& out);

void launch_mmq_q4_gemm_kmajor_wmma_v5_gfx1100(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t,
    at::Tensor& out);

void launch_mmq_q4_gemm_kmajor_wmma_v6_gfx1100(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t,
    at::Tensor& out);

void launch_mmq_q4_gemm_kmajor_wmma_v7_gfx1100(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t,
    at::Tensor& out);

void launch_mmq_q4_gemm_kmajor_wmma_v8_gfx1100(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t,
    at::Tensor& out);

void launch_mmq_q4_gemm_kmajor_wmma_v9_gfx1100(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t,
    at::Tensor& out);

namespace {

constexpr int64_t kPackFactor = 8;
constexpr int64_t kGroupSize = 32;

at::Tensor mmq_q4_gemm_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    int64_t version) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_packed.is_cuda(), "w_packed must be CUDA");
    TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");

    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed.scalar_type() == at::kInt, "w_packed must be int32");
    TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be fp16");

    TORCH_CHECK(x.dim() == 2 && w_packed.dim() == 2 && scales.dim() == 2,
                "all inputs must be 2D");
    TORCH_CHECK(x.is_contiguous() && w_packed.is_contiguous() && scales.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = w_packed.size(0);

    TORCH_CHECK(w_packed.size(1) * kPackFactor == K,
                "w_packed last dim mismatch: expected K/8 = ", K / kPackFactor,
                " got ", w_packed.size(1));
    TORCH_CHECK(scales.size(0) == N && scales.size(1) * kGroupSize == K,
                "scales shape mismatch: expected (", N, ", ", K / kGroupSize,
                ") got (", scales.size(0), ", ", scales.size(1), ")");
    TORCH_CHECK(K % kGroupSize == 0, "K must be divisible by group_size (32)");
    TORCH_CHECK(version == 0 || version == 1 || version == 2, "version must be 0 (scalar), 1 (WMMA), or 2 (small-M decode)");

    if (w_zeros.defined() && w_zeros.numel() > 0) {
        // Packed [N/8, K/32] int32, 8 uint4 zeros per int32 (TritonW4A16 layout).
        TORCH_CHECK(w_zeros.is_cuda() && w_zeros.scalar_type() == at::kInt,
                    "w_zeros must be CUDA int32 (packed)");
        TORCH_CHECK(w_zeros.dim() == 2 && w_zeros.size(0) * 8 == N && w_zeros.size(1) * kGroupSize == K,
                    "w_zeros shape mismatch: expected (", N / 8, ", ", K / kGroupSize, ")");
        TORCH_CHECK(w_zeros.is_contiguous(), "w_zeros must be contiguous");
    }

    auto out = at::empty({M, N}, x.options());
    launch_mmq_q4_gemm_gfx1100(x, w_packed, scales, w_zeros, out, version);
    return out;
}



at::Tensor mmq_q4_gemm_kmajor_wmma_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_packed_t.is_cuda(), "w_packed_t must be CUDA");
    TORCH_CHECK(scales_t.is_cuda(), "scales_t must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed_t.scalar_type() == at::kInt, "w_packed_t must be int32");
    TORCH_CHECK(scales_t.scalar_type() == at::kHalf, "scales_t must be fp16");
    TORCH_CHECK(x.dim() == 2 && w_packed_t.dim() == 2 && scales_t.dim() == 2,
                "all inputs must be 2D");
    TORCH_CHECK(x.is_contiguous() && w_packed_t.is_contiguous() && scales_t.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = scales_t.size(1);
    TORCH_CHECK(w_packed_t.size(0) == K && w_packed_t.size(1) * kPackFactor == N,
                "w_packed_t shape mismatch: expected (", K, ", ", N / kPackFactor,
                ") got (", w_packed_t.size(0), ", ", w_packed_t.size(1), ")");
    TORCH_CHECK(scales_t.size(0) * kGroupSize == K,
                "scales_t first dim mismatch: expected K/32");
    TORCH_CHECK(K % kGroupSize == 0, "K must be divisible by group_size (32)");

    if (zeros_t.defined() && zeros_t.numel() > 0) {
        TORCH_CHECK(zeros_t.is_cuda() && zeros_t.scalar_type() == at::kInt,
                    "zeros_t must be CUDA int32");
        TORCH_CHECK(zeros_t.dim() == 2 && zeros_t.size(0) * kGroupSize == K && zeros_t.size(1) * 8 == N,
                    "zeros_t shape mismatch: expected (K/32, N/8)");
        TORCH_CHECK(zeros_t.is_contiguous(), "zeros_t must be contiguous");
    }

    auto out = at::empty({M, N}, x.options());
    launch_mmq_q4_gemm_kmajor_wmma_gfx1100(x, w_packed_t, scales_t, zeros_t, out);
    return out;
}

at::Tensor mmq_q4_gemm_kmajor_wmma_v5_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_packed_t.is_cuda(), "w_packed_t must be CUDA");
    TORCH_CHECK(scales_t.is_cuda(), "scales_t must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed_t.scalar_type() == at::kInt, "w_packed_t must be int32");
    TORCH_CHECK(scales_t.scalar_type() == at::kHalf, "scales_t must be fp16");
    TORCH_CHECK(x.dim() == 2 && w_packed_t.dim() == 2 && scales_t.dim() == 2,
                "all inputs must be 2D");
    TORCH_CHECK(x.is_contiguous() && w_packed_t.is_contiguous() && scales_t.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = scales_t.size(1);
    TORCH_CHECK(w_packed_t.size(0) == K && w_packed_t.size(1) * kPackFactor == N,
                "w_packed_t shape mismatch: expected (", K, ", ", N / kPackFactor,
                ") got (", w_packed_t.size(0), ", ", w_packed_t.size(1), ")");
    TORCH_CHECK(scales_t.size(0) * kGroupSize == K,
                "scales_t first dim mismatch: expected K/32");
    TORCH_CHECK(K % kGroupSize == 0, "K must be divisible by group_size (32)");

    if (zeros_t.defined() && zeros_t.numel() > 0) {
        TORCH_CHECK(zeros_t.is_cuda() && zeros_t.scalar_type() == at::kInt,
                    "zeros_t must be CUDA int32");
        TORCH_CHECK(zeros_t.dim() == 2 && zeros_t.size(0) * kGroupSize == K && zeros_t.size(1) * 8 == N,
                    "zeros_t shape mismatch: expected (K/32, N/8)");
        TORCH_CHECK(zeros_t.is_contiguous(), "zeros_t must be contiguous");
    }

    auto out = at::empty({M, N}, x.options());
    launch_mmq_q4_gemm_kmajor_wmma_v5_gfx1100(x, w_packed_t, scales_t, zeros_t, out);
    return out;
}

at::Tensor mmq_q4_gemm_kmajor_wmma_v6_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_packed_t.is_cuda(), "w_packed_t must be CUDA");
    TORCH_CHECK(scales_t.is_cuda(), "scales_t must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed_t.scalar_type() == at::kInt, "w_packed_t must be int32");
    TORCH_CHECK(scales_t.scalar_type() == at::kHalf, "scales_t must be fp16");
    TORCH_CHECK(x.dim() == 2 && w_packed_t.dim() == 2 && scales_t.dim() == 2,
                "all inputs must be 2D");
    TORCH_CHECK(x.is_contiguous() && w_packed_t.is_contiguous() && scales_t.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = scales_t.size(1);
    TORCH_CHECK(w_packed_t.size(0) == K && w_packed_t.size(1) * kPackFactor == N,
                "w_packed_t shape mismatch: expected (", K, ", ", N / kPackFactor,
                ") got (", w_packed_t.size(0), ", ", w_packed_t.size(1), ")");
    TORCH_CHECK(scales_t.size(0) * kGroupSize == K,
                "scales_t first dim mismatch: expected K/32");
    TORCH_CHECK(K % kGroupSize == 0, "K must be divisible by group_size (32)");

    if (zeros_t.defined() && zeros_t.numel() > 0) {
        TORCH_CHECK(zeros_t.is_cuda() && zeros_t.scalar_type() == at::kInt,
                    "zeros_t must be CUDA int32");
        TORCH_CHECK(zeros_t.dim() == 2 && zeros_t.size(0) * kGroupSize == K && zeros_t.size(1) * 8 == N,
                    "zeros_t shape mismatch: expected (K/32, N/8)");
        TORCH_CHECK(zeros_t.is_contiguous(), "zeros_t must be contiguous");
    }

    auto out = at::empty({M, N}, x.options());
    launch_mmq_q4_gemm_kmajor_wmma_v6_gfx1100(x, w_packed_t, scales_t, zeros_t, out);
    return out;
}

at::Tensor mmq_q4_gemm_kmajor_wmma_v7_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_packed_t.is_cuda(), "w_packed_t must be CUDA");
    TORCH_CHECK(scales_t.is_cuda(), "scales_t must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed_t.scalar_type() == at::kInt, "w_packed_t must be int32");
    TORCH_CHECK(scales_t.scalar_type() == at::kHalf, "scales_t must be fp16");
    TORCH_CHECK(x.dim() == 2 && w_packed_t.dim() == 2 && scales_t.dim() == 2,
                "all inputs must be 2D");
    TORCH_CHECK(x.is_contiguous() && w_packed_t.is_contiguous() && scales_t.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = scales_t.size(1);
    TORCH_CHECK(w_packed_t.size(0) == K && w_packed_t.size(1) * kPackFactor == N,
                "w_packed_t shape mismatch: expected (", K, ", ", N / kPackFactor,
                ") got (", w_packed_t.size(0), ", ", w_packed_t.size(1), ")");
    TORCH_CHECK(scales_t.size(0) * kGroupSize == K,
                "scales_t first dim mismatch: expected K/32");
    TORCH_CHECK(K % kGroupSize == 0, "K must be divisible by group_size (32)");

    if (zeros_t.defined() && zeros_t.numel() > 0) {
        TORCH_CHECK(zeros_t.is_cuda() && zeros_t.scalar_type() == at::kInt,
                    "zeros_t must be CUDA int32");
        TORCH_CHECK(zeros_t.dim() == 2 && zeros_t.size(0) * kGroupSize == K && zeros_t.size(1) * 8 == N,
                    "zeros_t shape mismatch: expected (K/32, N/8)");
        TORCH_CHECK(zeros_t.is_contiguous(), "zeros_t must be contiguous");
    }

    auto out = at::empty({M, N}, x.options());
    launch_mmq_q4_gemm_kmajor_wmma_v7_gfx1100(x, w_packed_t, scales_t, zeros_t, out);
    return out;
}
at::Tensor mmq_q4_gemm_kmajor_wmma_v8_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_packed_t.is_cuda(), "w_packed_t must be CUDA");
    TORCH_CHECK(scales_t.is_cuda(), "scales_t must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed_t.scalar_type() == at::kInt, "w_packed_t must be int32");
    TORCH_CHECK(scales_t.scalar_type() == at::kHalf, "scales_t must be fp16");
    TORCH_CHECK(x.dim() == 2 && w_packed_t.dim() == 2 && scales_t.dim() == 2,
                "all inputs must be 2D");
    TORCH_CHECK(x.is_contiguous() && w_packed_t.is_contiguous() && scales_t.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = scales_t.size(1);
    TORCH_CHECK(w_packed_t.size(0) == K && w_packed_t.size(1) * kPackFactor == N,
                "w_packed_t shape mismatch: expected (", K, ", ", N / kPackFactor,
                ") got (", w_packed_t.size(0), ", ", w_packed_t.size(1), ")");
    TORCH_CHECK(scales_t.size(0) * kGroupSize == K,
                "scales_t first dim mismatch: expected K/32");
    TORCH_CHECK(K % kGroupSize == 0, "K must be divisible by group_size (32)");

    if (zeros_t.defined() && zeros_t.numel() > 0) {
        TORCH_CHECK(zeros_t.is_cuda() && zeros_t.scalar_type() == at::kInt,
                    "zeros_t must be CUDA int32");
        TORCH_CHECK(zeros_t.dim() == 2 && zeros_t.size(0) * kGroupSize == K && zeros_t.size(1) * 8 == N,
                    "zeros_t shape mismatch: expected (K/32, N/8)");
        TORCH_CHECK(zeros_t.is_contiguous(), "zeros_t must be contiguous");
    }

    auto out = at::empty({M, N}, x.options());
    launch_mmq_q4_gemm_kmajor_wmma_v8_gfx1100(x, w_packed_t, scales_t, zeros_t, out);
    return out;
}
at::Tensor mmq_q4_gemm_kmajor_wmma_v9_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_packed_t.is_cuda(), "w_packed_t must be CUDA");
    TORCH_CHECK(scales_t.is_cuda(), "scales_t must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed_t.scalar_type() == at::kInt, "w_packed_t must be int32");
    TORCH_CHECK(scales_t.scalar_type() == at::kHalf, "scales_t must be fp16");
    TORCH_CHECK(x.dim() == 2 && w_packed_t.dim() == 2 && scales_t.dim() == 2,
                "all inputs must be 2D");
    TORCH_CHECK(x.is_contiguous() && w_packed_t.is_contiguous() && scales_t.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = scales_t.size(1);
    TORCH_CHECK(w_packed_t.size(0) == K && w_packed_t.size(1) * kPackFactor == N,
                "w_packed_t shape mismatch: expected (", K, ", ", N / kPackFactor,
                ") got (", w_packed_t.size(0), ", ", w_packed_t.size(1), ")");
    TORCH_CHECK(scales_t.size(0) * kGroupSize == K,
                "scales_t first dim mismatch: expected K/32");
    TORCH_CHECK(K % kGroupSize == 0, "K must be divisible by group_size (32)");

    if (zeros_t.defined() && zeros_t.numel() > 0) {
        TORCH_CHECK(zeros_t.is_cuda() && zeros_t.scalar_type() == at::kInt,
                    "zeros_t must be CUDA int32");
        TORCH_CHECK(zeros_t.dim() == 2 && zeros_t.size(0) * kGroupSize == K && zeros_t.size(1) * 8 == N,
                    "zeros_t shape mismatch: expected (K/32, N/8)");
        TORCH_CHECK(zeros_t.is_contiguous(), "zeros_t must be contiguous");
    }

    auto out = at::empty({M, N}, x.options());
    launch_mmq_q4_gemm_kmajor_wmma_v9_gfx1100(x, w_packed_t, scales_t, zeros_t, out);
    return out;
}

at::Tensor mmq_q4_gemm_kmajor_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed_t,
    const at::Tensor& scales_t,
    const at::Tensor& zeros_t) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_packed_t.is_cuda(), "w_packed_t must be CUDA");
    TORCH_CHECK(scales_t.is_cuda(), "scales_t must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed_t.scalar_type() == at::kInt, "w_packed_t must be int32");
    TORCH_CHECK(scales_t.scalar_type() == at::kHalf, "scales_t must be fp16");
    TORCH_CHECK(x.dim() == 2 && w_packed_t.dim() == 2 && scales_t.dim() == 2,
                "all inputs must be 2D");
    TORCH_CHECK(x.is_contiguous() && w_packed_t.is_contiguous() && scales_t.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = scales_t.size(1);
    TORCH_CHECK(w_packed_t.size(0) == K && w_packed_t.size(1) * kPackFactor == N,
                "w_packed_t shape mismatch: expected (", K, ", ", N / kPackFactor,
                ") got (", w_packed_t.size(0), ", ", w_packed_t.size(1), ")");
    TORCH_CHECK(scales_t.size(0) * kGroupSize == K,
                "scales_t first dim mismatch: expected K/32");
    TORCH_CHECK(K % kGroupSize == 0, "K must be divisible by group_size (32)");

    if (zeros_t.defined() && zeros_t.numel() > 0) {
        TORCH_CHECK(zeros_t.is_cuda() && zeros_t.scalar_type() == at::kInt,
                    "zeros_t must be CUDA int32");
        TORCH_CHECK(zeros_t.dim() == 2 && zeros_t.size(0) * kGroupSize == K && zeros_t.size(1) * 8 == N,
                    "zeros_t shape mismatch: expected (K/32, N/8)");
        TORCH_CHECK(zeros_t.is_contiguous(), "zeros_t must be contiguous");
    }

    auto out = at::empty({M, N}, x.options());
    launch_mmq_q4_gemm_kmajor_gfx1100(x, w_packed_t, scales_t, zeros_t, out);
    return out;
}

}  // namespace

TORCH_LIBRARY(awq_mmq_gfx1100, m) {
    m.def("mmq_q4_gemm(Tensor x, Tensor w_packed, Tensor scales, Tensor w_zeros, int version) -> Tensor");
    m.def("mmq_q4_gemm_kmajor(Tensor x, Tensor w_packed_t, Tensor scales_t, Tensor zeros_t) -> Tensor");
    m.def("mmq_q4_gemm_kmajor_wmma(Tensor x, Tensor w_packed_t, Tensor scales_t, Tensor zeros_t) -> Tensor");
    m.def("mmq_q4_gemm_kmajor_wmma_v5(Tensor x, Tensor w_packed_t, Tensor scales_t, Tensor zeros_t) -> Tensor");
    m.def("mmq_q4_gemm_kmajor_wmma_v6(Tensor x, Tensor w_packed_t, Tensor scales_t, Tensor zeros_t) -> Tensor");
    m.def("mmq_q4_gemm_kmajor_wmma_v7(Tensor x, Tensor w_packed_t, Tensor scales_t, Tensor zeros_t) -> Tensor");
    m.def("mmq_q4_gemm_kmajor_wmma_v8(Tensor x, Tensor w_packed_t, Tensor scales_t, Tensor zeros_t) -> Tensor");
    m.def("mmq_q4_gemm_kmajor_wmma_v9(Tensor x, Tensor w_packed_t, Tensor scales_t, Tensor zeros_t) -> Tensor");
}

TORCH_LIBRARY_IMPL(awq_mmq_gfx1100, CUDA, m) {
    m.impl("mmq_q4_gemm", &mmq_q4_gemm_forward);
    m.impl("mmq_q4_gemm_kmajor", &mmq_q4_gemm_kmajor_forward);
    m.impl("mmq_q4_gemm_kmajor_wmma", &mmq_q4_gemm_kmajor_wmma_forward);
    m.impl("mmq_q4_gemm_kmajor_wmma_v5", &mmq_q4_gemm_kmajor_wmma_v5_forward);
    m.impl("mmq_q4_gemm_kmajor_wmma_v6", &mmq_q4_gemm_kmajor_wmma_v6_forward);
    m.impl("mmq_q4_gemm_kmajor_wmma_v7", &mmq_q4_gemm_kmajor_wmma_v7_forward);
    m.impl("mmq_q4_gemm_kmajor_wmma_v8", &mmq_q4_gemm_kmajor_wmma_v8_forward);
    m.impl("mmq_q4_gemm_kmajor_wmma_v9", &mmq_q4_gemm_kmajor_wmma_v9_forward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "AWQ-INT4 MMQ kernel for gfx1100. torch.ops.awq_mmq_gfx1100.mmq_q4_gemm(x, w_packed, scales, version)";
}
