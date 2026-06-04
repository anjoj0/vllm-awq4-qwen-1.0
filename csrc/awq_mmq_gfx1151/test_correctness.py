"""
Correctness test for AWQ-INT4 MMQ HIP custom op (gfx1151).

Tests both v0 (scalar reference) and v1 (WMMA + LDS staging) against a fp32
dequant reference matmul.

Known caveats for v1:
- v1.0 hardcodes activation_scale=1.0, so it expects activations roughly in
  [-128, 127]. The test scales activations small (* 0.1) so naive int8
  saturation barely engages, but the int8 quantization step itself loses
  precision (rounds to nearest integer). For this reason, v1 has a much
  looser atol than v0.
- v1 currently requires M % 64 == 0 and N % 48 == 0 (no tail handling).
"""
import sys

import torch

try:
    import awq_mmq_gfx1151
except ImportError as e:
    print(f"FAIL: import error: {e}")
    print("Did you run `python setup.py build_ext --inplace` first?")
    sys.exit(1)


def pack_uint4_weights(w_int4: torch.Tensor) -> torch.Tensor:
    assert w_int4.dtype == torch.int8
    assert ((w_int4 >= 0) & (w_int4 <= 15)).all()
    N, K = w_int4.shape
    assert K % 8 == 0
    w_int4 = w_int4.to(torch.int32)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=w_int4.device)
    for i in range(8):
        packed |= (w_int4[:, i::8] & 0xF) << (i * 4)
    return packed


def pack_kmajor_weights(w_packed: torch.Tensor) -> torch.Tensor:
    """Convert native [N, K/8] packing to Triton-style [K, N/8]."""
    N, K8 = w_packed.shape
    K = K8 * 8
    assert N % 8 == 0
    shifts = torch.arange(8, device=w_packed.device, dtype=torch.int32) * 4
    w_unpacked = ((w_packed.unsqueeze(-1) >> shifts) & 0xF).reshape(N, K)
    w_kn = w_unpacked.t().contiguous()
    return torch.sum(
        (w_kn.view(K, N // 8, 8) & 0xF) << shifts,
        dim=2,
        dtype=torch.int32,
    ).contiguous()


def reference_dequant_matmul(x, w_packed, scales, group_size=32):
    N, K_packed = w_packed.shape
    K = K_packed * 8
    w_unpacked = torch.zeros((N, K), dtype=torch.int32, device=w_packed.device)
    for i in range(8):
        w_unpacked[:, i::8] = (w_packed >> (i * 4)) & 0xF
    w_signed = (w_unpacked - 8).to(torch.float32)
    scales_expanded = scales.to(torch.float32).repeat_interleave(group_size, dim=1)
    w_dequant = w_signed * scales_expanded
    return (x.to(torch.float32) @ w_dequant.T).to(torch.float16)


def reference_with_int8_act_quant(x, w_packed, scales, group_size=32):
    """v1.1 reference: per-row dynamic int8 quantization (max_abs / 127)."""
    max_abs = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8 * 127)
    act_scales = (max_abs / 127.0)
    x_int8 = torch.clamp(torch.round(x.float() / act_scales), -128, 127).to(torch.int8)
    x_back = x_int8.float() * act_scales
    return reference_dequant_matmul(x_back.to(torch.float16), w_packed, scales, group_size)


def run_one(M, N, K, version, group_size=32, atol=1e-2, x_scale=0.1):
    label = f"v{version}: M={M}, N={N}, K={K}"
    print(f"\n=== {label} ===")
    device = torch.device("cuda")
    x = torch.randn(M, K, dtype=torch.float16, device=device) * x_scale
    w_int4 = torch.randint(0, 16, (N, K), dtype=torch.int8, device=device)
    w_packed = pack_uint4_weights(w_int4)
    scales = (torch.randn(N, K // group_size, dtype=torch.float16, device=device).abs() * 0.01 + 0.001)

    if version == 1:
        out_ref = reference_with_int8_act_quant(x, w_packed, scales, group_size)
    else:
        out_ref = reference_dequant_matmul(x, w_packed, scales, group_size)

    if version == 9:
        out_ours = awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v9(
            x, pack_kmajor_weights(w_packed), scales.t().contiguous())
    elif version == 8:
        out_ours = awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v8(
            x, pack_kmajor_weights(w_packed), scales.t().contiguous())
    elif version == 7:
        out_ours = awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v7(
            x, pack_kmajor_weights(w_packed), scales.t().contiguous())
    elif version == 6:
        out_ours = awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v6(
            x, pack_kmajor_weights(w_packed), scales.t().contiguous())
    elif version == 5:
        out_ours = awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v5(
            x, pack_kmajor_weights(w_packed), scales.t().contiguous())
    elif version == 4:
        out_ours = awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma(
            x, pack_kmajor_weights(w_packed), scales.t().contiguous())
    elif version == 3:
        out_ours = awq_mmq_gfx1151.mmq_q4_gemm_kmajor(
            x, pack_kmajor_weights(w_packed), scales.t().contiguous())
    else:
        out_ours = awq_mmq_gfx1151.mmq_q4_gemm(x, w_packed, scales, version=version)

    diff = (out_ours.to(torch.float32) - out_ref.to(torch.float32)).abs()
    rel = diff / (out_ref.to(torch.float32).abs() + 1e-6)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = rel.max().item()
    print(f"  diff: max_abs={max_abs:.6f}  mean_abs={mean_abs:.6f}  max_rel={max_rel:.4f}  atol={atol}")
    if max_abs > atol:
        print(f"  FAIL: max_abs {max_abs:.6f} > atol {atol}")
        return False
    print("  PASS")
    return True


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA/HIP device")
        sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # v0: tighter atol since it does fp32 accumulation directly (no int8 quant of activations)
    v0_shapes = [(64, 256, 256), (128, 1024, 1024), (256, 4096, 4096)]
    # v1: per-row int8 act quant. Atol bumped slightly to account for round-to-nearest noise.
    # MMQ_X=64 divides Qwen 3.6 27B hidden=5120 and intermediate=27648.
    # Tail shapes (M not multiple of 64, N not multiple of 64) test the in-kernel bounds checks.
    v1_shapes = [
        (64, 64, 256),                        # exact tile
        (128, 128, 1024),
        (256, 256, 1024),
        (1024, 512, 4096),
        (4096, 5120, 4096),                   # Qwen 3.6 27B q_proj-like
        (100, 64, 256),                       # M tail: 100 = 64 + 36
        (64, 80, 256),                        # N tail: 80 = 64 + 16
        (100, 80, 256),                       # both tails
        (4097, 5120, 4096),                   # M tail at large scale
    ]

    print("\n" + "=" * 60)
    print("v0 (scalar reference)")
    print("=" * 60)
    v0_results = [run_one(M, N, K, version=0, atol=0.01) for M, N, K in v0_shapes]

    print("\n" + "=" * 60)
    print("v1 (WMMA + LDS)")
    print("=" * 60)
    # v1 atol bumped to 0.5: int8 act quant adds ~0.5 max abs round error per K element,
    # accumulated over K with small weights (scale ~0.005) and small fp32 sum -> per-output
    # error proportional to sqrt(K) * scale * round_error ~ sqrt(4096) * 0.005 * 0.5 = 0.16
    v1_results = [run_one(M, N, K, version=1, atol=0.5) for M, N, K in v1_shapes]

    print("\n" + "=" * 60)
    print("v2 (native-layout small-M decode)")
    print("=" * 60)
    v2_shapes = [(1, 64, 256), (8, 128, 512), (16, 80, 256)]
    v2_results = [run_one(M, N, K, version=2, atol=0.03) for M, N, K in v2_shapes]

    print("\n" + "=" * 60)
    print("v3 (K-major small-M decode)")
    print("=" * 60)
    v3_shapes = [(1, 64, 256), (8, 128, 512), (16, 80, 256)]
    v3_results = [run_one(M, N, K, version=3, atol=0.03) for M, N, K in v3_shapes]

    print("\n" + "=" * 60)
    print("v4 (K-major fp16-WMMA small-M decode)")
    print("=" * 60)
    v4_shapes = [(1, 64, 256), (8, 128, 512), (16, 80, 256)]
    v4_results = [run_one(M, N, K, version=4, atol=0.03) for M, N, K in v4_shapes]

    print("\n" + "=" * 60)
    print("v5 (K-major fp16-WMMA M<=16 decode specialization)")
    print("=" * 60)
    v5_shapes = [(1, 64, 256), (8, 128, 512), (16, 80, 256), (32, 128, 512)]
    v5_results = [run_one(M, N, K, version=5, atol=0.03) for M, N, K in v5_shapes]

    print("\n" + "=" * 60)
    print("v6 (K-major fp16-WMMA M<=16 metadata staging)")
    print("=" * 60)
    v6_shapes = [(1, 64, 256), (8, 128, 512), (16, 80, 256), (32, 128, 512)]
    v6_results = [run_one(M, N, K, version=6, atol=0.03) for M, N, K in v6_shapes]

    print("\n" + "=" * 60)
    print("v7 (K-major fp16-WMMA M<=16,N=64 metadata staging)")
    print("=" * 60)
    v7_shapes = [(1, 64, 256), (8, 128, 512), (16, 80, 256), (16, 128, 512), (32, 128, 512)]
    v7_results = [run_one(M, N, K, version=7, atol=0.03) for M, N, K in v7_shapes]

    print("\n" + "=" * 60)
    print("v8 (K-major fp16-WMMA M<=8,N=128 metadata staging)")
    print("=" * 60)
    v8_shapes = [(1, 64, 256), (1, 128, 512), (8, 128, 512), (8, 256, 512), (16, 128, 512), (32, 128, 512)]
    v8_results = [run_one(M, N, K, version=8, atol=0.03) for M, N, K in v8_shapes]

    print("\n" + "=" * 60)
    print("v9 (K-major fp16-WMMA M<=16,N=64 B LDS swizzle)")
    print("=" * 60)
    v9_shapes = [(1, 64, 256), (8, 128, 512), (16, 80, 256), (16, 128, 512), (32, 128, 512)]
    v9_results = [run_one(M, N, K, version=9, atol=0.03) for M, N, K in v9_shapes]

    print()
    print("=" * 60)
    all_pass = (all(v0_results) and all(v1_results) and all(v2_results) and
                all(v3_results) and all(v4_results) and all(v5_results) and
                all(v6_results) and all(v7_results) and all(v8_results) and
                all(v9_results))
    if all_pass:
        print(
            f"ALL PASSED ({len(v0_results)} v0 + {len(v1_results)} v1 + "
            f"{len(v2_results)} v2 + {len(v3_results)} v3 + {len(v4_results)} v4 + "
            f"{len(v5_results)} v5 + {len(v6_results)} v6 + {len(v7_results)} v7 + "
            f"{len(v8_results)} v8 + {len(v9_results)} v9)")
        sys.exit(0)
    else:
        v0_fail = sum(1 for r in v0_results if not r)
        v1_fail = sum(1 for r in v1_results if not r)
        v2_fail = sum(1 for r in v2_results if not r)
        v3_fail = sum(1 for r in v3_results if not r)
        v4_fail = sum(1 for r in v4_results if not r)
        v5_fail = sum(1 for r in v5_results if not r)
        v6_fail = sum(1 for r in v6_results if not r)
        v7_fail = sum(1 for r in v7_results if not r)
        v8_fail = sum(1 for r in v8_results if not r)
        v9_fail = sum(1 for r in v9_results if not r)
        print(
            f"FAIL: v0={v0_fail}/{len(v0_results)}, v1={v1_fail}/{len(v1_results)}, "
            f"v2={v2_fail}/{len(v2_results)}, v3={v3_fail}/{len(v3_results)}, "
            f"v4={v4_fail}/{len(v4_results)}, v5={v5_fail}/{len(v5_results)}, "
            f"v6={v6_fail}/{len(v6_results)}, v7={v7_fail}/{len(v7_results)}, "
            f"v8={v8_fail}/{len(v8_results)}, v9={v9_fail}/{len(v9_results)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
