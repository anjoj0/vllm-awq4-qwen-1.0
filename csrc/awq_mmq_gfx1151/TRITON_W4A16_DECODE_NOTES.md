# TritonW4A16 Decode IR Notes

Date: 2026-06-01
Target: ROCm gfx1151 / Strix Halo, Qwen3.6 AWQ W4A16 decode path.

## Cache Artifacts Studied

Generated Triton artifacts are in the persisted host cache:

- `.triton-cache/WVA6Z7EMLFNAQVIPBHJW4LLHZCVJDKE3WGBRO2PN6BRSLKUULOFQ/triton_w4a16_gemm_kernel.*`
- `.triton-cache/HNKAGL3LVVXS5K74PXO7UR7CO6UM5NRVV7FECV3QPH4QG36ATEZA/triton_w4a16_gemm_kernel.*`
- `.triton-cache/JZ5KORFTIT6VHH36WPSOTLNK272EGEFMOKTVNVHBNIO4JEKARUHQ/triton_w4a16_gemm_kernel.*`

The first and third cache entries are the small-decode shape of interest. Their metadata reports `backend=hip`, `arch=gfx1151`, `warp_size=32`, `num_warps=4`, `num_stages=2`; the small shape uses 4096 bytes of LDS/shared memory.

The vLLM source inside the container is:

- `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py`

## Launch Shape

The gfx1x-side launch selection in vLLM is shape dependent:

- `M <= 32`: `BLOCK_M=32`, `BLOCK_N=32`, `BLOCK_K=64`
- `M <= 64`: `BLOCK_M=64`, `BLOCK_N=64`, `BLOCK_K=32`
- otherwise: `BLOCK_M=128`, `BLOCK_N=32`, `BLOCK_K=64`

For this model, AWQ group size is 32, so the small decode shape clamps `BLOCK_K` to 32. The effective key tile is therefore `32x32x32` for decode.

## IR Structure

The important observation is that TritonW4A16 is not using a direct int4 dot instruction on gfx1151.

The IR path is:

1. Load packed int32 W4 data from K-major `b_q` layout, shaped as `[K, N / 8]`.
2. Use shift/mask to unpack eight 4-bit weights from each int32 word.
3. Load per-group scales from `[K / group_size, N]`.
4. Optionally load packed zero points from `[K / group_size, N / 8]` and unpack them with the same nibble pattern.
5. Convert unpacked weights to fp16, apply `(q - zero) * scale`.
6. Feed the resulting fp16 B tile and fp16 A tile to Triton `tt.dot`.
7. Lower to AMD WMMA fp16 input, fp32 accumulate instructions.

Representative TTIR/TTGIR lines show:

```mlir
%accumulator = tt.dot %a, %b_fp, %accumulator : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
```

Representative LLVM/AMDGPU lowering shows:

```llvm
@llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16(<16 x half>, <16 x half>, <8 x float>)
```

```asm
v_wmma_f32_16x16x16_f16 v[1:8], v[53:60], v[37:44], v[1:8]
```

So the key fast path is a fused dequant-to-fp16 plus fp16 WMMA matmul, not an integer MMA path.

## v4 Replication

A v4 experimental HIP decode path was added to replicate that key structure:

- Kernel: `mmq_q4_gemm_kernel_v4_kmajor_wmma_small_m`
- Launcher: `launch_mmq_q4_gemm_kmajor_wmma_gfx1151`
- Python op: `torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma`
- Runtime selector: `AWQ_MMQ_DECODE_BACKEND=hip AWQ_MMQ_HIP_DECODE_VERSION=4`

Implementation shape:

- `V4_M_TILE=32`, `V4_N_TILE=32`, `V4_K_TILE=32`
- 4 wave32s per CTA
- A tile staged in LDS as fp16 `[32, 32]`
- B tile unpacked/dequantized from K-major packed int4 into LDS as fp16 `[32, 32]`
- Each wave computes one 16x16 output subtile using `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`

The K-major tensors must be prepared for both v3 and v4. The v4 enablement therefore changed the preprocessing guard to:

```python
need_kmajor_decode = self._hip_decode_version() in (3, 4)
```

The container entrypoint was also fixed to run custom-op setup from the op directory:

```bash
(cd "$AWQ_MMQ_DIR" && python setup.py build_ext --inplace --build-temp "$AWQ_MMQ_DIR/build/temp.linux-x86_64-cpython-312")
```

Without this, `CUDAExtension` resolves relative sources from `/opt` and fails on `/opt/bindings.cpp`.

## Correctness

Inside the container, the custom-op correctness test passed after v4 was added:

```text
ALL PASSED (3 v0 + 9 v1 + 3 v2 + 3 v3 + 3 v4)
```

The small v4 cases had max absolute error within the existing tolerance (`atol=0.03`).

## End-to-End Benchmark

Saved result:

- `test/results/decode_backend/20260601-184457_hip_decode_v4_kmajor_wmma_no_dual_storage.json`
- `test/results/decode_backend/20260601-184457_v4_kmajor_wmma_comparison.json`
- `test/results/decode_backend/20260601-184457_v4_kmajor_wmma_comparison.md`

Summary against existing one-run baselines:

| backend | short total t/s | mid total t/s | long total t/s |
|---|---:|---:|---:|
| triton_fallback | 42.721 | 97.815 | 111.639 |
| hip_v2_smallm | 5.675 | 32.678 | 81.208 |
| hip_v3_kmajor | 6.541 | 36.180 | 84.270 |
| hip_v4_kmajor_wmma | 26.768 | 82.893 | 110.710 |

v4 improves over v3 by +309.2% short, +129.1% mid, and +31.4% long total tokens/s. It still trails Triton by -37.3% short and -15.3% mid, while long is nearly tied because prefill dominates.

## Optimization Direction

Do not switch the default from Triton fallback yet. v4 is useful as the next HIP optimization baseline.

The remaining gap is likely in the parts Triton still does better:

- vectorized/coalesced nibble unpack for B and zero points;
- LDS layout/swizzle to avoid bank conflicts when feeding WMMA fragments;
- reducing wasted lanes for `M < 16`, which is common in speculative decode;
- pipelining unpack/dequant of the next K tile while the current WMMA tile is executing;
- avoiding full `[32, 32]` staging when only a few M rows are live.

A practical v5 direction is to keep the K-major W4 layout and fp16 WMMA accumulation, but change B dequant staging to produce WMMA-ready fragments more directly, then add a specialized `M <= 8` or `M <= 16` path instead of always paying for a full 32-row tile.

## v5 Follow-up: M<=16 Specialization

After v4, a v5 experimental path was added:

- Kernel: `mmq_q4_gemm_kernel_v5_kmajor_wmma_m16`
- Launcher: `launch_mmq_q4_gemm_kmajor_wmma_v5_gfx1151`
- Python op: `torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v5`
- Runtime selector: `AWQ_MMQ_DECODE_BACKEND=hip AWQ_MMQ_HIP_DECODE_VERSION=5`

v5 keeps the same K-major W4 layout and fp16 WMMA accumulation as v4, but adds a 16x32 tile for `M <= 16`:

- 2 wave32s per CTA instead of 4
- A tile staged as fp16 `[16, 32]` instead of `[32, 32]`
- B tile remains `[32, 32]` because both N subtiles still need the same dequantized weights
- For `M > 16`, the v5 launcher falls back to the v4 32x32 kernel to avoid duplicating B dequant work

Correctness passed in-container:

```text
ALL PASSED (3 v0 + 9 v1 + 3 v2 + 3 v3 + 3 v4 + 4 v5)
```

Saved v5 benchmark results:

- `test/results/decode_backend/20260601-193056_hip_decode_v5_m16_wmma_no_dual_storage.json`
- `test/results/decode_backend/20260601-193056_v5_m16_wmma_comparison.json`
- `test/results/decode_backend/20260601-193056_v5_m16_wmma_comparison.md`

| backend | short total t/s | mid total t/s | long total t/s |
|---|---:|---:|---:|
| triton_fallback | 42.721 | 97.815 | 111.639 |
| hip_v4_kmajor_wmma | 26.768 | 82.893 | 110.710 |
| hip_v5_m16_wmma | 27.136 | 85.438 | 110.927 |

v5 is correct and slightly faster than v4: +1.4% short, +3.1% mid, and +0.2% long total tokens/s. This confirms that trimming unused M waves helps, but the modest size of the gain suggests the remaining gap is dominated by B unpack/dequant and memory/LDS behavior rather than only idle M lanes.

## v6 Follow-up: Metadata-Staged B Dequant

v6 targets the B dequant metadata path rather than the M tile shape. It keeps v5's `M<=16`, `N=32`, two-wave WMMA tile, but stages per-group metadata into LDS once per K group:

- `scale_tile[32]`: one scale per output column in the tile
- `zero_tile[4]`: one packed zero word per 8 output columns

This removes repeated scale/zero global loads inside the `kk x packed-N` B unpack loop. Correctness passed:

```text
ALL PASSED (3 v0 + 9 v1 + 3 v2 + 3 v3 + 3 v4 + 4 v5 + 4 v6)
```

Saved v6 results:

- `test/results/decode_backend/20260601-200248_hip_decode_v6_meta_wmma_no_dual_storage.json`
- `test/results/decode_backend/20260601-200248_v6_meta_wmma_comparison.json`
- `test/results/decode_backend/20260601-200248_v6_meta_wmma_comparison.md`

| backend | short total t/s | mid total t/s | long total t/s |
|---|---:|---:|---:|
| triton_fallback | 42.721 | 97.815 | 111.639 |
| hip_v5_m16_wmma | 27.136 | 85.438 | 110.927 |
| hip_v6_meta_wmma | 28.690 | 87.179 | 111.506 |

v6 improves over v5 by +5.7% short, +2.0% mid, and +0.5% long total tokens/s. It is still slower than Triton on short and mid decode, but long is essentially tied. This confirms that repeated scale/zero metadata loads were a real part of the B dequant bottleneck.



## v7 Follow-up: Wider N Tile With Metadata Staging

v7 combines the v6 metadata-staged B dequant path with a wider N tile for the common `M <= 16` decode shape:

- Kernel: `mmq_q4_gemm_kernel_v7_kmajor_wmma_m16n64_meta`
- Launcher: `launch_mmq_q4_gemm_kmajor_wmma_v7_gfx1151`
- Python op: `torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v7`
- Runtime selector: `AWQ_MMQ_DECODE_BACKEND=hip AWQ_MMQ_HIP_DECODE_VERSION=7`

The main change is moving from v6's `N=32` tile to `N=64` while keeping `M=16`, `K=32`, and fp16 WMMA. Four wave32s in the CTA compute four 16x16 N subtiles. The dequant metadata is staged once per K group for all 64 output columns:

- `scale_tile[64]`
- `zero_tile[8]`
- `a_tile[16][32]`
- `b_tile[32][64]`

This is closer to Triton's useful work shape for decode because more output columns share the same staged A tile and metadata loads. For `M > 16`, the launcher still falls back to the v4 path.

Correctness passed in-container:

```text
ALL PASSED (3 v0 + 9 v1 + 3 v2 + 3 v3 + 3 v4 + 4 v5 + 4 v6 + 5 v7)
```

Saved v7 results:

- `test/results/decode_backend/20260601-203311_hip_decode_v7_m16n64_meta_wmma.json`
- `test/results/decode_backend/20260601-203311_v7_m16n64_meta_wmma_comparison.json`
- `test/results/decode_backend/20260601-203311_v7_m16n64_meta_wmma_comparison.md`

| backend | short total t/s | mid total t/s | long total t/s |
|---|---:|---:|---:|
| triton_fallback | 42.721 | 97.815 | 111.639 |
| hip_v4_kmajor_wmma | 26.768 | 82.893 | 110.710 |
| hip_v5_m16_wmma | 27.136 | 85.438 | 110.927 |
| hip_v6_meta_wmma | 28.690 | 87.179 | 111.506 |
| hip_v7_m16n64_meta_wmma | 30.494 | 89.520 | 112.615 |

v7 improves over v6 by +6.3% short, +2.7% mid, and +1.0% long total tokens/s. It is still below Triton on short and mid decode, but long prefill/decode now measures above Triton by +0.9%. Treat this as a useful but narrow HIP win rather than a global replacement for Triton fallback.

Next useful directions:

- Add repeated-run validation for the long case to separate the +0.9% win from run-to-run noise.
- Inspect generated ISA for v7 to confirm whether the wider tile improved global load coalescing or only amortized metadata overhead.
- Try N=64 plus a lighter B staging layout/swizzle to reduce LDS bank pressure.
- Consider an M=8 variant for speculative decode, where half of v7's M lanes are still idle.


## v8 Follow-up: M<=8, N=128 Wider Tile

v8 tests a more speculative-decode-specific shape than v7:

- Kernel: `mmq_q4_gemm_kernel_v8_kmajor_wmma_m8n128_meta`
- Launcher: `launch_mmq_q4_gemm_kmajor_wmma_v8_gfx1151`
- Python op: `torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v8`
- Runtime selector: `AWQ_MMQ_DECODE_BACKEND=hip AWQ_MMQ_HIP_DECODE_VERSION=8`

The idea is to use an `M<=8`, `N=128` tile so very small-M decode amortizes A staging and scale/zero metadata across more output columns. The kernel uses 8 wave32s, one per 16-column WMMA N subtile, and stores only 8 A rows in LDS. WMMA lanes for rows 8..15 are fed zeros.

Correctness passed in-container:

```text
ALL PASSED (3 v0 + 9 v1 + 3 v2 + 3 v3 + 3 v4 + 4 v5 + 4 v6 + 5 v7 + 6 v8)
```

Saved v8 results:

- `test/results/decode_backend/20260601-211208_hip_decode_v8_m8n128_meta_wmma.json`
- `test/results/decode_backend/20260601-211208_v8_m8n128_meta_wmma_comparison.json`
- `test/results/decode_backend/20260601-211208_v8_m8n128_meta_wmma_comparison.md`

| backend | short total t/s | mid total t/s | long total t/s |
|---|---:|---:|---:|
| triton_fallback | 42.721 | 97.815 | 111.639 |
| hip_v6_meta_wmma | 28.690 | 87.179 | 111.506 |
| hip_v7_m16n64_meta_wmma | 30.494 | 89.520 | 112.615 |
| hip_v8_m8n128_meta_wmma | 30.629 | 89.356 | 111.902 |

v8 improves short decode over v7 by +0.4%, but regresses mid by -0.2% and long by -0.6%. The wider `N=128` tile is not a general replacement for v7. It suggests that N-tile widening has a small benefit for very short decode, but the larger LDS B tile and 8-wave CTA are probably hurting occupancy or scheduling enough to lose the long-case advantage.

Current recommendation: keep v7 as the main HIP candidate. v8 is useful evidence for policy tuning or for a more careful M<=8 kernel, but not as the default HIP decode version.


## v9 Follow-up: B LDS Swizzle and Runtime Shape Stats

Two changes were added after v8:

1. Runtime shape stats in `vllm_kernel.py`, controlled by:
   - `AWQ_MMQ_SHAPE_STATS=1`
   - `AWQ_MMQ_SHAPE_STATS_INTERVAL=...`
   - `AWQ_MMQ_SHAPE_STATS_PATH=...`

   The recorded key is `(route, backend, version, M, N, K)`, which makes it possible to see real DFlash target verify shapes.

2. v9 experimental kernel:
   - Kernel: `mmq_q4_gemm_kernel_v9_kmajor_wmma_m16n64_meta_swizzle`
   - Launcher: `launch_mmq_q4_gemm_kmajor_wmma_v9_gfx1151`
   - Python op: `torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v9`
   - Runtime selector: `AWQ_MMQ_DECODE_BACKEND=hip AWQ_MMQ_HIP_DECODE_VERSION=9`

v9 keeps v7's `M<=16,N=64` metadata-staged compute shape but stores B in LDS using:

```cpp
n ^ ((kk & 0x3) << 4)
```

Correctness passed in-container:

```text
ALL PASSED (3 v0 + 9 v1 + 3 v2 + 3 v3 + 3 v4 + 4 v5 + 4 v6 + 5 v7 + 6 v8 + 5 v9)
```

Saved v9 and shape stats results:

- `test/results/decode_backend/20260601-220629_hip_decode_v9_swizzle_shape_stats_rerun.json`
- `test/results/decode_backend/20260601-220629_shape_stats_v9.json`
- `test/results/decode_backend/20260601-220629_v9_swizzle_shape_stats_comparison.json`
- `test/results/decode_backend/20260601-220629_v9_swizzle_shape_stats_comparison.md`

| backend | short total t/s | mid total t/s | long total t/s |
|---|---:|---:|---:|
| triton_fallback | 42.721 | 97.815 | 111.639 |
| hip_v7_m16n64_meta_wmma | 30.494 | 89.520 | 112.615 |
| hip_v8_m8n128_meta_wmma | 30.629 | 89.356 | 111.902 |
| hip_v9_b_lds_swizzle | 29.236 | 87.392 | 111.024 |

v9 regresses versus v7 by -4.1% short, -2.4% mid, and -1.4% long. This B LDS xor swizzle should not replace v7.

Runtime shape stats from the v9 benchmark:

| Route/M | Count |
|---|---:|
| decode M=9 | 18432 |
| decode M=1 | 256 |
| prefill M=289 | 256 |
| prefill M=536 | 256 |
| prefill M=2193 | 256 |
| prefill M=8185 | 256 |
| prefill M=8192 | 256 |

The important finding is that DFlash target verification is dominated by `M=9`, which matches `num_speculative_tokens=8` plus the current token. Therefore v8's `M<=8` specialization misses the dominant decode shape. If continuing HIP kernel work, target exactly `M=9` / `M<=9` or keep v7's `M<=16`; do not spend more time on M<=8 as the main path.


## Hybrid Backend Policy Experiment

A dynamic backend policy was added as an opt-in mode:

```bash
AWQ_MMQ_DECODE_BACKEND=hybrid
AWQ_MMQ_HYBRID_LONG_PREFILL_THRESHOLD=4096
AWQ_MMQ_HYBRID_VERIFY_M=9
AWQ_MMQ_HYBRID_HIP_VERSION=7
```

Intended behavior:

- short/mid small-M decode stays on TritonW4A16;
- after long-context prefill (`M >= 4096`), DFlash target verify `M=9` routes to HIP v7;
- v8/v9 remain experimental and are not selected by default.

The policy uses process-local state and assumes the current `max_num_seqs=1` setup. It keeps long-context active through tail prefill chunks and clears it after decode has been seen and the next short/mid prefill starts.

Saved results:

- `test/results/decode_backend/20260601-222843_hybrid_triton_short_v7_long_shape_stats.json`
- `test/results/decode_backend/20260601-222843_shape_stats_hybrid.json`
- `test/results/decode_backend/20260601-224041_hybrid_triton_short_v7_long.json`
- `test/results/decode_backend/20260601-225106_triton_fallback_current.json`
- `test/results/decode_backend/20260601-224041_hybrid_policy_comparison.json`
- `test/results/decode_backend/20260601-224041_hybrid_policy_comparison.md`

Route stats verified the policy:

| route/backend/version | calls |
|---|---:|
| decode/triton/-1 | 14592 |
| decode/hip/7 | 4096 |
| prefill/hip/1 | 1280 |

Performance versus the current Triton baseline:

| backend | short total t/s | mid total t/s | long total t/s |
|---|---:|---:|---:|
| triton_current | 38.394 | 97.293 | 112.824 |
| v7_all_hip | 30.494 | 89.520 | 112.615 |
| hybrid_triton_short_v7_long | 38.170 | 97.676 | 111.232 |

The policy routed correctly, but did not improve end-to-end throughput. Current Triton is still stronger overall, including long. Keep hybrid as an opt-in experiment rather than the default. The next high-value work should move toward KV/cache and memory policy, or a genuinely M=9-specific HIP kernel rather than this coarse hybrid selection.
