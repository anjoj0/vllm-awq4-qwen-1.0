# Patch 20/21 TRITON_ATTN Long-Context Retest - 2026-06-05

## Runtime Validation

| item | observed |
|---|---|
| container | `vllm-awq4-qwen` |
| image | `vllm-awq4-qwen:patches-20-21` |
| serving port | host `8001` -> container `8000` |
| attention backend | `TRITON_ATTN` from startup log |
| KV cache dtype | `fp8` from startup log |
| DFlash speculative tokens | `4` from startup log |
| max model len | `65536` from startup log |
| max num batched tokens | `8192` from startup log |
| max num seqs | `1` from startup log |

Patch activation was checked inside `/opt/venv/lib/python3.12/site-packages/vllm`: Patch 20 adds `CAUSAL` to unified attention and helper masks; Patch 21 relaxes the 3D split-K gate to `max_seqlen_q > 16`; Patch 19 sets non-power-of-two iteration tile to 64.

## Benchmark Setup

| field | value |
|---|---|
| script | `test/bench_matrix.py` with runtime `BASE=http://127.0.0.1:8001` override |
| case | `normal` |
| contexts | `0, 8192, 16384, 32768` prompt tokens |
| runs | `1` per context |
| max generation | `512` completion tokens |
| raw result | `test/results/bench_matrix_patch20_21_retest_20260605.json` |

## Results

| ctx tokens | prompt tokens | decode tokens | TTFT s | total s | prefill t/s | retest decode t/s | old ROCM_ATTN baseline | user claimed new path | uplift vs old |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 96 | 512 | 1.3 | 36.4 | 76.7 | 14.57 | 14.08 | 14.35 | +3% |
| 8192 | 8293 | 512 | 77.8 | 120.4 | 106.5 | 12.04 | 9.03 | 12.82 | +33% |
| 16384 | 16485 | 512 | 173.3 | 221.8 | 95.1 | 10.56 | 6.71 | 10.38 | +57% |
| 32768 | 32869 | 512 | 438.0 | 507.9 | 75.0 | 7.33 | 3.67 | 7.35 | +100% |

## Conclusion

The retest reproduces the expected long-context improvement. The 32K case reaches 7.33 t/s, matching the claimed 7.35 t/s within noise and roughly doubling the old ROCM_ATTN baseline. 8K is lower than the previously claimed 12.82 t/s in this single-run 512-token test, but still above the old 9.03 t/s baseline. 16K is slightly higher than the claimed number.

This should be treated as a quick validation run, not a final statistical benchmark: one run per context, one prompt style, and the currently running service uses DFlash N=4 with fp8 KV cache.
