# Full Triton GPU Memory Utilization 0.60 Benchmark

Date: 2026-06-02

## Result Summary

`VLLM_GPU_MEMORY_UTIL=0.60` is now the best candidate long-context memory profile found so far. It preserves full Triton dual-storage, expands KV cache by 27.91%, and did not regress the standard short/mid/long benchmark in this run.

## Startup / KV Cache

| Config | gpu util | model GiB | KV available GiB | KV tokens | max concurrency |
|---|---:|---:|---:|---:|---:|
| baseline full Triton | 0.55 | 40.50 | 20.61 | 71,552 | 2.68x |
| full Triton 0.60 | 0.60 | 40.50 | 26.41 | 91,520 | 3.43x |

KV capacity delta: +19,968 tokens (27.91%).

## Benchmark

| Case | baseline total tok/s | 0.60 total tok/s | delta |
|---|---:|---:|---:|
| short_decode_128 | 38.394 | 38.380 | -0.04% |
| mid_prefill_512_decode_64 | 97.293 | 98.214 | +0.95% |
| long_prefill_2k_decode_32 | 112.824 | 114.040 | +1.08% |

## Interpretation

This is better than the shape-exclude memory policy as a default candidate. `shape_exclude_5120x34816` reached 93,184 KV tokens but paid a large short/mid throughput cost; 0.60 reaches 91,520 KV tokens while keeping full Triton and matching or slightly improving throughput in this single run.

Recommended policy direction:

- Default/competition candidate: full Triton with `VLLM_GPU_MEMORY_UTIL=0.60`, after repeated cold-start validation.
- Conservative fallback: current `.env` 0.55 full Triton.
- Extreme long-context profile: `shape_exclude_5120x34816`, only when KV capacity matters more than short/mid latency.

Remaining validation before changing `.env`:

- Repeat 0.60 cold-start + first real request 3 times.
- Run at least one real long-context request close to 64K after cold start.
- Watch first-request GDN/Triton autotuner headroom; compose comments explicitly mention this risk.

## Files

- Benchmark: `test/results/decode_backend/20260602-143500_triton_gpu_util_060.json`
- Baseline: `test/results/decode_backend/20260601-225106_triton_fallback_current.json`
- Startup probe: `test/results/decode_backend/20260602-112159_gpu_memory_util_060_probe.json`
