# KV Cache GPU Memory Utilization Probe: 0.60

Date: 2026-06-02

## Setup

- Candidate env: `VLLM_GPU_MEMORY_UTIL=0.60`
- Backend: `AWQ_MMQ_DECODE_BACKEND=triton`
- Decode policy: `AWQ_MMQ_DECODE_POLICY=all`
- Shape/weight stats disabled
- Purpose: check whether a modest GPU memory util increase expands KV capacity without breaking startup or first real request.

## Startup Result

| Setting | model GiB | KV available GiB | KV tokens | max concurrency |
|---|---:|---:|---:|---:|
| 0.55 default | 40.50 | 20.61 | 71,552 | 2.68x |
| 0.60 probe | 40.50 | 26.41 | 91,520 | 3.43x |

Delta vs 0.55 default:

- KV available: +5.80 GiB
- KV tokens: +19,968 (27.91%)
- Max concurrency: +0.75x (27.99%)

## First Request Probe

| result | wall seconds | prompt tokens | completion tokens | finish |
|---|---:|---:|---:|---|
| ok | 3.446 | 12 | 16 | length |

A first urllib attempt returned HTTP 502 because it inherited host proxy settings and did not reach vLLM logs. Retest with `ProxyHandler({})` succeeded.

## Interpretation

`VLLM_GPU_MEMORY_UTIL=0.60` is a stronger candidate than deleting more K-major duplicate storage for long-context mode: it keeps full Triton decode storage, increases KV capacity to 91,520 tokens, and survived one first-request probe. It should still remain opt-in until repeated first-request probes and the normal short/mid/long benchmark pass, because the compose comment explicitly calls out first-request autotuner headroom risk.

Next checks:

- Repeat 0.60 first-request probe after a fresh restart at least 3 times.
- Run the normal decode benchmark at 0.60 and compare to full Triton 0.55.
- If stable, test `shape_exclude_5120x34816 + 0.60` only as an aggressive long-context memory profile, not as default.
- Continue studying page/block padding waste, because logs still warn about 4.17% and 25.00% maximum waste in different KV groups.

## Files

- JSON: `test/results/decode_backend/20260602-112159_gpu_memory_util_060_probe.json`
- Baseline benchmark: `test/results/decode_backend/20260601-225106_triton_fallback_current.json`
