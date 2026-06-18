# 64K / 128K / 256K Context Pressure Summary

Source file: `/home/xqhpc/data/AI_project/combined_papers_for_llm.txt`

Full source token count: **353,478 tokens**. This means 256K is still a truncated-input test, not a full-document test.

| Context class | Test type | max_model_len | util | KV dtype | Prompt tokens | Max output | Result | Key metric |
|---|---|---:|---:|---|---:|---:|---|---:|
| 64K | real generation near-boundary | 65,536 | 0.60 | fp8/current best profile | 60,000 | 128 | pass | 762.486 s wall |
| 128K | startup/capacity | 131,072 | 0.90 | fp8 | - | - | pass | 235,936 KV tokens, 2.81x concurrency |
| 256K | startup/capacity | 262,144 | 0.90 | fp8 | - | - | pass | 235,936 KV tokens, 1.46x concurrency |

## Interpretation

- 32K is not the maximum. 64K real-document generation has already been validated using a 60K-token slice of `combined_papers_for_llm.txt`.
- 128K and 256K both successfully start under the high-memory single-user profile: `VLLM_GPU_MEMORY_UTIL=0.90`, `VLLM_KV_CACHE_DTYPE=fp8`, `VLLM_DFLASH_N=4`, `VLLM_MAX_NUM_BATCHED_TOKENS=16384`.
- README-level 256K capacity is valid for startup/admission on this configuration, but it should not be interpreted as good interactive latency.
- Full 128K/256K generation was not run in this pass. The prior 60K real-document request took 762.486 s for 128 output tokens, so full 128K/256K generation should be scheduled as a long/overnight latency test.

## Result Files

- `test/results/decode_backend/20260602-151019_gpu_util_060_60k_combined_papers_request.json`
- `test/results/context_pressure/20260606-154100_ctx128k_startup_capacity.json`
- `test/results/context_pressure/20260606-154736_ctx256k_startup_capacity.json`
