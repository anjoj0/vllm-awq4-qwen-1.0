# 60K Long-Context Request Probe: combined_papers_for_llm

Date: 2026-06-02

## Setup

- Source file: `/home/xqhpc/data/AI_project/combined_papers_for_llm.txt`
- Prompt construction: repeated/prefix-truncated to exactly 60,000 input tokens using `/tokenize`
- Prompt file: `/tmp/combined_papers_60k_prompt.txt`
- Runtime profile: full Triton dual-storage, `VLLM_GPU_MEMORY_UTIL=0.60`
- `max_tokens=128`, `temperature=0`

## Result

| ok | wall seconds | prompt tokens | completion tokens | total tokens | finish |
|---|---:|---:|---:|---:|---|
| True | 762.486 | 60,000 | 128 | 60,128 | length |

## Interpretation

The 0.60 profile successfully handled a real 60K-token long-context request from the precipitation-nowcasting paper pack and generated 128 tokens without OOM or API failure. This closes the main validation gap after the short/mid/long benchmark and the 3/3 cold-start first-request probes.

This supports using full Triton with `VLLM_GPU_MEMORY_UTIL=0.60` as the primary competition memory profile. It keeps full Triton decode performance while increasing KV capacity from 71,552 to 91,520 tokens.

Observed caveat: logs showed `Cannot use ROCm custom paged attention kernel, falling back to Triton implementation` during the long request. That is not a failure, but it suggests future long-context optimization should look at the ROCm paged-attention path and chunked prefill behavior.

## Files

- Result JSON: `test/results/decode_backend/20260602-151019_gpu_util_060_60k_combined_papers_request.json`
- 0.60 standard benchmark: `test/results/decode_backend/20260602-143500_triton_gpu_util_060.json`
- 0.60 cold-start stability: `test/results/decode_backend/20260602-145647_gpu_util_060_cold_start_stability.md`
