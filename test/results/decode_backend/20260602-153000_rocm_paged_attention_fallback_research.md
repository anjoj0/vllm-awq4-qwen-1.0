# ROCm Paged Attention Fallback Research

Date: 2026-06-02

## Observation

The 60K combined-papers request on `VLLM_GPU_MEMORY_UTIL=0.60` completed successfully, but logs emitted:

`Cannot use ROCm custom paged attention kernel, falling back to Triton implementation.`

## Source-Level Cause

The warning comes from `/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py`. The local patch forces Triton when `block_size` is not a power of two. The upstream ROCm custom paged attention guard on gfx1x is stricter:

- `sliding_window == 0 or (-1, -1)`
- query dtype fp16/bf16
- `head_size == 128`
- `block_size == 16`
- `3 <= gqa_ratio <= 16`
- `max_seq_len <= 128K`
- `kv_cache_dtype == auto`
- no ALiBi/sinks

For this Qwen3.6/DFlash model, vLLM adjusts attention block size to match hybrid Mamba/attention page sizing. Startup logs show attention page alignment and mamba padding, and the resulting effective cache block does not satisfy the ROCm custom paged attention `block_size == 16` constraint. Therefore the Triton fallback is expected and safe.

## Practical Conclusion

Do not force-enable `ops.paged_attention_rocm` for this model. It would require either a ROCm paged-attention kernel that supports the non-standard effective block/page geometry, or a deeper vLLM cache layout change that keeps attention block size compatible with the ROCm kernel while still satisfying Mamba page alignment.

## Next Architecture Work

- Instrument the actual decode-time `block_size`, `head_size`, `gqa_ratio`, `sliding_window`, and `kv_cache_dtype` for long prompts, so fallback reasons are recorded in result JSON instead of inferred from logs.
- Study whether Mamba cache mode or block-size settings can reduce page padding without breaking correctness. This should be a controlled startup-only experiment first.
- If kernel work is attempted, target the paged-attention fallback path, not the AWQ matmul path: 60K validation indicates long-context runtime is dominated by chunked prefill / paged attention behavior, while full Triton AWQ remains adequate.

## Related Results

- `test/results/decode_backend/20260602-151019_gpu_util_060_60k_combined_papers_request.json`
- `test/results/decode_backend/20260602-145647_gpu_util_060_cold_start_stability.md`
- `test/results/decode_backend/20260602-143500_triton_gpu_util_060_comparison.md`
