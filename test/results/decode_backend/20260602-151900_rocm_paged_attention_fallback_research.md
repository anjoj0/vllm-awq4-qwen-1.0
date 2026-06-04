# ROCm Paged Attention Fallback Investigation

Date: 2026-06-02

## Trigger

During the 60K combined-papers request, vLLM logged:

`Cannot use ROCm custom paged attention kernel, falling back to Triton implementation.`

## Source

- Runtime file: `/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py`
- Platform gate: `vllm.platforms.rocm.use_rocm_custom_paged_attention`

On gfx1x, the native ROCm custom paged attention path requires:

- `sliding_window == 0` or `(-1, -1)`
- query dtype fp16/bf16
- `head_size == 128`
- `block_size == 16`
- `3 <= gqa_ratio <= 16`
- `max_seq_len <= 128K`
- no alibi, kv cache dtype auto, no sinks

The local patched `chunked_prefill_paged_decode.py` additionally forces Triton fallback when `value_cache.shape[3]` is not a power of two. This was added for Qwen3-style non-standard block sizes.

## Root Cause

This model has hybrid attention/Mamba cache layout. vLLM adjusts attention block/page size so attention page size is at least the Mamba page size and then pads Mamba state to match attention page size. The startup logs show this alignment step, and the effective attention block size is no longer the ROCm custom kernel-supported `16` tokens.

Therefore the warning is expected and not caused by `VLLM_GPU_MEMORY_UTIL=0.60`. It is a kernel/layout compatibility issue between ROCm custom paged attention and Qwen3/DFlash hybrid cache geometry.

## Practical Decision

Do not force-enable ROCm custom paged attention for this model. The safe path is Triton fallback. The 60K request succeeded on this path in 762.486s without OOM.

Future architecture work should focus on one of:

- a ROCm custom paged attention kernel that supports the effective non-16/non-power-of-two cache block size;
- changing hybrid Mamba/attention page layout so attention can retain block size 16;
- reducing chunked prefill overhead under Triton fallback;
- exposing a memory/profile layer that selects 0.60 full Triton by default and shape-exclude only for extreme long-context memory mode.

## Related Results

- `test/results/decode_backend/20260602-151019_gpu_util_060_60k_combined_papers_request.md`
- `test/results/decode_backend/20260602-145647_gpu_util_060_cold_start_stability.md`
- `test/results/decode_backend/20260602-143500_triton_gpu_util_060_comparison.md`
