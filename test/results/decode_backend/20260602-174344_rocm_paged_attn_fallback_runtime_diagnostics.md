# ROCm paged attention fallback runtime diagnostics - 8K probe

## Purpose

Validate why Qwen3.6 AWQ4 + DFlash does not use the ROCm custom paged attention kernel, and record the actual runtime shape distribution instead of relying only on source-level inference.

## Probe

- Source: `/home/xqhpc/data/AI_project/combined_papers_for_llm.txt`
- Prompt target: 8192 tokens from `combined_papers_for_llm.txt`
- API prompt usage: 8223 tokens
- Completion tokens: 8
- HTTP status: 200
- Wall time: 69.780s
- Service profile: full Triton AWQ path, `VLLM_GPU_MEMORY_UTIL=0.60`, DFlash enabled

## Runtime fallback stats

- Total recorded fallback calls: 126
- Last call:
  - `native_use_custom`: False
  - `is_pow2`: False
  - `block_size`: 832
  - `head_size`: 128
  - `gqa_ratio`: 4
  - `max_seq_len`: 8254
  - `num_query_heads`: 32
  - `num_kv_heads`: 8
  - `q_dtype`: `torch.float16`

## Shape distribution

- count=96: `native_use_custom=False|is_pow2=False|block_size=832|head_size=256|gqa_ratio=6|sliding_window=0|kv_cache_dtype=auto|has_alibi=False|has_sinks=False`
- count=30: `native_use_custom=False|is_pow2=False|block_size=832|head_size=128|gqa_ratio=4|sliding_window=0|kv_cache_dtype=auto|has_alibi=False|has_sinks=False`

## Interpretation

The runtime blocker is the hybrid cache geometry, not an omitted environment flag. vLLM sets the attention block size to 832 tokens so the attention page size matches the Mamba page size. Because 832 is not a power of two and is not the native ROCm paged-attention block size expected by the kernel path, `use_custom` becomes false and the request falls back to Triton.

The probe observed two dominant fallback shapes:

- DFlash drafter path: `head_size=128`, `gqa_ratio=4`, `num_query_heads=32`, `num_kv_heads=8`
- target Qwen3.6/Qwen3.5 path: `head_size=256`, `gqa_ratio=6`, `num_query_heads=24`, `num_kv_heads=4`

This means forcing the existing ROCm custom paged attention kernel is unsafe for this model/service combination. A real architecture-level improvement would need either a ROCm paged-attention kernel that supports block size 832 and these two head/GQA shapes, or a cache-layout/page-size policy change that preserves the DFlash hybrid page alignment while recovering a native-kernel-compatible attention block size.

## Files

- Probe result: `20260602-174344_rocm_paged_attn_8k_combined_papers_probe.json`
- Stats result: `20260602-174344_rocm_paged_attn_stats.json`
- Earlier 32K parse-failure record, server-side 200 but client parse bug: `20260602-163309_rocm_paged_attn_32k_combined_papers_probe.json`
