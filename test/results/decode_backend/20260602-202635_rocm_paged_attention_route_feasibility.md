# ROCm paged attention optimization route feasibility

## Context

Previous runtime diagnostics on Qwen3.6 AWQ4 + DFlash recorded ROCm paged-attention fallback with:

- `block_size=832`
- DFlash drafter shape: `head_size=128`, `gqa_ratio=4`, `num_query_heads=32`, `num_kv_heads=8`
- target Qwen3.6/Qwen3.5 shape: `head_size=256`, `gqa_ratio=6`, `num_query_heads=24`, `num_kv_heads=4`
- stats file: `20260602-174344_rocm_paged_attn_stats.json`

This report evaluates two possible architecture routes.

## Route 1: write a ROCm paged attention kernel for block_size=832

Difficulty: very high.

Validation evidence:

- `vllm.platforms.rocm.use_rocm_custom_paged_attention` on gfx1x only accepts `head_size == 128`, `block_size == 16`, `gqa_ratio in [3, 16]`, no sliding window, `kv_cache_dtype == auto`, no alibi/sinks.
- Current DFlash drafter shape with `head_size=128, block_size=832, gqa=4` returns `use_rocm_custom=False`.
- Current target Qwen3.6/Qwen3.5 shape with `head_size=256, block_size=832, gqa=6` returns `use_rocm_custom=False`.
- Even a hypothetical `head_size=256, block_size=16` returns `use_rocm_custom=False` on gfx1x because the native path only accepts `head_size == 128`.
- The native branch in `chunked_prefill_paged_decode.py` uses `_PARTITION_SIZE_ROCM = 256` and asserts `256 % block_size == 0`. `block_size=832` violates this immediately.
- `RocmAttentionImpl.do_kv_cache_update` uses native paged-cache write only for `block_size in (16, 32)` and routes non-standard blocks through Triton cache write.

Direct predicate validation:

```json
[
  {"name":"dflash_drafter","head_size":128,"block_size":832,"gqa_ratio":4,"use_rocm_custom":false,"partition_256_mod_block":256},
  {"name":"target_qwen3_6","head_size":256,"block_size":832,"gqa_ratio":6,"use_rocm_custom":false,"partition_256_mod_block":256},
  {"name":"native_possible_drafter_if_block16","head_size":128,"block_size":16,"gqa_ratio":4,"use_rocm_custom":true,"partition_256_mod_block":0},
  {"name":"head256_even_if_block16","head_size":256,"block_size":16,"gqa_ratio":6,"use_rocm_custom":false,"partition_256_mod_block":0}
]
```

Interpretation:

This is not a one-line dispatch change. A real route-1 implementation needs a new/modified ROCm paged decode kernel, revised partitioning for `block_size=832`, support for at least target `head_size=256/gqa=6`, and likely a separate path for drafter `head_size=128/gqa=4` or a policy that leaves one shape on Triton. It also needs compatible non-standard KV cache write/read handling.

Expected benefit:

Potentially high only if the new kernel beats Triton for long decode/page attention. Risk is also high because the current Triton fallback is already the validated safe path and the new kernel would need numerical validation across DFlash speculative verify.

## Route 2: page-size/cache policy to recover native-compatible attention block size

Difficulty: high, but lower than writing a full new HIP paged-attention kernel if limited to scheduler/cache-policy research.

Validation evidence:

The hybrid alignment code in `vllm/platforms/interface.py` computes, without prefix caching:

```text
attn_block_size = kernel_block_alignment_size * ceil(
    mamba_page_size / (kernel_block_alignment_size * attn_page_size_1_token)
)
```

For the observed run:

- attention block size became `832`
- attention page size is approximately `832 * 4096 = 3,407,872` bytes
- startup log reported Mamba padding by `1.09%`
- estimated Mamba page size is about `3,371,127` bytes, equivalent to `823.03` attention tokens
- native `block_size=16` attention page is only `65,536` bytes, about `1/51.4` of the Mamba page requirement
- native `block_size=32` attention page is only `131,072` bytes, about `1/25.7` of the Mamba page requirement

Policy simulation using the same formula:

```json
[
  {"user_block_size":16,"computed_attn_block_size":832,"final_block_size":832,"native_gfx1x_target_possible":false},
  {"user_block_size":32,"computed_attn_block_size":832,"final_block_size":832,"native_gfx1x_target_possible":false},
  {"user_block_size":64,"computed_attn_block_size":832,"final_block_size":832,"native_gfx1x_target_possible":false},
  {"user_block_size":128,"computed_attn_block_size":896,"final_block_size":896,"native_gfx1x_target_possible":false},
  {"user_block_size":256,"computed_attn_block_size":1024,"final_block_size":1024,"native_gfx1x_target_possible":false},
  {"user_block_size":512,"computed_attn_block_size":1024,"final_block_size":1024,"native_gfx1x_target_possible":false},
  {"user_block_size":832,"computed_attn_block_size":832,"final_block_size":832,"native_gfx1x_target_possible":false},
  {"user_block_size":1024,"computed_attn_block_size":1024,"final_block_size":1024,"native_gfx1x_target_possible":false}
]
```

Interpretation:

Current vLLM policy intentionally overrides user block settings for hybrid models. Simply setting `--block-size 16` or `--block-size 32` will not recover the ROCm native path; it is pushed back to `832` to cover the Mamba page. A real route-2 implementation must decouple attention page/block geometry from Mamba page geometry, or allow Mamba state to span many small attention blocks while preserving scheduler/hash/block-table correctness.

Expected benefit:

Partial. If route 2 restores attention `block_size=16`, the DFlash drafter path (`head_size=128/gqa=4`) can use native ROCm paged attention. The target Qwen3.6/Qwen3.5 path (`head_size=256/gqa=6`) still cannot use the current gfx1x native kernel, so target attention would remain Triton unless route 1 also extends head-size support.

## Recommendation

Do not start with route 1 as the next 20-day mainline. It is a large HIP kernel project with correctness and integration risk.

Route 2 is the better research mainline, but not as a quick parameter tuning task. The smallest meaningful experiment is a prototype cache-policy patch that creates separate attention and Mamba block/page sizes, then measures whether scheduler/hash/block-table assumptions still hold. If that is too invasive, keep the current Triton fallback and focus on KV-cache memory/scheduler knobs that already showed measurable gains.

Practical priority:

1. Keep current full Triton + `gpu_memory_utilization=0.60` default.
2. Explore route 2 with a source-level prototype only, not default runtime.
3. Treat route 1 as a longer-term custom-kernel project, useful for resume depth but unlikely to be a safe competition mainline unless isolated to the DFlash drafter `head_size=128/gqa=4` path first, or a new target `head_size=256/gqa=6` kernel is built.
