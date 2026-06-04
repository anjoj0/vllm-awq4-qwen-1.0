# DFlash drafter block_size=16 framework prototype research

## Goal

Explore whether DFlash drafter cache can use an independent `block_size=16` while target Qwen3.6/Qwen3.5 hybrid cache remains `block_size=832` and Triton fallback.

## Key source findings

1. DFlash config says `block_size=16`, but runtime draft attention still sees `block_size=832`.

2. The reason is not the DFlash config itself. In `SpecDecodeBaseProposer._create_draft_vllm_config()`, the default implementation returns the target/global `self.vllm_config`. Therefore the DFlash draft model inherits the already-aligned global `cache_config.block_size=832`.

3. `Attention.get_kv_cache_spec()` always reads:

```python
block_size = vllm_config.cache_config.block_size
```

So any draft AttentionSpec created with the global config becomes `block_size=832`.

4. Even if we prototype a draft-only config with `cache_config.block_size=16`, current KV cache grouping will still tend to erase it. For normal attention specs, `group_and_unify_kv_cache_specs()` does not apply because it is only for DeepSeekV4/SWA-MLA. The fallback path calls `unify_kv_cache_spec_page_size()`.

5. `unify_kv_cache_spec_page_size()` increases smaller page sizes to the maximum page size by increasing block size. Current numbers are exact:

```text
DFlash drafter block16 page size = 2 * 16 * 8 * 128 * 2 = 65,536 bytes
Target block832 page size        = 2 * 832 * 4 * 256 * 2 = 3,407,872 bytes
Ratio                            = 52
Draft block after unification    = 16 * 52 = 832
```

This means a naive draft-only `block_size=16` override will be transformed back to `832` before allocation.

6. `UniformTypeKVCacheSpecs.is_uniform_type()` rejects different block sizes inside the same uniform group. The allocator/scheduler path also contains assumptions that physical memory per block is the same across groups.


## Direct source-object validation

A minimal simulation using vLLM `FullAttentionSpec` and `unify_kv_cache_spec_page_size()` produced:

```json
{
  "before": {
    "draft_block16": {"block_size": 16, "page_size_bytes": 65536},
    "target_block832": {"block_size": 832, "page_size_bytes": 3407872}
  },
  "uniform_type_before": false,
  "after_unify_page_size": {
    "draft_block16": {"block_size": 832, "page_size_bytes": 3407872},
    "target_block832": {"block_size": 832, "page_size_bytes": 3407872}
  }
}
```

This directly validates that a draft-only `block_size=16` spec is not preserved by the current page-size unification path.

## Minimal viable prototype boundary

A real prototype needs allocator/cache-group changes, not just DFlash model config changes.

Required changes:

1. Create and store a draft-specific `VllmConfig` for DFlash with `cache_config.block_size=16`.
2. Use that draft config when collecting draft attention layer KV specs.
3. Keep draft layers in a separate KV cache group that is allowed to have a different page size from target groups.
4. Prevent `unify_kv_cache_spec_page_size()` from upscaling the draft group to target page size.
5. Update allocation/accounting paths that assume one physical memory-per-block size across groups.
6. Ensure DFlash context pre-insert slot mappings are generated for the draft KV group/block size, not target block size.

## Risk assessment

A small local override in `DFlashProposer` or `_create_draft_vllm_config()` is insufficient and likely incorrect, because metadata block size and actual cache tensor shape would diverge.

The framework route is still more realistic than a full `block_size=832/head_size=256` HIP kernel, but it is an allocator/scheduler prototype, not a one-file patch.

## Practical recommendation

Do not promote this to default runtime now.

Next safe step, if we want to continue this line, is a diagnostic-only patch that logs per-layer KV specs before and after page-size unification, with an optional experimental branch that returns separate groups without allocation changes only to observe where assertions fail. That would map the exact set of allocator assumptions before attempting a working implementation.
