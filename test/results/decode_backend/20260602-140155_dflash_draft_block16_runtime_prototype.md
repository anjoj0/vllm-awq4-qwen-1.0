# DFlash draft block16 runtime prototype

Date: 2026-06-02
Service: `vllm-awq4-qwen`, model `Qwen3.6-27B-AWQ4`, DFlash drafter `z-lab/Qwen3.6-27B-DFlash`.

## Goal

Prototype whether DFlash drafter attention layers can use native-friendly `block_size=16` while the target hybrid cache keeps the existing `block_size=832`/Triton path.

## Runtime patches tested

1. Added KV spec diagnostics in `GpuModelRunner.get_kv_cache_spec()` under `AWQ_DFLASH_DRAFT_BLOCK16_PROTOTYPE=1`.
2. Corrected draft layer matching from the wrong `draft_model.*` prefix to the actual DFlash draft layers:
   - `model.layers.64.self_attn.attn`
   - `model.layers.65.self_attn.attn`
   - `model.layers.66.self_attn.attn`
   - `model.layers.67.self_attn.attn`
   - `model.layers.68.self_attn.attn`
3. Forced those 5 draft attention layers to `block_size=16` in `Attention.get_kv_cache_spec()`.
4. Temporarily bypassed page-size unification for those draft layers to see how far the allocator path gets.

## Observed spec distribution

Before forcing draft block16, actual runtime KV specs were:

| Spec | block_size | page_size_bytes | heads | head_size | count |
|---|---:|---:|---:|---:|---:|
| MambaSpec | 65536 | 3,407,872 | n/a | n/a | 48 |
| FullAttentionSpec target | 832 | 3,407,872 | 4 | 256 | 16 |
| FullAttentionSpec draft | 832 | 3,407,872 | 8 | 128 | 5 |

After forcing draft block16:

| Spec | block_size | page_size_bytes | heads | head_size | count |
|---|---:|---:|---:|---:|---:|
| MambaSpec | 65536 | 3,407,872 | n/a | n/a | 48 |
| FullAttentionSpec target | 832 | 3,407,872 | 4 | 256 | 16 |
| FullAttentionSpec draft | 16 | 65,536 | 8 | 128 | 5 |

The forcing itself succeeded. The five expected draft layers emitted `AWQ_DFLASH_DRAFT_BLOCK16_PROTOTYPE forcing ... block_size 832->16`.

## Failure point

Startup failed after KV memory profiling:

```text
AWQ_DFLASH_DRAFT_BLOCK16_PROTOTYPE: preserving mixed page sizes for dflash draft layers; draft_layers=['model.layers.64.self_attn.attn', 'model.layers.65.self_attn.attn', 'model.layers.66.self_attn.attn', 'model.layers.67.self_attn.attn', 'model.layers.68.self_attn.attn'] page_sizes=[65536, 3407872]
AssertionError
  File "vllm/v1/core/kv_cache_utils.py", line 1794, in _max_memory_usage_bytes_from_groups
    page_size = get_uniform_page_size([group.kv_cache_spec for group in kv_cache_groups])
  File "vllm/v1/core/kv_cache_utils.py", line 943, in get_uniform_page_size
    assert len(page_sizes) == 1
```

## Conclusion

The framework-level `draft block16 + target 832` idea is not blocked at layer-name discovery or attention spec generation. It is blocked by deeper vLLM V1 KV cache assumptions:

- `unify_kv_cache_spec_page_size()` normally erases draft block16 by upscaling it to the target/mamba page size.
- If page-size unification is bypassed, memory accounting asserts that KV groups share one uniform page size.
- `get_kv_cache_config_from_groups()` also uses a global `num_blocks` and a uniform `page_size` for the general allocation path.

So a production implementation needs per-group KV cache geometry, not just a backend policy or `block_size` override:

- independent draft KV group;
- per-group page size;
- likely per-group `num_blocks` or equivalent capacity accounting;
- scheduler/block-table/slot-mapping changes so target and draft can have different token-per-block geometry.

## Practical recommendation

Do not continue treating draft block16 as a small config/policy fix. For the competition timeline, keep the stable default at full Triton/832 and use this result to justify either:

1. a larger KV allocator architecture change, or
2. a lower-risk cache memory policy that keeps the current uniform page-size invariant.
