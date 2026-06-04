# DFlash KV cache architecture probe: draft block16 feasibility

Date: 2026-06-02
Project: `vllm-awq4-qwen-1.0`
Focus: whether DFlash draft can use native-friendly 16/32-token paged-attention kernel geometry without breaking target hybrid 832-token cache geometry.

## Context

Previous runtime prototype proved that forcibly changing the DFlash draft attention layers to `block_size=16` does produce the intended KV specs:

| group | layers | block_size | page_size_bytes | shape |
|---|---:|---:|---:|---|
| target mamba/linear-attn | 48 | 65536 | 3,407,872 | MambaSpec |
| target full attention | 16 | 832 | 3,407,872 | head_size=256, kv_heads=4 |
| DFlash draft attention | 5 | 16 | 65,536 | head_size=128, kv_heads=8 |

But startup fails because `kv_cache_utils._max_memory_usage_bytes_from_groups()` and `get_kv_cache_config_from_groups()` assume a uniform page size in the general multi-group allocation path.

## Source path map

The active vLLM source is inside the container:

- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/worker/block_table.py`
- `vllm/v1/worker/utils.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/attention/backends/rocm_attn.py`

## Current KV cache pipeline

1. `Attention.get_kv_cache_spec()` emits per-layer `KVCacheSpec`.
2. `GpuModelRunner.get_kv_cache_spec()` collects layer specs into `dict[layer_name, KVCacheSpec]`.
3. `get_kv_cache_groups()` groups layers and usually calls `unify_kv_cache_spec_page_size()` for hybrid models.
4. `get_kv_cache_config_from_groups()` creates:
   - scalar `KVCacheConfig.num_blocks`;
   - `KVCacheTensor(size, shared_by)` entries;
   - `KVCacheGroupSpec(layer_names, kv_cache_spec)` entries.
5. `KVCacheManager` builds one `BlockPool(kv_cache_config.num_blocks, ...)`.
6. `KVCacheCoordinator` creates one single-type manager per KV group, but all managers share the same `BlockPool`.
7. `InputBatch` owns `MultiGroupBlockTable`, one `BlockTable` per KV group.
8. `GpuModelRunner._get_slot_mappings()` builds `slot_mappings_by_gid`, then maps each layer name to the slot mapping of its KV group.
9. Attention backends consume per-layer/group block table and slot mapping.

## Already-supported pieces

Several pieces are more advanced than expected and already point toward heterogeneous group support:

1. `KVCacheBlocks.blocks` is a tuple indexed by KV group. Its source comment explicitly says token-block outer dimension is avoided because different group block sizes may be supported in the future.
2. `HybridKVCacheCoordinator` already computes `lcm_block_size` across group block sizes.
3. `resolve_kv_cache_block_sizes()` already supports:
   - scheduler block size = LCM of group block sizes;
   - hash block size = GCD of group block sizes when prefix caching needs finer hash granularity.
4. `BlockPool.cache_full_blocks()` supports converting request hashes from `hash_block_size` to larger group block sizes via `BlockHashListWithBlockSize`.
5. `MultiGroupBlockTable` already accepts per-group `block_sizes` and `kernel_block_sizes`.
6. `BlockTable` supports virtual block splitting when allocation `block_size != kernel_block_size`.
7. `GpuModelRunner._get_slot_mappings()` is already group-aware.
8. `_reshape_kv_cache_tensors()` computes each layer's `num_blocks` from that layer's raw tensor size, not only from global `KVCacheConfig.num_blocks`.

## Hard blockers for true draft block16 allocation

True draft allocation `block_size=16/page=65,536` while target remains `block_size=832/page=3,407,872` still needs deeper changes:

1. `KVCacheConfig.num_blocks` is a scalar, not per group.
2. `BlockPool` owns one global block-id namespace sized by that scalar.
3. `get_kv_cache_config_from_groups()` general path calls `get_uniform_page_size()` and allocates tensors using a single page size.
4. `_max_memory_usage_bytes_from_groups()` also calls `get_uniform_page_size()` in the general path.
5. If we simply make one global pool large enough for draft block16 capacity, target tensors would need to handle high block ids too. Allocating target pages for every draft-sized block id would explode memory.
6. Therefore a correct design needs either:
   - per-group block pools and per-group block-id namespaces; or
   - block-id remapping from logical group-local ids to tensor-local offsets.

This is bigger than a policy patch.

## More promising lower-risk route discovered

A lower-risk alternative exists and is probably worth testing before any per-group allocator rewrite:

**Keep the KV manager allocation block size at 832, but force DFlash draft attention's kernel block size to 16 or 32.**

Why this matters:

- `BlockTable` already supports `allocation block_size != kernel_block_size`.
- Example: allocation block `832`, kernel block `16` -> one manager block maps to 52 kernel blocks.
- This preserves uniform page size and avoids changing `KVCacheConfig.num_blocks`, `BlockPool`, and allocator accounting.
- It may still activate the ROCm native paged-attention kernel for draft layers if the backend dispatch uses `kernel_block_size` rather than manager `block_size` in the critical predicate.

Current blocker to this lower-risk route:

`RocmAttentionBackend.get_supported_kernel_block_sizes()` returns `[MultipleOf(16)]`. Because 832 is a multiple of 16, `select_common_block_size()` treats 832 as supported and never chooses 16/32. So current runtime keeps `kernel_block_size=832`.

Minimal experiment:

1. Add diagnostics to log `kernel_block_sizes` after `prepare_kernel_block_sizes()`.
2. For only the DFlash draft KV group (`model.layers.64-68`), override selected kernel block size to `16` or `32` while keeping `kv_cache_spec.block_size=832`.
3. Restart and check:
   - whether service initializes;
   - whether draft group block table uses virtual splitting;
   - whether ROCm paged attention native path sees kernel block 16/32;
   - whether output correctness/API remain intact;
   - whether fallback stats decrease for draft `head_size=128/gqa=4`.

Expected risk: medium, much lower than per-group allocator.

Expected success probability:

- Kernel-block override prototype starts: 55%-65%.
- It actually routes draft paged attention to native ROCm and improves speed: 30%-45%.
- True per-group draft block16 allocator starts and is correct within competition timeline: 15%-25%.

## Recommended next step

Do not pursue full per-group `num_blocks` yet.

Next experiment should be the lower-risk kernel-block split route:

- keep DFlash draft `KVCacheSpec.block_size=832`;
- keep target/mamba uniform page geometry unchanged;
- force only draft attention group's `kernel_block_size` to `16` or `32`;
- collect ROCm paged attention fallback stats on a short/mid and long prompt.

If this does not reduce fallback or improve throughput, then the remaining useful KV work should move back to cache/page policy and memory utilization rather than allocator surgery.
