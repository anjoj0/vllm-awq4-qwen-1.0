# DFlash draft kernel block16 A/B experiment

Date: 2026-06-02

## Variant

Runtime-only patch in `vllm/v1/worker/utils.py`: keep DFlash draft KV manager allocation block size at `832`, but force the DFlash draft attention group kernel block size to `16`. Target attention groups kept `kernel_block_size=832`.

Startup confirmed the intended group split:

- target attention groups `gid=10..13`: `manager_block_size=832`, `selected_kernel_size=832`;
- DFlash draft group `gid=14`, layers `model.layers.64..68.self_attn.attn`: `manager_block_size=832`, `selected_kernel_size=832 -> 16`;
- service started successfully; KV cache stayed at `91,520 tokens`, max concurrency `3.43x`.

## A/B results

| case | prompt tokens | completion tokens | default elapsed s | patched elapsed s | default tok/s | patched tok/s | latency ratio | tok/s ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short_decode_96 | 31 | 96 | 5.547 | 20.801 | 17.306 | 4.615 | 3.750x | 0.267x |
| paper_prefix_32kchars_decode_96 | 6711 | 96 | 60.656 | 82.623 | 1.583 | 1.162 | 1.362x | 0.734x |

## Interpretation

This route is functionally viable but performance-negative in the tested configuration. Forcing draft virtual kernel block splitting to `16` did not improve end-to-end throughput; it made both short decode and the 6.7K-token long-context request slower.

Likely reasons:

- virtual block splitting expands each 832-token allocation block into 52 kernel-table blocks, increasing block table/slot-mapping pressure;
- ROCm native dispatch may still not be selected for the full DFlash path, or the saved native-kernel work is smaller than the added indexing overhead;
- DFlash target verification remains dominated by target hybrid `832` geometry and Triton/linear-attn behavior, so optimizing only the 5 draft attention layers has limited leverage;
- short decode is especially sensitive to framework overhead, and patched short decode regressed strongly.

## Decision

Do not continue with draft kernel-block splitting as a mainline optimization. Keep the default `kernel_block_size=832` path. This experiment further supports shifting KV work toward cache memory policy, scheduling, and DFlash parameter tuning rather than trying to force draft native paged attention.

Raw benchmark files:

- patched: `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/decode_backend/20260602-145620_kernel_block16_patched_benchmark_safe.json`
- default baseline: `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/decode_backend/20260602-150200_kernel_block16_default_baseline_benchmark_safe.json`
