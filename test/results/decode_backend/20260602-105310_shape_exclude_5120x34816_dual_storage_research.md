# Shape-Aware Dual-Storage Experiment: exclude 5120x34816 K-major

Date: 2026-06-02

## Policy

- Backend: `AWQ_MMQ_DECODE_BACKEND=triton`
- Policy: `AWQ_MMQ_DECODE_POLICY=shape_exclude_5120x34816`
- Meaning: keep Triton K-major decode storage for all AWQ linear shapes except `K=5120,N=34816`. That excluded shape corresponds to 64 large layers and is the single largest duplicate-memory block.

## Duplicate Weight Memory

- Full dual-storage K-major duplicate: 13.10 GiB
- Shape-exclude K-major duplicate: 6.96 GiB
- Saved duplicate weight memory: 6.14 GiB
- Native AWQ tensor memory is unchanged: 13.10 GiB

| Shape KxN | layers | K-major layers | K-major GiB | note |
|---:|---:|---:|---:|---|
| 17408x5120 | 64 | 64 | 3.07 | kept |
| 5120x16384 | 48 | 48 | 2.17 | kept |
| 6144x5120 | 64 | 64 | 1.08 | kept |
| 5120x14336 | 16 | 16 | 0.63 | kept |
| 5120x34816 | 64 | 0 | 0.00 | excluded |

## Startup And KV Cache

| Policy | model GiB | KV available GiB | KV tokens | max concurrency |
|---|---:|---:|---:|---:|
| full_triton | 40.50 | 20.61 | 71,552 | 2.68x |
| no_dual_all_hip | 27.27 | 33.84 | 118,144 | 4.40x |
| partial_n_lt_16384 | 32.09 | 29.02 | 100,672 | 3.77x |
| partial_n_ge_16384 | 36.39 | 24.72 | 85,696 | 3.21x |
| shape_exclude_5120x34816 | 34.24 | 26.87 | 93,184 | 3.49x |

Compared with full Triton dual-storage, `shape_exclude_5120x34816` reduces reported model memory by 6.26 GiB, increases available KV cache by 6.26 GiB, and raises KV capacity by 21,632 tokens (30.2%).

## Decode Benchmark

| Policy | short total tok/s | mid total tok/s | long total tok/s |
|---|---:|---:|---:|
| full_triton | 38.394 | 97.293 | 112.824 |
| no_dual_all_hip | 22.022 | 77.275 | 108.230 |
| partial_n_lt_16384 | 26.504 | 85.069 | 111.982 |
| partial_n_ge_16384 | 29.823 | 87.174 | 108.845 |
| shape_exclude_5120x34816 | 27.993 | 89.133 | 112.181 |

Relative to current full Triton:
- short: -27.09%
- mid: -8.39%
- long: -0.57%

## Conclusion

This policy is a useful Pareto point but should not become the default. It saves about 6.14 GiB of duplicate K-major weight storage and expands KV capacity by about 30%, while keeping long-context total throughput nearly unchanged (-0.57%). The cost is too visible on short and mid decode (-27.09% and -8.39%), which matches the expectation that the excluded `5120x34816` projection is still performance-critical outside the long-prefill regime.

Recommended next experiments:

- Keep full Triton as default for normal benchmark scoring.
- Keep `shape_exclude_5120x34816` as an opt-in long-context memory mode.
- Try a runtime policy that only disables this shape when the request needs extra KV capacity, instead of making it static at model load. If static load-time allocation is unavoidable, expose it as a memory profile rather than a decode backend default.
- Investigate KV cache sizing knobs next: `gpu_memory_utilization`, `max_num_batched_tokens`, `max_num_seqs`, DFlash draft/target KV coexistence, and page/block padding waste. Current logs show padding warnings up to 4.17% and 25.00% for different groups, so page geometry may be a cleaner optimization target than removing more K-major layers.

## Files

- Benchmark: `test/results/decode_backend/20260602-105601_shape_exclude_5120x34816.json`
- Weight stats: `test/results/decode_backend/20260602-105310_weight_stats_shape_exclude_5120x34816.json`
- Full Triton baseline: `test/results/decode_backend/20260601-225106_triton_fallback_current.json`
- Full Triton weight stats: `test/results/decode_backend/20260602-091326_weight_stats_triton.json`
