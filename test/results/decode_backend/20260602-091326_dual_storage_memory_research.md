# Dual-storage and KV cache memory research

## Measured Weight Duplication

- Source: `test/results/decode_backend/20260602-091326_weight_stats_triton.json`
- AWQ linear layers recorded: `256`
- Native AWQ tensor bytes: `13.098` GiB
- K-major duplicate bytes: `13.098` GiB
- This matches the previous startup-log delta: full Triton model memory 40.5 GiB vs no-dual-storage 27.27 GiB, about 13.23 GiB saved.

| layers | K | N | native GiB | K-major duplicate GiB |
|---:|---:|---:|---:|---:|
| 64 | 5120 | 34816 | 6.143 | 6.143 |
| 64 | 17408 | 5120 | 3.071 | 3.071 |
| 48 | 5120 | 16384 | 2.168 | 2.168 |
| 64 | 6144 | 5120 | 1.084 | 1.084 |
| 16 | 5120 | 14336 | 0.632 | 0.632 |

## Existing Policy Data

| Mode | Model mem GiB | Delta vs full Triton | KV tokens | 64K concurrency |
|---|---:|---:|---:|---:|
| triton_all | 40.50 | 0.00 | 71,552 | 2.68x |
| hip_none / no dual-storage | 27.27 | -13.23 | 118,144 | 4.40x |
| triton_n_ge_16384 | 36.39 | -4.11 | 85,696 | 3.21x |
| triton_n_lt_16384 | 32.09 | -8.41 | 100,672 | 3.77x |

Mapping the measured K-major duplicate bytes to the two N-threshold policies:

- `N >= 16384` keeps about `8.311` GiB duplicate tensors.
- `N < 16384` keeps about `4.788` GiB duplicate tensors.

## Interpretation

- The largest duplicate block is `K=5120,N=34816`: 64 layers, 6.143 GiB. This is the first target for memory reduction.
- The two `N=5120` shapes together cost about 4.155 GiB and likely represent attention/output-style projections; keeping only these was close to `triton_n_lt_16384` behavior.
- Previous partial policies were too coarse for throughput: they saved memory, but short/mid decode regressed significantly. The next policy should be shape-aware rather than only N-threshold based.
- KV cache capacity scales almost directly with removed duplicate weight memory: removing all K-major duplicate tensors increased KV tokens from 71,552 to 118,144.

## Next Experiment Matrix

1. `shape_exclude_mlp_big`: keep Triton/K-major for all shapes except `K=5120,N=34816`. Expected memory save around 6.14 GiB before allocator effects.
2. `shape_keep_attn_small`: keep only `N=5120` shapes. Expected duplicate kept around 4.16 GiB; similar memory to `n_lt_16384`, but more explicit.
3. `shape_keep_attention_plus_down`: keep `N=5120` and `K=5120,N=14336`, drop the largest MLP expansion shapes. Expected duplicate kept around 4.79 GiB.
4. Re-run endpoint benchmark and startup-log parsing for each policy, then compare model memory, KV tokens, 64K concurrency, and short/mid/long throughput.

Recommended first experiment: `shape_exclude_mlp_big`, because it removes the single largest 6.14 GiB duplicate block while preserving Triton for the rest of decode.
