# A/B Restartability Check: DFlash N=3 and MBT=16384 - 2026-06-05
## Baseline Reference

The previous fast result came from `test/results/bench_matrix_patch20_21_retest_20260605.json`: `N=4`, `MBT=8192`, `KV=fp8`, `max_model_len=65536`, with 32K decode `7.33 t/s`. That container was already running before this A/B cycle.
## Attempts

| attempt | KV dtype | N | MBT | startup | attention block | result |
|---|---|---:|---:|---|---:|---|
| N3_MBT8192_fp8 | fp8 | 3 | 8192 | failed | 1600 | `AssertionError` in `unify_kv_cache_spec_page_size` |
| N4_MBT16384_fp8 | fp8 | 4 | 16384 | failed | 1616 | same page-size assertion |
| N4_MBT8192_fp8_recreate | fp8 | 4 | 8192 | failed | 1616 | same page-size assertion after force-recreate |

## Restored Service

| field | value |
|---|---|
| status | ready |
| DFlash N | 4 |
| MBT | 8192 |
| KV dtype | auto |
| attention block size | 816 |
| GPU KV cache | 92,208 tokens |
| max concurrency for 65,536 tokens | 3.73x |

## Conclusion

The two requested candidates cannot be benchmarked under the fp8-KV baseline because they fail during KV cache initialization. More importantly, after force-recreate, even the previous `N=4, MBT=8192, KV=fp8` runtime fails with the same page-size assertion. This means the earlier fp8 speed number is a valid measurement of the already-running container, but it is not a restartable/deployable configuration until the hybrid page-size unification issue is fixed.

For now, the safe restartable configuration is `N=4, MBT=8192, KV=auto`. Do not select `N=3` or `MBT=16384` as the best config on the current code path.
