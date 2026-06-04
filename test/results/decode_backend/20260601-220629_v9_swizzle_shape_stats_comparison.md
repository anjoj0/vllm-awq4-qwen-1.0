# v9 LDS B swizzle + shape stats comparison

- Metric: `avg_total_tps_e2e` from one run per scenario.
- v9 design: same M<=16/N=64 metadata-staged shape as v7, but B tile uses `n ^ ((kk & 0x3) << 4)` in LDS.
- Shape stats were collected with `AWQ_MMQ_SHAPE_STATS=1`.

| Scenario | Triton | v7 | v8 | v9 | v9 vs Triton | v9 vs v7 |
|---|---:|---:|---:|---:|---:|---:|
| short_decode_128 | 42.721 | 30.494 | 30.629 | 29.236 | -31.6% | -4.1% |
| mid_prefill_512_decode_64 | 97.815 | 89.520 | 89.356 | 87.392 | -10.7% | -2.4% |
| long_prefill_2k_decode_32 | 111.639 | 112.615 | 111.902 | 111.024 | -0.6% | -1.4% |

## Shape Stats

- Total AWQ linear calls recorded: `19968`
- `decode` calls: `18688`
- `prefill` calls: `1280`

| Route/M | Count |
|---|---:|
| decode M=9 | 18432 |
| decode M=1 | 256 |
| prefill M=289 | 256 |
| prefill M=536 | 256 |
| prefill M=2193 | 256 |
| prefill M=8185 | 256 |
| prefill M=8192 | 256 |

| Decode N,K | Count |
|---|---:|
| N=5120, K=6144 | 4672 |
| N=5120, K=17408 | 4672 |
| N=34816, K=5120 | 4672 |
| N=16384, K=5120 | 3504 |
| N=14336, K=5120 | 1168 |

## Readout

- DFlash target verification is dominated by `decode M=9`: `18432 / 18688` decode calls. This matches `num_speculative_tokens=8` plus the current token.
- v8 M<=8 misses the dominant DFlash decode shape, which explains why its M<=8 specialization did not become the main winner.
- v9 swizzle is a clear regression versus v7 across all three benchmark cases, so this swizzle should not replace v7.
- The next HIP-specific shape to test, if any, should target exactly `M=9` or `M<=9`, not M<=8.
