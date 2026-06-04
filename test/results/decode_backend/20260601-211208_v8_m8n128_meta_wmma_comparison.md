# v8 M8N128 metadata WMMA decode comparison

- Metric: `avg_total_tps_e2e` from one run per scenario.
- v8 design: K-major fp16 WMMA, M<=8/N=128 metadata staging; M>8 falls back through v7.
- Note: v4-v8 require K-major decode tensors, so these are not memory-saving no-dual-storage kernels.

| Scenario | Triton | v6 | v7 | v8 | v8 vs Triton | v8 vs v7 |
|---|---:|---:|---:|---:|---:|---:|
| short_decode_128 | 42.721 | 28.690 | 30.494 | 30.629 | -28.3% | +0.4% |
| mid_prefill_512_decode_64 | 97.815 | 87.179 | 89.520 | 89.356 | -8.6% | -0.2% |
| long_prefill_2k_decode_32 | 111.639 | 111.506 | 112.615 | 111.902 | +0.2% | -0.6% |

## Readout

- v8 slightly improves short decode over v7: `30.629` vs `30.494` total tok/s, `+0.4%`.
- v8 regresses mid and long relative to v7: mid `-0.2%`, long `-0.6%`.
- The N=128 tile is not a clear replacement for v7. It may still be useful as a policy candidate for very small-M short decode, but v7 remains the better general HIP candidate.
