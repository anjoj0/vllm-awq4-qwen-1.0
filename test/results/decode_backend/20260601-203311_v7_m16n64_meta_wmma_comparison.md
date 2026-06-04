# v7 M16N64 metadata WMMA decode comparison

- Metric: `avg_total_tps_e2e` from one run per scenario.
- v7 design: K-major fp16 WMMA, M<=16/N=64 tile, metadata staging; M>16 falls back to v4 launcher path.
- Note: v4-v7 require K-major decode tensors, so these are not memory-saving no-dual-storage kernels despite some older filenames.

| Scenario | Triton | v4 | v5 | v6 | v7 | v7 vs Triton | v7 vs v6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_decode_128 | 42.721 | 26.768 | 27.136 | 28.690 | 30.494 | -28.6% | +6.3% |
| mid_prefill_512_decode_64 | 97.815 | 82.893 | 85.438 | 87.179 | 89.520 | -8.5% | +2.7% |
| long_prefill_2k_decode_32 | 111.639 | 110.710 | 110.927 | 111.506 | 112.615 | +0.9% | +1.0% |

## Readout

- v7 is the first pure HIP path in this series with a measured win over Triton on the long-prefill case: `112.615` vs `111.639` total tok/s, `+0.9%`.
- v7 recovers most of the v6 regression: short `+6.3%` vs v6, mid `+2.7%` vs v6, long `+1.0%` vs v6.
- Short and mid decode are still meaningfully below Triton, so the HIP path is not globally faster yet.
