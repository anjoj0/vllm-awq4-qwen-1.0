# v6 Metadata-Staged WMMA Decode Comparison

| backend | short total t/s | short vs Triton | short vs v5 | mid total t/s | mid vs Triton | mid vs v5 | long total t/s | long vs Triton | long vs v5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| triton_fallback | 42.721 | +0.0% | +57.4% | 97.815 | +0.0% | +14.5% | 111.639 | +0.0% | +0.6% |
| hip_v4_kmajor_wmma | 26.768 | -37.3% | -1.4% | 82.893 | -15.3% | -3.0% | 110.710 | -0.8% | -0.2% |
| hip_v5_m16_wmma | 27.136 | -36.5% | +0.0% | 85.438 | -12.7% | +0.0% | 110.927 | -0.6% | +0.0% |
| hip_v6_meta_wmma | 28.690 | -32.8% | +5.7% | 87.179 | -10.9% | +2.0% | 111.506 | -0.1% | +0.5% |

## Conclusion

- v6 improves over v5 by +5.7% short, +2.0% mid, and +0.5% long total tokens/s.
- v6 remains below Triton on short (-32.8%) and mid (-10.9%), but long is nearly tied (-0.1%).
- This validates metadata staging for B dequant; next useful step is reducing B tile LDS traffic or changing the B layout to feed WMMA fragments with fewer LDS reads.
