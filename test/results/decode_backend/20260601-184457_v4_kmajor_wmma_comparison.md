# v4 K-major WMMA Decode Comparison

| backend | short wall | short total t/s | mid wall | mid total t/s | long wall | long total t/s |
|---|---:|---:|---:|---:|---:|---:|
| triton_fallback | 8.614 | 42.721 (+0.0%) | 23.074 | 97.815 (+0.0%) | 78.405 | 111.639 (+0.0%) |
| hip_v2_smallm | 64.845 | 5.675 (-86.7%) | 69.067 | 32.678 (-66.6%) | 107.786 | 81.208 (-27.3%) |
| hip_v3_kmajor | 56.258 | 6.541 (-84.7%) | 62.382 | 36.180 (-63.0%) | 103.868 | 84.270 (-24.5%) |
| hip_v4_kmajor_wmma | 13.748 | 26.768 (-37.3%) | 27.228 | 82.893 (-15.3%) | 79.063 | 110.710 (-0.8%) |

## Conclusion

- v4 K-major WMMA decode is the first HIP decode path to substantially close the gap to TritonW4A16.
- It improves over v3 by +309.2% short, +129.1% mid, and +31.4% long total tokens/s.
- It still trails Triton by -37.3% short and -15.3% mid total tokens/s; long is effectively tied (-0.8%) because prefill dominates.
- Keep Triton fallback as default. Use v4 as the next optimization baseline, focusing on LDS layout, vectorized unpack/dequant, and fewer idle lanes for M < 16.
