# v3 K-major Decode Comparison

| backend | short wall | short total t/s | mid wall | mid total t/s | long wall | long total t/s |
|---|---:|---:|---:|---:|---:|---:|
| triton_fallback | 8.614 | 42.721 (0.0%) | 23.074 | 97.815 (0.0%) | 78.405 | 111.639 (0.0%) |
| hip_v2_smallm | 64.845 | 5.675 (-86.7%) | 69.067 | 32.678 (-66.6%) | 107.786 | 81.208 (-27.3%) |
| hip_v3_kmajor | 56.258 | 6.541 (-84.7%) | 62.382 | 36.18 (-63.0%) | 103.868 | 84.27 (-24.5%) |

## Conclusion

- v3 K-major decode is correct but much slower than Triton fallback and v2 end-to-end.
- The bottleneck is not just native layout coalescing; the small-M HIP decode still lacks a tensor-core/WMMA-style structure comparable to TritonW4A16.
- Do not ship v3 as the default. Keep Triton fallback for decode until a real fused/decode-specialized kernel is implemented.
